# coding=utf-8
# Copyright 2023 Microsoft, clefourrier The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch GraphMert model."""


import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F


from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithNoAttention,
    SequenceClassifierOutput,
    MultipleChoiceModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers import logging

from graphmert_model import GraphMertConfig
from graphmert_model import GraphMertDataCollator


logger = logging.get_logger(__name__)


def quant_noise(module, p, block_size):
    """
    From:
    https://github.com/facebookresearch/fairseq/blob/dd0079bde7f678b0cd0715cbd0ae68d661b7226d/fairseq/modules/quant_noise.py

    Wraps modules and applies quantization noise to the weights for subsequent quantization with Iterative Product
    Quantization as described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights, see "And the Bit Goes Down:
          Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper which consists in randomly dropping
          blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    if not isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
        raise NotImplementedError("Module unsupported for quant_noise.")

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        if module.weight.size(1) % block_size != 0:
            raise AssertionError("Input features must be a multiple of block sizes")

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            if module.in_channels % block_size != 0:
                raise AssertionError("Input channels must be a multiple of block sizes")
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            if k % block_size != 0:
                raise AssertionError("Kernel size must be a multiple of block size")

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = torch.zeros(in_features // block_size * out_features, device=weight.device)
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(weight.size(0), weight.size(1), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])

            # scale weights and apply mask
            mask = mask.to(torch.bool)  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module


class LayerDropModuleList(nn.ModuleList):
    """
    From:
    https://github.com/facebookresearch/fairseq/blob/dd0079bde7f678b0cd0715cbd0ae68d661b7226d/fairseq/modules/layer_drop.py
    A LayerDrop implementation based on [`torch.nn.ModuleList`]. LayerDrop as described in
    https://arxiv.org/abs/1909.11556.

    We refresh the choice of which layers to drop every time we iterate over the LayerDropModuleList instance. During
    evaluation we always iterate over all layers.

    Usage:

    ```python
    layers = LayerDropList(p=0.5, modules=[layer1, layer2, layer3])
    for layer in layers:  # this might iterate over layers 1 and 3
        x = layer(x)
    for layer in layers:  # this might iterate over all layers
        x = layer(x)
    for layer in layers:  # this might not iterate over any layers
        x = layer(x)
    ```

    Args:
        p (float): probability of dropping out each layer
        modules (iterable, optional): an iterable of modules to add
    """

    def __init__(self, p, modules=None):
        super().__init__(modules)
        self.p = p

    def __iter__(self):
        dropout_probs = torch.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if not self.training or (dropout_probs[i] > self.p):
                yield m


class GraphMertGraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.atom_encoder = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        self.root_nodes = config.root_nodes
        self.num_leaves = config.max_nodes // config.root_nodes - 1

        self.max_positions = 128 # maximum number of leaf token or head token in leaves and head spans

        self.relation_dim_projection = config.hidden_size
        self.relation_matrix_encoder = nn.Embedding(config.num_relationships + 1, config.hidden_size * self.relation_dim_projection, padding_idx=0)
        self.relation_encoder = nn.Embedding(config.num_relationships + 1, 2 * config.hidden_size, padding_idx=0)
        self.rel_embedding_dropout = nn.Dropout(p=config.relation_emb_dropout)

        self.graph_token = nn.Embedding(1, config.hidden_size)


    def forward(self, input_nodes, leaf_relationships, head_lengths):

        """leaf nodes encoded with relations based on https://ieeexplore.ieee.org/document/9679113"""
        atom_encoder = self.atom_encoder(input_nodes) # [n_graph, n_node, n_hidden]
        node_feature = atom_encoder.sum(dim=-2)
        n_graph, n_node, hidden_dim = node_feature.size()

        if leaf_relationships is not None: # in downstream tasks we may not have relations tensor
            # find non-empty <h, r, t> -- head, rel, tail
            # and run H-GAT on them
            rel_idx = torch.nonzero(leaf_relationships)
            if rel_idx.size(0) != 0:
                # leaf_idx & rel_idx counts for input_nodes[:, self.root_nodes: ] !!!
                head_starts = torch.nonzero(head_lengths)
                head_span_lengths = head_lengths[head_starts[:, 0], head_starts[:, 1]]
                heads = self._extract_spans_padded(input_nodes[:, :self.root_nodes, :], head_starts, head_span_lengths)
                try:
                    tails, tail_span_lengths, tail_starts = self._extract_tail_spans_padded(input_nodes)

                    new_heads = [0] * heads.size(0)
                    for i, head in enumerate(heads):
                        new_heads[i] = head[:head_span_lengths[i], :]
                    heads = new_heads

                    new_tails = [0] * tails.size(0)
                    for i, tail in enumerate(tails):
                        new_tails[i] = tail[:tail_span_lengths[i], :]
                    tails = new_tails

                    if len(tails) != len(heads):
                        logger.error(
                            f'ERROR: len(heads)={len(heads)}; len(tails)={len(tails)}; they must be equal'
                            f'heads: {heads}'
                            f'tails: {tails}'
                            f'roots: {input_nodes[:, :self.root_nodes, :]}'
                        )
                    else:
                        rels = leaf_relationships[rel_idx[:, 0], rel_idx[:, 1]].to(torch.long)
                        rel_embeds = self.relation_encoder(rels)
                        rel_embeds = self.rel_embedding_dropout(rel_embeds)
                        matrix_rel_embeds = self.relation_matrix_encoder(rels)
                        matrix_rel_embeds = self.rel_embedding_dropout(matrix_rel_embeds)
                        matrix_rel_embeds = matrix_rel_embeds.view(len(rels), hidden_dim, self.relation_dim_projection)

                        # list of tensors with shape (num tail tokens, num head tokens, 2)
                        heads = [self.atom_encoder(entry) for entry in heads]
                        tails = [self.atom_encoder(entry) for entry in tails]
                        
                        combined_embeds = self._list_cartesian_products(tails, heads)
                        heads = [entry.squeeze(1) for entry in heads]

                    
                        for triple_idx, triple_embed in enumerate(combined_embeds):
                            # triple_embed.shape: (num tail tokens, num head tokens, 2, hidden_dim)
                            num_tails, num_heads = triple_embed.size(0), triple_embed.size(1)
                            relation_embed = rel_embeds[triple_idx]
                            matrix_rel_embed = matrix_rel_embeds[triple_idx]
                            matrix = matrix_rel_embed.unsqueeze(0).expand(num_heads, hidden_dim, self.relation_dim_projection)
                            triple_embed = triple_embed.transpose(2, 3) # (num tail tokens, num head tokens, hidden_dim, 2)

                            for tail_idx in range(num_tails):
                                tail = torch.bmm(matrix, triple_embed[tail_idx]) # num head tokens, self.relation_dim_projection, 
                                tail = tail.view(num_heads, 2 * hidden_dim)
                                tail = torch.matmul(relation_embed, tail.T) # num heads
                                tail = F.leaky_relu(tail, negative_slope=0.01)
                                alphas = F.softmax(tail, dim=0) # num head tokens
                                head_embed = heads[triple_idx].transpose(0, 1)
                                final_embed = torch.matmul(matrix_rel_embed, head_embed) # W_r * h_i
                                final_embed = torch.matmul(final_embed, alphas) # sum alpha(r)_i * (W_r * h_i)
                                
                                tail_start = tail_starts[triple_idx]
                                node_feature[tail_start[0], tail_start[1] + tail_idx + self.root_nodes] += final_embed

                except Exception as e:
                    logger.error(
                        f'ERROR: Error in GraphMertGraphNodeFeature: {e}'
                        f'rel_idx: {rel_idx}'
                        f'rel_idx.size: {rel_idx.size}'
                        f'leaf_idx: {torch.nonzero(input_nodes[:, self.root_nodes:, :][..., 0] != self.pad_token_id)}'
                        f'you may ignore it if this is a rare error'
                    )
        
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        return graph_node_feature
    

    def _extract_spans_padded(self, nodes, starts, span_lengths):
        """
        returns: 
        all span tokens padded to max_len
        max_len: max_len
        """
        batch_idxs = starts[:, 0]                # (N,)
        start_idxs = starts[:, 1]                # (N,)
        max_len = int(span_lengths.max().item())  # length to pad to  

        offsets = torch.arange(max_len, device=nodes.device).unsqueeze(0)  # (1, max_len)
        pos_idxs = start_idxs.unsqueeze(1) + offsets # (N, max_len)
        batch_idxs_expanded = batch_idxs.unsqueeze(1).expand(-1, max_len)  # (N, max_len)

        # safeguard: clamp to legal range just in case (optional)
        L = nodes.size(1)
        pos_idxs = pos_idxs.clamp(0, L-1)

        spans_padded = nodes[batch_idxs_expanded, pos_idxs, :]
        # mask out the “overshoot” positions so they become zero
        mask = (offsets < span_lengths.unsqueeze(1)).unsqueeze(2)  # (N, max_len, 1)
        spans_padded = spans_padded * mask # Now `spans_padded[i]` contains exactly input_nodes[b_i, s_i:e_i, :] in its first `lengths[i]` rows
        spans_padded = torch.where(spans_padded != 0, spans_padded, self.pad_token_id)
        return spans_padded


    def _extract_tail_spans_padded(self, input_nodes):
        """
        From `input_nodes` of shape [B, N, F], extract all “leaf” spans 
        starting at positions where (pos % span_size)==0 in the sub‐tensor
        A span ends just before the next span start (or at the very last leaf).
        Returns:
        tails_padded: Tensor of shape [S, max_span_len, F]
        max_len
        """
        nodes = input_nodes[:, self.root_nodes:, :]                # [B, L, F]
        leaf_idx = torch.nonzero(nodes[..., 0] != self.pad_token_id)        # [M, 2] pairs (batch, pos)

        first_leaf = int(leaf_idx[0, 1])
        if first_leaf % self.num_leaves != 0:
            raise ValueError(
                f"Bad input: first leaf at pos={first_leaf} is not divisible by {self.num_leaves}"
                f"Either the leaf sequence is empty, or its contiogiuos and fills leaves starting with the 1st leaf "
                f"num_leaves={self.num_leaves}. Upstream data mis-alignment. "
                f"Check the dataset preprocessing pipeline. You can ignore this message if this err is rare"
            )

        start_mask = (leaf_idx[:, 1] % self.num_leaves) == 0                   # [M]
        end_mask   = torch.cat([start_mask[1:], torch.tensor([True], device=nodes.device)])
        tail_starts = leaf_idx[start_mask]                         # [S, 2]
        tail_ends   = leaf_idx[end_mask]                           # [S, 2]

        start_idxs  = tail_starts[:, 1]                            # [S]
        end_idxs    = tail_ends[:,   1] + 1                         # [S]  (+1 because exclusive)
        tail_span_lengths = end_idxs - start_idxs                       # [S]

        tails_padded = self._extract_spans_padded(nodes, tail_starts, tail_span_lengths) # [S, max_len, F]
        return tails_padded, tail_span_lengths, tail_starts


    def _list_cartesian_products(self, tails: list, heads: list) -> list:
        """
        Takes cartesian product of list entries and concats them.
        len(tails) == len(heads) -- one tail for each head.
        For each pair (t, h) in tails, heads produce a tensor of shape (len(t), len(h), 2).

        Example:
        tails = [tensor([[4],[5]])]
        heads = [tensor([[1],[2],[3]])]

        Output:
        [
            tensor([
                [[4, 1],[4, 2],[4, 3]],
                [[5, 1],[5, 2],[5, 3]]
            ])
        ]
        row index: tail tokens, column index: head tokens
        """
        results = []
        for t, h in zip(tails, heads):
            T = t.size(0)
            H = h.size(0)
            t_expanded = t.unsqueeze(1).expand(T, H, 1, -1)
            h_expanded = h.unsqueeze(0).expand(T, H, 1, -1)

            # Concatenate along last dimension -> (T,H,2,-1)
            combined = torch.cat([t_expanded, h_expanded], dim=-2)
            results.append(combined)
        return results


class GraphMertMultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.kdim = config.kdim if config.kdim is not None else config.hidden_size
        self.vdim = config.vdim if config.vdim is not None else config.hidden_size
        self.qkv_same_dim = self.kdim == config.hidden_size and self.vdim == config.hidden_size

        self.num_heads = config.num_attention_heads
        self.dropout_module = torch.nn.Dropout(p=config.dropout, inplace=False)

        self.head_dim = config.hidden_size // config.num_attention_heads
        if not (self.head_dim * config.num_attention_heads == self.hidden_size):
            raise AssertionError("The hidden_size must be divisible by num_heads.")
        self.scaling = self.head_dim**-0.5

        self.self_attention = True  # config.self_attention
        if not (self.self_attention):
            raise NotImplementedError("The GraphMert model only supports self attention for now.")
        if self.self_attention and not self.qkv_same_dim:
            raise AssertionError("Self-attention requires query, key and value to be of the same size.")

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, config.hidden_size, bias=config.bias),
            config.q_noise,
            config.qn_block_size,
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, config.hidden_size, bias=config.bias),
            config.q_noise,
            config.qn_block_size,
        )
        self.q_proj = quant_noise(
            nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias),
            config.q_noise,
            config.qn_block_size,
        )
        self.out_proj = quant_noise(
            nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias),
            config.q_noise,
            config.qn_block_size,
        )

        self.onnx_trace = False

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        decay_mask: Optional[torch.Tensor],
        attn_bias: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            key_padding_mask (Bytetorch.Tensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attention_mask (Bytetorch.Tensor, optional): typically used to
                implement causal attention, where the mask prevents the attention from looking forward in time
                (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default: return the average attention weights over all
                heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, hidden_size = query.size()
        src_len = tgt_len
        if not (hidden_size == self.hidden_size):
            raise AssertionError(
                f"The query embedding dimension {hidden_size} is not equal to the expected hidden_size"
                f" {self.hidden_size}."
            )
        if not (list(query.size()) == [tgt_len, bsz, hidden_size]):
            raise AssertionError("Query size incorrect in GraphMert, compared to model dimensions.")

        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                if (key_bsz != bsz) or (value is None) or not (src_len, bsz == value.shape[:2]):
                    raise AssertionError(
                        "The batch shape does not match the key or value shapes provided to the attention."
                    )

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)

        q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if (k is None) or not (k.size(1) == src_len):
            raise AssertionError("The shape of the key generated in the attention is incorrect")

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            if key_padding_mask.size(0) != bsz or key_padding_mask.size(1) != src_len:
                raise AssertionError(
                    "The shape of the generated padding mask for the key does not match expected dimensions."
                )
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        if list(attn_weights.size()) != [bsz * self.num_heads, tgt_len, src_len]:
            raise AssertionError("The attention weights generated do not match the expected dimensions.")

        if attn_bias is not None:
            attn_weights = attn_weights + attn_bias
        if decay_mask is not None:
            attn_weights = attn_weights * decay_mask

        if attention_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            # that would mask all the att btw pad tokens and other tokens
            attention_mask = attention_mask.to(torch.bool)
            query_mask = attention_mask.unsqueeze(2)  # [bsz, src_len, 1]
            key_mask = attention_mask.unsqueeze(1)    # [bsz, 1, src_len]
            combined_mask = ~(query_mask & key_mask)  # [bsz, src_len, src_len]
            combined_mask[:, 1:, 0] = True  # other tokens cannot attend graph token
            attention_mask = combined_mask.unsqueeze(1).expand(bsz, self.num_heads, tgt_len, src_len)

            attn_weights = attn_weights.masked_fill(attention_mask, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = torch.nn.functional.softmax(attn_weights, dim=-1)
        # replace nan values with 0
        attn_weights_float = torch.where(torch.isnan(attn_weights_float), torch.zeros_like(attn_weights_float), attn_weights_float)

        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        if v is None:
            raise AssertionError("No value generated")
        attn = torch.bmm(attn_probs, v)
        if list(attn.size()) != [bsz * self.num_heads, tgt_len, self.head_dim]:
            raise AssertionError("The attention generated do not match the expected dimensions.")

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, hidden_size)
        attn = self.out_proj(attn)

        attn_weights = None
        if need_weights:
            attn_weights = attn_weights_float.contiguous().view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights
        

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights


class GraphMertGraphEncoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        # Initialize parameters
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_dropout = config.attention_dropout
        self.q_noise = config.q_noise
        self.qn_block_size = config.qn_block_size
        self.pre_layernorm = config.pre_layernorm

        self.dropout_module = torch.nn.Dropout(p=config.dropout, inplace=False)

        self.activation_dropout_module = torch.nn.Dropout(p=config.dropout, inplace=False)

        # Initialize blocks
        self.activation_fn = ACT2FN[config.activation_fn]
        self.self_attn = GraphMertMultiheadAttention(config)

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.hidden_size)

        self.fc1 = self.build_fc(
            self.hidden_size,
            config.intermediate_size,
            q_noise=config.q_noise,
            qn_block_size=config.qn_block_size,
        )
        self.fc2 = self.build_fc(
            config.intermediate_size,
            self.hidden_size,
            q_noise=config.q_noise,
            qn_block_size=config.qn_block_size,
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.hidden_size)


    def build_fc(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def forward(
        self,
        input_nodes: torch.Tensor,
        decay_mask: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
        self_attention_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        nn.LayerNorm is applied either before or after the self-attention/ffn modules similar to the original
        Transformer implementation.
        """
        residual = input_nodes
        n_node, n_graph = input_nodes.size()[:2]
        if self.pre_layernorm:
            input_nodes = self.self_attn_layer_norm(input_nodes)

        input_nodes, attn = self.self_attn(
            query=input_nodes,
            key=input_nodes,
            value=input_nodes,
            attn_bias=self_attn_bias,
            decay_mask=decay_mask,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attention_mask=self_attention_mask,
        )
        input_nodes = self.dropout_module(input_nodes)
        input_nodes = residual + input_nodes
        if not self.pre_layernorm:
            input_nodes = self.self_attn_layer_norm(input_nodes)

        residual = input_nodes
        if self.pre_layernorm:
            input_nodes = self.final_layer_norm(input_nodes)
        input_nodes = self.activation_fn(self.fc1(input_nodes))
        input_nodes = self.activation_dropout_module(input_nodes)
        input_nodes = self.fc2(input_nodes)
        input_nodes = self.dropout_module(input_nodes)
        input_nodes = residual + input_nodes
        if not self.pre_layernorm:
            input_nodes = self.final_layer_norm(input_nodes)

        return input_nodes, attn


class GraphMertGraphEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dropout_module = torch.nn.Dropout(p=config.dropout, inplace=False)
        self.layerdrop = config.layerdrop
        self.hidden_size = config.hidden_size
        self.apply_graphmert_init = config.apply_graphmert_init
        self.traceable = config.traceable

        self.graph_node_feature = GraphMertGraphNodeFeature(config)

        self.embed_scale = config.embed_scale

        if config.q_noise > 0:
            self.quant_noise = quant_noise(
                nn.Linear(self.hidden_size, self.hidden_size, bias=False),
                config.q_noise,
                config.qn_block_size,
            )
        else:
            self.quant_noise = None

        if config.encoder_normalize_before:
            self.emb_layer_norm = nn.LayerNorm(self.hidden_size)
        else:
            self.emb_layer_norm = None

        if config.pre_layernorm:
            self.final_layer_norm = nn.LayerNorm(self.hidden_size)

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend([GraphMertGraphEncoderLayer(config) for _ in range(config.num_hidden_layers)])

        # Apply initialization of model params after building the model
        if config.freeze_embeddings:
            raise NotImplementedError("Freezing embeddings is not implemented yet.")

        for layer in range(config.num_trans_layers_to_freeze):
            m = self.layers[layer]
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        self.graph_token_virtual_distance = nn.Parameter(torch.ones(1, config.max_nodes))

        self.sp = nn.Parameter(torch.ones(config.num_attention_heads, 1, 1))
        self.exp_mask_base = config.exp_mask_base
        self.sp_activation = nn.GELU()

        self.num_attention_heads = config.num_attention_heads


    def forward(
        self,
        input_nodes,
        spatial_pos,
        attention_mask: Optional[torch.Tensor] = None,
        leaf_relationships: Optional[torch.Tensor] = None,
        head_lengths: Optional[torch.Tensor] = None,
        perturb=None,
        last_state_only: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.torch.Tensor, torch.Tensor]:
        n_graph, n_node = input_nodes.size()[:2]

        # Add class token for attention mask
        attention_mask_cls = torch.ones(n_graph, 1, device=attention_mask.device, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask_cls, attention_mask), dim=1)


        if token_embeddings is not None:
            input_nodes = token_embeddings
        else:
            input_nodes = self.graph_node_feature(input_nodes, leaf_relationships, head_lengths)


        if self.exp_mask_base is None:
            decay_mask = None
        else:
            n_graph, n_node = input_nodes.size()[:2]
            virtual_token_dist_to_itself = torch.tensor([[0]]).to(self.graph_token_virtual_distance.device)
            # we need the distance to be positve to take decimal powers
            graph_token_virtual_distance = nn.functional.softplus(self.graph_token_virtual_distance, threshold=1)
            t = torch.cat([virtual_token_dist_to_itself, graph_token_virtual_distance], dim=1)
            t = t.unsqueeze(0).expand(n_graph, 1, n_node)
            spatial_pos = torch.cat([t[:, :, 1:], spatial_pos], dim=1)
            spatial_pos = torch.cat([t.transpose(1, 2), spatial_pos], dim=2)
            spatial_pos = spatial_pos.unsqueeze(1).expand(n_graph, self.num_attention_heads, n_node, n_node)

            spatial = spatial_pos ** 0.5
            decay_mask = self.exp_mask_base ** self.sp_activation(spatial - self.sp)
            decay_mask = decay_mask.contiguous().view(n_graph * self.num_attention_heads, n_node, n_node)

        if perturb is not None:
            input_nodes[:, 1:, :] += perturb

        if self.embed_scale is not None:
            input_nodes = input_nodes * self.embed_scale

        if self.quant_noise is not None:
            input_nodes = self.quant_noise(input_nodes)

        if self.emb_layer_norm is not None:
            input_nodes = self.emb_layer_norm(input_nodes)

        input_nodes = self.dropout_module(input_nodes)

        input_nodes = input_nodes.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(input_nodes)

        for layer in self.layers:
            input_nodes, _ = layer(
                input_nodes,
                self_attn_padding_mask=None,
                self_attention_mask=attention_mask,
                self_attn_bias=None,
                decay_mask=decay_mask,
            )
            if not last_state_only:
                inner_states.append(input_nodes)

        graph_rep = input_nodes[0, :, :]

        if last_state_only:
            inner_states = [input_nodes]

        if self.traceable:
            return torch.stack(inner_states), graph_rep
        else:
            return inner_states, graph_rep


class GraphMertDecoderHead(nn.Module):
    def __init__(self, activation_fn, hidden_size, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.decoder = nn.Linear(hidden_size, num_labels)
        self.bias = nn.Parameter(torch.zeros(num_labels))
        self.decoder.bias = self.bias
        self.activation_fn = ACT2FN[activation_fn]


    def forward(self, input_nodes, **unused):
        input_nodes = self.dense(input_nodes)
        input_nodes = self.activation_fn(input_nodes)
        input_nodes = self.layer_norm(input_nodes)
        input_nodes = self.decoder(input_nodes)

        return input_nodes

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        # For accelerate compatibility and to not break backward compatibility
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias


class GraphMertMLPWithLayerNorm(nn.Module):
    def __init__(self, input_size, activation_fn, hidden_size):
        super().__init__()
        self.mlp_layer_norm = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            ACT2FN[activation_fn],
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            ACT2FN[activation_fn],
            nn.LayerNorm(hidden_size),
        )


    def forward(self, pairs_hidden):
        return self.mlp_layer_norm(pairs_hidden)


class GraphMertPairDecoderHead(nn.Module):
    """
    Decoder head for Span Boundary Objective
    """
    def __init__(self, activation_fn, hidden_size, num_labels, max_targets, position_embedding_size=None):
        super().__init__()
        self.max_targets = max_targets

        if position_embedding_size is None:
            position_embedding_size = hidden_size
        self.position_embeddings = nn.Embedding(max_targets, position_embedding_size)
        self.mlp_layer_norm = GraphMertMLPWithLayerNorm(hidden_size * 2 + position_embedding_size, activation_fn, hidden_size)
        self.decoder = nn.Linear(hidden_size, num_labels)

        # we don't use implement bias tying to avoid serialization issues when saving
        # it may be implemented later
        # self.bias = nn.Parameter(torch.zeros(num_labels))
        # self.decoder.bias = self.bias


    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        # For accelerate compatibility and to not break backward compatibility
        # GraphMertPairDecoderHead doesn't use bias tying, so nothing to do here
        pass

    
    def forward(self, input_nodes, pairs, **unused):
        bs, num_pairs, _ = pairs.size()
        bs, n_node, dim = input_nodes.size()
        # pair indices: (bs, num_pairs)
        left, right = pairs[:, :, 0], pairs[:, :, 1]
        # +1: shift because of graph token at the postion 0 concatenated after creating pairs, 
        left, right = left + 1, right + 1

        # (bs, num_pairs, dim)
        left_hidden = torch.gather(input_nodes, 1, left.unsqueeze(2).repeat(1, 1, dim))
        # pair states: bs * num_pairs, max_targets, dim
        left_hidden = left_hidden.contiguous().view(bs * num_pairs, dim).unsqueeze(1).repeat(1, self.max_targets, 1)
        right_hidden = torch.gather(input_nodes, 1, right.unsqueeze(2).repeat(1, 1, dim))
        # bs * num_pairs, max_targets, dim
        right_hidden = right_hidden.contiguous().view(bs * num_pairs, dim).unsqueeze(1).repeat(1, self.max_targets, 1)
        # (max_targets, dim)
        position_embeddings = self.position_embeddings.weight
        pair_embeddings_with_positions = torch.cat((left_hidden, right_hidden, position_embeddings.unsqueeze(0).repeat(bs * num_pairs, 1, 1)), -1)
        pair_scores = self.mlp_layer_norm(pair_embeddings_with_positions)
        # target scores : bs * num_pairs, max_targets, vocab_size
        pair_scores = self.decoder(pair_scores)
        return pair_scores



class GraphMertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GraphMertConfig
    base_model_prefix = "graphmert"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    main_input_name_nodes = "input_nodes"
    main_input_name_edges = "input_edges"

    def normal_(self, data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    def init_graphmert_params(self, module):
        """
        Initialize the weights specific to the GraphMert Model.
        """
        if isinstance(module, nn.Linear):
            self.normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            self.normal_(module.weight.data)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, GraphMertMultiheadAttention):
            self.normal_(module.q_proj.weight.data)
            self.normal_(module.k_proj.weight.data)
            self.normal_(module.v_proj.weight.data)

    def _init_weights(self, module):
        """
        Initialize the weights
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # We might be missing part of the Linear init, dependant on the layer num
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, GraphMertMultiheadAttention):
            module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.v_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.reset_parameters()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, GraphMertGraphEncoder):
            if module.apply_graphmert_init:
                module.apply(self.init_graphmert_params)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GraphMertModel):
            module.gradient_checkpointing = value

    def update_keys_to_ignore(self, config, del_keys_to_ignore):
        """Remove some keys from ignore list"""
        if not config.tie_word_embeddings:
            # must make a new list, or the class variable gets modified!
            self._keys_to_ignore_on_save = [k for k in self._keys_to_ignore_on_save if k not in del_keys_to_ignore]
            self._keys_to_ignore_on_load_missing = [
                k for k in self._keys_to_ignore_on_load_missing if k not in del_keys_to_ignore
            ]


class GraphMertModel(GraphMertPreTrainedModel):
    """The GraphMert model is a graph-encoder model.

    It goes from a graph to its representation. If you want to use the model for a downstream classification task, use
    GraphMertForGraphClassification instead. For any other downstream task, feel free to add a new class, or combine
    this model with a downstream model of your choice, following the example in GraphMertForGraphClassification.
    """
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.root_nodes = config.root_nodes
        self.graph_types = config.graph_types

        self.graph_encoder = GraphMertGraphEncoder(config)

        if config.fixed_graph_architecture:
            # Get tensors for the fixed architecture
            # TD del it later
            graph_types_no_kg = set(self.graph_types).difference({'amr', 'conceptnet', 'wikipedia'})
            data_collator = GraphMertDataCollator(config, None, graph_types_no_kg, process_arch_tensors=True)
            item = data_collator.preprocess_items({'input_ids': [[0] * self.root_nodes]})
            item = {k: v[0] for k, v in item.items()}

            # Set architecture tensors
            self.spatial_pos = torch.tensor(item["spatial_pos"], dtype=torch.long).unsqueeze(0)

            # Batch size used to make architecture tensors
            self.batch_size = 1

        self.lm_output_learned_bias = None

        # Remove head is set to true during fine-tuning
        self.load_softmax = not getattr(config, "remove_head", False)

        self.lm_head_transform_weight = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation_fn = ACT2FN[config.activation_fn]
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        self.post_init()

    def update_arch_tensors(self, input_nodes):
        batch_size = input_nodes.shape[0]
        
        self.spatial_pos = self.spatial_pos[0].unsqueeze(0).expand(batch_size, -1, -1)
        self.batch_size = batch_size

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))

    def get_input_embeddings(self):
        return self.graph_encoder.graph_node_feature.atom_encoder

    def set_input_embeddings(self, value):
        self.graph_encoder.graph_node_feature.atom_encoder = value

    def forward(
        self,
        input_nodes,
        spatial_pos: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        leaf_relationships: Optional[torch.Tensor] = None,
        head_lengths: Optional[torch.Tensor] = None,
        perturb=None,
        masked_tokens=None,
        return_dict: Optional[bool] = None,
        **unused
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        device = input_nodes.device
        self.spatial_pos = self.spatial_pos.to(device)
        if input_nodes.shape[0] != self.batch_size:
            # Update architecture tensors
            self.update_arch_tensors(input_nodes)


        inner_states, graph_rep = self.graph_encoder(
            input_nodes, 
            spatial_pos if spatial_pos is not None else self.spatial_pos, 
            attention_mask=attention_mask,
            leaf_relationships=leaf_relationships,
            head_lengths=head_lengths,
            perturb=perturb, 
        )

        # last inner state, then revert Batch and Graph len
        input_nodes = inner_states[-1].transpose(0, 1)

        # project masked tokens only
        if masked_tokens is not None:
            raise NotImplementedError

        input_nodes = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(input_nodes)))

        if not return_dict:
            return tuple(x for x in [input_nodes, inner_states] if x is not None)
        return BaseModelOutputWithNoAttention(last_hidden_state=input_nodes, hidden_states=inner_states)
    

    def max_nodes(self):
        """Maximum output length supported by the encoder."""
        return self.max_nodes


class GraphMertForMaskedLM(GraphMertPreTrainedModel):
    """
    This model can be used for graph-level masked language modeling.
    """
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    # _keys_to_ignore_on_save = [r"lm_pair_head.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.graphmert = GraphMertModel(config)
        self.activation_fn = config.activation_fn
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.lm_head = GraphMertDecoderHead(self.activation_fn, self.hidden_size, self.vocab_size)
        max_targets = config.span_upper_length if config.span_upper_length is not None else config.max_nodes // config.root_nodes - 1
        self.use_sbo = getattr(config, "mlm_sbo", True)

        if self.use_sbo:
            max_targets = config.span_upper_length if config.span_upper_length is not None else config.max_nodes // config.root_nodes - 1
            self.lm_pair_head = GraphMertPairDecoderHead(self.activation_fn, self.hidden_size, self.vocab_size, max_targets=max_targets)
        else:
            self.lm_pair_head = None

        self.is_encoder_decoder = True
        # The LM head weights require special treatment only when they are tied with the word embeddings
        # self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_nodes,
        spatial_pos: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        leaf_relationships: Optional[torch.Tensor] = None,
        head_lengths: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pairs: Optional[torch.Tensor] = None,
        pair_labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **unused,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.graphmert(
            input_nodes,
            spatial_pos,
            return_dict=True,
            attention_mask=attention_mask,
            leaf_relationships=leaf_relationships,
            head_lengths=head_lengths,
        )
        outputs, hidden_states = encoder_outputs["last_hidden_state"], encoder_outputs["hidden_states"]

        prediction_scores = self.lm_head(outputs)
        prediction_scores = prediction_scores[:, 1:, :].contiguous() # remove graph token score -- we don't have label for it

        masked_lm_loss = None
        if labels is not None:
            lm_loss_fct = CrossEntropyLoss()
            masked_lm_loss = lm_loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    
        sbo_prediction_scores, masked_sbo_loss = None, None
        if self.use_sbo:
            sbo_prediction_scores = self.lm_pair_head(outputs, pairs)
        
        if self.use_sbo and pair_labels is not None:
            sbo_loss_fct = CrossEntropyLoss()
            masked_sbo_loss = sbo_loss_fct(sbo_prediction_scores.view(-1, self.config.vocab_size), pair_labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        if masked_lm_loss is not None and masked_sbo_loss is not None:
            total_loss = masked_lm_loss + masked_sbo_loss
        else:
            total_loss = masked_lm_loss
    
        return {
            "loss": total_loss,
            "mlm_loss": masked_lm_loss,
            "sbo_loss": masked_sbo_loss,
            "logits": prediction_scores,
            "sbo_logits": sbo_prediction_scores,
            "hidden_states": hidden_states,
            "attentions": None,
        }


class GraphMertForSequenceClassification(GraphMertPreTrainedModel):
    """
    This model can be used for graph-level sequence classification or regression tasks.

    It can be trained on
    - regression (by setting config.num_labels to 1); there should be one float-type label per graph
    - one task classification (by setting config.num_labels to the number of classes); there should be one integer
      label per graph
    - binary multi-task classification (by setting config.num_labels to the number of labels); there should be a list
      of integer labels for each graph.
    """
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.graphmert = GraphMertModel(config)
        self.activation_fn = config.activation_fn
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.classifier = GraphMertDecoderHead(self.activation_fn, self.hidden_size, self.num_labels)
        self.is_encoder_decoder = True

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_nodes,
        spatial_pos: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **unused,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.graphmert(
            input_nodes,
            spatial_pos,
            return_dict=True,
            attention_mask=attention_mask
        )
        outputs, hidden_states = encoder_outputs["last_hidden_state"], encoder_outputs["hidden_states"]

        head_outputs = self.classifier(outputs)
        logits = head_outputs[:, 0, :].contiguous()

        loss = None
        if labels is not None:
            mask = ~torch.isnan(labels)

            if self.num_labels == 1:  # regression
                loss_fct = MSELoss()
                loss = loss_fct(logits[mask].squeeze(), labels[mask].squeeze().float())
            elif self.num_labels > 1 and len(labels.shape) == 1:  # One task classification
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits[mask].view(-1, self.num_labels), labels[mask].view(-1))
            else:  # Binary multi-task classification
                loss_fct = BCEWithLogitsLoss(reduction="sum")
                loss = loss_fct(logits[mask], labels[mask])

        if not return_dict:
            return tuple(x for x in [loss, logits, hidden_states] if x is not None)
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=hidden_states, attentions=None)


class GraphMertForMultipleChoice(GraphMertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.graphmert = GraphMertModel(config)
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.is_encoder_decoder = True

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_nodes,  # [batch_size, num_choices, max_num_nodes, node_feat_size]
        attention_mask: Optional[torch.Tensor] = None, 
        labels: Optional[torch.LongTensor] = None,  # [batch_size, ]
        return_dict: Optional[bool] = None,
        **unused,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        num_choices = input_nodes.shape[1]
        input_nodes = input_nodes.view(-1, *input_nodes.shape[2:]) if input_nodes is not None else None
        attention_mask = attention_mask.view(-1, *attention_mask.shape[2:]) if attention_mask is not None else None

        encoder_outputs = self.graphmert(
            input_nodes,
            attention_mask=attention_mask,
            return_dict=True
        )
        outputs, hidden_states = encoder_outputs["last_hidden_state"], encoder_outputs["hidden_states"]

        head_outputs = self.classifier(self.dropout(outputs))
        logits = head_outputs[:, 0, :].contiguous()
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=hidden_states,
            attentions=None,
        )
