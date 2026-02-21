# coding=utf-8
# Copyright 2023 Microsoft, clefourrier and The HuggingFace Inc. team. All rights reserved.
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
""" GraphMert model configuration"""

from typing import List

from transformers import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

GRAPHMERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "": "https://huggingface.co//resolve/main/config.json",
}


class GraphMertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~GraphMertModel`]. It is used to instantiate an
    GraphMert model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the GraphMert
    [](https://huggingface.co/) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50265):
            Vocabulary size of the GraphMERT model. Defines the number of different tokens that can be represented by the
            `inputs_nodes` passed when calling GraphMertModel.
        num_edges (`int`, *optional*, defaults to 1):
            Number of edges types in the graph.
        edge_type (`str`, *optional*, defaults to multihop):
            Type of edge relation chosen.
        max_nodes (`int`, *optional*, defaults to 512):
            Maximum number of nodes which can be parsed for the input graphs.
        share_input_output_embed (`bool`, *optional*, defaults to `False`):
            Shares the embedding layer between encoder and decoder - careful, True is not implemented.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of layers.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the embedding layer in encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (often named feed-forward) layer in encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads in the encoder.
        self_attention (`bool`, *optional*, defaults to `True`):
            Model is self attentive (False not implemented).
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for the attention weights.
        activation_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability after activation in the FFN.
        layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        bias (`bool`, *optional*, defaults to `True`):
            Uses bias in the attention module - unsupported at the moment.
        embed_scale(`float`, *optional*, defaults to None):
            Scaling factor for the node embeddings.
        num_trans_layers_to_freeze (`int`, *optional*, defaults to 0):
            Number of transformer layers to freeze.
        pre_layernorm (`bool`, *optional*, defaults to `False`):
            Apply layernorm before self attention and the feed forward network. Without this, post layernorm will be
            used.
        apply_graphmert_init (`bool`, *optional*, defaults to `False`):
            Apply a custom graphmert initialisation to the model before training.
        freeze_embeddings (`bool`, *optional*, defaults to `False`):
            Freeze the embedding layer, or train it along the model.
        pretrained_emb_dim (`int`, defaults to 0):
            If using pretrained embeddings for atom_encoder, the dim of pretrained embeddings. Used for projection. 0 means
            not using pretrained embeddings.
        encoder_normalize_before (`bool`, *optional*, defaults to `False`):
            Apply the layer norm before each encoder block.
        q_noise (`float`, *optional*, defaults to 0.0):
            Amount of quantization noise (see "Training with Quantization Noise for Extreme Model Compression"). (For
            more detail, see fairseq's documentation on quant_noise).
        qn_block_size (`int`, *optional*, defaults to 8):
            Size of the blocks for subsequent quantization with iPQ (see q_noise).
        kdim (`int`, *optional*, defaults to None):
            Dimension of the key in the attention, if different from the other values.
        vdim (`int`, *optional*, defaults to None):
            Dimension of the value in the attention, if different from the other values.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        traceable (`bool`, *optional*, defaults to `False`):
            Changes return value of the encoder's inner_state to stacked tensors.
        mlm: (`bool`, *optional*, defaults to `False`):
            Whether to use ONLY masked language modeling (MLM) pretraining objective. If set to `True`, the model is trained
            with a masked language modeling objective only.
        mlm_sbo: (`bool`, *optional*, defaults to `True`):
            Whether to use span boundary objective (SBO) pretraining objective in addition to MLM. If set to `True`, the model is trained
            with both mlm and a span boundary objective.
        exp_mask_base (`float`, *optional*, defaults to 0.9): base for the exponential mask. Must be less or equal to 1 or None. base ** Gelu(shortest_path(i,j) - p)

        Example:
            ```python
            >>> from transformers import GraphMertForGraphClassification, GraphMertConfig

            >>> # Initializing a GraphMert graphmert-base-pcqm4mv2 style configuration
            >>> configuration = GraphMertConfig()

            >>> # Initializing a model from the  style configuration
            >>> model = GraphMertForGraphClassification(configuration)

            >>> # Accessing the model configuration
            >>> configuration = model.config
            ```
    """
    model_type = "graphmert"
    keys_to_ignore_at_inference = ["past_key_values"]

    @property
    def exp_mask_base(self):
        return self._exp_mask_base
    
    @exp_mask_base.setter
    def exp_mask_base(self, value):
        if value is not None and value > 1:
            raise ValueError("Exp_mask_base must be 0<base<=1 or None")
        self._exp_mask_base = value


    def __init__(
        self,
        vocab_size: int = 50265,
        root_nodes: int = 128,
        max_nodes: int = 128 * 3 + 128,
        graph_types: List[str] = ['root_undirected', 'leaf_undirected', 'leaf_connected_undirected'],
        fixed_graph_architecture: bool = True,
        num_hidden_layers: int = 12,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_attention_heads: int = 12,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        relation_emb_dropout = 0.2,
        encoder_normalize_before: bool = False,
        pre_layernorm: bool = False,
        apply_graphmert_init: bool = False,
        activation_fn: str = "gelu",
        pretrained_emb_dim: int = 0,
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        num_trans_layers_to_freeze: int = 0,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        kdim: int = None,
        vdim: int = None,
        bias: bool = True,
        self_attention: bool = True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        num_relationships: int = None,
        exp_mask_base: float = 0.9,
        mlm_sbo: bool = True,
        span_upper_length: int = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.root_nodes = root_nodes
        self.max_nodes = max_nodes
        self.graph_types = graph_types
        self.fixed_graph_architecture = fixed_graph_architecture
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.layerdrop = layerdrop
        self.relation_emb_dropout = relation_emb_dropout
        self.encoder_normalize_before = encoder_normalize_before
        self.pre_layernorm = pre_layernorm
        self.apply_graphmert_init = apply_graphmert_init
        self.activation_fn = activation_fn
        self.pretrained_emb_dim = pretrained_emb_dim
        self.embed_scale = embed_scale
        self.freeze_embeddings = freeze_embeddings
        self.num_trans_layers_to_freeze = num_trans_layers_to_freeze
        self.traceable = traceable
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size
        self.num_relationships = num_relationships
        self.exp_mask_base = exp_mask_base
        self.span_upper_length = span_upper_length
        self.mlm_sbo = mlm_sbo

        # These parameters are here for future extensions
        # atm, the model only supports self attention
        self.kdim = kdim
        self.vdim = vdim
        self.self_attention = self_attention
        self.bias = bias

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
