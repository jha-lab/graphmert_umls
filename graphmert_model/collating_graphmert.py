# Copyright (c) Microsoft Corporation and HuggingFace
# Licensed under the MIT License.

import torch
import numpy as np
import logging
import json

from typing import Any, Dict, List, Mapping, Optional, Tuple
from itertools import chain

from transformers.utils import is_cython_available

logger = logging.getLogger(__name__)


if is_cython_available():
    import pyximport

    pyximport.install(setup_args={"include_dirs": np.get_include()})
    from . import algos_graphmert  # noqa E402

SUPPORTED_GRAPH_TYPES = {
    'root_directed': 'Root nodes are connected via directed edges from start to end',
    'root_undirected': 'Root nodes are connected via undirected edges',
    'root_fully_connected': 'Root nodes are fully connected',
    'leaf_directed': 'Leaf nodes are connected via directed edges from respective root nodes',
    'leaf_undirected': 'Leaf nodes are connected via undirected edges to the respective root nodes',
    'leaf_connected_undirected': 'Leaf nodes of same root node are sequentially connected; undirected connections;',
    'leaf_connected_directed': 'Leaf nodes of same root node are sequentially connected; directed connections;',
}


def load_json_file(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json_file(path, data):
    with open(path, 'w+') as f:
        json.dump(data, f)
    return


class GraphMertDataCollator:
    def __init__(self, config, tokenizer, graph_types: List[str], spatial_pos_max=20, process_arch_tensors=False,
                 on_the_fly_processing=False):
        if not is_cython_available():
            raise ImportError("GraphMert preprocessing needs Cython (pyximport)")

        self.config = config
        self.tokenizer = tokenizer

        self.arch_used, self.kg_used = False, False
        assert 'root_directed' in graph_types or 'root_undirected' in graph_types or 'root_fully_connected' in graph_types
        if 'leaf_directed' in graph_types or 'leaf_undirected' in graph_types:
            self.arch_used = True

        if 'amr' in graph_types:
            raise NotImplementedError("AMR graph constructor is not implemented yet")
        if 'conceptnet' in graph_types:
            raise NotImplementedError("ConceptNet graph constructor is not implemented yet")

        self.kg_maps = {'amr': {}, 'conceptnet': {}, 'wikipedia': {}}

        self.graph_types = graph_types

        self.spatial_pos_max = spatial_pos_max
        self.process_arch_tensors = process_arch_tensors
        self.on_the_fly_processing = on_the_fly_processing

        if self.process_arch_tensors is False:
            assert self.config.fixed_graph_architecture is True

        # adj and shortest_path_result stay the same for the selected graph_types
        self.adj, self.shortest_path_result = None, None


        if not is_cython_available():
            raise ImportError("GraphMert preprocessing needs Cython (pyximport)")

    def supported_graph_types(self):
        return SUPPORTED_GRAPH_TYPES
    

    def __call__(self, features: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """
        create batch with tensors from list of features;
        len(features) = batch_size

        Note: expects columns in the dataset were converted to torch tensors with
        dataset.set_format(type='torch', columns=['input_nodes', 'attention_mask'])
        """
        if self.on_the_fly_processing:
            raise NotImplementedError("'on_the_fly_processing' should be False. Use preprocess_items() instead.")

        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]

        batch = {}
        batch_size = len(features)

        max_node_num = self.config.max_nodes
        batch["input_nodes"] = torch.stack([f['input_nodes'] for f in features], dim=0)
        batch["input_nodes"] = batch["input_nodes"].unsqueeze(-1)

        batch["attention_mask"] = torch.stack([f['attention_mask'] for f in features], dim=0)

        if "head_lengths" in features[0].keys():
            batch["head_lengths"] = torch.stack([f['head_lengths'] for f in features], dim=0)

        if "start_indices" in features[0].keys():
            batch["start_indices"] = torch.stack([f['start_indices'] for f in features], dim=0)

        if "leaf_relationships" in features[0].keys():
            batch["leaf_relationships"] = torch.stack([f['leaf_relationships'] for f in features], dim=0)

        if "special_tokens_mask" in features[0].keys():
            batch["special_tokens_mask"] = torch.stack([f['special_tokens_mask'] for f in features], dim=0)
            batch["special_tokens_mask"] = batch["special_tokens_mask"].unsqueeze(-1)

        if self.process_arch_tensors:
            batch["spatial_pos"] = torch.zeros(batch_size, max_node_num, max_node_num, dtype=torch.int16)


        for ix, f in enumerate(features): 
            if self.process_arch_tensors:
                f["spatial_pos"] = torch.tensor(f["spatial_pos"])
                batch["spatial_pos"][ix, : f["spatial_pos"].shape[0], : f["spatial_pos"].shape[1]] = f["spatial_pos"]

        # Special handling for labels.
        first = features[0]
        if "label" in first.keys():
            label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)

        return batch

    
    def preprocess_arch_tensors(self, items, num_nodes: int, num_leaves: int, keep_features: bool):
        """num_nodes -- num roots, num_leaves -- num leaves per root"""
        max_nodes = self.config.max_nodes
        num_items = len(items["input_ids"])

        if not(keep_features and "edge_index" in items.keys()):
            # Create edge_index based on requested graph types
            if self.adj is None:
                self.adj = np.zeros([max_nodes, max_nodes], dtype=bool)
                # Connect root nodes based on graph type
                if 'root_directed' in self.graph_types:
                    self.adj[:num_nodes, :num_nodes] += np.eye(num_nodes, k=1, dtype=bool)
                elif 'root_undirected' in self.graph_types:
                    self.adj[:num_nodes, :num_nodes] += np.eye(num_nodes, k=1, dtype=bool) + np.eye(num_nodes, k=-1,
                                                                                                    dtype=bool)
                elif 'root_fully_connected' in self.graph_types:
                    self.adj[:num_nodes, :num_nodes] += np.ones((num_nodes, num_nodes), dtype=bool) ^ np.eye(num_nodes,
                                                                                                             dtype=bool)
                for i in range(num_nodes):
                    for j in range(0, num_leaves):
                        if 'leaf_directed' in self.graph_types:
                            self.adj[i, num_nodes + i * num_leaves + j] += True
                        elif 'leaf_undirected' in self.graph_types:
                            self.adj[i, num_nodes + i * num_leaves + j] += True
                            self.adj[num_nodes + i * num_leaves + j, i] += True

                    if 'leaf_connected_directed' in self.graph_types:
                        self.adj[num_nodes + i * num_leaves: num_nodes + (i + 1) * num_leaves,
                                 num_nodes + i * num_leaves: num_nodes + (i + 1) * num_leaves] += np.eye(num_leaves, k=1, dtype=bool)
                    elif 'leaf_connected_undirected' in self.graph_types:
                        self.adj[num_nodes + i * num_leaves: num_nodes + (i + 1) * num_leaves,
                                 num_nodes + i * num_leaves: num_nodes + (i + 1) * num_leaves] += (np.eye(num_leaves, k=1, dtype=bool) + 
                                                                                                   np.eye(num_leaves, k=-1, dtype=bool)
                                                                                                )

        if self.shortest_path_result is None:
            self.shortest_path_result, self.path = algos_graphmert.floyd_warshall(self.adj)
            
        items["spatial_pos"] = [self.shortest_path_result.astype(np.int16) + 1 for _ in range(num_items)]  


    def preprocess_items(
            self, items,
            keep_features=True,
        ):
        """
        add structure tensors to items dict;
        get input_nodes for dataset
        possible input items.keys() (probably, not all are listed):
            ['input_ids', 'attention_mask', 'special_tokens_mask', 'leaf_node_ids'];
        where
            input_ids -- root nodes from dataset; len(input_ids) == max_seq_length
            leaf_node_ids: [bs, num_nodes, num_leaves]

        currently adds keys on return: ['num_nodes', 'input_nodes']
        where
            input_nodes: concatenated input_ids + flatten leaves, len(input_nodes) == max_nodes; 
        """
        assert "input_ids" in items.keys()


        if self.kg_used and "leaf_node_ids" not in items.keys():
            if "text" in items.keys():
                items["sentence"] = items.pop("text")
            items["sentence"] = ["".join(sentence) for sentence in items["sentence"]]


        num_nodes = len(items["input_ids"][0])
        num_items = len(items["input_ids"])     # batch size
        max_nodes = self.config.max_nodes

        num_leaves = max_nodes // num_nodes - 1
        assert num_nodes <= max_nodes
        assert num_leaves >= 0

        if self.process_arch_tensors:
            self.preprocess_arch_tensors(items, num_nodes, num_leaves, keep_features)

        # we need this condition to distinguish call GraphMertModel by __init__()
        # when input_nodes aren't used and data_collator call during dataset preprcoessing in mlm.utils.py
        input_nodes = None

        if 'leaf_node_ids' in items.keys():
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            input_nodes = np.zeros((num_items, max_nodes), dtype=np.uint32)
            for i in range(num_items):
                if self.tokenizer is not None:
                    leaf_iter = (leaf if leaf != 0 else pad_id for sub in items['leaf_node_ids'][i] for leaf in sub)
                else:
                    leaf_iter = chain.from_iterable(items['leaf_node_ids'][i])

                input_nodes[i][:num_nodes] = items["input_ids"][i]
                input_nodes[i][num_nodes:] = np.fromiter(leaf_iter, dtype=np.uint32, count = max_nodes - num_nodes)

        # Update input nodes
        #TD del it later
        if self.kg_used:
            raise NotImplementedError

        if input_nodes is not None:
            items["input_nodes"] = input_nodes

        return items
    

class GraphMertDataCollatorForLanguageModeling():
    def __init__(self, config, tokenizer, graph_types: List[str], spatial_pos_max=20, process_arch_tensors=False,
                 on_the_fly_processing=False, mlm_sbo=True, mlm_probability=0.15, mlm_on_leaves_probability=None,
                 geometric_p=0.2, subword_token_start='##'):
        """
        Args:
            mlm_sbo: use both mlm and span boundaries objectives;
            mlm_probability: probability of masking tokens
            lower: minimum span length
            upper: maximum span length
            geometric_p: geometric distribution
        """
        self.data_collator = GraphMertDataCollator(config, tokenizer, graph_types, spatial_pos_max,
                                                   process_arch_tensors, on_the_fly_processing)
        self.config = config
        self.tokenizer = tokenizer
        self.mlm_sbo = mlm_sbo
        self.mlm_probability = mlm_probability
        self.mlm_on_leaves_probability = mlm_on_leaves_probability or mlm_probability

        # for span masking
        num_leaves = self.config.max_nodes // self.config.root_nodes - 1
        self.num_leaves = num_leaves
        self.span_upper_length = config.span_upper_length if config.span_upper_length is not None else num_leaves
        lower = 1
        upper = self.span_upper_length
        if upper > num_leaves:
            raise ValueError(f"upper span length {upper} is greater than the number of leaves {num_leaves}"
                             f" You can change this behavior if needed"
        )
        assert upper > lower, "upper should be greater than lower"
        self.lens = list(range(lower, upper + 1))
        self.len_distrib = [geometric_p * (1 - geometric_p)**(i - lower) for i in range(lower, upper + 1)] if geometric_p >= 0 else None
        self.len_distrib = [x / (sum(self.len_distrib)) for x in self.len_distrib]

        self.subword_token_start = subword_token_start


    def supported_graph_types(self):
        return SUPPORTED_GRAPH_TYPES
    

    def __call__(self, features: List[dict]) -> Dict[str, Any]:
        batch = self.data_collator(features)

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm_sbo:
            batch["input_nodes"], batch["labels"], batch["pairs"], batch["pair_labels"] = self.mask_spans(
                batch["input_nodes"],  batch["start_indices"], batch["attention_mask"], special_tokens_mask=special_tokens_mask
            )
        else:
            batch["input_nodes"], batch["labels"] = self.mask_tokens(
                batch["input_nodes"], batch["attention_mask"], special_tokens_mask=special_tokens_mask
            )
            batch["pairs"] = None
            batch["pair_labels"] = None
        
        return batch
    

    def preprocess_items(self, items, keep_features=True):
        return self.data_collator.preprocess_items(items, keep_features)
    

    def set_on_the_fly_processing(self, val=False):
        self.data_collator.on_the_fly_processing = False


    def mask_tokens(self, inputs, attention_mask, special_tokens_mask: Optional[Any] = None):
        """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original."""
        labels = inputs.clone()

        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        probability_matrix[:, self.config.root_nodes:, :] = self.mlm_on_leaves_probability

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool).unsqueeze(-1)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        
        attention_mask = attention_mask.bool().unsqueeze(-1)

        probability_matrix.masked_fill_(special_tokens_mask | ~attention_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    

    def _is_valid_word_start(self, token):
            """
            valid starts for spans with len > 1
            Since the aim is to predict semantic leaves as a span, we want to restrct
            the examples of what is a valid start of a span.
            """
            if token.startswith(self.subword_token_start):
                return False
            if token[0].isalnum():
                return True
            return False


    def get_start_indices(self, items):
        """Get start indices of words in the input_ids for maksing spans longer than 1 token"""
        start_indices = []
        for idx, input_ids in enumerate(items['input_ids']):
            seq = self.tokenizer.convert_ids_to_tokens(input_ids)
            start_inds = [self._is_valid_word_start(token) for token in seq]
            start_indices.append(start_inds)

        items['start_indices'] = start_indices
        return items
        

    def _find_right_end_of_word(self, input, start_idx, e_idx, seq_len):
        """
        searches for the token that is not a subword token right to the e_idx
        if the span goes beyond the end of the sequence, returns end of sequence.
        If the span longer than the model span_upper_length we omit it to match model layers dimensions.
        len(input) != seq_len (because of leaf nodes).
        input[seq_len - 1] is [SEP] and not allowed.
        """
        flag = False
        while e_idx < seq_len - 1 and self.tokenizer.decode(input[e_idx][0]).startswith(self.subword_token_start):
            e_idx += 1
            flag = True

        if e_idx - start_idx + 1 > self.span_upper_length:
            return None
        if e_idx >= seq_len - 1:
            return seq_len - 2
        if flag:
            return e_idx - 1
        return e_idx
    

    def _find_leaves(self, leaf_indices: torch.Tensor):
        """
        find the index of the 1st leaf in the leaf sequence
        and the len of the leaf sequence
        """
        valid_mask = (leaf_indices % self.num_leaves == 0)
        if leaf_indices.flatten().shape[0] != 0:
            assert valid_mask.sum().item() >= 1, "Leaf sequence does not start at valid position"
        indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
        
        # positions of first leaves
        first = leaf_indices[indices]
        
        # "second": for indices[1:], use the previous element (i - 1) of leaf_indices
        # then append the last element of leaf_indices so that len(second)==len(first)
        if indices.numel() >= 2:
            second_mid = leaf_indices[indices[1:] - 1]
        else:
            second_mid = torch.tensor([], dtype=leaf_indices.dtype, device=leaf_indices.device)
        # positions of last leaves
        second = torch.cat((second_mid, leaf_indices[-1].unsqueeze(0))) # inclusive
        # leaf_lengths = second - first + 1
        return first, second
    

    # only for test
    @classmethod
    def spans_are_overlapping(cls, all_spans):
        """
        Check if any inner list of span tensors contains overlapping spans.
        Each inner list is a list of PyTorch tensors of shape [2] ([start, end]).
        """
        for span_list in all_spans:
            if not span_list:
                continue  # Skip empty inner lists

            # Stack the list of span tensors into a single tensor of shape [N, 2]
            spans_tensor = torch.stack(span_list, dim=0)
            # Sort spans by their start value.
            sorted_indices = torch.argsort(spans_tensor[:, 0])
            sorted_spans = spans_tensor[sorted_indices]
            
            # Check consecutive spans for overlap.
            for i in range(1, sorted_spans.size(0)):
                prev_start, prev_end = sorted_spans[i - 1]
                curr_start, curr_end = sorted_spans[i]
                if (prev_end >= curr_start).item():
                    return True
        return False


    # TD implement and replace code in 3 places    
    def _get_special_tokens_mask(self, inputs, special_tokens_mask: Optional[Any] = None):
        """
        Returns a mask of special tokens in the inputs.
        """
        pass


    def span_mask_for_mlm(self, inputs, all_spans, attention_mask, special_tokens_mask):
        """
        Prepare masked tokens inputs/labels for usual MLM loss based on span boundaries.
        80% MASK, 10% random, 10% original.
        """
        
        labels = inputs.clone()
        if special_tokens_mask is None:
            special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool).unsqueeze(-1)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        attention_mask = attention_mask.bool().unsqueeze(-1)

        batch_size = inputs.shape[0]
        mask = torch.zeros_like(inputs, dtype=torch.bool, device=inputs.device)
        for b in range(batch_size):
            indices_to_mask_list = [torch.arange(span[0], span[1] + 1) for span in all_spans[b]]
            if len(indices_to_mask_list) > 0: # we could fail to find span in a sequence
                indices_to_mask = torch.cat(indices_to_mask_list, dim=0)
                mask[b, indices_to_mask, 0] = True
        
        mask.masked_fill_(special_tokens_mask | ~attention_mask, value=False)
        labels[~mask] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & mask
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & mask & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels
    

    def span_mask_for_sbo(self, all_spans, mlm_labels):
        """
        Prepare masked labels for Span Boundaries Objective (SBO loss) 
        based on span boundaries.
        sbo_labels will be subset of mlm_labels -- for some spans we dont compute SBO
        Returns:
        pairs: tensor of shape [batch_size, max_num_pairs, 2] - start and end indices for the SBO (boundaries)
        pair_labels: tensor of shape [batch_size, max_num_pairs * self.span_upper_length, 1] - labels for the SBO
        """
        batch_size = mlm_labels.shape[0]
        max_num_pairs = max([len(spans) for spans in all_spans])
        # -1 in pairs corresponds to padding to the max length
        pairs = -torch.ones((batch_size, max_num_pairs, 2), dtype=torch.long, device=mlm_labels.device)
        pair_labels = torch.full((batch_size, max_num_pairs * self.span_upper_length, 1), -100, dtype=torch.long, device=mlm_labels.device)

        for b in range(batch_size):
            for i, span in enumerate(all_spans[b]):
                # we don't calculate SBO loss for spans that fall on start and end of the root sequence, and for all leaf sequences
                if span[0] > 0 and span[1] < self.config.root_nodes:
                    pairs[b, i, 0] = span[0] - 1
                    pairs[b, i, 1] = span[1] + 1
                    pair_labels[b, i * self.span_upper_length: i * self.span_upper_length + (span[1] + 1 - span[0])] = mlm_labels[b, span[0]: span[1] + 1]

        return pairs, pair_labels
    

    def mask_spans(self, inputs, start_indices, attention_mask, special_tokens_mask: Optional[Any] = None):
            """Prepare masked tokens inputs/labels for masked language modeling with span masking.
            For roots:
            length of span is sampled from a geometric distribution with p=0.2 truncated at num_leaves
            start of span is sampled uniformly from the start_indices of the sequence.
            start_inds is a list of bools for root nodes, where True indicates a valid start of a word.
            For spans with len > 1 we make sure that the span is a valid word.
            if the span goes beyond the unmasked tokens, resample start of span.
            If span goes beyond the sequence, truncate it.
            For leaves:
            First, sample a leaf sequence to mask with probability p=0.15.
            Second, length of span is sampled from a reverse geometric distribution with p=0.2
            Only valid start of words are considered.

            spans [start_idx, e_idx] are inclusive
            """
            def sample_random_allowed_index(allowed_inds):
                """
                allowed_inds is a list of bools, where True indicates an allowed postion.
                samples a random index from the allowed positions.
                """
                allowed_indices = torch.nonzero(allowed_inds, as_tuple=False).squeeze()
                if allowed_indices.dim() == 0:
                    allowed_indices = allowed_indices.unsqueeze(0)
                if allowed_indices.numel() == 0:
                    return -1
                rand_idx = torch.randint(0, allowed_indices.size(0), (1,))
                sampled_idx = allowed_indices[rand_idx].item()
                return sampled_idx


            all_spans = []
            masked_root_num_ceil = int(self.mlm_probability * self.config.root_nodes) # for root nodes only
            for batch_idx in range(inputs.shape[0]):
                input, start_inds = inputs[batch_idx], start_indices[batch_idx]
                mask = set()
                spans = [] # list of torch tensors with range to be masked
                # mask root nodes first
                iter_num = 0
                iter_num_max = 100
                while len(mask) < masked_root_num_ceil and iter_num < iter_num_max:
                    span_len = np.random.choice(self.lens, p=self.len_distrib)
                    e_idx = None
                    iter_num_per_one_span = 0
                    iter_num_per_one_span_max = 50
                    while e_idx is None and iter_num_per_one_span < 50:
                        iter_num_per_one_span += 1
                        start_idx = sample_random_allowed_index(start_inds)
                        if start_idx == -1:  # no valid start index in start_inds
                            break
                        e_idx = start_idx + span_len - 1
                        e_idx = self._find_right_end_of_word(input, start_idx, e_idx, seq_len=self.config.root_nodes)
                        if e_idx is None: # span is longer than permitted span_upper_length
                            continue
                        masked_range = set(range(start_idx, e_idx + 1))
                        if masked_range & mask:
                            e_idx = None # span overlaps with already masked tokens
                            continue
                        mask |= masked_range

                    if iter_num_per_one_span >= iter_num_per_one_span_max:
                        logger.error(
                            f"currently masked root nodes: {len(mask)};"
                            f"masked spans: {spans};"
                            f"required span_len: {span_len};"
                            f"input: {self.tokenizer.convert_ids_to_tokens(input[:128])};"
                            f"input: {self.tokenizer.decode(input[:128].flatten())};"
                            f"Cannot mask span of len {span_len} of root nodes after {iter_num_per_one_span} iterations"
                        )

                    if start_idx == -1: # no valid start index in start_inds
                        logger.error(
                            f"currently masked root nodes: {len(mask)};"
                            f"masked spans: {spans};"
                            f"input: {self.tokenizer.convert_ids_to_tokens(input[:128])};"
                            f"input: {self.tokenizer.decode(input[:128].flatten())};"
                            f"Cannot mask required number of root nodes ({masked_root_num_ceil}): not enough valid start indices"
                        )
                        break

                    if e_idx is not None:
                        start_inds[start_idx: e_idx + 1] = False
                        spans.append(torch.tensor([start_idx, e_idx]))

                    iter_num += 1
                    if iter_num >= iter_num_max:
                        logger.error(
                            f"currently masked root nodes: {len(mask)};"
                            f"masked spans: {spans};"
                            f"input: {self.tokenizer.convert_ids_to_tokens(input[:128])};"
                            f"input: {self.tokenizer.decode(input[:128].flatten())};"
                            f"Cannot mask required number of root nodes ({masked_root_num_ceil}) after {iter_num} iterations"
                        )
                
                # mask leaves next
                leaf_indices = torch.nonzero(input[self.config.root_nodes:] != self.tokenizer.pad_token_id, as_tuple=True)[0]
                if leaf_indices.shape[0] != 0:
                    first_leaves, last_leaves = self._find_leaves(leaf_indices)
                    first_leaves += self.config.root_nodes
                    last_leaves += self.config.root_nodes
                    prob = torch.full(first_leaves.shape, self.mlm_on_leaves_probability)
                    leaf_seq_to_mask = torch.bernoulli(prob).bool()
                    chosen_leaf_indices = first_leaves[leaf_seq_to_mask]
                    chosen_last_leaves_indices = last_leaves[leaf_seq_to_mask]

                    if chosen_leaf_indices.shape[0]:
                        leaf_spans = torch.stack([chosen_leaf_indices, chosen_last_leaves_indices], dim=1)
                        for leaf_span in leaf_spans:
                            spans.append(leaf_span)
                            
                all_spans.append(spans)

            # all_spans = [[] for _ in range(inputs.shape[0])]
            inputs, labels = self.span_mask_for_mlm(inputs, all_spans, attention_mask, special_tokens_mask)
            pairs, pair_labels = self.span_mask_for_sbo(all_spans, labels)

            return inputs, labels, pairs, pair_labels



class GraphMertDataCollatorForMultipleChoice():
    def __init__(self, config, tokenizer, graph_types: List[str], process_arch_tensors=False,
                 on_the_fly_processing=False, num_choices=0):
        self.config = config
        self.tokenizer = tokenizer
        self.graph_types = graph_types
        self.num_choices = num_choices

        assert num_choices > 0

        self.arch_used, self.kg_used = False, False
        assert 'root_directed' in graph_types or 'root_undirected' in graph_types or 'root_fully_connected' in graph_types
        if 'leaf_directed' in graph_types or 'leaf_undirected' in graph_types:
            self.arch_used = True
        if 'leaf_connected_directed' in graph_types in graph_types:
            assert 'leaf_connected_undirected' not in graph_types
        
        if 'leaf_connected_undirected' in graph_types:
            assert 'leaf_connected_directed' not in graph_types


        self.process_arch_tensors = process_arch_tensors
        self.on_the_fly_processing = on_the_fly_processing
        if self.kg_used is True:
            raise NotImplementedError("'kg_used' should be False")
        if process_arch_tensors is True:
            raise NotImplementedError("'process_arch_tensors' should be False")
        if on_the_fly_processing is True:
            raise NotImplementedError("'on_the_fly_processing' should be False")

    def __call__(self, features: List[dict]) -> Dict[str, Any]:
        """
        create batch with tensors from list of features;
        len(features) = batch_size
        """
        if self.on_the_fly_processing:
            raise NotImplementedError("'on_the_fly_processing' should be False. Use preprocess_items() instead.")
        else:
            for i in features:
                i["input_nodes"] = np.asarray(i.pop("input_nodes"), dtype=np.int64)

        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]

        batch = {}
        batch_size = len(features)
        max_node_num = self.config.max_nodes
        node_feat_size = features[0]["input_nodes"][0].shape[-1]
        batch["input_nodes"] = torch.zeros(batch_size, self.num_choices, max_node_num, node_feat_size, dtype=torch.long)
        batch["attention_mask"] = torch.zeros(batch_size, self.num_choices, max_node_num)

        for ix, f in enumerate(features):
            f["input_nodes"] = torch.tensor(f["input_nodes"])
            batch["input_nodes"][ix, :] = f["input_nodes"]

            for i, choice in enumerate(f["attention_mask"]):
                batch["attention_mask"][ix, i, :len(choice)] = torch.tensor(choice)

        # Special handling for labels.
        first = features[0]
        if "label" in first.keys():
            label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)

        return batch
        

    def preprocess_items(self, items):

        num_nodes = self.config.root_nodes
        num_items = len(items["input_ids"])
        max_nodes = self.config.max_nodes

        num_leaves = max_nodes // num_nodes - 1
        assert num_nodes <= max_nodes
        assert num_leaves >= 0

        input_nodes = []
        for n in range(num_items):
            choices = []
            for c in items["input_ids"][n]:
                choices.append(np.asarray(c + [self.tokenizer.pad_token_id if self.tokenizer is not None else 0] * (
                    max_nodes - len(c)), dtype=np.int64).reshape(-1, 1))
            input_nodes.append(choices)

        items["num_nodes"] = [max_nodes for _ in range(num_items)]
        items["input_nodes"] = input_nodes

        return items
