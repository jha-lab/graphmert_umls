import logging
import os
import json
from pathlib import Path

import yaml
import hashlib

import datasets
from datasets import load_dataset, load_from_disk, DatasetDict, Value, Sequence


import numpy as np
from collections import defaultdict

from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from filelock import FileLock


from training_arguments import ModelArguments, DataTrainingArguments, PreprocessingArguments
from graphmert_model import GraphMertDataCollatorForLanguageModeling, GraphMertConfig
AutoConfig.register("graphmert", GraphMertConfig)


logger = logging.getLogger(__name__)


hf_auth_token = os.getenv('HF_AUTH_TOKEN')
if hf_auth_token is None:
    logger.warning("HF_AUTH_TOKEN not found. Private datasets will be inaccessible.")


def inject_leaves(dataset, split: str, relation_map, injections_dict, data_args, training_args, config, tokenizer, cache_file_name, num_leaves):
    def inject_leaves_into_batch(batch):
        batch_len = len(batch['input_ids'])
        leaf_node_ids = np.zeros((batch_len, config.root_nodes, num_leaves), dtype=np.uint32)
        leaf_relationships = np.zeros((batch_len, config.root_nodes), dtype=np.int16)
        head_lengths = np.zeros((batch_len, config.root_nodes), dtype=np.int16)
        
        for i in range(batch_len):
            id = batch['id'][i]
            if id in injections_dict:
                inj = injections_dict[id]
                head_positions = json.loads(batch['head_positions'][i])
                for triple in inj:
                    head, rel, tail = triple['head'], triple['relation_type'], triple['tail']
                    # Skip if head entity wasn't found in the text
                    if head not in head_positions:
                        continue
                    leaves = np.array(tokenizer.encode(tail, add_special_tokens=False))
                    if len(leaves) > num_leaves:
                        continue
                    head_pos = head_positions.pop(head)
                    head_tokens = tokenizer.encode(head, add_special_tokens=False)
                    num_head_tokens = len(head_tokens)
                    head_lengths[i, head_pos] = num_head_tokens
                    relation_num = relation_map[rel]
                    assert relation_num != 0
                    leaf_relationships[i, head_pos] = relation_num
                    leaf_node_ids[i, head_pos, :len(leaves)] = leaves
                
        batch['leaf_node_ids'] = leaf_node_ids
        batch['leaf_relationships'] = leaf_relationships
        batch['head_lengths'] = head_lengths
        batch.pop('head_positions')
        return batch
    
    
    new_features = dataset.features.copy()
    new_features['head_lengths'] = Sequence(Value(dtype='uint8'), length=config.root_nodes)
    new_features['leaf_relationships'] = Sequence(Value(dtype='uint8'), length=config.root_nodes)

    new_features["leaf_node_ids"] = Sequence(
            Sequence(Value("uint32"), length=num_leaves),
            length=config.root_nodes
    )
    new_features.pop('head_positions')

    with training_args.main_process_first(desc="preprocessing texts for graphmert"):
        dataset = dataset.map(
            inject_leaves_into_batch,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Injecting leaves into {split}",
            cache_file_name=f'{cache_file_name[split]}_{num_leaves}leaves',
            features=new_features,
        )
        return dataset


def preprocess_items_for_graphmert(dataset, split: str, data_args, training_args, config, tokenizer, cache_file_name, data_collator, max_seq_length):
    """Run preprocessing to obtain input_nodes, special_tokens_mask and leaf attention_mask"""

    def preprocess_function(items):
        items = data_collator.preprocess_items(items)
        items = data_collator.get_start_indices(items)

        # set attention mask 0 for pads and 1 otherwise in leaf nodes
        input_nodes = items['input_nodes'] # take the numpy arr of input_nodes
        bsz = len(items['input_ids'])
        attention_mask = np.ones((bsz, max_seq_length), dtype=np.uint8)
        leaf_att_mask = (input_nodes[:, config.root_nodes:] != tokenizer.pad_token_id).astype(np.uint8)
        attention_mask = np.concatenate((attention_mask, leaf_att_mask), axis=1)
        items["attention_mask"] = attention_mask
        items['special_tokens_mask'] = np.array(
            [
                tokenizer.get_special_tokens_mask(input_ids_row.tolist(), already_has_special_tokens=True)
                for input_ids_row in items['input_nodes']
            ],
            dtype=np.uint8
        )
        items.pop('leaf_node_ids')
        return items
    
    # set feature typs to speed up processing
    new_features = dataset.features.copy()
    new_features['input_nodes'] = Sequence(Value(dtype='int64'), length=config.max_nodes)
    new_features['attention_mask'] = Sequence(Value(dtype='uint8'), length=config.max_nodes)
    new_features['special_tokens_mask'] = Sequence(Value(dtype='uint8'), length=config.max_nodes)
    new_features['start_indices'] = Sequence(Value(dtype='uint16'))
    new_features.pop('leaf_node_ids')
    
    with training_args.main_process_first(desc="preprocessing texts for graphmert"):
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Preprocessing grouped {split} texts with collator",
            cache_file_name=f'{cache_file_name[split]}_input_nodes',
            features=new_features,
        )

    return dataset


def main(yaml_file: str):        
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, PreprocessingArguments))
    
    # Read YAML and disable bf16/fp16 for CPU preprocessing before parsing
    yaml_config = yaml.safe_load(Path(yaml_file).read_text())
    yaml_config['bf16'] = False
    yaml_config['fp16'] = False
    
    model_args, data_args, training_args, preprocessing_args = parser.parse_dict(yaml_config, allow_extra_keys=True)

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)

    set_seed(training_args.seed)

    config = CONFIG_MAPPING['graphmert']()
    logger.warning("You are instantiating a new config instance from scratch.")

    # If a knowledge graph is not used, set max nodes to sequence length so leaf nodes are not created
    if not ('leaf_directed' in model_args.graph_types or 'leaf_undirected' in model_args.graph_types):
        config.max_nodes = data_args.max_seq_length
    else:
        config.root_nodes = data_args.max_seq_length
    config.graph_types = model_args.graph_types
    config.num_relationships = model_args.num_relationships

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "token": hf_auth_token,
    }

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, add_prefix_space=True, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if tokenizer.mask_token_id is None or tokenizer.mask_token is None:
        raise ValueError("tokenizer.mask_token or tokenizer.mask_token_id is None for mlm task")
    
    # that would work only for tokenizers  with bos / cls and eos / sep tokens (e.g roberta and bert)
    if (tokenizer.bos_token_id is None and tokenizer.cls_token_id is None) or (tokenizer.eos_token_id is None and tokenizer.sep_token_id is None):
        raise ValueError("set tokenizer.bos_token_id or/and tokenizer.eos_token_id for mlm task")
    
    if model_args.config_overrides is not None:
        config.update_from_string(model_args.config_overrides)
        logger.info(f"Overriding config: {model_args.config_overrides}")
        logger.info(f"New config: {config}")

    # Preprocessing the datasets.

    def unique_cache_filename(file_path):
        cache_filename = str.split(str.split(file_path, "/")[-1], ".")[0]
        # Create a hash of the full file path to ensure unique cache file names
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        return f'{cache_filename}_{file_hash}'
    

    cache_file_name = {
        'train': f"{data_args.dataset_name}_train" if data_args.dataset_name is not None else unique_cache_filename(data_args.train_file),
        'validation': f"{data_args.dataset_name}_validation" if data_args.dataset_name is not None else unique_cache_filename(data_args.validation_file),
    }
    processed_dataset_name = {
        'train': os.path.join(model_args.cache_dir, cache_file_name["train"], 'ready_for_training'),
        'validation': os.path.join(model_args.cache_dir, cache_file_name["validation"], 'ready_for_training'),
    }        

    if training_args.local_rank in [-1, 0]:
        os.makedirs(model_args.cache_dir, exist_ok=True)


    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)



    for split, name in cache_file_name.items():
        cache_file_name[split] = os.path.join(model_args.cache_dir, name)
        if training_args.local_rank in [-1, 0]:
            os.makedirs(cache_file_name[split], exist_ok=True)
        cache_file_name[split] = os.path.join(cache_file_name[split], "cache.tmp") # dataset lib will not work for multiple processes w/o dot in filename

    
    num_leaves = config.max_nodes // config.root_nodes - 1
    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = GraphMertDataCollatorForLanguageModeling(
        config=config,
        tokenizer=tokenizer,
        graph_types=model_args.graph_types,
        process_arch_tensors=not config.fixed_graph_architecture,
        on_the_fly_processing=True,
        subword_token_start=preprocessing_args.subword_token_start,
    )

    def get_injection_dict(path_to_csv):
        injections = load_dataset('csv', data_files=path_to_csv)
        injections = injections['train']
        if 'sequence' in injections.column_names:
            injections = injections.remove_columns('sequence')
        injections_dict = defaultdict(list)
        for record in injections:
            id = record.pop('id')
            injections_dict[id].append(record)
        return injections_dict
    
    tokenized_datasets = DatasetDict()
    tokenized_datasets["train"] = load_from_disk(preprocessing_args.train_dataset_with_heads)
    tokenized_datasets["validation"] = load_from_disk(preprocessing_args.train_dataset_with_heads)

    # Code for testing only (reduces dataset size)
    if preprocessing_args.cut_dataset_for_testing:
        tokenized_datasets["train"] = tokenized_datasets["train"].select(range(min(1000, len(tokenized_datasets["train"]))))
        tokenized_datasets["validation"] = tokenized_datasets["validation"].select(range(min(200, len(tokenized_datasets["validation"]))))
    
    injections_dict = {'train': None, 'validation': None}
    injections_dict['train'] = get_injection_dict(preprocessing_args.injections_train_path)
    injections_dict['validation'] = get_injection_dict(preprocessing_args.injections_eval_path)
    with open(preprocessing_args.relation_map_path, 'r') as f:
        relation_map = json.load(f)

    relation_map_values = list(map(int, relation_map.values()))
    assert max(relation_map_values) < model_args.num_relationships, f'Please set model_args.num_relationships >= {max(relation_map_values) + 1}'
    assert 0 not in relation_map_values, '0 is reserved for empty relation, assign no-zero to relations at relation_map'

    with open(os.path.join(training_args.output_dir, 'relation_map.json'), 'w') as f:
        json.dump(relation_map, f, indent=2)


    for key in tokenized_datasets.keys():    
        tokenized_datasets[key] = inject_leaves(tokenized_datasets[key], key, relation_map, injections_dict[key], data_args, \
                                                training_args, config, tokenizer, cache_file_name, num_leaves)

    if training_args.do_train:
        tokenized_datasets["train"] = preprocess_items_for_graphmert(tokenized_datasets["train"], 'train', data_args, training_args, config, 
                                                                     tokenizer, cache_file_name, data_collator, max_seq_length)
        if training_args.local_rank in [-1, 0] and (not os.path.exists(processed_dataset_name['train']) or data_args.overwrite_cache):
            with FileLock(f"{processed_dataset_name['train']}.lock", timeout=60):
                tokenized_datasets["train"].save_to_disk(processed_dataset_name['train'])
        logger.info(f"Saved processed training dataset to {processed_dataset_name['train']}")
    if training_args.do_eval:
        tokenized_datasets["validation"] = preprocess_items_for_graphmert(tokenized_datasets["validation"], 'validation', 
                                                        data_args, training_args, config, tokenizer, cache_file_name, data_collator, max_seq_length)
        if training_args.local_rank in [-1, 0] and (not os.path.exists(processed_dataset_name['validation']) or data_args.overwrite_cache):
            with FileLock(f"{processed_dataset_name['validation']}.lock", timeout=60):
                tokenized_datasets["validation"].save_to_disk(processed_dataset_name['validation'])
        logger.info(f"Saved processed evaluation dataset to {processed_dataset_name['validation']}")

    logger.info("Preprocessing completed.")
