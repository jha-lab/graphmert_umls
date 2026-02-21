import os
import torch
from datasets import load_from_disk, concatenate_datasets
import logging
import json

from transformers import AutoTokenizer
import yaml

# Load shared entity discovery config and take needed keys
ECONF_PATH = os.path.join(os.path.dirname(__file__), 'entity_discovery_args.yaml')
if not os.path.exists(ECONF_PATH):
    raise FileNotFoundError(f"Missing config: {ECONF_PATH}")
with open(ECONF_PATH, 'r', encoding='utf-8') as _f:
    config = yaml.safe_load(_f)


tokenizer_name = config['model']['tokenizer_path']
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


TAKE_SUBSET = config.get('data', {}).get('take_subset', False)
subset_size = config.get('data', {}).get('subset_size', None)
if TAKE_SUBSET:
    assert subset_size is not None, "subset_size must be set if TAKE_SUBSET is True"


# to list the output chunks
batch_size = config['data']['batch_size']
num_batches = config['data']['num_batches']
chunk_size = batch_size * num_batches
model_name = config['model']['model_name']
original_dataset_size = config['data']['original_dataset_size']


dataset_path = config.get('data', {}).get('output_path')
output_path = f'{dataset_path}_{model_name}_all'

def unite_output(dataset_path, original_dataset_size, chunk_size, model_name):
    """
    original_dataset_size -- how many records in the dataset before running the entity discovery;
    it's also in the filename in the folder -- the largest end index
    """
    start_idx = 0
    end_idx = 0

    datasets = []
    if TAKE_SUBSET:
        path = os.path.join(dataset_path, f'{model_name}_subset_{subset_size}_0-{subset_size}')
        dataset = load_from_disk(path)
        logger.info(f'loaded from {path}')
        datasets.append(dataset)
    else:
        while end_idx < original_dataset_size:
            end_idx = min(start_idx + chunk_size, original_dataset_size)
            path = os.path.join(dataset_path, f'{model_name}_{start_idx}-{end_idx}')
            dataset = load_from_disk(path)
            logger.info(f'loaded from {path}')
            datasets.append(dataset)
            start_idx += chunk_size
    
    united_dataset = concatenate_datasets(datasets)
    if not TAKE_SUBSET:
        assert original_dataset_size == len(united_dataset), f'total of chunks {len(united_dataset)} is not equal to the original dataset size'
    return united_dataset


dataset_heads = unite_output(dataset_path, original_dataset_size, chunk_size, model_name)
logger.info(f"Whole dataset:\n{dataset_heads}")


if TAKE_SUBSET:
    dataset_heads = dataset_heads.select(range(subset_size))


def find_head_positions(example, idx):
    """
    For each record, finds the starting token index of each head (key) in the response JSON.
    Considers hyphen token mismatches: if the head token id equals 17, we allow dataset token 17 or 226.
    Prints an error message if a head is not found.
    """
    input_ids = example["input_ids"]

    if isinstance(input_ids, str):
        response_str = example["response"].strip() if example["response"] else ""
        try:
            response_list = json.loads(response_str) if response_str else []
        except Exception as e:
            logger.info(f"{idx}: Error parsing response JSON - {e}")
            response_list = []

    # if isinstance(input_ids, list):
    response_list = example["response"]
    
    head_positions = {}
    
    # Process each head in the original response.
    for head in response_list:
        head_lower = head.lower()
        head_token_ids = tokenizer.encode(head_lower, add_special_tokens=False)
        
        match_index = -1
        # Search for the head token sequence in the input_ids.
        for i in range(len(input_ids) - len(head_token_ids) + 1):
            found = True
            for j, token in enumerate(head_token_ids):
                dataset_token = input_ids[i + j]
                # If token is the hyphen token (assumed to be 17), allow dataset token 17 or 226.
                if token == 17:
                    if dataset_token not in (17, 226):
                        found = False
                        break
                else:
                    if dataset_token != token:
                        found = False
                        break
            if found:
                match_index = i
                break
        
        if match_index == -1:
            logger.info(f"{idx}: head '{head_lower}' not found.")
        else:
            span_token_ids = input_ids[match_index: match_index + len(head_token_ids)]
            matched_text = tokenizer.decode(span_token_ids, skip_special_tokens=True).strip()
            head_positions[matched_text] = match_index
    
    example["head_positions"] = json.dumps(head_positions)
    return example


dataset_heads_with_positions = dataset_heads.map(find_head_positions, num_proc=100, with_indices=True)
dataset_heads_with_positions = dataset_heads_with_positions.remove_columns(["response"])


dataset_heads_with_positions = dataset_heads_with_positions.map(
        lambda example, idx: {"id": idx}, with_indices=True,
        desc='Indexing dataset',
        num_proc=100,
)

if TAKE_SUBSET:
    path_to_save = f'{output_path}_subset_{subset_size}'
else:
    path_to_save = output_path
    
dataset_heads_with_positions.save_to_disk(path_to_save)
print(f'saved to {path_to_save}')
print(dataset_heads_with_positions)
