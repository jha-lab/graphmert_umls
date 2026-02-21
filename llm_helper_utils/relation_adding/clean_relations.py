import os
from datasets import load_from_disk, concatenate_datasets
import logging
import json
import yaml

from transformers import AutoTokenizer

from add_relations_prompts import ALLOWED_RELATIONS


ECONF_PATH = os.path.join(os.path.dirname(__file__), 'add_relations_args.yaml')
if not os.path.exists(ECONF_PATH):
    raise FileNotFoundError(f"Missing config: {ECONF_PATH}")
with open(ECONF_PATH, 'r', encoding='utf-8') as _f:
    config = yaml.safe_load(_f)

tokenizer_name = config['tokenizer_path']
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TAKE_SUBSET = config['take_subset']
subset_size = config.get('subset_size', None)
if TAKE_SUBSET:
    assert subset_size is not None, "subset_size must be set if TAKE_SUBSET is True"

# to list the output chunks
batch_size = config['batch_size']
num_batches = config['num_batches']
chunk_size = batch_size * num_batches
model_name = config['model_name']
original_dataset_size = config['original_dataset_size']


dataset_path = config['output_path']
output_path = os.path.join(dataset_path, f'{model_name}_relations_clean')

def unite_output(dataset_path, original_dataset_size, chunk_size, model_name):
    """
    original_dataset_size -- how many records in the dataset before running the entity discovery;
    it's also in the filename in the folder -- the largest end index
    """
    start_idx = 0
    end_idx = 0

    datasets = []
    if TAKE_SUBSET:
        path = os.path.join(dataset_path, f'{model_name}_relations_subset_{subset_size}_0-{subset_size}')
        dataset = load_from_disk(path)
        logger.info(f'loaded from {path}')
        datasets.append(dataset)
    else:
        while end_idx < original_dataset_size:
            end_idx = min(start_idx + chunk_size, original_dataset_size)
            path = os.path.join(dataset_path, f'{model_name}_relations_{start_idx}-{end_idx}')
            dataset = load_from_disk(path)
            logger.info(f'loaded from {path}')
            datasets.append(dataset)
            start_idx += chunk_size
    
    united_dataset = concatenate_datasets(datasets)
    if not TAKE_SUBSET:
        assert original_dataset_size == len(united_dataset), f'total of chunks {len(united_dataset)} is not equal to the original dataset size'
    return united_dataset


dataset_rels = unite_output(dataset_path, original_dataset_size, chunk_size, model_name)
logger.info(f"Whole dataset:\n{dataset_rels}")


if TAKE_SUBSET:
    dataset_rels = dataset_rels.select(range(subset_size))


def find_heads_with_relations(example, idx):
    """
    returns a cleaned_response column which includes only the heads with matched relations, 
    also checks if heads preserve spelling in the llm response.
    """
    response_dict = None
    if isinstance(example["response"], dict):
        response_dict = example["response"]
    elif isinstance(example["response"], str):
        response_str = example["response"].strip()
    elif isinstance(example["response"], list) and len(example["response"]) == 1:
        response_str = example["response"][0].strip()
    else:
        response_dict = {}
    
    if response_dict is None:
        try:
            response_dict = json.loads(response_str) if response_str else {}
        except Exception as e:
            logger.info(f"{idx}: Error parsing response JSON - {e}")
            response_dict = {}

    if response_dict is None:
        raise ValueError(f"id {example['id']}: Unexpected response type {type(example['response'])} with value {example['response']}")
    
    head_positions = json.loads(example["head_positions"])
    cleaned_response = {}
    
    # Process each head in the original response JSON.
    for head in response_dict.keys():
        head_lower = head.lower().strip()
        if head_lower not in head_positions:
            logger.info(f"{idx}: head {head_lower} is not in the orginal dict.")
            continue
        head_token_ids = tokenizer.encode(head_lower, add_special_tokens=False)
        if len(head_token_ids) == 0:
            logger.info(f"{idx}: head is empty.")
            continue

        # response_dict[head] expected ti be well-formed after running the previous cleaning step
        relations = response_dict[head]
        good_relations = []
        for relation in relations:
            if relation not in ALLOWED_RELATIONS:
                logger.info(f"{idx}: head '{head}' has an invalid relation '{relation}'")
            else:
                good_relations.append(relation)
        if not good_relations:
            logger.info(f"{idx}: head '{head}' has no valid relations")
        else:
            # drop duplicates while preserving order
            good_relations = list(dict.fromkeys(good_relations))
            # Include the original head and its relations in cleaned_response.
            cleaned_response[head_lower] = good_relations
    
    example["cleaned_response"] = json.dumps(cleaned_response)
    return example


dataset_rels_with_positions = dataset_rels.map(find_heads_with_relations, num_proc=100, with_indices=True)
dataset_rels_with_positions = dataset_rels_with_positions.remove_columns(["response"])

# filter out examples with only empty heads
def filter_fn(examples):
    return [cr != '{}' for cr in examples['cleaned_response']]


dataset_rels_with_positions = dataset_rels_with_positions.filter(filter_fn, num_proc=100, batched=True,
    desc='Filter out examples with empty heads'
)
logger.info("After dropping empty heads:\n%s", len(dataset_rels_with_positions))


if TAKE_SUBSET:
    output_path = f'{output_path}_subset_{subset_size}'
    
dataset_rels_with_positions.save_to_disk(output_path)
print(f'saved to {output_path}')
print(dataset_rels_with_positions)
