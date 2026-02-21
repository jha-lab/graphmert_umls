import os
import argparse
import torch
from torch.nn import functional as F
import json
import logging
import yaml

from graphmert_model import GraphMertForMaskedLM
from datasets import load_from_disk, concatenate_datasets

from transformers import AutoTokenizer
import shutil

from utils.job_utils import get_job_info

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_config(yaml_file):
    """Load configuration from YAML file."""
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


class NamedModel:
    def __init__(self, model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        graphmert_model = GraphMertForMaskedLM.from_pretrained(model_path).to(device)
        graphmert_model.use_sbo = False
        graphmert_model = graphmert_model.to(device)

        graphmert_model.eval()
        for param in graphmert_model.parameters(): param.requires_grad = False       
        self.model = graphmert_model
        self.path = model_path


def get_top_predictions(examples, model, tokenizer, top_k=15, num_leaves=7, root_nodes=128):
    device = next(model.parameters()).device

    input_ids = torch.tensor(examples['input_ids'], dtype=torch.long, device=device)
    batch_size = input_ids.size(0) # to correcntly handle the remainder in the last batch

    attention_mask = torch.tensor(examples['attention_mask'], dtype=torch.long, device=device)
    head_pos = torch.tensor(examples['position'], dtype=torch.long, device=device)
    new_relation_num = torch.tensor(examples['relation_num'], dtype=torch.long, device=device)
    head_len = torch.tensor(examples['head_len'], dtype=torch.long, device=device)

    total_nodes = root_nodes * (num_leaves + 1)

    input_nodes = torch.zeros((batch_size, total_nodes, 1), dtype=torch.long, device=device)
    input_nodes[:, :root_nodes, :] = input_ids.unsqueeze(-1)

    masked_pos = root_nodes + num_leaves * head_pos
    idx = torch.arange(batch_size, device=device)
    input_nodes[idx, masked_pos, 0] = tokenizer.mask_token_id

    attention_mask_new = torch.zeros((batch_size, total_nodes), dtype=torch.long, device=device)
    attention_mask_new[:, :root_nodes] = attention_mask[:, :root_nodes]
    attention_mask_new[idx, masked_pos] = 1

    leaf_relationships = torch.zeros((batch_size, root_nodes), dtype=torch.long, device=device)
    head_lengths = torch.zeros((batch_size, root_nodes), dtype=torch.long, device=device)
    leaf_relationships[idx, head_pos] = new_relation_num
    head_lengths[idx, head_pos] = head_len

    with torch.no_grad():
        outputs = model(
            input_nodes=input_nodes, 
            attention_mask=attention_mask_new, 
            leaf_relationships=leaf_relationships,
            head_lengths=head_lengths,
            pairs=None,
        )

    masked_probs = F.softmax(outputs["logits"], dim=-1)        # (B, total_nodes, V)
    selected_probs = masked_probs[idx, masked_pos]        # (B, V)
    top_k_probs, top_k_indices = torch.topk(selected_probs, top_k, dim=-1)  # (B, K)
    top_k_tokens = [tokenizer.convert_ids_to_tokens(ids.tolist()) for ids in top_k_indices]  # List[List[str]]
    top_k_predictions = [" ".join(tokens) for tokens in top_k_tokens]
    
    filtered_examples = {
        'predictions': top_k_predictions,
        'probabilities': top_k_probs.tolist(),
    }
    return filtered_examples


def expand_dicts(examples, relation_map, tokenizer):
    """
    Expand dicts with heads and lists with relations for these heads for easy iteration.
    Drop examples with empty heads. Expand lists of relations for every non-empty head 
    to create a new record for each relation.
    """
    expanded = {k: [] for k in ["id", "input_ids", "attention_mask", "head", "head_len", "position", "relation", "relation_num"]}

    for batch_idx in range(len(examples["input_ids"])):
        response = examples["cleaned_response"][batch_idx]
        try:
            response = json.loads(response) if response else {}
        except Exception as e:
            logger.info(f"Error parsing response JSON - {e}")
            response = {}

        head_positions = examples["head_positions"][batch_idx]
        try:
            head_positions = json.loads(head_positions) if head_positions else {}
        except Exception as e:
            logger.info(f"Error parsing head_positions JSON - {e}")
            head_positions = {}
        if response == {}:
            continue

        id = examples["id"][batch_idx]

        for head, relations in response.items():
            head_pos = head_positions[head]
            head_len = len(tokenizer.encode(head, add_special_tokens=False))
            if head_len == 0:
                print(f"head_len == 0 for id {id}; head {head}")
                continue
            
            for relation in relations:  # Create a new record for each relation
                try: 
                    relation_num = relation_map[relation]
                except KeyError as e:
                    print(f"{relation} is not in relation_map")
                    continue
                
                relation_num = relation_map[relation]
                expanded["relation"].append(relation)
                expanded["relation_num"].append(relation_num)
                expanded["input_ids"].append(examples["input_ids"][batch_idx])
                expanded["attention_mask"].append(examples["attention_mask"][batch_idx])
                expanded["head"].append(head)
                expanded["head_len"].append(head_len)
                expanded["position"].append(head_pos)
                expanded["id"].append(id)

    return expanded


def main(yaml_file: str):
    # Load configuration
    config = load_config(yaml_file)
    
    # Extract config values
    model_path = config['model_path']
    tokenizer_path = config['tokenizer_path']
    preprocessed_dataset_path = config['preprocessed_dataset_path']
    relation_map_path = config.get('relation_map_path') or os.path.join(model_path, "relation_map.json")
    output_dir = config.get('output_dir') or os.path.join(model_path, 'predictions')
    
    top_k = config.get('top_k', 15)
    batch_size = config.get('batch_size', 128)
    chunk_size = config.get('chunk_size', 100_000)
    NUM_PARALLEL_JOBS = config.get('num_parallel_jobs', 1)
    TAKE_SUBSET = config.get('take_subset', False)
    subset_size = config.get('subset_size', 1000)
    num_leaves = config.get('num_leaves', 7)
    root_nodes = config.get('root_nodes', 128)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Load relation map
    with open(relation_map_path, 'r') as f:
        relation_map = json.load(f)
    
    # Load model
    model = NamedModel(model_path)
    logger.info(f"Loaded model from {model_path}")
    predictions_path = output_dir
    os.makedirs(predictions_path, exist_ok=True)

    dataset = load_from_disk(preprocessed_dataset_path)
    logger.info(f"Loaded dataset from {preprocessed_dataset_path}")


    if TAKE_SUBSET:
        dataset = dataset.select(range(subset_size))

    # later on pass head_lengths from the dataset to speed up -- get head and head_len from there
    keep = ["id", "input_ids", "attention_mask", "head", "head_len", "position", "relation", "relation_num"]

    to_remove = [c for c in dataset.column_names if c not in keep]

    dataset = dataset.map(expand_dicts, batched=True, batch_size=1000, num_proc=100,
        remove_columns=to_remove,
        fn_kwargs={
            "relation_map": relation_map,
            "tokenizer": tokenizer,
        },
        desc='Expand head lists to put each head and relation in a separate example',
    )


    # to run on NUM_PARALLEL_JOBS gpus
    job_id, task_id, is_slurm = get_job_info()
    part_size = len(dataset) // NUM_PARALLEL_JOBS # number of parts = number of array jobs
    start_idx = task_id * part_size
    end_idx = (task_id + 1) * part_size if task_id < NUM_PARALLEL_JOBS - 1 else len(dataset)
    logger.info(f"This job predicts tails for examples from {start_idx} to {end_idx} out of {len(dataset)}")

    chunk_paths = []
    for start in range(start_idx, end_idx, chunk_size):

        end = min(start + chunk_size, len(dataset))
        chunk_dir = os.path.join(predictions_path, f'top_{top_k}_{start}-{end}')
        # Check if chunk already exists and skip if so
        if os.path.exists(chunk_dir) and len(os.listdir(chunk_dir)) > 0:
            print(f'Chunk {start}-{end} already exists, skipping...')
            chunk_paths.append(chunk_dir)
            continue

        ds_chunk = dataset.select(range(start, end))
        result_chunk = ds_chunk.map(
            get_top_predictions,
            batched=True,
            batch_size=batch_size,
            num_proc=None,  # Changed from num_proc=1 to disable multiprocessing with CUDA
            desc=f"Predicting tails with top {top_k} tokens {start_idx}-{end_idx}",
            load_from_cache_file=False,
            fn_kwargs={
                "model": model.model,
                "tokenizer": tokenizer,
                "top_k": top_k,
                "num_leaves": num_leaves,
                "root_nodes": root_nodes,
            },
        )
        result_chunk.save_to_disk(chunk_dir)
        print(f'saved chunk {start}-{end} to {chunk_dir}')
        chunk_paths.append(chunk_dir)

    datasets_list = [load_from_disk(p) for p in chunk_paths]
    result = concatenate_datasets(datasets_list)

    if TAKE_SUBSET:
        path_to_save = os.path.join(predictions_path, f'top_{top_k}_subset_{subset_size}seq')
    else:
        if NUM_PARALLEL_JOBS == 1:
            path_to_save = os.path.join(predictions_path, f'top_{top_k}')
        else:
            path_to_save = os.path.join(predictions_path, f'top_{top_k}_{start_idx}-{end_idx}')

    result.save_to_disk(path_to_save)

    print(f'saved to {path_to_save}')
    print(result)

    # Delete individual chunks after successful concatenation
    for chunk_path in chunk_paths:
        if os.path.exists(chunk_path):
            shutil.rmtree(chunk_path)
            logger.info(f'Deleted chunk {chunk_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict tail entities using trained GraphMERT model')
    parser.add_argument('--yaml_file', type=str, required=True, help='Path to YAML configuration file')
    args = parser.parse_args()
    main(args.yaml_file)