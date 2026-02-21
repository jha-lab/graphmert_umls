"""
After running this script, you should combine the separate outputs from each job into one dataset.

Configuration is loaded from fact_score_config.yaml
"""

import os, sys
from pathlib import Path
import logging
import yaml

from vllm import LLM, SamplingParams
from datasets import Dataset

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.job_utils import get_job_info
from prompts_scores import system_prompt_fact_score_seq_only, system_prompt_fact_score_general


# ===== Load Configuration =====
def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "fact_score_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


config = load_config()

# ===== Setup Logging =====
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

job_id, task_id, is_slurm = get_job_info()

# ===== Load Config Variables =====

TAKE_SUBSET = config.get('take_subset', False)
SEQUENCE_ONLY = config.get('sequence_only', True)
subset_size = config.get('subset_size', 2000)

logger.info(f'SEQUENCE_ONLY: {SEQUENCE_ONLY}, TAKE_SUBSET: {TAKE_SUBSET}, task_id: {task_id}')

# ===== Dataset Settings =====
predictions_path = config['predictions_path']
path_to_save = config.get('path_to_save') or os.path.dirname(predictions_path)

logger.info(f"loading dataset from {predictions_path}")
if predictions_path.endswith('.csv'):
    dataset = Dataset.from_csv(predictions_path)
else:
    dataset = Dataset.load_from_disk(predictions_path)

if 'relation_type' in dataset.column_names:
    dataset = dataset.rename_column('relation_type', 'relation')
if 'sequence' in dataset.column_names:
    dataset = dataset.rename_column('sequence', 'text')

# ===== Batch Settings =====
batch_size = config.get('batch_size', 1000)
num_batches = config.get('num_batches', 20)
offset = config.get('offset', 0)

start_idx = batch_size * num_batches * task_id + offset
end_idx = start_idx + batch_size * num_batches

if TAKE_SUBSET:
    start_idx = 0
    end_idx = subset_size

end_idx = min(end_idx, len(dataset))
logger.info(f"taking dataset range {start_idx} - {end_idx}")
dataset = dataset.select(range(start_idx, end_idx))

# ===== Load LLM =====
model_id = config['model_id']
model_name = config.get('model_name', 'llm')
tensor_parallel_size = config.get('tensor_parallel_size', 1)
max_model_len = config.get('max_model_len', 8192)

logger.info(f"loading model from {model_id}")
llm = LLM(
    model=model_id,
    trust_remote_code=config.get('trust_remote_code', False),
    tensor_parallel_size=tensor_parallel_size,
    max_model_len=max_model_len,
)

# ===== Sampling Parameters =====
sampling_config = config.get('sampling', {})
logger.info('Initializing sampling parameters')
sampling_params = SamplingParams(
    temperature=sampling_config.get('temperature', 0.6),
    top_p=sampling_config.get('top_p', 0.95),
    top_k=sampling_config.get('top_k', 20),
    max_tokens=sampling_config.get('max_tokens', 8192),
    min_p=sampling_config.get('min_p', 0),
)

# ===== Select System Prompt =====
if SEQUENCE_ONLY:
    system_prompt = system_prompt_fact_score_seq_only
else:
    system_prompt = system_prompt_fact_score_general


def extract_answer(response: str) -> list:
    """
    Extracts the rightmost JSON list from an LLM response string
    and returns it as a Python list.
    """
    start, end = response.rfind('['), response.rfind(']')
    if start == -1 or end == -1 or start > end:
        return ""
    candidate = response[start+1:end].strip().lower()
    if candidate in ("yes", "no") :
        return candidate
    else:
        return ""
    

def format_vllm_chat_messages(examples):
    prompts = []

    for i in range(len(examples["head"])):
        head = examples["head"][i]
        relation = examples["relation"][i]
        tail = examples["tail"][i]
        sequence = examples["text"][i]

        messages = [{"role": "system", "content": system_prompt}]

        query = f"Input:\n{sequence}\n\nhead: {head}\nrelation: {relation}\ntail: {tail}\n\nOutput:"
        messages.append({"role": "user", "content": query})

        prompts.append(messages)

    return prompts



def evaluate_tails(examples, indices, sampling_params=sampling_params):
    prompts = format_vllm_chat_messages(examples)
    outputs = llm.chat(prompts, sampling_params=sampling_params, use_tqdm=False)

    accepted = []
    for out, example_idx in zip(outputs, indices):
        response = out.outputs[0].text
        clean_response = extract_answer(response)
        accepted.append(clean_response)

        print(f"{example_idx} Generated text: {response!r}")
        print("-" * 40)

    return {"accepted": accepted}


dataset = dataset.map(evaluate_tails, batched=True, batch_size=batch_size, with_indices=True,
                        desc='Evaluating tails'
                      )


dataset = dataset.filter(lambda ex: ex["accepted"] == "yes",
                             desc="Keep only accepted=yes")

dataset = dataset.remove_columns(['accepted'])
if SEQUENCE_ONLY:
        prefix = 'accepted_seq_only'
else:    
    prefix = 'accepted'
if TAKE_SUBSET:
    path_to_save = os.path.join(path_to_save, f'{prefix}_{model_name}_subset_{subset_size}_{start_idx}-{end_idx}')
else:
    path_to_save = os.path.join(path_to_save, f'{prefix}_{model_name}_{start_idx}-{end_idx}')

dataset.save_to_disk(path_to_save)
logger.info(f'Accepted: saved to {path_to_save}')
logger.info(dataset)