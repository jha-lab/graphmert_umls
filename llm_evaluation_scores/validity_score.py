import os, sys
import yaml
from vllm import LLM, SamplingParams
from datasets import Dataset

from pathlib import Path

import logging

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.job_utils import get_job_info
from prompts_scores import system_prompt_validity_score


# ===== Load Configuration =====
def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "fact_score_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


config = load_config()

job_id, task_id, is_slurm = get_job_info()

# ===== Setup Logging =====
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logger.info('initializing model')

TAKE_SUBSET = config.get('take_subset', False)
subset_size = config.get('subset_size', 500)

predictions_path = config['predictions_path']
path_to_save = config.get('path_to_save') or os.path.dirname(predictions_path) 

logger.info(f"loading dataset from {predictions_path}")
if predictions_path.endswith('.csv'):
    dataset = Dataset.from_csv(predictions_path, cache_dir=os.environ["HF_DATASETS_CACHE"])
else:
    dataset = Dataset.load_from_disk(predictions_path)
if 'relation_type' in dataset.column_names:
    dataset = dataset.rename_column('relation_type', 'relation')


batch_size = config['batch_size'] # take at one model inference pass
num_batches = config.get('num_batches', 4)

offset = config.get('offset', 0)
start_idx = batch_size * num_batches * task_id + offset
end_idx = start_idx + batch_size * num_batches

if TAKE_SUBSET:
    subset_size = min(subset_size, len(dataset))
    start_idx = 0
    end_idx = subset_size

end_idx = min(end_idx, len(dataset))
logger.info(f"taking dataset range {start_idx} - {end_idx}")
dataset = dataset.select(range(start_idx, end_idx))


# ==== loading the LLM ========
model_id = config['model_id']
model_name = config.get('model_name', 'llm')
tensor_parallel_size = config.get('tensor_parallel_size', 1)
max_model_len = config.get('max_model_len', 8192)

logger.info(f"loading model from {model_id}")
llm = LLM(
        model=model_id, trust_remote_code=config.get('trust_remote_code', False),
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
    )

logger.info('Initializing sampling parameters')
sampling_params = SamplingParams(
    temperature=config.get('sampling', {}).get('temperature', 0.6),
    top_p=config.get('sampling', {}).get('top_p', 0.95),
    top_k=config.get('sampling', {}).get('top_k', 20),
    max_tokens=config.get('sampling', {}).get('max_tokens', 8192),
    min_p=config.get('sampling', {}).get('min_p', 0),
)
# ==== loading the LLM ========


system_prompt = system_prompt_validity_score

def extract_answer(response: str) -> list:
    """
    Extracts the rightmost JSON list from an LLM response string
    and returns it as a Python list.
    """
    start, end = response.rfind('['), response.rfind(']')
    if start == -1 or end == -1 or start > end:
        return ""
    candidate = response[start+1:end].strip().lower()
    if candidate in ("yes", "no", "maybe") :
        return candidate
    else:
        return ""
    

def format_vllm_chat_messages(examples, pos_examples=None, neg_examples=None):
    prompts = []

    for i in range(len(examples["head"])):
        head = examples["head"][i]
        relation = examples["relation"][i]
        tail = examples["tail"][i]

        messages = [{"role": "system", "content": system_prompt}]

        # Add positive examples
        if pos_examples is not None:
            for u, a in pos_examples:
                messages += [
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": a},
                ]

        query = f"{head} {relation} {tail}\nOutput:"
        messages.append({"role": "user", "content": query})

        prompts.append(messages)

    return prompts



def evaluate_tails(examples, indices, pos_examples=None, sampling_params=sampling_params):
    prompts = format_vllm_chat_messages(examples, pos_examples)
    outputs = llm.chat(prompts, sampling_params=sampling_params, use_tqdm=False)

    accepted = []
    for out, example_idx in zip(outputs, indices):
        response = out.outputs[0].text
        clean_response = extract_answer(response)
        accepted.append(clean_response)

        example_idx = example_idx - indices[0]
        head = examples["head"][example_idx]
        relation = examples["relation"][example_idx]
        tail = examples["tail"][example_idx]
        print(f"{example_idx} {head} | {relation} | {tail}\n Generated text: {response!r}")
        print("-" * 40)

    return {"verdict": accepted}


dataset = dataset.map(evaluate_tails, batched=True, batch_size=batch_size, with_indices=True,
                        desc='Evaluating triples'
                    )


selected_columns = ['id', 'head', 'relation', 'tail', 'verdict'] if 'id' in dataset.column_names else ['head', 'relation', 'tail', 'verdict']
dataset = dataset.select_columns(selected_columns)

prefix = 'validated'
if TAKE_SUBSET:
    path_to_save = os.path.join(path_to_save, f'{prefix}_{model_name}_subset_{subset_size}_{start_idx}-{end_idx}')
else:
    path_to_save = os.path.join(path_to_save, f'{prefix}_{model_name}_{start_idx}-{end_idx}')

dataset.save_to_disk(path_to_save)
logger.info(f'Accepted: saved to {path_to_save}')
logger.info(dataset)