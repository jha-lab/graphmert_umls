# The code is designed to run in a distributed environment, such as SLURM.

import yaml
from dataclasses import dataclass, field
import argparse
from typing import Optional

from vllm import LLM, SamplingParams
from datasets import Dataset
import json
import ast, re
import os
import logging

from transformers import AutoTokenizer

from entity_discovery_prompts import SYSTEM_CONTEXT, POSITIVE_PROMPT_EXAMPLES, NEGATIVE_PROMPT_EXAMPLES

# Ensure repository root is on sys.path so top-level imports like `utils` work
# when running this file directly (e.g. `python llm_helper_utils/entity_discovery/entity_discovery.py`).
# Running with `-m` still works and this is a no-op in that case.
import sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.job_utils import get_job_info


@dataclass
class ModelConfig:
    """Configuration for LLM model."""
    model_id: str = field(
        default=None,
        metadata={"help": "Path or identifier of the model to be used."},)
    model_name: str = field(
        default="qwen3-32b",
        metadata={"help": "Name of the model being used for generating output paths."},
    )
    tokenizer_path: str = field(default=None)
    tensor_parallel_size: int = field(default=1)
    max_model_len: int = field(default=8192)
    trust_remote_code: bool = field(default=False)
    dtype: str = field(
        default="auto",
        metadata={"help": "Data type for model weights (e.g., 'auto', 'float16', 'bfloat16', 'float32')"},
    )

@dataclass
class SamplingConfig:
    """Configuration for sampling parameters."""
    temperature: float = field(default=0.6)
    top_p: float = field(default=0.95)
    top_k: int = field(default=20)
    max_tokens: int = field(default=8192)
    min_p: float = field(default=0)

@dataclass
class DataConfig:
    """Configuration for data processing."""
    dataset_path: str = field(default=None)
    output_path: Optional[str] = field(default=None)
    batch_size: int = field(default=10_000)
    num_batches: int = field(default=10)
    subset_size: Optional[int] = field(default=None)
    take_subset: bool = field(default=False)
    original_dataset_size: int = field(default=None)


    def __post_init__(self):
        if self.output_path is None and self.dataset_path is not None:
            self.output_path = f"{self.dataset_path}_heads"


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None, help='Path to config YAML (default: llm_helper_utils/entity_discovery/entity_discovery_args.yaml)')
args = parser.parse_args()

config_path = args.config if args.config is not None else 'llm_helper_utils/entity_discovery/entity_discovery_args.yaml'
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}")
with open(config_path, 'r') as f:
    user_config = yaml.safe_load(f)

config = {
    'model': ModelConfig(**user_config.get('model', {})),
    'sampling': SamplingConfig(**user_config.get('sampling', {})),
    'data': DataConfig(**user_config.get('data', {})),
    'printout': user_config.get('printout', True),
}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger.info('initializing model')

PRINTOUT = config['printout']
TAKE_SUBSET = config['data'].take_subset if hasattr(config['data'], 'take_subset') else False


if config['data'].output_path is None:
    raise ValueError("output_path must be set")

logger.info(f"loading dataset from {config['data'].dataset_path}")
dataset = Dataset.load_from_disk(config['data'].dataset_path)


job_id, task_id, is_slurm = get_job_info()
batch_size = config['data'].batch_size
num_batches = config['data'].num_batches
start_idx = batch_size * num_batches * task_id
end_idx = start_idx + batch_size * num_batches

if TAKE_SUBSET:
    subset_size = config['data'].subset_size
    assert subset_size is not None, "subset_size must be set if TAKE_SUBSET is True"
    start_idx = 0
    end_idx = subset_size
    logger.info(f"TAKE_SUBSET is True, using subset_size={subset_size}")

end_idx = min(end_idx, len(dataset))
logger.info(f"taking dataset range {start_idx} - {end_idx}")
dataset = dataset.select(range(start_idx, end_idx))


logger.info(f"loading model from {config['model'].model_id}")

llm = LLM(
    model=config['model'].model_id,
    trust_remote_code=config['model'].trust_remote_code,
    tensor_parallel_size=config['model'].tensor_parallel_size,
    max_model_len=config['model'].max_model_len,
    dtype=config['model'].dtype, 
)

logger.info('Initializing sampling parameters')
sampling_params = SamplingParams(
    temperature=config['sampling'].temperature,
    top_p=config['sampling'].top_p,
    top_k=config['sampling'].top_k,
    max_tokens=config['sampling'].max_tokens,
    min_p=config['sampling'].min_p,
)

tokenizer = AutoTokenizer.from_pretrained(
    config['model'].tokenizer_path,
    trust_remote_code=config['model'].trust_remote_code,
)
    

def extract_rightmost_list(response: str) -> list:
    # 1) Strip any lines that start with ``` (with or without language tag)
    #    but keep what's between them.
    response = re.sub(r'(?m)^```.*\n?', "", response)

    # 2) Find all [ ... ] spans (non-greedy, across newlines)
    matches = re.findall(r"\[.*?\]", response, flags=re.DOTALL)
    if not matches:
        return []
    candidate = matches[-1]

    lst = None
    try:
        lst = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            lst = ast.literal_eval(candidate)
        except Exception:
            return []
    cleaned = []
    for item in lst:
        if item is Ellipsis:
            continue
        cleaned.append(str(item))

    return cleaned


def format_vllm_chat_messages(examples, pos_examples=POSITIVE_PROMPT_EXAMPLES, neg_examples=NEGATIVE_PROMPT_EXAMPLES):
    prompts = []

    for i in range(len(examples["input_ids"])):
        messages = [
            {"role": "system", "content": SYSTEM_CONTEXT},
            {"role": "user", "parts": [{"text": "I will provide you with examples"}]},
            {"role": "model", "parts": [{"text": "Understood — send the sample and I’ll output entities"}]}
        ]
        if pos_examples is not None:
            for u, a, e in pos_examples:
                messages += [
                    {"role": "user", "content": [{"type": "text", "text": u}]},
                    {"role": "assistant", "content": a},
                    {"role": "user", "content": [{"type": "text", "text": "Explanation of the previous output:"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": e}]},
                ]

        # Add negative examples
        if neg_examples is not None:
            for u, a, e in neg_examples:
                messages += [
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": a},
                    {"role": "user", "content": "Explanation of what is wrong with the previous output:"},
                    {"role": "assistant", "content": e},
                ]
        messages.append({"role": "user", "content": [{"type": "text", "text": "**End of examples**.\nNow read the actual input:"}]})        

        input_ids = examples['input_ids'][i]
        sequence = tokenizer.decode(input_ids, skip_special_tokens=True)

        query = {"role": "user", "content": [{"type": "text", "text": f"Input:\n{sequence}"}]}
        messages.append(query)

        prompts.append(messages)

    return prompts



def find_heads(examples, idx):
    responses = []

    messages = format_vllm_chat_messages(examples, pos_examples=POSITIVE_PROMPT_EXAMPLES, neg_examples=NEGATIVE_PROMPT_EXAMPLES)

    output = llm.chat(messages, sampling_params=sampling_params, use_tqdm=False)
    logger.info("batch idx count=%d, outputs count=%d", len(idx), len(output))
    for out, example_idx in zip(output, idx):
        response = out.outputs[0].text
        clean_response = extract_rightmost_list(response)
        responses.append(clean_response)
        if PRINTOUT:
            print(f"{example_idx} Generated text: {response!r}")
            print("-" * 40)
    
    examples['response'] = responses
    return examples



dataset = dataset.map(find_heads, batched=True, batch_size=batch_size, with_indices=True,
                        desc='Filtering tails'
                    )

if TAKE_SUBSET:
    path_to_save = os.path.join(
        config['data'].output_path,
        f"{config['model'].model_name}_subset_{subset_size}_{start_idx}-{end_idx}"
    )
else:
    path_to_save = os.path.join(
        config['data'].output_path,
        f"{config['model'].model_name}_{start_idx}-{end_idx}"
    )
dataset.save_to_disk(path_to_save)
logger.info(f'saved to {path_to_save}')
logger.info(dataset)