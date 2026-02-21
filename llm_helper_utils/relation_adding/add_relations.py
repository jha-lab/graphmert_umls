from vllm import LLM, SamplingParams
from datasets import Dataset
import json
import logging
import yaml
import os, sys
from transformers import AutoTokenizer

# Ensure repository root is on sys.path so top-level imports like `utils` work
# when running this file directly (e.g. `python llm_helper_utils/relation_adding/add_relations.py`).
# Running with `-m` still works and this is a no-op in that case.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.job_utils import get_job_info

from add_relations_prompts import SYSTEM_CONTEXT, positive_examples


# Default config path next to this file
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'add_relations_args.yaml')

def load_config(config_path: str | None = None) -> dict:
    path = config_path or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {path} did not contain a mapping (dict)")
    return cfg

# Load configuration (default file)
cfg = load_config(None)

tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer_path'])

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logger.info('initializing model')


model_id = cfg['model_id']
model_name = cfg['model_name']


# load the daatset
dataset_path = cfg['dataset_path']
output_path = cfg['output_path']
logger.info(f"loading dataset from {dataset_path}")
dataset = Dataset.load_from_disk(dataset_path)

job_id, task_id, is_slurm = get_job_info()
batch_size = cfg['batch_size']
num_batches = cfg['num_batches']
start_idx = batch_size * num_batches * task_id
end_idx = start_idx + batch_size * num_batches

PRINTOUT = cfg['printout']
TAKE_SUBSET = cfg['take_subset']
if TAKE_SUBSET:
    subset_size = cfg['subset_size']
    start_idx = 0
    end_idx = subset_size

end_idx = min(end_idx, len(dataset))
logger.info(f"taking dataset range {start_idx} - {end_idx}")
dataset = dataset.select(range(start_idx, end_idx))


logger.info(f"loading model from {model_id}")
llm = LLM(
        model=model_id,
        trust_remote_code=cfg['trust_remote_code'],
        tensor_parallel_size=cfg['tensor_parallel_size'],
        max_model_len=cfg['max_model_len'],
    )

logger.info('Initializing sampling parameters')
sampling_cfg = cfg['sampling']
sampling_params = SamplingParams(
    temperature=sampling_cfg['temperature'],
    top_p=sampling_cfg['top_p'],
    top_k=sampling_cfg['top_k'],
    max_tokens=sampling_cfg['max_tokens'],
    min_p=sampling_cfg['min_p'],
)


def extract_answer(response: str) -> str:
    """
    Extracts the rightmost JSON object from an LLM response string
    and returns it as a JSON string.
    """
    start, end = response.rfind('{'), response.rfind('}')
    if start == -1 or end == -1 or start > end:
        return ""  # No valid JSON found; return empty string
    candidate = response[start:end+1]
    try:
        # Validate that candidate is valid JSON by loading it
        obj = json.loads(candidate)
        # Return a standardized JSON string
        return json.dumps(obj, separators=(',', ':'))
    except json.JSONDecodeError:
        return ""  # Fallback to empty string if decoding fails
    

def format_vllm_chat_messages(examples, pos_examples, neg_examples=None):
    prompts = []
    to_call_indices = []

    for i in range(len(examples["input_ids"])):
        messages = [{"role": "system", "content": SYSTEM_CONTEXT}]

        heads = examples['head_positions'][i]
        heads = json.loads(heads)
        heads = list(heads.keys())
        if heads is None or len(heads) == 0:
            continue
        
        to_call_indices.append(i)

        input_ids = examples['input_ids'][i]
        sequence = tokenizer.decode(input_ids, use_special_tokens=False)
        query = {"role": "user", "content": [{"type": "text", "text": f"Input: \nsequence: {sequence}\nheads: {heads}\n\nOutput:"}]}

        # Add positive examples
        if pos_examples is not None:
            for u, a, e in pos_examples:
                messages += [
                    {"role": "user", "content": [{"type": "text", "text": u}]},
                    {"role": "assistant", "content": a},
                    {"role": "user", "content": [{"type": "text", "text": "Explanation:"}]},
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

        messages.append(query)
        prompts.append(messages)

    return prompts, to_call_indices



logger.info(f"systmm prompt used: {SYSTEM_CONTEXT}")
logger.info(f"positive examples used: {positive_examples}")


def find_heads(examples, idx):
    messages, to_call_indices = format_vllm_chat_messages(examples, positive_examples, neg_examples=None)
    output = llm.chat(messages, sampling_params=sampling_params, use_tqdm=False)

    responses = [""] * len(examples['input_ids'])
    for i in range(len(output)):
        out, output_idx, example_idx = output[i], to_call_indices[i], idx[i]
        response = out.outputs[0].text
        clean_response = extract_answer(response)
        responses[output_idx] = clean_response
        if PRINTOUT:
            print(f"{example_idx} Generated text: {response!r}")
            print("-" * 40)
    
    examples['response'] = responses
    return examples


dataset = dataset.map(find_heads, batched=True, batch_size=batch_size, with_indices=True,
                        desc='Filtering tails'
                      )

if TAKE_SUBSET:
    path_to_save = os.path.join(output_path, f'{model_name}_relations_subset_{subset_size}_{start_idx}-{end_idx}')
else:
    path_to_save = os.path.join(output_path, f'{model_name}_relations_{start_idx}-{end_idx}')
dataset.save_to_disk(path_to_save)
logger.info(f'saved to {path_to_save}')
logger.info(dataset)