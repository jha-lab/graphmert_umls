from vllm import LLM, SamplingParams
from datasets import Dataset, Features, Sequence, Value
import json
import os, sys
import logging
import re, ast
import yaml
from pathlib import Path
from transformers import AutoTokenizer


"""Before launching the script, check that the MEANING_EXPL in jupyter.prompt_library.combine_tokens_prompts doesn't have relation outside the one used in the dataset, 
and the example set does not contain any relations that are not in the dataset."""

# Ensure repository root is on sys.path so top-level imports like `utils` work
# when running this file directly (e.g. `python llm_helper_utils/relation_adding/add_relations.py`).
# Running with `-m` still works and this is a no-op in that case.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.job_utils import get_job_info
from combine_tails_prompts import SYSTEM_CONTEXT, PROMPT_EXAMPLES



# ===== Load Configuration =====
def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "combine_tails_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


config = load_config()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ===== Load Config Variables =====
job_id, task_id, is_slurm = get_job_info()

PRINTOUT = config.get('printout', True)
TAKE_SUBSET = config.get('take_subset', False)
subset_size = config.get('subset_size', 100)

# ===== Dataset Settings =====
predictions_path = config['predictions_path']

logger.info(f"loading dataset from {predictions_path}")
dataset = Dataset.load_from_disk(predictions_path)

tokenizer_path = config['tokenizer_path']
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# ===== Batch Settings =====
batch_size = config.get('batch_size', 1000)
num_batches = config.get('num_batches', 10)
offset = config.get('offset', 0)

start_idx = batch_size * num_batches * task_id + offset
end_idx = start_idx + batch_size * num_batches

if TAKE_SUBSET:
    start_idx = 0
    end_idx = start_idx + subset_size

end_idx = min(end_idx, len(dataset))
logger.info(f"taking dataset range {start_idx} - {end_idx}")
dataset = dataset.select(range(start_idx, end_idx))

# ===== Load LLM =====
model_id = config['model_id']
model_name = config.get('model_name', 'llm')
ENABLE_THINKING = config.get('enable_thinking', True)

logger.info(f"loading model from {model_id}")
llm = LLM(
    model=model_id,
    trust_remote_code=config.get('trust_remote_code', False),
    tensor_parallel_size=config.get('tensor_parallel_size', 1),
    max_model_len=config.get('max_model_len', 8192),
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


def extract_rightmost_list(response: str) -> list:
    # 1) Strip any lines that start with ``` (with or without language tag)
    #    but kepp what's between them.
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


def format_vllm_chat_messages(examples, pos_examples=None, neg_examples=None):
    prompts = []

    for i in range(len(examples["input_ids"])):
        messages = []
        # Add system prompt
        messages.append({"role": "system", "content": SYSTEM_CONTEXT})
        # Add user prompt with instructions
        messages.append({"role": "assistant", "content": "I understand. I will help you complete triples for the medical knowledge graph by filtering and combining candidate tokens into precise, medically-specific tails. I will avoid vague tails. I'll follow the steps you outlined and provide my reasoning."})
        messages.append({"role": "user","parts": [{"text": "I will provide you with examples"}]})
        messages.append({"role": "assistant","parts": [{"text": "Understood — send the sample and I’ll output tails"}]})

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

        head = examples['head'][i]
        relation = examples['relation'][i]
        predictions = examples['predictions'][i]
        sequence = tokenizer.decode(examples['input_ids'][i], add_special_tokens=False)
        query = {"role": "user", "content": [{"type": "text", "text": f"Input: \n{sequence}\nhead: {head}\nrelation: {relation}\npredictions: {predictions}\n\nOutput:"}]}
        messages.append(query)

        prompts.append(messages)

    return prompts


def add_tails(examples, idx, pos_examples=None, neg_examples=None):
    tails = []

    messages = format_vllm_chat_messages(examples, pos_examples, neg_examples)

    output = llm.chat(messages, sampling_params=sampling_params, use_tqdm=False, chat_template_kwargs={"enable_thinking": ENABLE_THINKING},)
    for out, example_idx in zip(output, idx):
        response = out.outputs[0].text
        clean_response = extract_rightmost_list(response)
        tails.append(clean_response)
        if PRINTOUT:
            print(f"{example_idx} Generated text: {response!r}")
            print("-" * 40)
    
    examples['tails'] = tails
    return examples


new_features = Features({
    "id":      dataset.features["id"],
    "input_ids":      dataset.features["input_ids"],
    "head":           dataset.features["head"],
    "position":       dataset.features["position"],
    "relation":       dataset.features["relation"],
    "predictions":    dataset.features["predictions"],
    "tails":          Sequence(feature=Value("string"), length=-1),
})


keep = new_features.keys()
to_remove = [c for c in dataset.column_names if c not in keep]

dataset = dataset.map(add_tails, batched=True, batch_size=batch_size, with_indices=True,
                      remove_columns=to_remove,
                      features=new_features, 
                        desc='Filtering tails with llm',
                        fn_kwargs={'pos_examples': PROMPT_EXAMPLES},
                      )

logger.info("Finished filtering dataset with llm")


prefix = model_name if ENABLE_THINKING else f'{model_name}_no_thinking'
if TAKE_SUBSET:
    path_to_save = os.path.join(predictions_path, f'{prefix}_subset_{subset_size}_{start_idx}-{end_idx}')
else:
    path_to_save = os.path.join(predictions_path, f'{prefix}_{start_idx}-{end_idx}')
dataset.save_to_disk(path_to_save)
logger.info(f'saved to {path_to_save}')

raise SystemExit(0)  # Exit after saving the expanded dataset
