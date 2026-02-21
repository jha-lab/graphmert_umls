# LLM Helper Utils

This directory contains utilities for LLM-based dataset processing: entity discovery, relation adding, and token combining.

### Creating vLLM Environment

The files are `vllm_environment.yml` and `vllm_requirements.txt`. To create and activate the vLLM conda environment, run:
```bash
conda env create -n vllm -f vllm_environment.yml
conda activate vllm
python -m pip install -r vllm_requirements.txt --no-input
```
Note that we will use the same vLLM environment for all three utilities in graphmert/llm_helper_utils and later for graphmert/llm_evaluation_scores.

If you face an error like: `Value error, invalid literal for int() with base 10: 'MIG-bb21f702-c580-5a71-a5e1-d39872153ff3'...`, please update your NVIDIA driver to the latest version or switch to GPUs with later versions.

### Prerequisites
1. vLLM environment set up.
2. Access to a compatible LLM model.
3. GPU resources for vLLM inference.

For running any of these scripts, make sure to activate your vLLM environment first.
Qwen3-32B works well as a helper LLM for these tasks, but perhaps a smaller model, e.g. Qwen3-14B or other compact one is sufficient depending on your domain. Qwen3-7B is likely too small.

**Important**: For each of the utilities, you need to create your own prompt examples in the corresponding `*_prompts.py` file to fit your domain. Then, update the corresponding list at the end of each prompt file. For prompts, choose diverse examples that cover what you expect in your dataset. 2-3 examples are recommended. You may generate examples with an advanced LLM and edit them.

### Adding Your Own Prompt Examples

To customize the prompt examples used for LLM filtering, edit the `PROMPT_EXAMPLES` list in `combine_tails_prompts.py`. Each example should be a tuple of (user_input, assistant_output, explanation) and will be included in the prompt for every batch.

**Example:**
```python
PROMPT_EXAMPLES = [
    (example_user_1, example_assistant_1, example_explanation_1),
    # Add your own examples here
]
```

Be sure to provide enough positive and negative examples (when applicable) to get the desired output quality. It is a good idea to make test runs and inspect the outputs to see if your prompts produce high quality results.

### Main Setup Variables (each utility)
| Variable              | Type    | Description |
|----------------------|---------|-------------|
| dataset_path     | str     | Path to the input dataset |
| tokenizer_path       | str     | Name on a HuggingFace  or local path to the tokenizer|
| take_subset          | bool    | If true, only process a subset of the data |
| subset_size          | int     | Number of samples to process if take_subset is true |
| printout             | bool    | If true, print LLM outputs for each example |
| batch_size           | int     | Number of samples per LLM inference batch |
| num_batches          | int     | Number of batches per job |
| offset               | int     | Starting offset for dataset indexing |
| model_id             | str     | Path to the LLM model weights |
| model_name           | str     | Model name for output file naming |
| tensor_parallel_size | int     | Number of GPUs for tensor parallelism |
| max_model_len        | int     | Maximum model context length |
| trust_remote_code   | bool    | Whether to trust remote code for the model; default is false |
| sampling.*           | dict    | Sampling parameters for vLLM (see vLLM docs) |


### Additional Variables for entity_discovery and relation_adding
| Variable              | Type    | Description |
|----------------------|---------|-------------|
| output_path        | str     | Path to save the output dataset |
| original_dataset_size | int | Length of the dataset |

- `original_dataset_size`: Total number of samples in the original dataset. Used for merging output chunks in the subsequent cleaning scripts.



### Variable Details

- `take_subset` / `subset_size`: For quick tests, process only a subset of the data.

- `printout`: Print LLM generated output for each example (for debugging/inspection).
- `batch_size`: Number of examples processed per LLM call. Decide based on your GPU memory capacity. Larger batch sizes lead to faster processing.
- `num_batches`: How many batches to process per job. Choose based on length on the input dataset and number of jobs your run in parallel:
one job processes ``batch_size × num_batches`` samples. You have to process the whole dataset, so choose these two parameters accordingly considering number of jobs. It is also recommended to split lage dataset into multiple chunks and submit multiple jobs to handle failures. In case if a single job fails, re-running will take less time. E.g., avoid running jobs longer than 10h unless you are sure they will complete successfully.
- `offset`: Start index in the dataset (useful for skipping processed data).
- `model_id`: Local path to the vLLM-compatible model weights.
- `model_name`: Used only in output directory naming.
- `tensor_parallel_size`: Number of GPUs to use for model parallelism.
- `max_model_len`: Maximum token context window for the LLM.
- `sampling.*`: For vLLM sampling parameters (temperature, top_p, top_k, max_tokens, min_p, etc.) see [vLLM SamplingParams documentation](https://docs.vllm.ai/en/latest/dev/sampling_params.html). Choose the recommended settings for your LLM.

It is highly recommended to inspect the quality of LLM-generated outputs in the generated log by setting `printout: true` in the config file.

# Utilities

## 1. entity_discovery

`entity_discovery.py` finds head entity spans in the tokenized dataset following the prompt guidlines. You can obtain tokenized dataset after the very 1st pipeline step (`run_tokenization.py`). Some of identified entities may be hallucinations, so you need to further filter them using `filter_heads.py`. 

In prompts, include both positive and negative domain-speicific head entity examples. ~2 positive and ~1 negative examples are recommended. 

### Usage
By default, both utilities read configuration from `entity_discovery_args.yaml`.  Set up your variables there before running.
To run the entity discovery script (uses GPU):

```bash
python llm_helper_utils/entity_discovery/entity_discovery.py 
```

Next, run `find_heads_positions.py` to filter out hallucinated head entities that are not in the dataset (cpu-only job):

```bash
python llm_helper_utils/entity_discovery/find_heads_positions.py 
```
Read the log to see the path where the cleaned dataset is saved.


## 2. relation_adding
We add relations to the discovered head entities for prediction with the trained model. Note that they are required for prediction only, but not for training. For training we use relations from the seed KG only, but for prediction we add more relations (same set of allowed relations) to increase coverage.
`add_relations.py` matches allowed relations to head entities found on the previous `entity_discovery` step. The list of allowed relations can be obtained after running `seed_kg_injection_algorithm.ipynb` on previous step: these are all relations included in the seed KG. **You must add list of your allowed relations in`add_relations_prompts.py`**, MEANING_EXPL, and REL_USAGE_EXAMPLES. However, it is ok if you decide to remove MEANING_EXPL and the corresponding part in the prompt. 

### Usage
By default, both utilities read configuration from `add_relations_args.yaml`. Set up your variables there before running.
To run the relation adding script (uses GPU):
```bash
python llm_helper_utils/relation_adding/add_relations.py 
```

Next, run `filter_relations.py` to filter out hallucinated relations (cpu-only job). It removes all relations but in the ALLOWED_RELATIONS list defined in `filter_relations_prompts.py`:

```bash
python llm_helper_utils/relation_adding/filter_relations.py 
```
Read the log to see the path where the cleaned dataset is saved.

## 3. combine_tails

Before running this utility, please check that `predict_tail.py` predicted candidate tokens that you can use for tail combination: they are relevant to your domain and have good coverage of tail entities. If not (e.g. all the predictions are stop words), consider re-training GraphMERT with more data or a adjust training hyperparameters.

| Variable              | Type    | Description |
|----------------------|---------|-------------|
| predictions_path | str     | Path to the dataset with graphmert-predicted candidate tokens for tails |

`combine_tails.py` filters and combine candidate tokens after graphmert prediction into multi-token tail entities for each triple. 
It works on the dataset that you get upon running `predict_tail.py`. The dataset in `predictions_path` should contain columns: `head`, `relation`, `candidate_tokens`, and `text` (or `sequence`). **You must create your own prompts in `combine_tails_prompts.py`.** 
Some of the generated tails will be helper LLM hallucinations, so you can further filter them using `unite_chunks_and_clean.ipynb`.

### Usage
By default, the script reads configuration from `combine_tails_config.yaml`. Set up you vairables there before running.

To run the tail combination script (uses GPU):

```bash
python llm_helper_utils/combine_tails/combine_tails.py 
```

Second, run `llm_helper_utils/combine_tails/unite_chunks_and_clean.ipynb` (cpu-only job) to merge output chunks and clean hallucinated tails. Please review the qulaity of the tails in this notebook. This is your final resulting dataset with head-relation-tail triples, so make sure that the quality is satisfactory.
