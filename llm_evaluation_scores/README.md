# LLM Evaluation Scores

This module provides LLM-based evaluation utilities for assessing the quality of predicted knowledge graph triples.

### Prerequisites

1. vLLM environment set up (created in `llm_helper_utils/README.md`)
2. A dataset with predicted triples containing columns: `head`, `relation`, `tail`, and `text` (or `sequence`)
3. Access to a compatible evaluator LLM model (e.g., Qwen3-32B)
4. GPU resources for vLLM inference


Before running any of these scripts, edit the prompt examples in `prompts_scores.py` to fit your domain.
For running any of these scripts, activate your vLLM environment. 

## Prompts
`prompts_scores.py`

Contains the following system prompts used in the LLM evaluation:
- `system_prompt_fact_score_seq_only`: Strict factual evaluation based on source text only
- `system_prompt_fact_score_general`: Factual evaluation allowing general knowledge
- `system_prompt_validity_score`: Simple validity check with reasoning

## Fact Score Evaluation
`fact_score.py`

### Overview

This utility evaluates whether predicted triples are factually correct given a source text sequence. It filters predictions by asking the LLM to accept or reject each triple based on context support and logical alignment.

### Evaluation Modes

- **Sequence Only** (set `sequence_only: true`): Evaluates triples strictly based on the source text. Rejects triples that aren't explicitly supported by the context.
- **General Knowledge** (set `sequence_only: false`): Allows triples that are factually correct based on general knowledge of the model, even if not perfectly aligned with the source text.

### Usage

To run the evaluation script:

```bash
python llm_evaluation_scores/fact_score.py
```

### Configuration Variables

Configure the evaluation in `fact_score_config.yaml` or directly in the script:

| Variable | Type | Description |
|----------|------|-------------|
| `predictions_path` | `str` | Path to the dataset with triples (`.csv` or HF dataset) |
| `path_to_save` | `str` | Output directory. Defaults to `null`, which uses dirname of `predictions_path` |
| `sequence_only` | `bool` | If `true`, evaluate based only on source text; if `false`, allow LLM's general knowledge (default: `true`) |
| `take_subset` | `bool` | Set `true` to test on a small sample (default: `false`) |
| `subset_size` | `int` | Number of samples when `take_subset` is `true` |
| `batch_size` | `int` | Samples per model inference pass|
| `num_batches` | `int` | Number of batches per one script run;|
| `offset` | `int` | Starting offset for dataset indexing (default: `0`) |
| `model_id` | `str` | Local path to the LLM model |
| `model_name` | `str` | Human-readable name used only in output directory naming (e.g., `qwen3-32b`). |
| `tensor_parallel_size` | `int` | Number of GPUs for tensor parallelism (default: `1`). Set to match your available GPUs |
| `max_model_len` | `int` | Maximum token context length for LLM (default: `8192`) |
| `sampling.*`           | `dict`    | Sampling parameters for vLLM |

#### Variable Details

- `sequence_only`: Uses two different prompts. When `true`, the LLM only accepts triples that are explicitly supported by the source text. When `false`, it can accept triples based on general knowledge of the model even if not directly stated in the text.
- `take_subset` / `subset_size`: For testing purposes. Set `take_subset: true` to run on only `subset_size` samples instead of the full dataset.
- `num_batches`/ `batch_size`: Together determine how many samples each job handles: `batch_size × num_batches` samples per job.
- `sampling.*`: vLLM sampling parameters. See [vLLM SamplingParams documentation](https://docs.vllm.ai/en/stable/).


### Output

The script:
1. Evaluates each triple and adds an `accepted` column (`yes` or `no`). See the reasoning trace in the generated log.
2. Filters to keep only accepted triples.
3. Saves the filtered dataset. The len of the saved dataset corresponds to the number of accepted triples.

### Post-Processing

After running multiple jobs, combine the separate outputs into one dataset, calculate FactScore and delete the individual chunks using `unite_score_chunks.ipynb`.

---

## Validity Score Evaluation
`validity_score.py`

An evaluation that checks if triples are valid (yes/no/maybe) with a brief explanation. Uses the prompt from `prompts_scores.py`.

---

### Usage
To run the validity score evaluation script:

```bash
python llm_evaluation_scores/validity_score.py
```
### Configuration Variables

Similar to `fact_score.py`. Configure the evaluation in `validity_score_config.yaml`.

### Output
The script:
1. Evaluates each triple and adds a `verdict` column with values: `yes`, `no`, or `maybe`. See the reasoning trace in the generated log.
2. Saves the dataset.   

### Post-Processing
After running multiple jobs, combine the separate outputs into one dataset, calculate ValidityScore and delete the individual chunks using `unite_score_chunks.ipynb`.