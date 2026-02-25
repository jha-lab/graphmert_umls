# GraphMert

This folder contains the main GraphMert model implementation and utilities for training. 

## Citation

If you use GraphMERT (code, models, data or data processing scripts) in your work, please cite our TMLR paper:

```
@article{
    belova2026graphmert,
    title={Graph{MERT}: {E}fficient and Scalable Distillation of Reliable Knowledge Graphs from Unstructured Data},
    author={Margarita Belova and Jiaxin Xiao and Shikhar Tuli and Niraj Jha},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2026},
    url={https://openreview.net/forum?id=tnXSdDhvqc},
}
```

## Creating GraphMERT Conda Environment

### Setup Script

To set up the conda environment named `graphmert`, run the following command:

```bash
./setup_graphmert_env.sh
```

### Alternative installation using `environment.yml` and `requirements.txt`

```bash
# Create the env once (skip if you already have it)
conda env create -n graphmert -f environment.yml
conda activate graphmert
# Install pip deps
python -m pip install -r requirements.txt --no-input
python -m spacy download en_core_web_sm
```

### To register the conda environment as a kernel in Jupyter notebooks, run:

```bash
conda activate graphmert
conda install -c anaconda ipykernel -y
python -m ipykernel install --user --name=graphmert
```


# Project Structure

![Description](images/pipeline-methodology.svg "Pipeline")

```
graphmert/
├── graphmert_model/                # Core model implementation
│   ├── __init__.py
│   ├── algos_graphmert.pyx         # Cython algorithms for graph processing
│   ├── collating_graphmert.py      # Data collator for MLM with graph injections;  Masked Node Modelling (MNM) implementation
│   ├── configuration_graphmert.py  # Model configuration
│   └── modeling_graphmert.py       # Model architecture;  SBO loss implementation
├── utils/                          # Training and data processing utilities
│   ├── mlm_utils.py # MLM training logic
│   ├── tokenization_utils.py       # Dataset tokenization
│   ├── trainer_utils.py            # Custom TrainerWithCosineAnealing class for MLM with graph injections
│   └── training_arguments.py       # Custom dataclasses for training with graph injections
│   ├── dataset_preprocessing_utils.py      # Dataset preprocessing: injecting seed KG, forming chain graphs
|   ├── seed_kg_injection_algorithm.ipynb   # Jupyter notebook for generating csv with seed KG injections
│   └── ...
├── llm_helper_utils/               # steps that require LLM helper;
├── llm_evaluation_scores/          # LLM-based evaluation utilities
├── launch_configs/                 # YAML configuration files for run_*.py s
├── run_mlm.py                      # Training entry point
├── run_tokenization.py             # Tokenization entry point
├── run_dataset_preprocessing.py    # Dataset preprocessing entry point
├── predict_tails.py                # Predict tail tokens with trained GraphMERT model
└── README.md
```
---
# Usage

For following steps, set up the YAML configuration files in the `launch_configs/` directory first. Let's assume you have a configuration file named `args_mlm.yaml`training.
We have three main steps: **tokenization**, **dataset preprocessing**, and **model training**. All steps uses the same YAML config file; it is not necessary to set some variables for the latest steps from the beginning. You can set them later as needed.

If using slurm, put the running commands in a slurm script `your_slurm_script.slurm` and submit the job using `sbatch your_slurm_script.slurm`.

The datasets used in this project are available on the Hugging Face Hub:

- [GraphMERT PubMED diabetes abstracts](https://huggingface.co/datasets/jha-lab/GraphMERT_data)
- [Filtered UMLS](https://huggingface.co/datasets/jha-lab/filtered_UMLS)

---
## Environment Variables

Set these environment variables before running scripts:

```bash
export HF_AUTH_TOKEN=your_huggingface_token  # For private datasets
export HF_HOME=/path/to/hf_home              # HuggingFace home
```
---

### 1. Tokenize Dataset

```bash
python run_tokenization.py --yaml_file launch_configs/args_mlm.yaml
```

This step tokenizes the dataset specified in the YAML config file and saves the tokenized dataset to the cache directory, or to `tokenized_dataset_output_path` if specified.
It is a cpu-only step. Do not allocate gpu resources for this step.

### 2. Seed KG Injections
After tokenization, you should discover domain-specific heads for you dataset on the tokenized dataset. Check entity_discovery section in `llm_helper_utils/README.md`.
Next, obtain the seed KG injections using the algorithm in `utils/seed_kg_injection_algorithm.ipynb` (check section below). The seed KG injections will be used in the next preprocessing step (Step 3). Please explicitly exclude relations for which predicting tails with MLM objective is impossible: e.g., custom defined relations in the seed KG that cannot be inferred from the training dataset, or relations with mostly numerical tails. To exclude, put them in `undesired_relations` list in the `seed_kg_injection_algorithm.ipynb`.
Specify the paths to the generated csv files in the YAML config file under `injections_train_path` and `injections_eval_path`; specify the generated JSON with relation mapping file path under `relation_map_path`.


### Entity Linking
Link the discovered heads to UMLS terms and get candidate triples for injection. check README.md and scripts in [umls_mapping](umls_mapping/README.md).

Also describe how to add allowed relations to the discovered heads.

### 3. Preprocess Dataset

```bash
python run_dataset_preprocessing.py --yaml_file launch_configs/args_mlm.yaml
```
This step preprocesses the tokenized dataset for GraphMERT-compatible training: it forms chain graphs using the tokenized dataset and seed KG injections specified by `injections_train_path` and `injections_eval_path`, and saves the processed dataset to the cache directory.
It is a cpu-only step. Do not allocate gpu resources for it.

### 4. Train Model

This step trains the GraphMERT model using the preprocessed dataset of chain graphs. The trained model checkpoints will be saved to the `output_dir` specified in the YAML config file.
This step requires gpu resources for training.
```bash
python run_mlm.py --yaml_file launch_configs/args_mlm.yaml
```
Distributed training on multiple GPUs is supported via HuggingFace [Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer) and [Accelerate](https://huggingface.co/docs/accelerate/en/index).
If you use multiple gpus, set up Accelerate coonfiguration beforehand according to official HuggingFace Accelerate documentation. By default, it will be saved to your `$HF_HOME/accelerate/default_config.yaml`. Then run training as:

```bash
accelerate launch run_mlm.py --yaml_file launch_configs/args_mlm.yaml
```

Here is an example Accelerate configuration for multi-GPU training using 2 GPUs on a single machine:

```yaml
compute_environment: LOCAL_MACHINE
debug: true
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

## Configuration Variables

Here we will explain the most important variables used in the YAML configuration files for tokenization and training. For the full list of variables, refer to the setup files, e.g. `graphmert/utils/training_arguments.py`, `graphmert/graphmert_model/configuration_graphmert.py`.

Note: If a variable is mentioned twice below, e.g. in both tokenization and training setups, that means you may reset it for each step as needed.

### Main Variables for `tokenization_utils.py`

Here are the most important parameters to set before running the tokenization script:

| Variable | Type | Required | Description |
|----------|------|----------|-------------|
| `tokenizer_name` | `str` | Yes | Path or name of the tokenizer |
| `cache_dir` | `str` | Yes | Directory for caching downloads and processed data |
| `tokenized_dataset_output_path` | `str` | No | Output path for tokenized dataset (default: `cache_dir`) |
| `cut_dataset_for_testing` | `bool` | No | Reduce dataset size for testing (default: `false`) |
| `subword_token_start` | `str` | Yes | Subword token start (should be: `##` for BERT tokenizers (default), `Ġ` for RoBERTa tokenizers) |
| `dataset_name` | `str` | No | HuggingFace dataset name (e.g., `wikitext`) |
| `dataset_config_name` | `str` | No | Dataset config (e.g., `wikitext-103-raw-v1`) |
| `train_file` | `str` | No | Path to local training file (`.json`, `.csv`, `.txt`) |
| `validation_file` | `str` | No | Path to local validation file |
| `validation_split_percentage` | `int` | No | % of train to use for validation if there is no a separate validation set (default: `5`) |
| `preprocessing_num_workers` | `int` | No | Number of workers for preprocessing (default: `None`) |
| `overwrite_cache` | `bool` | No | Overwrite cached preprocessed data (default: `false`) |

* `cut_dataset_for_testing`: If true, reduces the dataset size to a tiny fraction (1000 records for train, 200 for eval) for testing and debugging. 
* `tokenizer_name`: if possible, find and use a roberta/bert tokenizer trained for your paritcular domain, e.g. medical, financial.
You should use the tokenizer consistently thorought the project. If you plan to use a pretrained model checkpoint, make sure the tokenizer matches the model architecture.
If you download the tokenizer locally, specify the path. Otherwise, specify the HuggingFace model hub identifier for the tokenizer (e.g., `bert-base-uncased`, `roberta-base`).
* `cache_dir`: used in `tokenization_utils.py` and `dataset_preprocessing_utils.py` to store all preprocessing stages of dataset. It also stores the processed dataset version with injected seed KG in the folder `ready_for_training`. Choose a directory with enough disk space to store all intermediate files. Use it consistently throughout the project.
* `tokenized_dataset_output_path`: path to save the tokenized dataset.
* `dataset_name`, `dataset_config_name`: use only for datasets on hugging face hub.
* `train_file`, `validation_file`: paths to your train and validation files in csv or json format. Used in `dataset_preprocessing_utils.py` to prepare data for training.
* `validation_split_percentage`: used in `dataset_preprocessing_utils.py` to split the training data into train and validation sets only when no separate validation file is provided.
* `preprocessing_num_workers`: used to set up num_proc in map operation (parallel) in tokenization and dataset preprocessing scripts. Set it to the number of cpu cores available for faster processing. If `None`, it will use a single core.


### Main Variables for `dataset_preprocessing_utils.py`
Here are the most important parameters to set before running the dataset preprocessing script:

| Variable | Type | Required | Description |
|----------|------|----------|-------------|
| `train_dataset_with_heads` | `str` | Yes | Path to the dataset with discovered heads
| `injections_train_path` | `str` | Yes | Path to seed KG injections for training set |
| `injections_eval_path` | `str` | Yes | Path to seed KG injections for evaluation set |
| `relation_map_path` | `str` | Yes | Path to relation type mapping file |
| `preprocessing_num_workers` | `int` | No | Number of workers for preprocessing (default: `None`) |
| `overwrite_cache` | `bool` | No | Overwrite cached preprocessed data (default: `false`) |
| `cut_dataset_for_testing` | `bool` | No | Reduce dataset size for testing (default: `false`) |
| `num_relationships` | `int` | Yes | ≥ Number of relationship types in the knowledge graph + 1 |
| `overwrite_cache` | `bool` | No | Overwrite cached preprocessed data (default: `false`) |
| `output_dir` | `str` | Yes | Output directory for training checkpoints |


* `injections_train_path`, `injections_eval_path`, `relation_map_path`: paths to the files containing the knowledge graph injections for training and evaluation datasets.  Used to load the injections.

    * `injections_train_path`, `injections_eval_path`: path to csv files containing the seed KG injections generated from `seed_kg_injection_algorithm.ipynb`.
    * `relation_map_path`: JSON file mapping relation names to integer IDs for embedding lookup generated from `seed_kg_injection_algorithm.ipynb`.
    * `num_relationships`: used in `graphmert_model/configuration_graphmert.py` to set the number of trainable embeddings in the model. Should be more or equal to  #unique relationship types in the knowledge graph + 1 (for 'no relation' type). 
    * `output_dir`: used in `dataset_preprocessing_utils.py` to save the relatiion_map.json and later all training checkpoints and info. Make sure it has enough disk space to store the model checkpoints.


### Main Variables for `mlm_utils.py`
Here are the most important parameters to set before running the MLM training script:

| Variable | Type | Required | Description |
|----------|------|----------|-------------|
| `relation_emb_dropout` | `float` | No | Dropout rate for relation embeddings (default: `0.3`) |
| `mlm_probability` | `float` | No | Probability of masking tokens for MLM (default: `0.15`) |
| `mlm_sbo` | `bool` | No | Enable span boundary objective (default: `true`) |
| `exp_mask_base` | `float` | No | Base for exponential masking distribution |
| `config_overrides` | `str` | No | Override model config values (comma-separated key=value pairs) |
| `do_train` | `bool` | No | Whether to run training |
| `do_eval` | `bool` | No | Whether to run evaluation |
| `resume_from_checkpoint` | `str` | No | Path to checkpoint to resume from |
| `learning_rate` | `float` | No | Learning rate for optimizer |
| `per_device_train_batch_size` | `int` | Yes | Batch size per device for training |
| `per_device_eval_batch_size` | `int` | Yes | Batch size per device for evaluation |
| `num_train_epochs` | `float` | Yes | Number of training epochs |
| `eval_steps` | `int` | No | Evaluate every X steps (default: `500`) |
| `logging_steps` | `int` | No | Log every X steps (default: `500`) |
| `save_steps` | `int` | No | Save checkpoint every X steps (default: `500`) |
| `weight_decay` | `float` | No | Weight decay for optimizer (default: `0.0`) |


* learning_rate, per_device_train_batch_size, per_device_eval_batch_size, num_train_epochs, logging_steps, save_steps, weight_decay: standard training hyperparameters used in `mlm_utils.py` to configure the training process. Also, if you use slurm and run batch jobs you may set `learning_rates` as list of floats instead of `learning_rate` to use different learning rates for batch jobs in slurm. `len(learning_rates)` should match the number of tasks in the slurm job array.

* `config_overrides`: used in `mlm_utils.py` to modify any of model configuration parameters set in `configuration_graphmert.py` on-the-fly without changing the original config file. Sets model hyperparameters like hidden size, number of layers, attention heads, etc. **You must set transformer parameters so that the model size matches to your data training size.** Check research papers to figure  out which model size is recommended for your training dataset size. For example, for small datasets (e.g., <100k samples), it is recommended to use a smaller model (e.g., 4-6 layers, hidden size 256-512) to prevent overfitting. For larger datasets, you can use a larger model. See `graphmert_model/configuration_graphmert.py` for full list of configuration parameters.
**Important** you can adjust number of leaves here using max_nodes and root_nodes. Max_nodes must be a power or 2, and root_nodes must be less than max_nodes. Num_leaves is calculated as `max_nodes // root_nodes - 1`. E.g., possible configurations are:
max_nodes = 1024, root_nodes = 128: 7 leaves per head;
max_nodes = 2048, root_nodes = 256: 7 leaves per head;
max_noides = 1024, root_nodes = 256: 3 leaves per head. 

The number of leaves, 3 vs 7, (2^n - 1) depends on the number of tokens in your triple tails. If triple tails in your seed KG fit well into 3 tokens on average, this is fine. If your tails have elaborated long phrases, choose number of leaves equal to 7.

Adjust these parameters based on your dataset size and GPU memory. The higher `root_nodes` is, the longer is your actual traiing sequence and context. High `max_nodes` increases GPU memory usage.

```
* `relation_emb_dropout`: used in `graphmert_model/configuration_graphmert.py` to set the dropout rate for relation embeddings. Set higher values (e.g., 0.3-0.5) if semantic vocabulary size is small to prevent overfitting.
* `mlm_sbo`: used in `mlm_utils.py` to enable span boundary objective during MLM training. Set to true for training with masked node objective.
* `exp_mask_base`: used in `mlm_utils.py` to set the base for exponential masking of tokens. Smaller values focus on nearby tokens, larger values spread attention more evenly and distantly.

Training checkpoints will be saved to the `output_dir` specified in the YAML config file.

---

## Variables you need to set for training

Here are variables you need to set for you model based on your dataset size and training preferences, and adjust them as needed on evaluation results:
* `learning_rate`: Start with lower, e.g. 5e-5 or 1e-4. Adjust based on validation loss: while loss decreases, you may try increasing learning rate. The larger the dataset, the higher learning rate you may try. The smaller the dataset, the lower learning rate you should use to prevent overfitting. The higher is the learning rate, the faster the training converges, but also the more unstable it may be.
* `per_device_train_batch_size`: set max that fits your GPU memory. Start with 8 or 16.
* `num_train_epochs`: start with 3-5 epochs. Monitor validation loss for overfitting. Train until validation loss stops decreasing.
* `relation_emb_dropout`: set between 0.2-0.4 based on semantic vocabulary size.
* `mlm_probability`: typically 0.15 is a good starting point. Increase to 0.2-0.3 for smaller datasets to encourage more learning.
* `exp_mask_base`: default is 0.6, should be less than 1. Larger values spread attention more evenly and distantly; smaller values focus on nearby tokens.
* `logging_steps`: set how often to log training loss. Set based on dataset size. For large datasets, log every 1000-5000 steps.
* `eval_steps`: set how often to evaluate the model during training. The larger the value, the less frequent evaluation. Very low values (e.g., 100) may slow down training. Ideally, set it so that evaluation runs 3-5 times per epoch.
* `save_steps`: set based on dataset size. For large datasets, save every 1000-5000 steps. For smaller datasets, save more frequently (e.g., every 500 steps) to capture model checkpoints.

There are many other standard training hyperparameters you may set in the YAML config file, including inherited from Hugging Face TrainingArguments. For the full list, refer to `graphmert/utils/training_arguments.py` and official Hugging Face Trainer documentation.


## Seed KG Injection Algorithm
`graphmert/utils/seed_kg_injection/seed_kg_injection.ipynb`


This utility implements the seed knowledge graph (KG) injection algorithm for GraphMERT. Before using that, uou should:
1. Discover head entities from your dataset using as explained in `llm_helper_utils/README.md`.
2. Match the discovered head entities to an external knowledge graph.

This dataset serves as an input to the seed KG injection algorithm. It must contain the following columns:

['input_ids', 'leaf_node_ids', 'leaf_relationships', 'attention_mask', 'input_nodes', 'start_indices', 'special_tokens_mask', 'head_positions', 'id', 'top_k_relations_with_scores']

If you followed the pipeline instructions in `../README.md` and `llm_helper_utils/README.md`, you should already have a file with necesary columns containing the discovered head entities. After mathcing those entities to an external knowledge graph, you should have gotten `top_k_relations_with_scores` column added to the dataset. Now you can proceed to use the seed KG injection algorithm.



### Usage

Fill in the setup parameters in the beginning of the notebook, and run the notebook to perform the seed KG injection.

### Output
The algorithm produces three files per dataset split (train/eval):
1. csv file with injected triples for selected head entities.
2. relation_id_map: A mapping from relation names to relation IDs.
3. seed_kg: a set of unique triples (head, relation, tail) injected into the dataset. It is not used in training, but can be useful for analysis.


# Prediction with the trained GraphMERT model

`predict_tails.py` utility performs tail entity prediction using a trained GraphMERT model. Given head entities and their relationships from the seed knowledge graph, the model predicts the most likely tail entities to complete the knowledge graph triples.

## Prerequisites

1. A trained GraphMERT model checkpoint
2. A dataset with relation matching completed  (output from `llm_helper_utils/match_relations.py`, see `llm_helper_utils/README.md` for details)

## Usage
```bash
python predict_tails.py --yaml_file launch_configs/args_predict.yaml
```

This job uses GPU for inference. For large datasets, you can parallelize prediction across multiple GPUs. Set the `num_slurm_jobs` parameter in the config file to the number of jobs, and If using SLURM, submit slurm job array with number of tasks equal to `num_slurm_jobs`.

It is recommended to first run a small test prediction by setting `take_subset` to `true` and `subset_size` to a small number (e.g., `1000`) to verify that the job completes correctly before launching full prediction. Decide on `top_k` hypeparameter (see below) based on manual prediction quality check before launching the full prediction.

## Configuration Variables

Configure the prediction parameters in `utils/args_predict.yaml`. Below are the available variables:

| Variable | Type | Required | Description |
|----------|------|----------|-------------|
| `model_path` | `str` | Yes | Path to the trained GraphMERT model checkpoint (relative to repo root) |
| `relation_map_path` | `str` | No | Path to the relation ID mapping JSON file. Defaults to `null` |
| `preprocessed_dataset_path` | `str` | Yes | Path to the preprocessed dataset with seed KG injections |
| `tokenizer_path` | `str` | Yes | Path to the tokenizer used during training |
| `output_dir` | `str` | No | Directory to save prediction results. If `null`, defaults to `{model_path}/predictions` |
| `top_k` | `int` | No | Number of top tail predictions to return per (head, relation) pair (default: `15`) |
| `batch_size` | `int` | No | Batch size for inference |
| `chunk_size` | `int` | No | Save results in chunks of this size (default: `100000`) |
| `num_parallel_jobs` | `int` | No | Number of parallel jobs to split the dataset into for parallel prediction (default: `1`) |
| `take_subset` | `bool` | No | Set `true` to test on a small sample (default: `false`) |
| `subset_size` | `int` | No | Number of samples when `take_subset` is `true` (default: `1000`) |
| `num_leaves` | `int` | No | Number of leaf nodes per head entity (default: `7`). Must match trained model |
| `root_nodes` | `int` | No | Number of root nodes / sequence length (default: `128`). Must match trained model |

* `model_path`: Should point to the directory containing your trained model checkpoint (with `config.json`, `model.safetensors` or `pytorch_model.bin`).
* `relation_map_path`: If not provided, the script assumes the relation map is located at `{model_path}/relation_map.json` (generated during dataset preprocessing step).
* `preprocessed_dataset_path`: This should be a dataset after relation matching step that contains columns `cleaned_response` and `head_positions` from the relation matching step, where `cleaned_response` conintains the matched relation for each head entity in `head_positions`.
* `top_k`: Adjust based on your application needs. Higher values yield more predictions but less reliable. Recommended range is 10-20. It is advised to launch a test run with smaller `top_k` first to verify that these top_k predictions are reasonable. 
Larger `top_k` values will increase computation time and memory usage on later steps (combine tails).
* `num_leaves`, `root_nodes`: These architecture parameters must match the values used during model training. Check your training config if unsure.
* `num_parallel_jobs`: Use this for large datasets to parallelize prediction across multiple GPUs. When using SLURM, set to match your SLURM array job count. If using num_parallel_jobs > 1, you have to manually unite the output chunks later (see Output section below).
* `batch_size`: Select the largest batch size that fits your GPU memory.
* `chunk_size`: Results are saved incrementally in chunks. If a chunk already exists, it will be skipped on re-run, allowing resumable predictions. After all chunks are generated, they will be combined into a final output dataset and erased.

## Output

The script generates:
1. **Chunk directories**: `{output_dir}/top_{top_k}_{start}-{end}/` - Intermediate prediction chunks; will be combined and erased after completion.
2. **Final dataset**: `{output_dir}/top_{top_k}/` - Consolidated predictions with columns:
   - `id`: Original example ID
   - `head`: Head entity text
   - `relation`: Relation type
   - `predictions`: Space-separated top-k predicted tail tokens
   - `probabilities`: Corresponding prediction 
   

If used with `num_parallel_jobs > 1`, you will need to manually combine the num_parallel_jobs datasets output datasets into the final dataset. Here is a sample code snippet to do that:

```python
datasets = []
suffices = []

# === set your parameters here
dataset_path = os.path.join('../outputs/test/span/rel29/bs256_lr_0.0001/predictions/')
#  # get this one from output log:
# This job predicts tails for examples from .. to .. out of <dataset_len>
dataset_len = 43828
NUM_PARALLEL_JOBS = 4
top_k = 10
# =========

part_size = dataset_len // NUM_PARALLEL_JOBS
for task_id in range(0, NUM_PARALLEL_JOBS):
    start_idx = task_id * part_size
    end_idx = (task_id + 1) * part_size if task_id < NUM_PARALLEL_JOBS - 1 else dataset_len
    suffices.append(f'top_{top_k}_{start_idx}-{end_idx}')

for suf in suffices:
    path = os.path.join(dataset_path, suf)
    dataset = load_from_disk(path)            
    print(f'loaded from {path}')
    datasets.append(dataset)
united_dataset = concatenate_datasets(datasets)
output_path = os.path.join(dataset_path, f'top_{top_k}')
united_dataset.save_to_disk(output_path)
print(f'saved to {output_path} with total examples {len(united_dataset)}')
```

# Evaluation with GraphRAG
After getting combined tails by LLM, we use embedding based method to filter tails. The script and instructions are in [umls_mapping](umls_mapping/README.md). It produces the final KG csv file for evaluation.

For evaluation with [GraphRAG](https://github.com/microsoft/graphrag), check [graphrag](graphrag/README.md) folder.
