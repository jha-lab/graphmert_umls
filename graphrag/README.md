# GraphRAG Pipeline
This folder contains two methods built upon the **GraphRAG** framework but optimized for local inference without relying on external APIs. 
1. **Local LLM Knowledge Graph Extractor** \
`extract_kg.py` provides a pipeline for extracting Knowledge Graphs (KG) from text files using local Large Language Models (LLMs) via **vLLM**. 
2. **Medical QA for KG Evaluation** \
This folder contains a 4-step pipeline designed to evaluate KGs using GraphRAG on medical datasets. The pipeline transforms raw graph data, retrieves context using GraphRAG's local search, runs LLM inference via vLLM, and computes accuracy metrics.


## 🛠️ Environment Setup
Create the conda environment and install dependencies:
```bash
conda create -n graphrag python=3.11
conda activate graphrag
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.7.3
pip install poetry
poetry install
pip install sentence-transformers
```

## 📂 Data & Directory Setup

You need to prepare a specific data directory structure for your project.

1.  **Create Root Directory**: Create a root directory for your project (e.g., `my_kg_project`).
2.  **Add Configuration**: Move `settings.yaml`, `extraction_config.yaml`, `.env` into this root.
3.  **Prepare Input**: Create an `input` folder inside the root and place your `.txt` data files there.
4.  **Prepare Output**: Create an empty `output` folder inside the root.
5.  **Prepare Queries**: Copy the `queries` folder into the root. It contains the filtered QA datasets.
6.  **Prepare Prompts**: Copy the `prompts` folder into the root. It contains the default GraphRAG prompt.

**Your folder structure should look like this:**

```text
my_kg_project/
├── .env                       # Just for GraphRAG to work, no need to change
├── extraction_config.yaml     # Prompts, LLM params, Entity/Relation schema
├── settings.yaml              # GraphRAG settings (chunk size, overlap)
├── input/
│   ├── data1.txt
│   └── data2.txt
├── output/                    # Generated graph data will appear here
└── queries/                   # Query files for each dataset
    ├── qa_icd.json
    ├── qa_mmlu.json
    └── ...
```

## ⚙️ Configuration
1. Extraction Logic (extraction_config.yaml): Modify this file to customize the LLM behavior:
- LLM Parameters: Adjust `model_path`, `temperature`, `top_p`, etc.
- Schema: Define your target `entity_types` and `relation_types`.
- Prompts: Adjust the `prompt_template` or add examples for few-shot learning.

2. Indexing Settings (settings.yaml): Modify this file to control standard GraphRAG behaviors.
- Chunks: Only adjust `size` and `overlap` under the `chunks` section. This controls the length of the LLM input context and the overlap between segments.
- Note: Keep all other parameters unchanged to ensure GraphRAG functions correctly with the local pipeline.


## 🚀 Usage 1: Local LLM Knowledge Graph Extractor

Run the extraction script pointing to your root directory. You can run the entire pipeline at once or step-by-step.

### Run All Steps
```bash
python extract_kg.py --root /path/to/my_kg_project --pipeline all
```

### Run Individual Pipelines
Use the `--pipeline` argument with numbers `1-5` to debug or resume specific stages:

- 1: Input Processing: Generates `text_units.parquet` in `output` folder.
- 2: Document Finalization: Generates `documents.parquet` in `output` folder.
- 3: LLM Extraction: Performs batched inference via vLLM. Generates `extracted_graph_responses.parquet` in `output` folder. 
  
  Note: This stage requires a GPU for vLLM acceleration. This environment pins a specific version of vLLM to maintain compatibility with GraphRAG dependencies. If you prefer to use a newer version of vLLM, you can execute the `extract_graph()` function within a separate, updated vLLM environment.
- 4: Graph Parsing: Parses LLM responses to KG, generates `entities.parquet` and `relationships.parquet` in `output` folder.
- 5: Graph Finalization & Manual Overrides: Clean the final KGand applies manual overrides. Generates `final_entities.parquet` and `final_relationships.parquet` in `output` folder.


## 🚀 Usage 2: Medical QA for KG Evaluation
### Step 1: Data Formatting & Indexing
Script `1_to_graphrag_format.py` transforms raw CSV graphs or existing GraphRAG Parquet files into the specific format required by GraphRAG (LanceDB index + Parquet files). The graphs will be saved in `graphs` folder in `root` by default.

**Usage:**
```bash
# Example: Process a Seed graph and a New graph, creating 'seed', 'new', and 'combined' indices
python 1_to_graphrag_format.py \
    --root "/path/to/my_kg_project" \
    --seed_graph_path "/path/to/raw_seed_graph.csv" \
    --new_graph_path "/path/to/raw_new_graph.csv" \
    --model_path "nomic-ai/nomic-embed-text-v1" \
```

**Supported Scenarios:**
The pipeline automatically handles the following graph combinations:
- SEED_ONLY: Only processes the seed graph.
- NEW_ONLY: Only processes the new graph.
- SEED_NEW_COMBINED: Processes Seed, New, and a merged "Combined" graph if both `--seed_graph_path` and `--new_graph_path` are provided.
- LLM_ONLY: Processes LLM KG.
- ALL_FOUR: Processes all the above scenarios if all paths are provided.

### Step 2: Context Retrieval
Script `2_run_graphrag.py` runs GraphRAG's Local Search to retrieve relevant context for every query in your dataset. It iterates through all subfolders generated in Step 1 (seed, new, combined, llm). The contexts will be saved in `contexts` folder in `root` by default.

**Usage:**
```bash
python 2_run_graphrag.py \
    --root "/path/to/my_kg_project" \
    --dataset "icd" \
    --embedding_model "nomic-embed-text" \
    --api_base "http://localhost:11434/v1" #(local Ollama instance)
```

### Step 3: LLM Inference
Script `3_llm_inference.py` uses vLLM to generate answers based on the context retrieved in Step 2. The LLM responses will be saved in `responses` folder in `root` by default. 

**Usage:**
```bash
python 3_llm_inference.py \
    --root "/path/to/my_kg_project" \
    --dataset "icd" \ # choose the dataset from icd, mmlu, medmcqa, medqa4.
    --model_id "Qwen/Qwen3-14B" \
    --prompt_template_file "/path/to/prompt/txt/file" # default is "/root/prompts/local_search_system_prompt.txt" \
    --enable_thinking
    --seed 1
```

### Step 4: Evaluation
Script `4_evaluate.py` parses the LLM responses, extracts the answer, and computes accuracy. The evaluation tables will be printed and saved in `evaluations` folder in `root` by default.

**Usage:**
```bash
python 4_evaluate.py \
    --root "/path/to/my_kg_project" \
    --dataset "icd" \
    --run_name "default_run" 
```

Your data directory will look like this after all steps:

```text
my_kg_project/
├── ...
├── graphs/                        # [STEP 1 OUTPUT] Indexed graph data
│   ├── seed/                      #    ├── Generated from seed_graph.csv
│   │   ├── entities.parquet
│   │   ├── relationships.parquet
│   │   └── lancedb/               #    └── Vector database files
│   ├── new/
│   ├── combined/
│   └── llm/
├── contexts/                      # [STEP 2 OUTPUT] Retrieved contexts per graph
│   ├── seed/
│   │   ├── icd.json               #    └── Contexts for ICD queries using Seed graph
│   │   └── mmlu.json
│   ├── new/
│   └── ...
├── responses/                     # [STEP 3 OUTPUT] LLM Inference results
│   └── default_run/               #    ├── Organized by --run_name
│       ├── seed_42/               #    │   ├── Organized by --seed
│       │   ├── seed/              #    │   │   ├── Graph Source
│       │   │   └── icd_temp_0.6_responses.json
│       │   ├── new/
│       │   └── ...
│       └── seed_1/
└── evaluations/                   # [STEP 4 OUTPUT] Final accuracy metrics
    └── eval_default_run_icd.csv   #    └── Summary table (CSV)
```