# UMLS Entity Mapper & Triple Ranker

This folder provides a pipeline for mapping entities from biomedical text to the Unified Medical Language System (UMLS), extracting relevant triples from the UMLS Knowledge Graph, and ranking those triples based on semantic similarity to the source text using embedding models (SapBERT and Gemini). 

## Overview
This folder contains scripts for 2 stages: entity linking before the injection, and tail filtering after GraphMERT prediction. \
### Entity Linking 
1.  **Entity Mapping:** Maps "head entities" found in your dataset to UMLS CUIs (Concept Unique Identifiers) using SapBERT embeddings and FAISS for efficient similarity search.
2.  **Result Merging:** Consolidates mapping results from parallel tasks and filters the UMLS relation table to include only relevant relations connected to the mapped entities.
3.  **Embedding Generation:** Generates high-quality embeddings for both the filtered UMLS relations and the source dataset text using the Gemini API.
4.  **Triple Ranking:** Ranks the UMLS triples for each document by computing the cosine similarity between the document text embedding and the relation embeddings, returning the top-k most relevant triples.

### Tail Filtering
`filter_tails.py`: Rank the tails based on their similarity to the sequence.

## Setup

### 1. Environment
The dependencies are listed in `requirements.txt`.

### 2. Configuration
Modify `config.yaml` to set your paths and API key.

- `data_dir`: The main directory where intermediate files and outputs will be stored.
- `dataset`: Path to the dataset from previous pipeline.
- `api_key`: Your Google Gemini API key. API Key is loaded from ```os.getenv("GEMINI_API_KEY")``` in the script, if not found, `api_key` in `config.yaml` will be used.

### 3. Data Preparation
1. Create your `data_dir` (as defined in `config.yaml`).
2. Down load the UMLS files here: (placeholder). It contains 2 files: `filtered_mrconso_limit_vocab.csv` (filtered UMLS concepts) and `filtered_mrrel.csv` (filtered UMLS relations). Put the 2 csv files in `umls_data` folder in your `data_dir`. 

## Entity Linking Usage
### Step 1: Map Entities
Maps entities in your dataset to UMLS CUIs. This script supports parallel execution (slurm array jobs) for large datasets.
```bash
python 1_map_entities.py --task_id 0 --num_tasks 1

# For parallel execution (e.g., 10 chunks), run 10 separate jobs:
# python 1_map_entities.py --task_id 0 --num_tasks 10
# python 1_map_entities.py --task_id 1 --num_tasks 10
# ...
```

### Step 2: Merge Results
Consolidates the partial mapping files from Step 1 and filters the UMLS relation table.
```bash
python 2_merge_results.py
```

### Step 3: Generate Embeddings
Generates embeddings for the filtered relations and your input dataset using the Gemini API.
```bash
python 3_generate_embeddings.py
```

### Step 4: Rank Triples
Calculates similarity between document embeddings and relation embeddings to find the most relevant knowledge graph triples for each document.
```bash
python 4_rank_triples.py
```
The final output (dataset with ranked triples) will be saved to the path defined in `config.yaml` under `paths.final_output`.


## Tail Filtering Usage
Run `filter_tails.py` with arguments (see argument details in the script):
```bash
python filter_tails.py \
  --input_dataset /path/to/your/input_dataset \
  --output_dir /path/to/save/results \
  --thresholds 0.65 0.67 # similarity threshold to keep the tails, will create one csv for each input
```