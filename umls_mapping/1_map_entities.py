import argparse
import os
import json
import pickle
import pandas as pd
import numpy as np
import torch
import faiss
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize

from utils import load_config, setup_logger, jaccard_similarity, cosine_similarity

config = load_config()
logger = setup_logger("EntityMapper")

def get_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(config['models']['sapbert'])
    model = AutoModel.from_pretrained(config['models']['sapbert'])
    return tokenizer, model

def generate_embeddings(text_list, tokenizer, model, batch_size, device):
    model.to(device)
    model.eval()
    embeddings = []
    
    for i in tqdm(range(0, len(text_list), batch_size), desc="Embedding"):
        batch = text_list[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=50, return_tensors="pt").to(device)
        with torch.no_grad():
            cls_rep = model(**inputs)[0][:, 0, :]
            embeddings.append(cls_rep.cpu().numpy())
            
    return np.concatenate(embeddings, axis=0)

def main(args):
    data_dir = config['paths']['data_dir']
    os.makedirs(data_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ---------------------------------------------------------
    # 1. Preprocessing: Build/Load UMLS Index (One time setup)
    # ---------------------------------------------------------
    faiss_path = os.path.join(data_dir, config['paths']['faiss_index'])
    map_path = os.path.join(data_dir, config['paths']['term_map'])
    
    if not os.path.exists(faiss_path) or not os.path.exists(map_path):
        logger.info("Building UMLS Index...")
        df = pd.read_csv(os.path.join(config['paths']['data_dir'], config['paths']['filtered_ent_src']), usecols=['CUI', 'NAME']).dropna()
        df = df[df['NAME'].str.strip() != '']
        df = df.astype({'CUI': str, 'NAME': str})
        names = df['NAME'].tolist()
        term_map = list(df[['CUI', 'NAME']].itertuples(index=False, name=None))
        
        tok, model = get_model_and_tokenizer()
        embs = generate_embeddings(names, tok, model, config['params']['batch_size_sapbert'], device)
        embs = normalize(embs)
        
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        
        faiss.write_index(index, faiss_path)
        with open(map_path, 'wb') as f: pickle.dump(term_map, f)
        logger.info("Index built and saved.")
    else:
        logger.info("Loading existing UMLS Index...")
        index = faiss.read_index(faiss_path)
        with open(map_path, 'rb') as f: term_map = pickle.load(f)

    # ---------------------------------------------------------
    # 2. Extract Heads from Dataset
    # ---------------------------------------------------------
    dataset_path = config['paths']['dataset']
    logger.info(f"Loading dataset: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    # Extract unique heads directly
    unique_heads = set()
    for row in tqdm(dataset['head_positions'], desc="Extracting Heads"):
        if row:
            data = json.loads(row)
            if isinstance(data, dict):
                unique_heads.update(data.keys())
    
    queries = sorted(list(unique_heads))
    logger.info(f"Found {len(queries)} unique head entities.")

    # ---------------------------------------------------------
    # 3. Parallel Search Logic (Chunking)
    # ---------------------------------------------------------
    # If running array job, slice the queries
    chunk_size = (len(queries) + args.num_tasks - 1) // args.num_tasks
    start_idx = args.task_id * chunk_size
    end_idx = min(start_idx + chunk_size, len(queries))
    
    queries_batch = queries[start_idx:end_idx]
    logger.info(f"Processing chunk {args.task_id}/{args.num_tasks}: {len(queries_batch)} queries.")

    if not queries_batch:
        return

    # Embed Queries
    tok, model = get_model_and_tokenizer()
    query_embs = generate_embeddings(queries_batch, tok, model, config['params']['batch_size_sapbert'], device)
    query_embs = normalize(query_embs)
    
    # Search
    D, I = index.search(query_embs, config['params']['top_k_ann'])
    
    # ---------------------------------------------------------
    # 4. Filter & Rerank (Logic from 3_mapping_result.py)
    # ---------------------------------------------------------
    final_mapping = {}
    
    for i, query in enumerate(queries_batch):
        candidates = []
        for rank, idx in enumerate(I[i]):
            if idx < 0: continue
            
            cui, name = term_map[idx]
            sim_score = float(D[i][rank])
            
            # Metric Filters
            j_score = jaccard_similarity(query, name)
            c_score = cosine_similarity(query, name)
            
            if (sim_score > config['params']['similarity_threshold'] and 
                j_score > config['params']['jaccard_threshold'] and 
                c_score > config['params']['cosine_threshold']):
                
                candidates.append({
                    'cui': cui,
                    'name': name,
                    'jaccard': j_score,
                    'ann_score': sim_score
                })
        
        # Rerank: Keep only the candidates with the HIGHEST Jaccard score
        if candidates:
            candidates.sort(key=lambda x: x['jaccard'], reverse=True)
            best_score = candidates[0]['jaccard']
            best_matches = [c for c in candidates if c['jaccard'] == best_score]
            final_mapping[query] = best_matches

    # Save Partial Results
    output_file = os.path.join(data_dir, f"mapping_part_{args.task_id}.json")
    with open(output_file, 'w') as f:
        json.dump(final_mapping, f, indent=2)
    logger.info(f"Saved partial mapping to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=0) # should start from 0
    parser.add_argument("--num_tasks", type=int, default=1)
    args = parser.parse_args()
    main(args)