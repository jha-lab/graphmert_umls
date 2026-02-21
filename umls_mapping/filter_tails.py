import argparse
import asyncio
import os
from functools import partial
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from gemini_client import GeminiEmbedder
from utils import load_config, setup_logger

config = load_config()
logger = setup_logger("ScoreTriples")


def flatten_triples_for_batch(head_batch, relation_batch, tails_batch):
    """
    Flattens the nested structure (1 head -> N tails) into a flat list of strings 
    for embedding: "head relation tail".
    """
    flat_triples = []
    lengths = []
    
    for head, rel, tails in zip(head_batch, relation_batch, tails_batch):
        lengths.append(len(tails))
        rel_clean = rel.replace('_', ' ')
        
        # Construct strings
        flat_triples.extend([f"{head} {rel_clean} {tail}" for tail in tails])
        
    return flat_triples, lengths


def compute_cosine_similarity(text_embs, tail_embs_batch, device):
    """
    Computes cosine similarity between document text embedding and list of triple embeddings.
    Done on GPU for speed.
    """
    results = []
    for i, text_vec_list in enumerate(text_embs):
        tails_list = tail_embs_batch[i]
        
        if not tails_list or text_vec_list is None:
            results.append([])
            continue
            
        # To Tensor
        text_vec = torch.tensor(text_vec_list, dtype=torch.float32, device=device).unsqueeze(0)
        tails = torch.tensor(tails_list, dtype=torch.float32, device=device)
        
        # Normalize
        text_vec = torch.nn.functional.normalize(text_vec, dim=1)
        tails = torch.nn.functional.normalize(tails, dim=1)
        
        # Dot Product
        sims = (text_vec @ tails.T).squeeze(0).cpu().tolist()
        
        # Handle scalar output case if only 1 tail
        if isinstance(sims, float):
            sims = [sims]
            
        results.append(sims)
    return results


def process_batch_unified(batch, tokenizer, embedder, device):
    """
    1. Decodes input_ids to text.
    2. Flattens triples.
    3. Calls Gemini API for both.
    4. Computes similarity.
    """
    # 1. Decode Text
    texts = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
    
    # 2. Flatten Triples
    all_triples_flat, lengths = flatten_triples_for_batch(batch['head'], batch['relation'], batch['tails'])
    
    # 3. Async Embedding
    async def run_embeddings():
        t_task = embedder.embed_texts_async(texts)
        r_task = embedder.embed_texts_async(all_triples_flat)
        return await asyncio.gather(t_task, r_task)

    text_embeddings, triple_embeddings_flat = asyncio.run(run_embeddings())
    
    # 4. Unflatten
    triple_embeddings = []
    if triple_embeddings_flat:
        idx = 0
        for l in lengths:
            triple_embeddings.append(triple_embeddings_flat[idx:idx+l])
            idx += l
    else:
        triple_embeddings = [[] for _ in lengths]

    # 5. Score
    sims = compute_cosine_similarity(text_embeddings, triple_embeddings, device)
    
    # Return both text and similarities to avoid a second pass
    return {'text': texts, 'triple_similarities': sims}

def export_results(dataset_path, output_dir, thresholds):
    """
    Loads the scored dataset, converts to Pandas, explodes lists, filters by threshold, and saves CSVs.
    """
    ds = load_from_disk(dataset_path)
    
    # Keep only necessary columns
    keep_cols = ['id', 'text', 'head', 'relation', 'tails', 'triple_similarities']
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])
    
    logger.info("Converting to Pandas (this may take memory)...")
    df = ds.to_pandas()
    
    logger.info("Filtering empty rows...")
    df = df[df['tails'].map(lambda x: len(x) > 0)]
    
    logger.info("Exploding lists (1 row per triple)...")
    # tails and triple_similarities are parallel lists
    df = df.explode(['tails', 'triple_similarities'])
    
    # Clean up types
    df['triple_similarities'] = pd.to_numeric(df['triple_similarities'], errors='coerce')
    df = df.rename(columns={'tails': 'tail'})
    
    logger.info("Sorting globally by similarity score...")
    df = df.sort_values(by='triple_similarities', ascending=False)
    
    for thr in thresholds:
        logger.info(f"Processing threshold > {thr}...")
        filtered = df[df['triple_similarities'] > thr].copy()
        
        out_name = f"filtered_triples_beta_{thr}.csv"
        out_path = os.path.join(output_dir, out_name)
        
        filtered[['id', 'text', 'head', 'relation', 'tail']].to_csv(out_path, index=False)
        logger.info(f"Saved {len(filtered)} rows to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Score and Filter KG Triples using Gemini Embeddings")
    
    # IO Arguments
    parser.add_argument("--input_dataset", type=str, required=True, help="Path to input dataset from previous pipeline stage")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--tokenizer", type=str, default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", help="tokenizer for decoding input_ids")
    
    # Processing Arguments
    parser.add_argument("--stage", type=str, choices=['all', 'embed', 'filter'], default='all', 
                        help="Run full pipeline, only embedding, or only filtering (if embedding is done)")
    parser.add_argument("--batch_size", type=int, default=2000, help="dataset Map batch size")
    parser.add_argument("--rpm", type=int, default=150, help="Gemini API Request Per Minute limit")
    parser.add_argument("--thresholds", type=float, nargs='+', default=[0.67], help="List of thresholds for final KG csv export")
    
    args = parser.parse_args()
    
    # Config
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        api_key = config['gemini']['api_key']

    if not api_key:
        raise ValueError("No Gemini API key found. Set GEMINI_API_KEY env var or check config.yaml")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(args.output_dir, exist_ok=True)
    embedded_dataset_path = os.path.join(args.output_dir, args.input_dataset.split('/')[-1] + "_with_scores")

    # --- STAGE 1: EMBED & SCORE ---
    if args.stage in ['all', 'embed']:
        
        logger.info(f"Starting Embedding Stage on {device}...")
        dataset = load_from_disk(args.input_dataset)
        
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        embedder = GeminiEmbedder(api_key, rpm=args.rpm)
        
        # Partial function to inject dependencies
        process_fn = partial(process_batch_unified, tokenizer=tokenizer, embedder=embedder, device=device)
        
        processed_dataset = dataset.map(
            process_fn,
            batched=True,
            batch_size=args.batch_size,
            desc="Embedding & Scoring"
        )
        
        logger.info(f"Saving scored dataset to {embedded_dataset_path}...")
        processed_dataset.save_to_disk(embedded_dataset_path)

    # --- STAGE 2: FILTER & EXPORT ---
    if args.stage in ['all', 'filter']:
        logger.info("Starting Filter & Export Stage...")
        
        if not os.path.exists(embedded_dataset_path):
            raise FileNotFoundError(f"Could not find scored dataset at {embedded_dataset_path}. Run with --stage embed first.")
            
        export_results(embedded_dataset_path, args.output_dir, args.thresholds)

    logger.info("Done.")

if __name__ == "__main__":
    main()