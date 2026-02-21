import os
import uuid
import json
import argparse
import logging
import pandas as pd
import networkx as nx
import lancedb
from sentence_transformers import SentenceTransformer
from graphrag.index.operations.compute_edge_combined_degree import compute_edge_combined_degree

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Transform Graph CSVs or existing Parquets to GraphRAG format.")
    
    # Path Arguments (Optional arguments, validation handled in main)
    parser.add_argument("--seed_graph_path", type=str, help="Path to the seed graph CSV.")
    parser.add_argument("--new_graph_path", type=str, help="Path to the new graph CSV.")
    parser.add_argument("--llm_graph_path", type=str, help="Path to folder containing existing GraphRAG parquets (final_entities.parquet, final_relationships.parquet).")
    
    # Required Root
    parser.add_argument("--root", type=str, required=True, help="Root data directory for GraphRAG project")
    # Model path
    parser.add_argument("--model_path", type=str, default="nomic-ai/nomic-embed-text-v1", help="Path to local embedding model or HuggingFace model name.")
    
    # Output Directory
    parser.add_argument("--graph_dir", type=str, default="graphs", help="directory name for saving embedded graphs.")

    # Embedding Config
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size for embedding generation.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (cuda/cpu).")

    return parser.parse_args()

def load_graph_csv(path: str) -> pd.DataFrame:
    """Reads a graph CSV and standardizes columns to ['head', 'relation', 'tail']."""
    logger.info(f"Reading CSV from: {path}")
    df = pd.read_csv(path, na_values=['NULL'])
    
    if 'relation_type' in df.columns:
        df = df.rename(columns={'relation_type': 'relation'})
    
    if not {'head', 'relation', 'tail'}.issubset(df.columns):
        raise ValueError(f"File {path} missing required columns. Found: {df.columns}")

    return df[['head', 'relation', 'tail']].dropna()

def process_csv_graph(df: pd.DataFrame, output_path: str, model: SentenceTransformer, batch_size: int):
    """Standard logic for processing raw CSV graphs (Seed/New)."""
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Processing CSV graph ({len(df)} edges) -> {output_path}")

    # 1. Prepare Edge List
    df_grouped = df.groupby(['head', 'tail'], sort=False).agg(description=("relation", set)).reset_index()
    df_grouped['description'] = df_grouped['description'].apply(lambda x: ", ".join(sorted(x)))
    df_grouped = df_grouped.rename(columns={"head": "source", "tail": "target"})

    # 2. Build Graph & Entities
    graph = nx.from_pandas_edgelist(df_grouped, edge_attr='description', create_using=nx.DiGraph())
    entities_data = [{"title": node, "description": node, "degree": int(degree)} for node, degree in graph.degree]
    entities = pd.DataFrame(entities_data).drop_duplicates(subset="title").sort_values(by='degree', ascending=False).reset_index(drop=True)
    
    entities["human_readable_id"] = entities.index
    entities["id"] = entities["human_readable_id"].apply(lambda _x: str(uuid.uuid4()))
    entities.to_parquet(os.path.join(output_path, 'entities.parquet'), index=False)

    # 3. Relationships
    df_grouped["combined_degree"] = compute_edge_combined_degree(
        df_grouped, entities[['title', 'degree']], 
        node_name_column="title", node_degree_column="degree",
        edge_source_column="source", edge_target_column="target",
    )
    relationships = df_grouped[df_grouped['source'] != df_grouped['target']].reset_index(drop=True)
    relationships = relationships.sort_values(by='combined_degree', ascending=False).reset_index(drop=True)
    relationships["human_readable_id"] = relationships.index
    relationships["id"] = relationships["human_readable_id"].apply(lambda _x: str(uuid.uuid4()))
    
    final_rels = relationships[["id", "human_readable_id", "source", "target", "description", "combined_degree"]]
    final_rels.to_parquet(os.path.join(output_path, 'relationships.parquet'), index=False)
    
    # 4. Embeddings
    save_embeddings_to_lancedb(entities, model, output_path, batch_size, text_column='description')

def process_llm_graph(input_dir: str, output_path: str, model: SentenceTransformer, batch_size: int):
    """
    Specific logic for processing LLM-generated graphs from existing Parquets.
    """
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Processing LLM graph from {input_dir} -> {output_path}")

    entities_path = os.path.join(input_dir, 'final_entities.parquet')
    relationships_path = os.path.join(input_dir, 'final_relationships.parquet')
    
    if not os.path.exists(entities_path) or not os.path.exists(relationships_path):
        raise FileNotFoundError(f"Missing final_entities.parquet or final_relationships.parquet in {input_dir}")

    # Load Data
    entities = pd.read_parquet(entities_path)
    relationships = pd.read_parquet(relationships_path)

    # 1. Filter and Lowercase Entities
    entities_df = entities[entities['degree'] > 0].reset_index(drop=True)
    entities_df['title'] = entities_df['title'].str.lower()
    entities_df = entities_df.sort_values(by='degree', ascending=False)
    entities_df['description'] = entities_df['title'] 

    # 2. Process Relationships
    # Ensure description is string (handle lists if present)
    relationships['description'] = relationships['description'].apply(lambda x: ', '.join(x) if isinstance(x, (list, tuple, set)) else str(x))
    
    relationships['source'] = relationships['source'].str.lower()
    relationships['target'] = relationships['target'].str.lower()
    relationships = relationships.sort_values(by='combined_degree', ascending=False)

    # 3. Save Parquets
    entities_df.to_parquet(os.path.join(output_path, 'entities.parquet'), index=False)
    relationships.to_parquet(os.path.join(output_path, 'relationships.parquet'), index=False)
    
    logger.info(f"LLM Stats: {len(entities_df)} entities, {len(relationships)} relations")

    # 4. Embeddings
    save_embeddings_to_lancedb(entities_df, model, output_path, batch_size, text_column='title')

def save_embeddings_to_lancedb(entities: pd.DataFrame, model: SentenceTransformer, output_path: str, batch_size: int, text_column: str):
    """Generates embeddings and saves to LanceDB."""
    logger.info(f"Generating embeddings using column: '{text_column}'...")
    
    sentences = entities[text_column].astype(str).tolist()
    
    # Vectorized encoding
    embeddings = model.encode(sentences, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    
    entities['vector'] = list(embeddings)
    entities['attributes'] = entities['title'].apply(lambda title: json.dumps({"title": title}))
    
    # Prepare final DF
    final_db_df = entities[['id', 'description', 'vector', 'attributes']].rename(columns={'description': 'text'})
    
    # Connect to LanceDB
    db_path = os.path.join(output_path, 'lancedb')
    db = lancedb.connect(db_path)
    table = db.create_table("default-entity-description", final_db_df, mode="overwrite")
    logger.info("Embeddings saved to LanceDB.")

def main():
    args = parse_args()

    # --- Input Validation Logic ---
    has_seed = args.seed_graph_path is not None
    has_new = args.new_graph_path is not None
    has_llm = args.llm_graph_path is not None

    scenario = None

    if has_seed and not has_new and not has_llm:
        scenario = "SEED_ONLY"
    elif not has_seed and has_new and not has_llm:
        scenario = "NEW_ONLY"
    elif has_seed and has_new and not has_llm:
        scenario = "SEED_NEW_COMBINED"
    elif not has_seed and not has_new and has_llm:
        scenario = "LLM_ONLY"
    elif has_seed and has_new and has_llm:
        scenario = "ALL_FOUR"
    else:
        logger.error("Invalid Argument Combination.")
        logger.error("Allowed combinations:")
        logger.error("1. --seed_graph_path ONLY")
        logger.error("2. --new_graph_path ONLY")
        logger.error("3. --seed_graph_path AND --new_graph_path")
        logger.error("4. --llm_graph_path ONLY")
        logger.error("5. --seed_graph_path AND --new_graph_path AND --llm_graph_path")
        raise ValueError("Invalid input arguments provided.")

    logger.info(f"Detected Scenario: {scenario}")

    # Load Model
    model = SentenceTransformer(args.model_path, trust_remote_code=True, device=args.device)

    # --- Execution Logic ---
    
    # 1. Seed Only
    if scenario == "SEED_ONLY":
        df_seed = load_graph_csv(args.seed_graph_path)
        process_csv_graph(df_seed, os.path.join(args.root, args.graph_dir, 'seed'), model, args.batch_size)

    # 2. New Only
    elif scenario == "NEW_ONLY":
        df_new = load_graph_csv(args.new_graph_path)
        process_csv_graph(df_new, os.path.join(args.root, args.graph_dir, 'new'), model, args.batch_size)

    # 3. Seed + New + Combined
    elif scenario == "SEED_NEW_COMBINED":
        # Seed
        df_seed = load_graph_csv(args.seed_graph_path)
        process_csv_graph(df_seed, os.path.join(args.root, args.graph_dir, 'seed'), model, args.batch_size)
        # New
        df_new = load_graph_csv(args.new_graph_path)
        process_csv_graph(df_new, os.path.join(args.root, args.graph_dir, 'new'), model, args.batch_size)
        # Combined
        df_combined = pd.concat([df_seed, df_new], ignore_index=True)
        process_csv_graph(df_combined, os.path.join(args.root, args.graph_dir, 'combined'), model, args.batch_size)

    # 4. LLM Only
    elif scenario == "LLM_ONLY":
        process_llm_graph(args.llm_graph_path, os.path.join(args.root, args.graph_dir, 'llm'), model, args.batch_size)

    # 5. All Four (Seed, New, Combined, LLM)
    elif scenario == "ALL_FOUR":
        # Seed
        df_seed = load_graph_csv(args.seed_graph_path)
        process_csv_graph(df_seed, os.path.join(args.root, args.graph_dir, 'seed'), model, args.batch_size)
        # New
        df_new = load_graph_csv(args.new_graph_path)
        process_csv_graph(df_new, os.path.join(args.root, args.graph_dir, 'new'), model, args.batch_size)
        # Combined
        df_combined = pd.concat([df_seed, df_new], ignore_index=True)
        process_csv_graph(df_combined, os.path.join(args.root, args.graph_dir, 'combined'), model, args.batch_size)
        # LLM
        process_llm_graph(args.llm_graph_path, os.path.join(args.root, args.graph_dir, 'llm'), model, args.batch_size)

if __name__ == "__main__":
    main()