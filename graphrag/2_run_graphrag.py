import os
import argparse
import json
import pickle
import asyncio
import logging
import pandas as pd
import tiktoken

from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
)
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.vector_stores.lancedb import LanceDBVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run GraphRAG context retrieval and extract context.")
    
    # Required Paths
    parser.add_argument("--root", type=str, required=True, help="Root data directory for GraphRAG project")
    parser.add_argument("--dataset", type=str, required=True, choices=["icd", "mmlu", "medmcqa", "medqa4"], help="Name of the dataset, choose from icd, mmlu, medmcqa, medqa4.")
    
    # Graph & Context Path
    parser.add_argument("--graph_dir", type=str, default="graphs", help="Directory name for embedded graphs, containing graph subfolders (seed, new, combined, llm). The conetxt files (json) will be saved in each subfolder.")
    parser.add_argument("--context_dir", type=str, default="contexts", help="Directory name for saving extracted context.")
    
    # API & Model Config
    parser.add_argument("--embedding_model", type=str, default="nomic-embed-text", help="Name of the embedding model.")
    parser.add_argument("--api_base", type=str, default="http://localhost:11434/v1", help="API base URL for embeddings (e.g., Ollama).")
    parser.add_argument("--api_key", type=str, default="PlaceHolder", help="API Key. Note: Not used for Ollama but required by other interface.")
    
    # Search Parameters
    parser.add_argument("--max_tokens", type=int, default=12000, help="Max token limit for context.")
    parser.add_argument("--top_k_entities", type=int, default=30, help="Top K mapped entities.")
    parser.add_argument("--top_k_relationships", type=int, default=10, help="Top K relationships per entity.")
    
    return parser.parse_args()

def extract_contexts_from_results(results):
    """Extracts relationship strings from GraphRAG results."""
    contexts = []
    for result in results:
        
        if result is None:
            contexts.append("")
            continue

        relation_str = '\n-----Relationships-----\nsource -- (relation) --> target\n'
        
        context_records = result.get('context_records', {})
        
        if context_records and 'relationships' in context_records:
            df = context_records['relationships']
            
            # Ensure it is a DataFrame and not empty
            if isinstance(df, pd.DataFrame) and not df.empty:
                for index, row in df.iterrows():
                    source = str(row['source'])
                    target = str(row['target'])
                    description = str(row['description'])
                    
                    # Formatting logic from user reference
                    relation = description.replace(source, '').replace(target, '').strip()
                    relation_str += f"{source} -- ({relation}) --> {target}\n"
        
        contexts.append(relation_str)

    return contexts


async def run_analysis_for_directory(
    input_dir: str,
    save_file: str,
    queries: list,
    token_encoder,
    args
):
    """
    Loads data from a specific directory, runs the context retrieval, and saves results.
    """
    logger.info(f"--- Starting analysis for: {input_dir} ---")

    # Define paths
    lancedb_uri = os.path.join(input_dir, "lancedb")
    entity_path = os.path.join(input_dir, "entities.parquet")
    relationship_path = os.path.join(input_dir, "relationships.parquet")

    # 1. Validation
    if not all(os.path.exists(p) for p in [entity_path, relationship_path, lancedb_uri]):
        logger.warning(f"Skipping {input_dir}: Missing entities, relationships, or lancedb folder.")
        return

    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    # 2. Load Data
    logger.info("Loading entities and relationships...")
    entity_df = pd.read_parquet(entity_path)
    entities = read_indexer_entities(entity_df, None, None)
    
    relationship_df = pd.read_parquet(relationship_path)
    relationships = read_indexer_relationships(relationship_df)
    logger.info(f"Loaded {len(entity_df)} entities and {len(relationship_df)} relationships.")

    # 3. Setup Vector Store & Embedder
    description_embedding_store = LanceDBVectorStore(collection_name="default-entity-description")
    description_embedding_store.connect(db_uri=lancedb_uri)

    text_embedder = OpenAIEmbedding(
        api_key=args.api_key,
        api_base=args.api_base,
        api_type=OpenaiApiType.OpenAI,
        model=args.embedding_model,
        deployment_name=args.embedding_model,
        max_retries=20,
    )

    # 4. Build Context Engine
    context_builder = LocalSearchMixedContext(
        community_reports=None,
        text_units=None,
        entities=entities,
        relationships=relationships,
        covariates=None,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )

    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.15,
        "conversation_history_max_turns": 5,
        "top_k_mapped_entities": args.top_k_entities,
        "top_k_relationships": args.top_k_relationships,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,
        "max_tokens": args.max_tokens,
    }

    # 5. Async Search Wrapper
    async def asearch(query_text):
        context_result = context_builder.build_context(
            query=query_text,
            conversation_history=None,
            **local_context_params,
        )
        return {
            'query': query_text,
            'context_chunks': context_result.context_chunks,
            'context_records': context_result.context_records,
        }

    # 6. Execution
    logger.info(f"Running search for {len(queries)} queries...")
    tasks = [asearch(q['query']) for q in queries]
    results = await asyncio.gather(*tasks)

    # 7. Save
    formatted_contexts = extract_contexts_from_results(results)
    
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    with open(save_file, 'w') as f:
        json.dump(formatted_contexts, f, indent=2)
    logger.info(f"Formatted contexts saved to: {save_file}")

async def main():
    args = parse_args()
    
    # Load Queries
    query_file = os.path.join(args.root, "queries", f"qa_{args.dataset}.json")
    logger.info(f"Loading queries from {query_file}")
    with open(query_file, "r") as file:
        queries = json.load(file)

    # Initialize Shared Tokenizer
    token_encoder = tiktoken.get_encoding("cl100k_base")

    # Define standard subfolders to look for
    target_subfolders = ["seed", "new", "combined", "llm"]
    
    # Scan and Run
    for subfolder in target_subfolders:
        input_dir = os.path.join(args.root, args.graph_dir, subfolder)
        
        # Check if this subfolder exists in the data root
        if os.path.exists(input_dir):
            save_path = os.path.join(args.root, args.context_dir, subfolder, f"{args.dataset}.json") # save context as json to the same folder
            try:
                await run_analysis_for_directory(input_dir, save_path, queries, token_encoder, args)
            except Exception as e:
                logger.error(f"Critical error processing {subfolder}: {e}")
        else:
            logger.info(f"Subfolder '{subfolder}' not found in data root. Skipping.")

if __name__ == "__main__":
    asyncio.run(main())