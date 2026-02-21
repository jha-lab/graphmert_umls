import asyncio
import pandas as pd
import os
import json
from datasets import load_from_disk
from transformers import AutoTokenizer
from utils import load_config, setup_logger
from gemini_client import GeminiEmbedder

config = load_config()
logger = setup_logger("GeminiGen")


async def embed_relations(embedder):
    """Generates embeddings for the relations CSV/Parquet"""
    rel_path = os.path.join(config['paths']['data_dir'], config['paths']['relations_to_embed'])
    out_path = os.path.join(config['paths']['data_dir'], config['paths']['relations_embeddings'])
    ent_path = os.path.join(config['paths']['data_dir'], config['paths']['filtered_ent_src'])
    
    if os.path.exists(out_path):
        logger.info("Relation embeddings already exist. Skipping.")
        return

    logger.info("Loading Relations...")
    rel_df = pd.read_csv(rel_path, usecols=['CUI1', 'AUI1', 'CUI2', 'AUI2', 'RELA'],
                            dtype={'CUI1': str, 'AUI1': str, 'CUI2': str, 'AUI2': str, 'RELA': str})
    
    undesired_relations = ['acted_on_by_process', 'active_ingredient_of', 'associated_procedure_of', 'basis_of_strength_substance_of', 'component_of',
        'direct_device_of', 'direct_substance_of', 'has_associated_finding', 'has_finding_context', 'has_interpretation', 'has_laterality',
        'has_laterality', 'has_realization', 'has_scale_type', 'has_specimen', 'has_subject_relationship_context', 'has_temporal_context',
        'inverse_was_a', 'mapped_from', 'mapped_to', 'moved_to', 'negatively_regulated_by', 'positively_regulated_by', 'precise_active_ingredient_of',
        'realization_of', 'regulated_by', 'replaced_by', 'replaces','was_a',
    ]

    ent_df = pd.read_csv(ent_path, usecols=['CUI', 'AUI', 'NAME'], dtype={'CUI': str, 'AUI': str, 'NAME': str})
    ent_df.dropna(subset=['NAME'], inplace=True) 
    cui_to_name = ent_df.drop_duplicates(subset=['CUI']).set_index('CUI')['NAME'].to_dict()
    aui_to_name = ent_df.dropna(subset=['AUI']).drop_duplicates(subset=['AUI']).set_index('AUI')['NAME'].to_dict()

    def get_target_name(aui, cui): # Renamed parameters for clarity
        """Gets the name, preferring AUI if available and not NaN."""
        aui_val = None
        if pd.notna(aui):
            if aui in aui_to_name: # Use renamed parameter
                aui_val = aui_to_name[aui]

        if aui_val is not None:
            return aui_val
        
        if pd.notna(cui) and cui in cui_to_name: # Use renamed parameter
            return cui_to_name[cui]
            
        return None

    def format_relation_string(row):
        # Get the names using the provided function
        name1 = get_target_name(row['AUI1'], row['CUI1'])
        name2 = get_target_name(row['AUI2'], row['CUI2'])

        # Get the relationship type, replacing underscores with spaces
        relation_type = row['RELA'].replace('_', ' ')

        if name1 is None or name2 is None:
            return None

        # Format the final string
        return f"{name2} | {relation_type} | {name1}"

    rel_df = rel_df[~rel_df['RELA'].isin(undesired_relations)] # Filter out undesired relations
    # Apply the function to each row (axis=1) and create the new column
    rel_df['formatted_relation'] = rel_df.apply(format_relation_string, axis=1)
    # drop NaN values in 'formatted_relation' column
    rel_df.dropna(subset=['formatted_relation'], inplace=True)


    logger.info(f"Embedding {len(rel_df)} relations...")

    embeddings = await embedder.embed_texts(rel_df['formatted_relation'].tolist())
    
    rel_df['embedding'] = embeddings
    #rel_df = rel_df.dropna(subset=['embedding'])
    rel_df.to_parquet(out_path)
    logger.info(f"Saved relation embeddings to {out_path}")

async def embed_dataset(embedder):
    """Generates embeddings for the main dataset"""
    input_path = config['paths']['dataset']
    output_path = os.path.join(config['paths']['data_dir'], config['paths']['dataset_with_embeddings'])
    
    if os.path.exists(output_path):
        logger.info("Dataset embeddings already exist. Skipping.")
        return

    dataset = load_from_disk(input_path)
    
    # Decode tokenized input_ids to text
    tokenizer = AutoTokenizer.from_pretrained(config['models']['biomedbert'])
    
    def decode_batch(batch):
        decoded_texts = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        return {"text": decoded_texts, 'head_positions': batch['head_positions']}

    # Decode if 'text' column missing
    if 'text' not in dataset.column_names:
        logger.info("Decoding input_ids to text...")
        dataset = dataset.map(decode_batch, batched=True, batch_size=5000, remove_columns=dataset.column_names)

    logger.info(f"Embedding {len(dataset)} documents...")
    text_list = dataset['text']
    embeddings = await embedder.embed_texts(text_list)
    
    dataset = dataset.add_column("embedding", embeddings)
    dataset.save_to_disk(output_path)
    logger.info(f"Saved dataset with embeddings to {output_path}")

async def main():
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        api_key = config['gemini']['api_key']

    if not api_key:
        raise ValueError("No Gemini API key found. Set GEMINI_API_KEY env var or check config.yaml")
    
    embedder = GeminiEmbedder(api_key, rpm=config['gemini']['rpm_limit'])
    
    # Run both (sequentially to respect rate limits globally, or choose one)
    await embed_relations(embedder)
    await embed_dataset(embedder)

if __name__ == "__main__":
    asyncio.run(main())