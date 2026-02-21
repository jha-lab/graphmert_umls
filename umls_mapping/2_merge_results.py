import os
import json
import glob
import pandas as pd
from utils import load_config, setup_logger

config = load_config()
logger = setup_logger("MergeResults")

def _create_filtered_relations(cui_list, data_dir):
    """Helper to generate the CSV/Parquet for the embedding step"""
    logger.info("Filtering relations based on mapped CUIs...")
    
    rel_src_path = os.path.join(config['paths']['data_dir'], config['paths']['filtered_rel_src'])
    if not os.path.exists(rel_src_path):
        logger.error(f"Filtered relation source not found at: {rel_src_path}")
        return

    # Load source relations
    rel_df = pd.read_csv(rel_src_path, dtype={'CUI1': str, 'CUI2': str, 'SAB': str})
    valid_vocab = ['SNOMEDCT_US', 'GO']
    
    # Logic: Keep relations where CUI2 (Target) is in our mapped list
    # Ensure strict string matching
    filtered = rel_df[
        (rel_df['CUI1'] != rel_df['CUI2']) &
        (rel_df['SAB'].isin(valid_vocab)) &
        (rel_df['CUI2'].isin(cui_list))
    ].copy()
    
    out_path = os.path.join(data_dir, config['paths']['relations_to_embed'])
    filtered.to_csv(out_path, index=False)
    logger.info(f"Saved {len(filtered)} filtered relations to {out_path}")

def main():
    data_dir = config['paths']['data_dir']
    
    # Find all partial result files
    search_pattern = os.path.join(data_dir, "mapping_part_*.json")
    files = glob.glob(search_pattern)
    
    if not files:
        logger.error(f"No partial mapping files found matching: {search_pattern}")
        return

    logger.info(f"Found {len(files)} partial files. Merging...")
    
    final_mapping = {}
    
    for f_path in files:
        try:
            with open(f_path, 'r') as f:
                data = json.load(f)
                # Since task partitions are distinct, simple update is safe
                final_mapping.update(data)
        except Exception as e:
            logger.error(f"Error reading file {f_path}: {e}")

    logger.info(f"Total unique head entities mapped: {len(final_mapping)}")
    
    # Save the consolidated mapping
    final_map_path = os.path.join(data_dir, config['paths']['head_to_cui_map'])
    with open(final_map_path, 'w') as f:
        json.dump(final_mapping, f, indent=4)
    logger.info(f"Saved final mapping to {final_map_path}")

    # Extract unique CUI set for relation filtering
    cui_set = set()
    for matches in final_mapping.values():
        for m in matches:
            if 'cui' in m:
                cui_set.add(m['cui'])
    
    logger.info(f"Extracted {len(cui_set)} unique CUIs.")

    # Generate the filtered relation file for next step
    if cui_set:
        _create_filtered_relations(list(cui_set), data_dir)
    else:
        logger.warning("No CUIs found. Skipping relation filtering.")

if __name__ == "__main__":
    main()