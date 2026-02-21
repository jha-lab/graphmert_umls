import json
import pandas as pd
import numpy as np
import torch
import os
from datasets import load_from_disk
from src.utils import load_config, setup_logger

config = load_config()
logger = setup_logger("Ranker")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    data_dir = config['paths']['data_dir']
    
    # Paths
    dataset_emb_path = os.path.join(data_dir, config['paths']['dataset_with_embeddings'])
    rel_emb_path = os.path.join(data_dir, config['paths']['relations_embeddings'])
    map_path = os.path.join(data_dir, config['paths']['head_to_cui_map'])
    output_path = os.path.join(data_dir, config['paths']['final_output'])

    logger.info("Loading data...")
    dataset = load_from_disk(dataset_emb_path)
    rel_df = pd.read_parquet(rel_emb_path)
    with open(map_path, 'r') as f:
        head_to_cui = json.load(f)

    # Convert relevant columns to NumPy for faster access
    relations_np = rel_df['formatted_relation'].to_numpy()
    cui1_np = rel_df['CUI1'].to_numpy()

    # Create a mapping from CUI2 to a list of its indices in the dataframe
    # This is the single most important optimization to avoid slow filtering.
    cui_to_indices = rel_df.groupby('CUI2').apply(lambda x: x.index.to_list()).to_dict()

    # Move all relation embeddings to GPU once and normalize
    all_relation_embeddings_gpu = torch.tensor(
        np.stack(rel_df['embedding'].to_numpy()), 
        dtype=torch.float32, 
        device=DEVICE
    )
    all_relation_embeddings_gpu = torch.nn.functional.normalize(all_relation_embeddings_gpu, p=2, dim=1)

    # Define Batch Function
    def find_top_k_relations(batch):
        """
        Processes a batch of examples to find top-k relations for each head entity.
        This function is designed to be used with dataset.map(batched=True).
        """
        text_embeddings = batch['embedding']
        head_positions_list = batch['head_positions']
        batch_results = []

        # Process each example within the batch
        for i in range(len(text_embeddings)):
            text_embedding = text_embeddings[i]
            heads = json.loads(head_positions_list[i])
            
            # Move text embedding to GPU and normalize
            text_emb_gpu = torch.tensor(text_embedding, dtype=torch.float32, device=DEVICE)
            text_emb_gpu = torch.nn.functional.normalize(text_emb_gpu.unsqueeze(0), p=2, dim=1)

            top_k_for_example = {}
            
            for head in heads.keys():
                candidate_indices = []
                list_of_cui_data = head_to_cui.get(head, [])
                
                # Efficiently gather all candidate indices for the current head
                for cui_data_item in list_of_cui_data:
                    source_cui = cui_data_item.get('cui')
                    if source_cui:
                        indices = cui_to_indices.get(source_cui, [])
                        candidate_indices.extend(indices)
                
                # Remove duplicate indices if any
                candidate_indices = list(set(candidate_indices))

                if not candidate_indices:
                    top_k_for_example[head] = []
                    continue

                # Select candidate embeddings from the global GPU tensor using indices
                candidate_embeddings_gpu = all_relation_embeddings_gpu[candidate_indices]
                
                # Compute similarities
                similarities = torch.mm(text_emb_gpu, candidate_embeddings_gpu.T).squeeze(0)
                
                # Get top K results
                k_actual = min(config['params']['top_k_final'], len(candidate_indices))
                top_scores, top_local_indices = torch.topk(similarities, k_actual)
                
                # Convert local indices to original DataFrame indices
                top_global_indices = [candidate_indices[j] for j in top_local_indices.cpu().tolist()]
                
                # Retrieve results using fast NumPy indexing
                top_relations = relations_np[top_global_indices]
                top_tail_cuis = cui1_np[top_global_indices]
                
                # Format results
                top_k_for_example[head] = [
                    {'relation': rel, 'score': score.item(), 'tail_cui': cui}
                    for rel, score, cui in zip(top_relations, top_scores, top_tail_cuis)
                ]
            
            batch_results.append(json.dumps(top_k_for_example))
            
        return {'top_k_relations_with_scores_cui': batch_results}

    # Use dataset.map for efficient, parallelized processing
    dataset = dataset.map(
        find_top_k_relations,
        batched=True,
        batch_size=config['params']['batch_size_gemini'],
        #remove_columns=['embedding', 'head_positions'] # Optional: clean up old columns
    )

    # --- Save Results ---

    target_dataset = load_from_disk(config['paths']['dataset'])
    final_dataset = target_dataset.add_column(
        name="top_k_relations_with_scores_cui",
        column=dataset["top_k_relations_with_scores_cui"]
    )

    final_dataset.save_to_disk(output_path)
    logger.info(f"Done. Saved to {output_path}")

if __name__ == "__main__":
    main()