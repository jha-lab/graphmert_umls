import re
import json
import os
import argparse
import logging
import glob
import pandas as pd


def load_json(path):
    """Helper to load JSON files safely."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_answer_content(response):
    """
    Extracts the content inside \boxed{} and removes <think> tags.
    """
    if not isinstance(response, str):
        return 'E'

    # 1. Remove <think> blocks (Chain of Thought)
    think_pattern = re.compile(r'</think>(.*)', re.DOTALL)
    think_match = think_pattern.search(response)
    if think_match:
        cleaned_response = think_match.group(1).strip()
    else:
        cleaned_response = response.strip()

    # 2. Extract \boxed{...}
    # Matches \boxed{ANSWER} and grabs ANSWER. 
    # Logic: takes the *last* boxed occurrence if multiple exist.
    boxed_pattern = re.compile(r'\\boxed\{([^}]+)\}')
    boxed_matches = boxed_pattern.findall(cleaned_response)
    
    if boxed_matches:
        return boxed_matches[-1].strip()
    
    return 'E' # Error / Not Found

def normalize_prediction(answer_str):
    """
    Maps the string answer (A, B, C, D) to an integer index (0, 1, 2, 3).
    Returns 4 for Empty/Error, 5 for 'maybe'.
    """
    if not answer_str or answer_str == 'E':
        return 4
    
    answer_str = answer_str.strip()
    
    # Check strict single letter
    response_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    
    # Check first character if the string is longer (e.g. "A)")
    if len(answer_str) > 0:
        first_char = answer_str[0].upper()
        if first_char in response_mapping:
            return response_mapping[first_char]
    
    if 'maybe' in answer_str.lower():
        return 5
        
    return 6 # Unknown format

def compute_accuracy(y_true, y_pred):
    """Computes accuracy and returns score + list of wrong indices."""
    if not y_true:
        return 0.0
        
    correct = 0
    for true_val, pred_val in zip(y_true, y_pred):
        # y_true are 0,1,2,3 from the dataset
        if true_val == pred_val:
            correct += 1
    
    return correct / len(y_true)

def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    # --- Path Construction ---
    # 1. Ground Truth Queries
    query_file = os.path.join(args.root, "queries", f"qa_{args.dataset}.json")
    logger.info(f"Loading ground truth from: {query_file}")
    qas = load_json(query_file)

    # 2. Base directory for this specific Run
    run_base_dir = os.path.join(args.root, args.response_dir, args.run_name)
    if not os.path.exists(run_base_dir):
        logger.error(f"Run directory not found: {run_base_dir}")
        return

    # --- Detect Available Seeds ---
    # Look for folders named 'seed_X'
    seed_dirs = sorted(glob.glob(os.path.join(run_base_dir, "seed_*")))
    if not seed_dirs:
        logger.error(f"No seed directories found in {run_base_dir}")
        return
    
    all_run_dfs = []

    # --- Branching Logic ---
    
    if args.dataset == 'icd':
        # === ICD Logic (Levels + All) ===
        raw_levels = set(qa.get('level') for qa in qas)
        sorted_levels = sorted(list(raw_levels), key=lambda x: (str(x) if x is not None else ""))
        logger.info(f"ICD Levels found: {sorted_levels}")

        for seed_path in seed_dirs:
            seed_name = os.path.basename(seed_path)
            logger.info(f"Processing {seed_name}...")
            
            graph_subfolders = [f for f in os.listdir(seed_path) if os.path.isdir(os.path.join(seed_path, f))]
            seed_scores = {}

            for graph in graph_subfolders:
                resp_file = os.path.join(seed_path, graph, f"{args.dataset}_temp_{args.temperature:.1f}_responses.json")
                if not os.path.exists(resp_file): continue
                
                responses = load_json(resp_file)
                
                preds = [normalize_prediction(extract_answer_content(r)) for r in responses]
                
                # Compute per-level accuracy
                graph_accs = {}
                for lvl in sorted_levels:
                    # Filter indices for this level
                    indices = [i for i, q in enumerate(qas) if q.get('level') == lvl]
                    if indices:
                        true_subset = [qas[i]['answer'] for i in indices]
                        pred_subset = [preds[i] for i in indices]
                        graph_accs[f"Level {lvl}"] = compute_accuracy(true_subset, pred_subset)
                
                # Compute overall Average
                total_true = [q['answer'] for q in qas]
                graph_accs['All'] = compute_accuracy(total_true, preds)
                seed_scores[graph] = graph_accs

            if seed_scores:
                df = pd.DataFrame.from_dict(seed_scores, orient='index')
                # Reorder: Levels first, then All
                cols = sorted([c for c in df.columns if c != "All"]) + ["All"]
                all_run_dfs.append(df[cols])
                print(f"--- {seed_name} Results ---")
                print(df[cols])

    else:
        # === Generic Logic (Single Accuracy) ===
        logger.info(f"Processing generic dataset: {args.dataset}")
        
        for seed_path in seed_dirs:
            seed_name = os.path.basename(seed_path)
            logger.info(f"Processing {seed_name}...")
            
            graph_subfolders = [f for f in os.listdir(seed_path) if os.path.isdir(os.path.join(seed_path, f))]
            seed_scores = {}

            for graph in graph_subfolders:
                resp_file = os.path.join(seed_path, graph, f"{args.dataset}_temp_{args.temperature:.1f}_responses.json")
                if not os.path.exists(resp_file): continue
                
                responses = load_json(resp_file)
                # Align lengths
                limit = min(len(responses), len(qas))
                
                preds = [normalize_prediction(extract_answer_content(r)) for r in responses]
                true_vals = [q['answer'] for q in qas]
                
                # Compute single accuracy
                acc = compute_accuracy(true_vals, preds)
                seed_scores[graph] = {"Accuracy": acc}

            if seed_scores:
                df = pd.DataFrame.from_dict(seed_scores, orient='index')
                all_run_dfs.append(df)
                print(f"--- {seed_name} Results ---")
                print(df)

    # --- 3. Final Aggregation (Shared) ---
    if all_run_dfs:
        print(f"\n{'='*25} AVERAGE ACROSS {len(all_run_dfs)} SEEDS {'='*25}")
        avg_df = pd.concat(all_run_dfs).groupby(level=0).mean()
        
        # Ensure nice column ordering for final print
        if "All" in avg_df.columns:
            cols = sorted([c for c in avg_df.columns if c != "All"]) + ["All"]
            avg_df = avg_df[cols]
            
        print(avg_df)
        
        save_path = os.path.join(args.root, args.evaluation_csv_dir, f"eval_{args.run_name}_{args.dataset}.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        avg_df.to_csv(save_path)
        logger.info(f"Saved summary to {save_path}")
    else:
        logger.warning("No results found.")

    if all_run_dfs:
        print(f"\n{'='*25} STATISTICS ACROSS {len(all_run_dfs)} SEEDS {'='*25}")
        
        combined_df = pd.concat(all_run_dfs)
        
        # Group by graph name (index) and compute mean + variance
        # Variance requires >1 seed; if 1 seed, variance will be NaN
        agg_df = combined_df.groupby(level=0).agg(['mean', 'var'])
        
        # Flatten MultiIndex columns: e.g. ('Level 1', 'mean') -> 'Level 1_mean'
        agg_df.columns = [f"{col[0]}_{col[1]}" for col in agg_df.columns]
        
        # --- Reorder columns for readability ---
        # 1. Identify base metrics (e.g., Level 1, Level 2, All OR Accuracy)
        # We grab the columns from the first dataframe to know the original order/names
        base_cols = list(all_run_dfs[0].columns)
        
        # If 'All' exists, ensure it is at the end (specifically for ICD logic)
        if "All" in base_cols:
             base_cols = sorted([c for c in base_cols if c != "All"], key=lambda x: str(x)) + ["All"]

        # 2. Interleave Mean and Var: [Acc_mean, Acc_var, Level1_mean, Level1_var...]
        final_cols = []
        for base in base_cols:
            mean_col = f"{base}_mean"
            var_col = f"{base}_var"
            if mean_col in agg_df.columns: final_cols.append(mean_col)
            if var_col in agg_df.columns: final_cols.append(var_col)
            
        agg_df = agg_df[final_cols]

        print(agg_df)
        
        save_path = os.path.join(args.root, args.evaluation_csv_dir, f"eval_{args.run_name}_{args.dataset}.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        agg_df.to_csv(save_path)
        logger.info(f"Saved summary (Mean & Variance) to {save_path}")
    else:
        logger.warning("No results found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root data directory for GraphRAG project")
    parser.add_argument("--dataset", type=str, required=True, choices=["icd", "mmlu", "medmcqa", "medqa4"], help="Dataset name") # one dataset at a time, all graphs
    parser.add_argument("--response_dir", type=str, default="responses", help="Directory name for saving LLM responses (relative to root)")    
    parser.add_argument("--run_name", type=str, default='default_run', help="Name of the current run (used for output folder naming)")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature used during LLM inference (for locating response files)")
    parser.add_argument("--evaluation_csv_dir", type=str, default="evaluations", help="Directory name for saving evaluation summaries (relative to root)")
    args = parser.parse_args()
    main(args)