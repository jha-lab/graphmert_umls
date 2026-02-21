import torch
from vllm import LLM, SamplingParams
import json
import os
import logging
import argparse

def load_json(path):
    """Helper to load JSON files safely."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main(args):
    """Main function to run the LLM inference pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    # --- Path Construction ---
    # Input directories
    context_dir = os.path.join(args.root, args.context_dir)
    query_path = os.path.join(args.root, "queries", f"qa_{args.dataset}.json")
    template_path = args.prompt_template_file if args.prompt_template_file else os.path.join(args.root, "prompts", "local_search_system_prompt.txt")

    # Output directories
    output_base_dir = os.path.join(args.root, args.response_dir, args.run_name, f"seed_{args.seed}")
    os.makedirs(output_base_dir, exist_ok=True)

    logger.info(f"Root Directory: {args.root}")
    logger.info(f"Context Directory: {context_dir}")
    logger.info(f"Output Directory: {output_base_dir}")

    # --- Automatic Subfolder Detection ---
    valid_tasks = []

    # Check which subfolders actually exist and have the data
    for sub in os.listdir(context_dir):
        sub_path = os.path.join(context_dir, sub)
        if os.path.isdir(sub_path):
            data_file = os.path.join(sub_path, f"{args.dataset}.json")
            
            # If subfolder is in our target list (or we accept all) and file exists
            if os.path.exists(data_file):
                valid_tasks.append({
                    "subfolder": sub,
                    "context_file": data_file
                })
            else:
                logger.warning(f"Subfolder '{sub}' found but missing dataset file: {data_file}")

    if not valid_tasks:
        logger.error(f"No valid context files found in {context_dir} for dataset '{args.dataset}'.")
        return

    logger.info(f"Found {len(valid_tasks)} valid context subfolders")

    # --- Load Resources ---
    # Load Queries
    logger.info(f"Loading queries from: {query_path}")
    queries = load_json(query_path)

    # Load Template
    logger.info(f"Loading prompt template from: {template_path}")
    with open(template_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    # --- Load LLM ---
    logger.info(f'Loading LLM: {args.model_id}')
    
    # Auto-detect GPU count if not provided
    if args.tensor_parallel_size is None:
        num_gpus = torch.cuda.device_count()
        logger.info(f"Tensor parallel size not specified. Defaulting to available GPUs: {num_gpus}")
    else:
        num_gpus = args.tensor_parallel_size

    try:
        llm = LLM(
            model=args.model_id,
            trust_remote_code=True,
            dtype=args.dtype,
            quantization=args.quantization,
            max_model_len=args.max_model_len,
            tensor_parallel_size=num_gpus
        )
    except Exception as e:
        logger.error(f"Failed to load LLM: {e}")
        raise e
    
    logger.info("LLM loaded successfully.")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed
    )

    # --- Inference Loop ---
    for task in valid_tasks:
        subfolder_name = task['subfolder']
        context_file_path = task['context_file']
        
        logger.info(f"Processing subfolder: {subfolder_name}")
        
        # Load Context
        contexts = load_json(context_file_path)
        assert len(contexts) == len(queries), f"Context count {len(contexts)} does not match query count {len(queries)}"

        #contexts = contexts[:100] # for testing purposes
        #queries = queries[:100]

        # Prepare Prompts
        formatted_prompts = []
        for context, query_obj in zip(contexts, queries):
            # Format system prompt
            system_content = prompt_template.format(
                context_data=context,
                response_type="multiple paragraphs"
            )
            
            formatted_prompts.append([
                {"role": "system", "content": system_content},
                {"role": "user", "content": query_obj['query']},
            ])

        # Generate
        logger.info(f"Generating responses for {len(formatted_prompts)} prompts...")
        
        
        if args.enable_thinking:
            chat_kwargs = {}
            chat_kwargs["enable_thinking"] = True
            outputs = llm.chat(formatted_prompts, sampling_params, chat_template_kwargs=chat_kwargs)
        else:
            outputs = llm.chat(formatted_prompts, sampling_params)
        
        responses = [output.outputs[0].text for output in outputs]

        # Save Results
        save_sub_dir = os.path.join(output_base_dir, subfolder_name)
        os.makedirs(save_sub_dir, exist_ok=True)
        
        output_filename = f"{args.dataset}_temp_{args.temperature:.1f}_responses.json"
        response_file_path = os.path.join(save_sub_dir, output_filename)

        with open(response_file_path, "w", encoding='utf-8') as output_file:
            json.dump(responses, output_file, indent=2)

        logger.info(f"Saved {len(responses)} responses to {response_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM evaluation on GraphRAG contexts.")
    
    # --- Path Arguments ---
    parser.add_argument("--root", type=str, required=True, help="Root data directory for GraphRAG project")
    parser.add_argument("--dataset", type=str, required=True, choices=["icd", "mmlu", "medmcqa", "medqa4"], help="Dataset name") # one dataset at a time, all graphs
    
    parser.add_argument("--prompt_template_file", type=str, default=None, help="Full path to the system prompt template .txt file")
    parser.add_argument("--context_dir", type=str, default="contexts", help="Directory name for extracted contexts (relative to root)")
    parser.add_argument("--response_dir", type=str, default="responses", help="Directory name for saving LLM responses (relative to root)")
    
    # --- Model Configuration ---
    parser.add_argument("--model_id", type=str, required=True, help="Path to model or HF Model ID")
    parser.add_argument("--tensor_parallel_size", type=int, default=None, help="Number of GPUs to use. Defaults to all available.")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization method (e.g., 'fp8', 'awq'). Default None.")
    parser.add_argument("--dtype", type=str, default="auto", help="Data type for model weights (e.g., 'auto', 'bfloat16').")
    parser.add_argument("--max_model_len", type=int, default=20000, help="Max context length for the model.")
    parser.add_argument("--enable_thinking", action='store_true', help="Enable thinking mode (specifically for reasoning models).")

    # --- Sampling Configuration ---
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling.")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling.")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Max tokens to generate.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    
    # --- Run Metadata ---
    parser.add_argument("--run_name", type=str, default='default_run', help="Name of the current run (used for output folder naming)")

    args = parser.parse_args()
    main(args)