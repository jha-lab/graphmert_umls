import os
import sys
import json
import asyncio
import logging
import re
import argparse
import yaml
from pathlib import Path
from dataclasses import asdict
from uuid import uuid4
from collections.abc import Mapping
from typing import Any

import pandas as pd
import numpy as np
import networkx as nx
from vllm import LLM, SamplingParams

# GraphRAG Imports
from graphrag.callbacks.workflow_callbacks_manager import WorkflowCallbacksManager
from graphrag.index.run.utils import create_run_context
from graphrag.config.load_config import load_config
from graphrag.storage.factory import StorageFactory
from graphrag.cache.factory import CacheFactory
from graphrag.index.context import PipelineRunStats
from graphrag.storage.pipeline_storage import PipelineStorage
from graphrag.utils.storage import load_table_from_storage, write_table_to_storage
from graphrag.index.utils.string import clean_str

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load prompt configuration from YAML
def load_extraction_config(yaml_path: str) -> dict[str, Any]:
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def extract_graph(
    text_units: pd.DataFrame,
    extraction_config: dict[str, Any],
) -> list[str]:
    """
    Batched inference using vLLM to extract entities and relationships.
    
    Args:
        text_units (pd.DataFrame): The input dataframe containing text to process.
        extraction_args (argparse.Namespace): Parsed arguments containing model configuration 
                                              (temperature, top_p, max_tokens, model_path, batch_size).
    Returns:
        List[str]: A list of extracted graph responses from the LLM.
    """

    # 1. System Prompt
    system_message = {"role": "system", "content": extraction_config['prompt_template'].format(
        completion_delimiter=extraction_config['completion_delimiter'],
        tuple_delimiter=extraction_config['tuple_delimiter'],
        record_delimiter=extraction_config['record_delimiter'],
        entity_types=", ".join(extraction_config['entity_types']),
        entity_types_examples=extraction_config['entity_types_examples'],
        relation_types=", ".join(extraction_config['relation_types']),
        relation_types_examples=extraction_config['relation_types_examples'],
    )}

    # 2. Few-Shot Examples (Dynamic Loop)
    few_shot_messages = []
    if 'examples' in extraction_config:
        for ex in extraction_config['examples']:
            few_shot_messages.append({"role": "user", "content": ex['user']})
            few_shot_messages.append({"role": "assistant", "content": ex['assistant'].format(
                completion_delimiter=extraction_config['completion_delimiter'],
                tuple_delimiter=extraction_config['tuple_delimiter'],
                record_delimiter=extraction_config['record_delimiter'],
            )})

    prompts_batch = []
    prompts = []
    
    # 3. User Prompt Template
    user_prompt_template = extraction_config['user_prompt']

    for _, row in text_units.iterrows():
        # Build the full conversation for this text unit
        conversation = [system_message] + few_shot_messages + [
            {"role": "user", "content": user_prompt_template.format(input_text=row['text'])}
        ]
        
        prompts_batch.append(conversation)

        if len(prompts_batch) == extraction_config['llm_config']['batch_size']:
            prompts.append(prompts_batch)
            prompts_batch = []
            
    if prompts_batch:
        prompts.append(prompts_batch)
    
    # Run Inference
    logger.info(f"Loading LLM: {extraction_config['llm_config']['model_path']}")
    
    llm = LLM(
        model=extraction_config['llm_config']['model_path'],
        trust_remote_code=True,
        max_model_len=extraction_config['llm_config']['max_model_len'],
        tensor_parallel_size=extraction_config['llm_config']['tensor_parallel_size'],
        enable_prefix_caching=True
    )
    
    sampling_params = SamplingParams(
        temperature=extraction_config['llm_config']['temperature'],
        top_p=extraction_config['llm_config']['top_p'],
        max_tokens=extraction_config['llm_config']['max_tokens']
    )
    logger.info("LLM loaded successfully.")

    all_responses = []
    for batch in prompts:
        outputs = llm.chat(batch, sampling_params=sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        all_responses.extend(responses)

    return all_responses


# --- PIPELINES ---

async def pipeline_1(config, context, callback_chain):
    """Pipeline 1: Create base text units from input documents."""
    from graphrag.index.input.factory import create_input
    from graphrag.index.workflows.create_base_text_units import (
        run_workflow as run_create_base_text_units,
    )

    dataset = await create_input(config.input, None, config.root_dir)
    logger.info(f"Final # of rows loaded: {len(dataset)}")
    
    context.stats.num_documents = len(dataset)
    await write_table_to_storage(dataset, "documents", context.storage)

    result = await run_create_base_text_units(config, context, callback_chain)
    logger.info(f"Pipeline 1 Result: {result}")

async def pipeline_2(config, context, callback_chain):
    """Pipeline 2: Create final documents."""
    from graphrag.index.workflows.create_final_documents import (
        run_workflow as run_create_final_documents,
    )
    result = await run_create_final_documents(config, context, callback_chain)
    logger.info(f"Pipeline 2 Result: {result}")

async def pipeline_3(context, extraction_config):
    """Pipeline 3: Extract graph using local LLM."""
    
    text_units = await load_table_from_storage("text_units", context.storage)
    
    all_responses = extract_graph(text_units=text_units, extraction_config=extraction_config)
    all_responses = pd.DataFrame(all_responses, columns=["response"])

    await write_table_to_storage(all_responses, "extracted_graph_responses", context.storage)
        
    logger.info(f"Extracted responses saved.")

# Helper functions for Graph processing
def _unpack_descriptions(data: Mapping) -> list[str]:
    value = data.get("description", None)
    return [] if value is None else value.split("\n")

def _unpack_source_ids(data: Mapping) -> list[str]:
    value = data.get("source_id", None)
    return [] if value is None else value.split(", ")

async def _process_results_directed(
    results: dict[int, str],
    tuple_delimiter: str,
    record_delimiter: str,
    _join_descriptions: bool = True,
) -> nx.DiGraph:
    """Parse result string to create a directed graph."""
    graph = nx.DiGraph()

    for source_doc_id, extracted_data in results.items():
        records = [r.strip() for r in extracted_data.split(record_delimiter)]

        for record in records:
            record = re.sub(r"^\(|\)$", "", record.strip())
            record_attributes = record.split(tuple_delimiter)

            # Entity Processing
            if record_attributes[0] == '"entity"' and len(record_attributes) >= 4:
                entity_name = clean_str(record_attributes[1].upper())
                entity_type = clean_str(record_attributes[2].upper())
                entity_description = clean_str(record_attributes[3])

                if graph.has_node(entity_name):
                    node = graph.nodes[entity_name]
                    if _join_descriptions:
                        node["description"] = "\n".join(
                            list({*_unpack_descriptions(node), entity_description})
                        )
                    elif len(entity_description) > len(node.get("description", "")):
                        node["description"] = entity_description
                    
                    node["source_id"] = ", ".join(
                        list({*_unpack_source_ids(node), str(source_doc_id)})
                    )
                    node["type"] = entity_type if entity_type != "" else node["type"]
                else:
                    graph.add_node(
                        entity_name,
                        type=entity_type,
                        description=entity_description,
                        source_id=str(source_doc_id),
                    )
            
            # Relationship Processing
            if record_attributes[0] == '"relationship"' and len(record_attributes) >= 5:
                source = clean_str(record_attributes[1].upper())
                target = clean_str(record_attributes[2].upper())
                edge_description = clean_str(record_attributes[3])
                edge_source_id = clean_str(str(source_doc_id))
                try:
                    weight = float(record_attributes[-1])
                except ValueError:
                    weight = 1.0

                if not graph.has_node(source):
                    graph.add_node(source, type="", description="", source_id=edge_source_id)
                if not graph.has_node(target):
                    graph.add_node(target, type="", description="", source_id=edge_source_id)

                if graph.has_edge(source, target):
                    edge_data = graph.get_edge_data(source, target)
                    if edge_data is not None:
                        weight += edge_data["weight"]
                        if _join_descriptions:
                            edge_description = "\n".join(
                                list({*_unpack_descriptions(edge_data), edge_description})
                            )
                        edge_source_id = ", ".join(
                            list({*_unpack_source_ids(edge_data), str(source_doc_id)})
                        )
                
                graph.add_edge(
                    source, target,
                    weight=weight,
                    description=edge_description,
                    source_id=edge_source_id,
                )
    return graph

async def pipeline_4(context, extraction_config):
    """Pipeline 4: Parse responses and create entities and relationships."""
    
    text_units = await load_table_from_storage("text_units", context.storage)
    all_responses = await load_table_from_storage("extracted_graph_responses", context.storage)
    
     # Regex to remove "Chain of Thought" if present
    think_pattern = re.compile(r'</think>(.*)', re.DOTALL)

    entity_dfs = []
    relationship_dfs = []
    
    for i, result in enumerate(all_responses["response"].tolist()):
        # Remove thinking tokens if present
        think_match = think_pattern.search(result)
        result = think_match.group(1).strip() if think_match else result.strip()

        # Process single graph
        graph = await _process_results_directed({i : result}, extraction_config['tuple_delimiter'], extraction_config['record_delimiter'])
        
        # Map the "source_id" back to the "id" field, the real id of the text_units
        for _, node in graph.nodes(data=True):
            if node:
                node["source_id"] = ",".join(
                    text_units['id'][int(id)] for id in node["source_id"].split(",")
                )

        for _, _, edge in graph.edges(data=True):
            if edge:
                edge["source_id"] = ",".join(
                    text_units['id'][int(id)] for id in edge["source_id"].split(",")
                )

        entities = [({"title": item[0], **(item[1] or {})}) for item in graph.nodes(data=True) if item]
        relationships = nx.to_pandas_edgelist(graph)
        
        entity_dfs.append(pd.DataFrame(entities))
        relationship_dfs.append(pd.DataFrame(relationships))
    

    from graphrag.index.operations.extract_graph.extract_graph import _merge_entities, _merge_relationships
    entities_merged = _merge_entities(entity_dfs)
    relationships_merged = _merge_relationships(relationship_dfs)
    
    logger.info(f"Extracted {len(entities_merged)} entities and {len(relationships_merged)} relationships.")
    
    await write_table_to_storage(entities_merged, "entities", context.storage)
    await write_table_to_storage(relationships_merged, "relationships", context.storage)

def finalize_entities_relationships_directed(
    entities: pd.DataFrame,
    relationships: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Finalize entities and relationships for directed graph."""
    
    graph = nx.from_pandas_edgelist(relationships, edge_attr='description', create_using=nx.DiGraph())
    
    entities_ = entities.copy()
    entities_.set_index("title", inplace=True)
    graph.add_nodes_from((n, dict(d)) for n, d in entities_.iterrows())

    degrees = pd.DataFrame([
        {"title": node, "degree": int(degree)}
        for node, degree in graph.degree
    ])

    final_entities = (
        entities.merge(degrees, on="title", how="left")
        .drop_duplicates(subset="title")
    )
    final_entities = final_entities.loc[entities["title"].notna()].reset_index(drop=True)
    final_entities["degree"] = final_entities["degree"].fillna(0).astype(int)
    final_entities.reset_index(inplace=True)
    # Generate IDs
    final_entities["human_readable_id"] = final_entities.index
    final_entities["id"] = final_entities["human_readable_id"].apply(lambda _x: str(uuid4()))
    
    final_entities = final_entities[["id","human_readable_id","title","type","description","text_unit_ids","degree"]]
    
    # Relationships
    final_relationships = relationships.drop_duplicates(subset=["source", "target"])
    
    from graphrag.index.operations.compute_edge_combined_degree import compute_edge_combined_degree
    final_relationships["combined_degree"] = compute_edge_combined_degree(
        final_relationships,
        degrees,
        node_name_column="title",
        node_degree_column="degree",
        edge_source_column="source",
        edge_target_column="target",
    )

    final_relationships.reset_index(inplace=True, drop=True)
    final_relationships["human_readable_id"] = final_relationships.index
    final_relationships["id"] = final_relationships["human_readable_id"].apply(lambda _x: str(uuid4()))

    final_relationships = final_relationships[["id", "human_readable_id", "source", "target", "description", "weight","combined_degree","text_unit_ids"]]
    
    return final_entities, final_relationships

async def apply_manual_overrides(
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    extraction_config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies manual overrides for entities and relationships from the few-shot examples defined in the YAML configuration.
    LLM tends to repeat the provided examples, so we use these to enforce specific entities/relationships.
    """
    
    examples = extraction_config.get('examples', [])
    if not examples:
        logger.info("No examples found in extraction config. Skipping manual overrides.")
        return entities, relationships

    logger.info(f"Processing {len(examples)} manual examples for overrides...")

    # 1. Parse Examples into DataFrames
    manual_graph = nx.DiGraph()
    
    for i, ex in enumerate(examples):
        formatted_response = ex['assistant'].format(
            completion_delimiter=extraction_config['completion_delimiter'],
            tuple_delimiter=extraction_config['tuple_delimiter'],
            record_delimiter=extraction_config['record_delimiter'],
        )
        
        # Parse this single example
        g = await _process_results_directed(
            {i: formatted_response},
            extraction_config['tuple_delimiter'],
            extraction_config['record_delimiter']
        )
        
        # Merge into the main manual graph
        # Note: This is a simple composition. If examples contradict each other, 
        # the later one might overwrite node attributes depending on nx.compose logic, 
        # but usually examples are distinct or complementary.
        manual_graph = nx.compose(manual_graph, g)

    # Convert Manual Graph to DataFrames
    manual_entities_list = [({"title": item[0], **(item[1] or {})}) for item in manual_graph.nodes(data=True) if item]
    manual_entities_df = pd.DataFrame(manual_entities_list)
    
    manual_relationships_list = nx.to_pandas_edgelist(manual_graph)
    manual_relationships_df = pd.DataFrame(manual_relationships_list)

    # 2. Apply Entity Overrides
    if not manual_entities_df.empty:
        # Standardize
        manual_entities_df['title'] = manual_entities_df['title'].astype(str).str.upper()
        # Ensure description is a list for consistency with pipeline_5 aggregation
        manual_entities_df['description'] = manual_entities_df['description'].apply(lambda x: [x] if isinstance(x, str) else x)
        
        logger.info(f"Found {len(manual_entities_df)} manual entities.")

        # Merge logic: Left join to keep all extracted entities
        # We use suffix '_man' for manual data
        merged = pd.merge(entities, manual_entities_df, on=['title'], how='left', suffixes=('', '_man'))
        
        # Overwrite Type and Description where manual data exists
        entities['type'] = merged['type_man'].fillna(entities['type'])
        entities['description'] = merged['description_man'].fillna(entities['description'])
    
    # 3. Apply Relationship Overrides
    if not manual_relationships_df.empty:
        # Standardize
        manual_relationships_df['source'] = manual_relationships_df['source'].astype(str).str.upper()
        manual_relationships_df['target'] = manual_relationships_df['target'].astype(str).str.upper()
        # Ensure description is a list
        manual_relationships_df['description'] = manual_relationships_df['description'].apply(lambda x: [x] if isinstance(x, str) else x)
        
        logger.info(f"Found {len(manual_relationships_df)} manual relationships.")

        # Identify the set of nodes involved in the manual overrides.
        # Logic: If we have manual data for specific nodes, we trust the manual connections
        # between them more than the LLM's output.
        manual_entity_titles = manual_entities_df['title'].unique().tolist() if not manual_entities_df.empty else []
        
        # A. Preserve 'text_unit_ids' from existing relationships if possible.
        # If the LLM found this edge too, it has text_unit_ids. We want to keep them.
        ids_to_add = relationships[['source', 'target', 'text_unit_ids']]
        manual_relationships_df = pd.merge(manual_relationships_df, ids_to_add, on=['source', 'target'], how='left')

        # B. Remove existing edges between manual nodes.
        # "Delete rows where both 'source' and 'target' are in the manual examples"
        if manual_entity_titles:
            condition_to_remove = relationships['source'].isin(manual_entity_titles) & \
                                  relationships['target'].isin(manual_entity_titles)
            relationships = relationships[~condition_to_remove]
        
        # C. Add the manual relationships
        relationships = pd.concat([relationships, manual_relationships_df], ignore_index=True)

    return entities, relationships


async def pipeline_5(config, context, extraction_config):
    """Pipeline 5: Clean and finalize entities and relationships."""

    entities = await load_table_from_storage("entities", context.storage)
    relationships = await load_table_from_storage("relationships", context.storage)
    
    # Load entity types and valid relations from Config
    valid_entity_types = [t.upper() for t in extraction_config.get('entity_types', [])]
    # Handle the list vs set requirement for relation types
    valid_relation_types = set(extraction_config.get('relation_types', []))

    # Filter invalid entities
    entities = entities[
        entities['type'].isin(valid_entity_types) &
        (entities['description'].str.len() > 0) &
        (entities['title'] != '')
    ]

    # Merge duplicate titles
    cleaned_entities = entities.groupby(["title"], sort=False).agg(
        description=("description", lambda x: np.concatenate(x.values)),
        text_unit_ids=("text_unit_ids", lambda x: np.concatenate(x.values)),
        type=('type', 'first'),
    ).reset_index()

    # Clean relationships description
    def clean_rel_description(desc_list):
        cleaned = []
        if isinstance(desc_list, (list, np.ndarray)):
            for des in desc_list:
                parts = des.split('\n') if isinstance(des, str) else []
                cleaned.extend([p.strip().lower() for p in parts])
        return cleaned

    relationships['description'] = relationships['description'].apply(clean_rel_description)

    # Filter against valid_relation_types from config
    relationships['description'] = relationships['description'].apply(
        lambda desc_list: sorted([item for item in set(desc_list) if item in valid_relation_types])
    )

    # Filter empty relationships
    relationships = relationships[
        (relationships['description'].str.len() > 0) &
        (relationships['source'] != '') &
        (relationships['target'] != '') &
        (relationships['source'] != relationships['target'])
    ].reset_index(drop=True)

    # Apply Manual Overrides using the YAML examples
    cleaned_entities, cleaned_relationships = await apply_manual_overrides(
        cleaned_entities,
        relationships,
        extraction_config
    )

    logger.info(f"Cleaned entities: {len(cleaned_entities)}")
    logger.info(f"Cleaned relationships: {len(cleaned_relationships)}")

    final_entities, final_relationships = finalize_entities_relationships_directed(cleaned_entities, cleaned_relationships)
    
    logger.info(f"Final entities: {len(final_entities)}")
    logger.info(f"Final relationships: {len(final_relationships)}")
    
    await write_table_to_storage(final_entities, "final_entities", context.storage)
    await write_table_to_storage(final_relationships, "final_relationships", context.storage)

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphRAG Extraction Pipeline")
    parser.add_argument("--root", type=str, required=True, help="Root data directory for GraphRAG project")
    parser.add_argument("--graphrag_config", type=str, default=None, help="Path to config yaml file. If not provided, looks for 'settings.yaml' in root directory.")
    parser.add_argument("--pipeline", type=str, choices=["1", "2", "3", "4", "5", "all"], default="all", help="Select pipeline stage to run")
    # the following arg is for pipeline 3-5 (LLM extraction and parseing)
    parser.add_argument("--extraction_config", type=str, default=None, help="Path to extraction YAML file. If not provided, looks for 'extraction_config.yaml' in root directory.")

    args = parser.parse_args()
    
    root_dir = Path(args.root)
    
    # Load Config
    cli_overrides = {}
    config = load_config(root_dir, args.graphrag_config, cli_overrides)
    if args.extraction_config is None:
        # find extraction_config.yaml in config root dir
        extraction_config_path = root_dir / "extraction_config.yaml"
        try:
            extraction_config = load_extraction_config(str(extraction_config_path))
        except FileNotFoundError:
            logger.error(f"Extraction config file not found at {extraction_config_path}. Please provide a valid path using --extraction_config.")
            sys.exit(1)
    else:
        extraction_config = load_extraction_config(args.extraction_config)

    # Setup Context
    storage_config = config.output.model_dump()
    storage = StorageFactory().create_storage(
        storage_type=storage_config["type"],
        kwargs=storage_config,
    )
    cache_config = config.cache.model_dump()
    cache = CacheFactory().create_cache(
        cache_type=cache_config["type"],
        root_dir=config.root_dir,
        kwargs=cache_config,
    )
    context = create_run_context(storage=storage, cache=cache, stats=None)
    callback_chain = WorkflowCallbacksManager()

    # Execution Flow
    if args.pipeline in ["1", "all"]:
        logger.info("Starting Pipeline 1...")
        asyncio.run(pipeline_1(config, context, callback_chain))
    
    if args.pipeline in ["2", "all"]:
        logger.info("Starting Pipeline 2...")
        asyncio.run(pipeline_2(config, context, callback_chain))

    if args.pipeline in ["3", "all"]:
        logger.info("Starting Pipeline 3...")
        asyncio.run(pipeline_3(context, extraction_config))

    if args.pipeline in ["4", "all"]:
        logger.info("Starting Pipeline 4...")
        asyncio.run(pipeline_4(context, extraction_config))
    
    if args.pipeline in ["5", "all"]:
        logger.info("Starting Pipeline 5...")
        asyncio.run(pipeline_5(config, context, extraction_config))