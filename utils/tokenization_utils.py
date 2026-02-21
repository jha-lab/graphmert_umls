import logging
import os
from itertools import chain
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import yaml
import datasets
from datasets import load_dataset, load_from_disk, DatasetDict, Value, Sequence

from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from training_arguments import ModelArguments, DataTrainingArguments, PreprocessingArguments

import spacy
from spacy.language import Language


logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed.
check_min_version("4.27.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

hf_auth_token = os.getenv('HF_AUTH_TOKEN')
if hf_auth_token is None:
    logger.warning("HF_AUTH_TOKEN not found. Private datasets will be inaccessible.")


def load_raw_datasets(data_args, model_args):
    """
    Load datasets either from HuggingFace Hub or from local files.
    
    Args:
        data_args: DataTrainingArguments containing dataset configuration
        model_args: ModelArguments containing cache and auth configuration
    
    Returns:
        DatasetDict: Dictionary containing 'train' and 'validation' splits
    """
    
    def split_validation_from_train(dataset_loader_func, loader_kwargs):
        """Helper to split validation set from train if it doesn't exist."""
        raw_datasets = dataset_loader_func(**loader_kwargs)
        
        if "validation" not in raw_datasets.keys():
            logger.info(f"Creating validation split ({data_args.validation_split_percentage}% of train)")
            splits = {
                "train": f"train[{data_args.validation_split_percentage}%:]",
                "validation": f"train[:{data_args.validation_split_percentage}%]"
            }
            raw_datasets = DatasetDict({
                key: dataset_loader_func(**loader_kwargs, split=split_str)
                for key, split_str in splits.items()
            })
        
        return raw_datasets
    
    if data_args.dataset_name is not None:
        # Downloading from hub
        if data_args.dataset_config_name == 'None': 
            data_args.dataset_config_name = None
        
        loader_kwargs = {
            "path": data_args.dataset_name,
            "name": data_args.dataset_config_name,
            "cache_dir": model_args.cache_dir,
            "streaming": data_args.streaming,
            "token": hf_auth_token,  # Use 'token' instead of deprecated 'use_auth_token'
        }
        
        raw_datasets = split_validation_from_train(load_dataset, loader_kwargs)
    
    else:
        # Loading from local files
        data_files = {}
        extension = None
        
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        
        if not data_files:
            raise ValueError("Must provide either dataset_name or train_file/validation_file")
        
        if extension == "txt":
            extension = "text"
        
        loader_kwargs = {
            "path": extension,
            "data_files": data_files,
            "cache_dir": model_args.cache_dir,
            "token": hf_auth_token,
        }
        
        raw_datasets = split_validation_from_train(load_dataset, loader_kwargs)
    
    return raw_datasets


def tokenize_dataset(raw_datasets, tokenizer, data_args, model_args, training_args, text_column_name, remove_columns, cache_file_name):
    """runs tokenize_function over dataset"""
    # Load spacy pipeline
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    @Language.component("lower_case_lemmas")
    def lower_case_lemmas(doc):
        for token in doc:
            token.lemma_ = token.lemma_.lower()
        return doc
    nlp.add_pipe("lower_case_lemmas", after="tagger")


    def tokenize_function(examples):
        """returns tokenized_examples"""
        # Runs spacy pipeline
        docs = list(nlp.pipe(examples[text_column_name]))
        word_lists = [[str(t) for t in doc] for doc in docs]
        tokenized_examples = tokenizer(word_lists, is_split_into_words=True, add_special_tokens=False, return_special_tokens_mask=False, return_token_type_ids=False, return_attention_mask=False)
        # Generate input sentences back
        tokenized_examples[text_column_name] = examples[text_column_name]
        return tokenized_examples

    
    processing_args = {
        'batched': True,
        'remove_columns': remove_columns,
        'desc': "Running tokenizer on every text in dataset",
    }
    if not data_args.streaming:
        processing_args['num_proc'] = data_args.preprocessing_num_workers
        processing_args['load_from_cache_file'] = not data_args.overwrite_cache

    tokenized_datasets = DatasetDict()
    # dataset name to save to disk; this one will be ready for trainng;
    
    with training_args.main_process_first(desc="dataset tokenization"):
        for split, dataset in raw_datasets.items():
            processing_args['cache_file_name'] = os.path.join(model_args.cache_dir, f'{cache_file_name[split]}_tokenized')
            if training_args.local_rank in [-1, 0]:
                os.makedirs(processing_args['cache_file_name'], exist_ok=True)
            processing_args['cache_file_name'] = os.path.join(processing_args['cache_file_name'], "cache.tmp") # dataset lib will not work for multiple processes w/o dot in filename

            tokenized_datasets[split] = dataset.map(
                tokenize_function,
                **processing_args,
            )
    return tokenized_datasets


def concatenate_texts_into_chunks(dataset, split: str, max_seq_length: int, tokenizer, data_args, training_args, cache_file_name):
    def group_texts(examples):
        """Concatenate all texts into chunks of max_seq_length, add special tokens and masks."""
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()} # flatten the list of lists
        total_length = len(concatenated["input_ids"])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop
        max_raw = max_seq_length - tokenizer.num_special_tokens_to_add()
        if total_length >= max_raw:
            total_length = (total_length // max_raw) * max_raw

        out = {}
        # Use numpy array so PyArrow infers fixed-size list from shape
        out["input_ids"] = np.array([
            tokenizer.build_inputs_with_special_tokens(concatenated["input_ids"][i : i + max_raw])
            for i in range(0, total_length, max_raw)
        ], dtype=np.int64)
        return out
    
    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
                # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
                # might be slower to preprocess.
    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            dataset = dataset.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                cache_file_name=f"{cache_file_name[split]}_grouped",
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping {split} texts in chunks of {max_seq_length}",
            )
        else:
            dataset = dataset.map(
                group_texts,
                batched=True,
            )
    return dataset


def main(yaml_file: str):        
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, PreprocessingArguments))
    
    # Read YAML and disable bf16/fp16 for CPU tokenization before parsing
    yaml_config = yaml.safe_load(Path(yaml_file).read_text())
    yaml_config['bf16'] = False
    yaml_config['fp16'] = False
    
    model_args, data_args, training_args, preprocessing_args = parser.parse_dict(yaml_config, allow_extra_keys=True)

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)

    set_seed(training_args.seed)

    raw_datasets = load_raw_datasets(data_args, model_args)
        
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "token": hf_auth_token,
    }

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, add_prefix_space=True, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if tokenizer.mask_token_id is None or tokenizer.mask_token is None:
        raise ValueError("tokenizer.mask_token or tokenizer.mask_token_id is None for mlm task")
    
    # that would work only for tokenizers  with bos / cls and eos / sep tokens (e.g roberta and bert)
    if (tokenizer.bos_token_id is None and tokenizer.cls_token_id is None) or (tokenizer.eos_token_id is None and tokenizer.sep_token_id is None):
        raise ValueError("set tokenizer.bos_token_id or/and tokenizer.eos_token_id for tokenization")

    def unique_cache_filename(file_path):
        cache_filename = str.split(str.split(file_path, "/")[-1], ".")[0]
        # Create a hash of the full file path to ensure unique cache file names
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        return f'{cache_filename}_{file_hash}'
    

    cache_file_name = {
        'train': f"{data_args.dataset_name}_train" if data_args.dataset_name is not None else unique_cache_filename(data_args.train_file),
        'validation': f"{data_args.dataset_name}_validation" if data_args.dataset_name is not None else unique_cache_filename(data_args.validation_file),
    }
    if preprocessing_args.tokenized_dataset_output_path is not None:
        tokenized_dataset_name = {
            'train': os.path.join(preprocessing_args.tokenized_dataset_output_path, cache_file_name["train"] + '_tokenized'),
            'validation': os.path.join(preprocessing_args.tokenized_dataset_output_path, cache_file_name["validation"] + '_tokenized'),
        }
    else:
        tokenized_dataset_name = {
            'train': os.path.join(model_args.cache_dir, cache_file_name["train"] + '_tokenized'),
            'validation': os.path.join(model_args.cache_dir, cache_file_name["validation"] + '_tokenized'),
        }        

    if training_args.local_rank in [-1, 0]:
        os.makedirs(model_args.cache_dir, exist_ok=True)

    # we tokenize all the texts.
    column_names = list(raw_datasets["train"].features) if raw_datasets["train"].features is not None else list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]
    # We need the sentence column for knowledge graph injection
    remove_columns = deepcopy(column_names)
    remove_columns.remove(text_column_name)

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Code for testing only (reduces dataset size)
    if preprocessing_args.cut_dataset_for_testing:
        raw_datasets["train"] = raw_datasets["train"].select(range(1000))
        raw_datasets["validation"] = raw_datasets["validation"].select(range(200))

    if data_args.line_by_line:
        raise NotImplementedError("The spacy pipeline is not implemented with line_by_line command")

    # we tokenize every text, then concatenate them together before splitting them in smaller parts.
    tokenized_datasets = tokenize_dataset(raw_datasets, tokenizer, data_args, model_args, training_args, text_column_name, remove_columns, cache_file_name)

    for split, name in cache_file_name.items():
        cache_file_name[split] = os.path.join(model_args.cache_dir, name)
        if training_args.local_rank in [-1, 0]:
            os.makedirs(cache_file_name[split], exist_ok=True)
        cache_file_name[split] = os.path.join(cache_file_name[split], "cache.tmp") # dataset lib will not work for multiple processes w/o dot in filename

    for column_name in ['sentence', 'text', 'idx']:
        # go over train, validation splits
        for key in tokenized_datasets.keys():
            if column_name in tokenized_datasets[key].features.keys():
                tokenized_datasets[key] = tokenized_datasets[key].remove_columns(column_name)
    
    for key in tokenized_datasets.keys():
        tokenized_datasets[key] = concatenate_texts_into_chunks(tokenized_datasets[key], key, max_seq_length, tokenizer, data_args, training_args, cache_file_name)

    for key in tokenized_datasets.keys():
        tokenized_datasets[key].save_to_disk(tokenized_dataset_name[key])
        logger.info(f'Tokenized dataset {key} saved to {tokenized_dataset_name[key]}')
