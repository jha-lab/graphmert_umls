#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""


import logging
import math
import os, sys
import multiprocessing

import datasets
from datasets import load_from_disk, DatasetDict

import torch

import evaluate

import transformers

from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from graphmert_model import GraphMertDataCollatorForLanguageModeling, GraphMertConfig, GraphMertForMaskedLM
AutoConfig.register("graphmert", GraphMertConfig)
AutoModelForMaskedLM.register(GraphMertConfig, GraphMertForMaskedLM)

from filelock import FileLock

from .trainer_utils import TrainerWithCosineAnealing

from .training_arguments import ModelArguments, DataTrainingArguments, PreprocessingArguments, unique_cache_filename

from .job_utils import get_job_info


logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(yaml_file: str):
    # See all possible arguments in src/transformers/training_args.py
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, PreprocessingArguments))
    model_args, data_args, training_args, preprocessing_args = parser.parse_yaml_file(yaml_file, allow_extra_keys=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    training_args.label_names = ["labels"]
    
    job_id, task_id, is_slurm = get_job_info()
    logger.info(f"Job ID: {job_id}, Task ID: {task_id}, SLURM: {is_slurm}")

    if is_slurm and data_args.lrs is not None:
        training_args.learning_rate = data_args.lrs[task_id]

    if training_args.output_dir.endswith('outputs/test') or training_args.output_dir.endswith('outputs/test/'):
        training_args.overwrite_output_dir = True

    # to create different output dirs for different model types
    suffix_for_output_dir = f'span/rel{model_args.num_relationships}' if model_args.mlm_sbo else f'mlm/rel{model_args.num_relationships}'
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    efficient_bs = num_gpus * training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    logger.info(f"Using {num_gpus} GPU(s), effective batch size: {efficient_bs}")
    training_args.output_dir = os.path.join(training_args.output_dir, suffix_for_output_dir, f'bs{efficient_bs}_lr_{training_args.learning_rate}')

    if training_args.local_rank in [-1, 0]:
        if not os.path.exists(training_args.output_dir):
            os.makedirs(training_args.output_dir)

    if data_args.preprocessing_num_workers is None:
        data_args.preprocessing_num_workers = multiprocessing.cpu_count()

    logger.info(f"Number of workers: {data_args.preprocessing_num_workers}")
   
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f'max_seq_length: {data_args.max_seq_length}') 
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Data parameters {data_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

        if training_args.local_rank in [-1, 0]:
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

    if training_args.local_rank in [-1, 0]:
        with open(os.path.join(training_args.output_dir, 'job_id'), 'w') as file:
            file.write(job_id)
            file.write('\n')


    set_seed(training_args.seed)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = CONFIG_MAPPING['graphmert']()
    logger.warning("You are instantiating a new config instance from scratch.")

    # If a knowledge graph is not used, set max nodes to sequence length so leaf nodes are not created
    if not ('leaf_directed' in model_args.graph_types or 'leaf_undirected' in model_args.graph_types):
        config.max_nodes = data_args.max_seq_length
    else:
        config.root_nodes = data_args.max_seq_length
    config.graph_types = model_args.graph_types
    config.num_relationships = model_args.num_relationships
    config.exp_mask_base = model_args.exp_mask_base
    config.mlm_sbo = model_args.mlm_sbo
    config.relation_emb_dropout = model_args.relation_emb_dropout

    config.keys_to_ignore_at_inference = ['hidden_states', 'attentions', 'mlm_loss', 'sbo_loss', 'sbo_logits']

    hf_auth_token = os.getenv('HF_AUTH_TOKEN')
    if hf_auth_token is None:
        logger.warning("HF_AUTH_TOKEN not found. Private datasets will be inaccessible.")
        
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
        raise ValueError("set tokenizer.bos_token_id or/and tokenizer.eos_token_id in config")
    config.bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    config.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    config.pad_token_id = tokenizer.pad_token_id
    
    if model_args.config_overrides is not None:
        config.update_from_string(model_args.config_overrides)
        logger.info(f"Overriding config: {model_args.config_overrides}")
        logger.info(f"New config: {config}")

    model = AutoModelForMaskedLM.from_config(config)


    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if data_args.pretrained_embeddings_path is not None:
        raise NotImplementedError("pretrained_embeddings_path not implemented yet")
        # pretrained_weights = torch.load(data_args.pretrained_embeddings_path)
        # new_atom_encoder = Embedding(model.config.vocab_size, model.config.pretrained_emb_dim)    # for full size
        # new_atom_encoder.weight.data.copy_(pretrained_weights)
        # model.set_input_embeddings(new_atom_encoder)
        # for param in model.graphmert.graph_encoder.graph_node_feature.atom_encoder.parameters():
        #     param.requires_grad = False


    # This one will take care of randomly masking the tokens.
    data_collator = GraphMertDataCollatorForLanguageModeling(
        config=config,
        tokenizer=tokenizer,
        graph_types=model_args.graph_types,
        process_arch_tensors=not config.fixed_graph_architecture,
        on_the_fly_processing=False,
        mlm_sbo=model_args.mlm_sbo,
        mlm_probability=data_args.mlm_probability,
        mlm_on_leaves_probability=data_args.mlm_on_leaves_probability,
        subword_token_start=preprocessing_args.subword_token_start,
        )


    cache_file_name = {
        'train': f"{data_args.dataset_name}_train" if data_args.dataset_name is not None else unique_cache_filename(data_args.train_file),
        'validation': f"{data_args.dataset_name}_validation" if data_args.dataset_name is not None else unique_cache_filename(data_args.validation_file),
    }

    processed_dataset_name = {
        'train': os.path.join(model_args.cache_dir, cache_file_name["train"], 'ready_for_training'),
        'validation': os.path.join(model_args.cache_dir, cache_file_name["validation"], 'ready_for_training'),
    }

    with FileLock(f"{processed_dataset_name['train']}.lock", timeout=60), FileLock(f"{processed_dataset_name['validation']}.lock", timeout=60):
        processed_dataset = DatasetDict({
            "train": load_from_disk(processed_dataset_name['train']),
            "validation": load_from_disk(processed_dataset_name['validation']),
        })

    if training_args.do_train:
        if "train" not in processed_dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = processed_dataset["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        train_dataset.set_format(type='torch', columns=['input_nodes', 'attention_mask', 'head_lengths', 'leaf_relationships', 'start_indices', 'special_tokens_mask'])


    if training_args.do_eval:
        if "validation" not in processed_dataset:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = processed_dataset["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))


        eval_dataset.set_format(type='torch', columns=['input_nodes', 'attention_mask', 'head_lengths', 'leaf_relationships', 'start_indices', 'special_tokens_mask'])


        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        experiment_id=str(int(job_id) % 100) if job_id != 'local_0' else 'local_0'
        metric = evaluate.load("accuracy", experiment_id=experiment_id)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics
            
            labels = labels.reshape(-1)
            preds = preds.reshape(-1)
            mask = labels != -100
            labels = labels[mask]
            preds = preds[mask]
            return metric.compute(predictions=preds, references=labels)

    
    # Initialize our Trainer
    trainer_init = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset if training_args.do_train else None,
        "eval_dataset": eval_dataset if training_args.do_eval else None,
        "tokenizer": tokenizer,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics if training_args.do_eval else None,
        "preprocess_logits_for_metrics": preprocess_logits_for_metrics if training_args.do_eval else None,
    }

    if training_args.lr_scheduler_type == 'cosine_with_restarts':
        trainer = TrainerWithCosineAnealing(
            **trainer_init,
            num_cycles=10,
        )
    else:
        trainer = Trainer(**trainer_init)

    # Training
    if training_args.do_train:

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        if training_args.local_rank in [-1, 0]:
            trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics = train_result.metrics

            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        if training_args.local_rank in [-1, 0]:
            max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
            print(metrics)
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")
            metrics["perplexity"] = perplexity

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    if training_args.local_rank in [-1, 0]:
        kwargs = {}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

    return metrics
