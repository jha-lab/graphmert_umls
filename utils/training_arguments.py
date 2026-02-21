from dataclasses import dataclass, field
from typing import Optional, List
import hashlib

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    num_relationships: Optional[int] = field(
        default=None,
        metadata={"help": "Max number of semantic relationship types in the knowledge graph"},
    )

    relation_emb_dropout: float = field(
        default=0.3,
        metadata={"help": "Dropout probability on all relation embeddings"},
    )

    exp_mask_base: Optional[float] = field(
        default=0.6,
        metadata={"help": "Base for the exponential mask on the graph attention weights, "},
    )
    mlm_sbo: bool = field(
        default=True,
        metadata={"help": "Whether to use span masking and span boundaries objective."},
    )

    span_upper_length: Optional[int] = field(
        default=None,
        metadata={"help": "Upper limit for the length of the masked span. If None, set to number of leaves."},
    )
    graph_types: Optional[List[str]] = field(
        default_factory=lambda: ['root_undirected', 'leaf_undirected', 'leaf_connected_undirected'],
        metadata={"help": "List of graph types to be used)."},
    )



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})

    mlm_on_leaves_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss on leaf nodes"}
    )

    pretrained_embeddings_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained node embeddings to initialize the model embeddings"}
    )

    lrs: List[float] = field(
        default_factory=lambda: [1e-4],
        metadata={"help": "Learning rates for different jobs/tasks"},
    )


@dataclass
class PreprocessingArguments:
    """
    Additional arguments specific to preprocessing only.
    """
    tokenized_dataset_output_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the tokenized dataset"}
    )
    
    subword_token_start: str = field(
        default='##',
        metadata={"help": "Prefix for subword tokens (e.g., '##' for BERT-style tokenizers)"}
    )
    
    cut_dataset_for_testing: bool = field(
        default=False,
        metadata={"help": "Whether to use only a small subset for testing"}
    )
    
    # Knowledge graph injection parameters
    injections_train_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to CSV file with training knowledge graph injections"}
    )
    
    injections_eval_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to CSV file with evaluation knowledge graph injections"}
    )
    
    relation_map_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to JSON file with relation type mappings"}
    )

    train_dataset_with_heads: Optional[str] = field(
        default=None,
        metadata={"help": "Path to preprocessed training dataset with entity heads"}
    )
    eval_dataset_with_heads: Optional[str] = field(
        default=None,
        metadata={"help": "Path to preprocessed evaluation dataset with entity heads"}
    )


def unique_cache_filename(file_path):
    cache_filename = str.split(str.split(file_path, "/")[-1], ".")[0]
    # Create a hash of the full file path to ensure unique cache file names
    file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
    return f'{cache_filename}_{file_hash}'