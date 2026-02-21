import yaml
import logging
import math

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logger(name):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(name)

def get_char_ngrams(text, n=3):
    text = str(text).lower()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def jaccard_similarity(str1, str2, n=3):
    set1 = get_char_ngrams(str1, n)
    set2 = get_char_ngrams(str2, n)
    union = len(set1 | set2)
    return len(set1 & set2) / union if union > 0 else 0.0

def cosine_similarity(str1, str2, n=3):
    set1 = get_char_ngrams(str1, n)
    set2 = get_char_ngrams(str2, n)
    denom = math.sqrt(len(set1)) * math.sqrt(len(set2))
    return len(set1 & set2) / denom if denom > 0 else 0.0