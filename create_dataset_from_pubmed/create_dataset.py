# papers in the output are shuffled, but paragraphs in them aren't

# check that it the final pmcid list is determenistic given the seed
# TO DO:   add arrow support;

import os
import glob
import re
import random
from typing import List
from math import ceil
import logging
import xml.etree.ElementTree as ET
from lxml.etree import XMLSyntaxError

# donwload only on the first launch
# nltk.download('punkt')

import pubmed_parser as pp
from nltk.tokenize import sent_tokenize
from tokenizers.normalizers import BertNormalizer

import json

FIELDS_TO_DROP = ("pmc", "pmid", "reference_ids", "section")
LOG_FILE = 'log.info'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8', mode='w')
formatter = logging.Formatter('%(levelname)s %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


FOLDER = "/scratch/gpfs/JHA/mb5157/large_data/diabetes_2025_6years"
# params to change
params = {
    'extension': 'json', # possible values: 'json',
    'abstracts_only': True,
    'split_into_sentences': False,
    'paper_folder': os.path.join(FOLDER, f'papers'),
    'dataset_folder': os.path.join(FOLDER, 'dataset'),    # output_folder
    'normalize': True,
    'drop_fields': True,
    'shuffle': True,
    'seed': 1331,   # seed for random choice
    'train_size': 350_000,     #num papers in train, output will be splitted into chunks of size params['batch_size']
    'eval_size': 39_000,
    'test_size': 0,
    'batch_size': 1000,  # num papers per one json file (chunk)
}

params['train_chunk_dataset_folder'] = os.path.join(params['dataset_folder'], 'pre_train') # output folder for train chunks
params['eval_chunk_dataset_folder'] = os.path.join(params['dataset_folder'], 'pre_eval') # output folder for eval chunks
params['test_chunk_dataset_folder'] = os.path.join(params['dataset_folder'], 'pre_test') # output folder for eval chunks
params['train_dataset_folder'] = os.path.join(params['dataset_folder'], 'train') # output folder train
params['eval_dataset_folder'] = os.path.join(params['dataset_folder'], 'eval') # output folder train
params['test_dataset_folder'] = os.path.join(params['dataset_folder'], 'test') # output folder train

random.seed = 1331

if not os.path.exists(params['paper_folder']):
    raise FileNotFoundError(f"{params['paper_folder']} doesn't exist")

if not os.path.exists(params['dataset_folder']):
    os.mkdir(params['dataset_folder'])

if not os.path.exists(params['eval_dataset_folder']):
    os.mkdir(params['eval_dataset_folder'])
    os.mkdir(os.path.join(params['eval_dataset_folder'], 'pmcid_index'))

if not os.path.exists(params['test_dataset_folder']):
    os.mkdir(params['test_dataset_folder'])
    os.mkdir(os.path.join(params['test_dataset_folder'], 'pmcid_index'))

if not os.path.exists(params['train_chunk_dataset_folder']):
    os.mkdir(params['train_chunk_dataset_folder'])
    os.mkdir(os.path.join(params['train_chunk_dataset_folder'], 'pmcid_index'))

if not os.path.exists(params['eval_chunk_dataset_folder']):
    os.mkdir(params['eval_chunk_dataset_folder'])
    os.mkdir(os.path.join(params['eval_chunk_dataset_folder'], 'pmcid_index'))

if not os.path.exists(params['test_chunk_dataset_folder']):
    os.mkdir(params['test_chunk_dataset_folder'])
    os.mkdir(os.path.join(params['test_chunk_dataset_folder'], 'pmcid_index'))


def save_to_json(records, chunk: int, added_pmcids, split, output_folder: str, remainder=0):
    """saves chunks to files and records pmcids in each chunk to file"""
    out_file = os.path.join(output_folder,  f"{split}{chunk}.{params['extension']}")
    with open(out_file, 'w', encoding="utf-8") as json_file:
        json.dump(records, json_file, indent=0, ensure_ascii=False)
        if not remainder:
            print(f"{params['batch_size']} papers added to file: {out_file}")
        else:
            print(f'{remainder} papers added to file: {out_file}')
    # log pmcids for split
    json_idx_file = os.path.join(output_folder, 'pmcid_index', f"{split}{chunk}_pmcid.json")
    with open(json_idx_file, 'w', encoding="utf-8") as idx_file:
        json.dump(added_pmcids, idx_file, indent=0, ensure_ascii=False)


normalizer_with_cleaning = BertNormalizer(clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=True)


def split_into_sentences(paragraph):
    """split into sentences and copy all other keys in the paragraph"""
    sentences = []
    paragraph_sentences = sent_tokenize(paragraph['text'])
    for sentence in paragraph_sentences:
        new_sentence = paragraph.copy()
        new_sentence['text'] = normalizer_with_cleaning.normalize_str(sentence)
        sentences.append(new_sentence)
    return sentences


def remove_starting_phrases_from_abstracts(text):
    """
    Remove a set of known starting phrases from the beginning of the text.
    The regex is case-insensitive and removes optional punctuation/whitespace following the phrase.
    """
    # Define the phrases you want to remove (in a non-capturing group)
    pattern = re.compile(
        r"^(?:background:?|abstract|purpose|key points|introduction|objective:?|aim(?:/hypothesis)?|graphical abstract|supplemental|summary|context|rationale)(?:s\b)?(?: and)?[\s:,-/]+(?:background|abstract|purpose|key points|introduction|objective|aim|hypothesis|supplemental|summary|context|rationale)?(?:s\b)?[\s:,-/]*",
        re.IGNORECASE
    )
    # Substitute the pattern with an empty string and strip leading/trailing whitespace.
    return pattern.sub("", text).strip()


def parse_one_paper(nxml_file, abstracts_only=False):
    """
        get paragraphs from a paper, process them
        one paragraph: Dict, keys: pmc, pmid, reference_ids, section, text
    """
    dataset_records = []
    if abstracts_only:
        try:
            abstract = pp.parse_pubmed_xml(nxml_file)['abstract']
        except (XMLSyntaxError, ET.ParseError) as e:
            logging.error(f"Error parsing abstract in file {nxml_file}: {e}")
            return None
        abstract = remove_starting_phrases_from_abstracts(abstract)
        list_of_paragraphs = [{'text': abstract}]

    else:
        try:
            list_of_paragraphs = pp.parse_pubmed_paragraph(nxml_file, all_paragraph=True)
        except (XMLSyntaxError, ET.ParseError) as e:
            logging.error(f"Error parsing paragraphs in file {nxml_file}: {e}")
            return None
        if params["drop_fields"]:
            for paragraph in list_of_paragraphs:
                for key in FIELDS_TO_DROP:
                    paragraph.pop(key, None)


    if params['split_into_sentences']:
        for paragraph in list_of_paragraphs:
            dataset_records.extend(split_into_sentences(paragraph))
    else:
        dataset_records = list_of_paragraphs

    return dataset_records


records = []
items = os.listdir(params['paper_folder'])
assert len(items) >= params['train_size'] + params['eval_size'] + params['test_size'], "not enough papers in folder"
logger.info(f"found {len(items)} papers in {params['paper_folder']}")
items = sorted(items) # to make items in determenistic oreder for reproducibility
choosen = random.sample(items, k=(params['train_size'] + params['eval_size'] + params['test_size']))
if params['shuffle']:
    random.shuffle(choosen) # the result of random.sample is in selection order

train = choosen[:params['train_size']]
eval = choosen[params['train_size'] : params['train_size'] + params['eval_size']]
test = choosen[params['train_size'] + params['eval_size'] : ]
logger.info(f"{len(train)} papers will be in train, {len(eval)} in eval, {len(test)} in test")

def add_papers(dataset: List[str], batch_size: int, output_folder: str, split: str) -> int:
    ''' parsing papers loop: saves paper chunks of size batch_size as json'''
    records = []
    added_papers = 0
    skipped_papers = 0
    added_pmcids  = []
    for item in dataset:
        pmcid = item

        item = os.path.join(params['paper_folder'], item)
        if not os.path.isdir(item):
            logger.warning(f"WARNING: no dir {item}")
            continue

        nxml_files = glob.glob(os.path.join(item, '*.nxml'))
        if len(nxml_files) != 1:
#             raise ValueError(f"num of nxml files is not equal to 1 in folder {pmcid}")
             logger.warning(f"num of nxml files is not equal to 1 in folder {pmcid}")
             continue
        
        dataset_records = parse_one_paper(nxml_files[0], abstracts_only=params['abstracts_only'])
        if dataset_records is not None:
            records.extend(dataset_records)
            added_papers += 1
            added_pmcids.append(pmcid)
        else:
            skipped_papers += 1

        if not added_papers % batch_size:
            if params['extension'] == 'json':
                chunk = added_papers // batch_size
                save_to_json(records, chunk, added_pmcids, split, output_folder=output_folder)
                records = []
                added_pmcids = []

    if added_pmcids != []:
        chunk = added_papers // batch_size + 1
        save_to_json(records, chunk, added_pmcids, split, output_folder, remainder=added_papers % batch_size)
        added_pmcids = []
    logger.info(f"total paper added: {added_papers}")
    logger.info(f"skipped_papers: {skipped_papers}")
    return chunk


# get number of chunks & save them to jsons
num_chunk_train  = add_papers(train, params['batch_size'], params['train_chunk_dataset_folder'], split='train')
if params['eval_size'] > 0:
    num_chunk_eval = add_papers(eval, params['batch_size'], params['eval_chunk_dataset_folder'], split='eval')
if params['test_size'] > 0:
    num_chunk_test = add_papers(test, params['batch_size'], params['test_chunk_dataset_folder'], split='test')

# ---------------
#combining dataset chunks

def combine_dataset(start: int, chunk: int, split: str) -> None:
    """combine dataset start-chunk into one file, chunk is included"""
    assert split in ('train', 'eval', 'test')
    # List of JSON file names
    json_files = [os.path.join(params[f'{split}_chunk_dataset_folder'], f"{split}{num}.json") for num in range(start, chunk + 1)]
    # Initialize an empty list to hold the combined data
    combined_data = []

    # Iterate through the JSON files
    for file_name in json_files:
        with open(file_name, 'r') as file:
            data = json.load(file)  # Parse JSON content into a Python list
            combined_data.extend(data)  # Combine the data

    if params['shuffle']:
        random.shuffle(combined_data)
    # Write the combined data to a new JSON file
    output_name = os.path.join(params[f'{split}_dataset_folder'], f"{split}_{start}_{chunk}.json")
    with open(output_name, 'w', encoding="utf-8") as output_file:
        json.dump(combined_data, output_file, indent=0, ensure_ascii=False)

    print(f'Combined 1-{chunk} JSON files; saved to {output_name}')


start = 1
if not os.path.exists(params['train_dataset_folder']):
    os.mkdir(params['train_dataset_folder'])
combine_dataset(start, num_chunk_train, 'train')

if not os.path.exists(params['eval_dataset_folder']) and params['eval_size'] > 0:
    os.mkdir(params['eval_dataset_folder'])
combine_dataset(start, num_chunk_eval, 'eval')

if not os.path.exists(params['test_dataset_folder']) and params['test_size'] > 0:
    os.mkdir(params['test_dataset_folder'])
combine_dataset(start, num_chunk_test, 'test')
