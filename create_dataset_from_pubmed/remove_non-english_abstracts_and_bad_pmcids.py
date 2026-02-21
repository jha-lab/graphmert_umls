"""""
This one detect langauage on abstracts only, for papers change sampled text

    parsed = pp.parse_pubmed_xml(nxml_file)['abstract']
to
    list_of_paragraphs = pp.parse_pubmed_paragraph(nxml_files[0], all_paragraph=True)
"""

import os
import glob
import logging
from lxml.etree import XMLSyntaxError
import xml.etree.ElementTree as ET
from langdetect import detect, DetectorFactory, LangDetectException
import pubmed_parser as pp

# Ensure reproducible language detection
DetectorFactory.seed = 0

PAPER_FOLDER            = '/scratch/gpfs/JHA/mb5157/large_data/diabetes_2025_6years/papers'
PMCID_NONENG_FILE       = '/scratch/gpfs/JHA/mb5157/large_data/diabetes_2025_6years/non-eng_and_bad_pmcids.txt'

MIN_ABSTRACT_CHARS = 200  # below this, we cannot precisely decide on Langauge

logging.basicConfig(
    filename=os.path.join(os.path.dirname(PMCID_NONENG_FILE), 'langcheck.log'),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)


def is_english(text: str, min_chars: int = MIN_ABSTRACT_CHARS) -> bool:
    if len(text) < min_chars:
        return False
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False


def main():
    paper_ids = sorted(os.listdir(PAPER_FOLDER))
    with open(PMCID_NONENG_FILE, 'w') as noneng_f:

        for idx, pmcid in enumerate(paper_ids, start=1):
            pmc_dir = os.path.join(PAPER_FOLDER, pmcid)
            nxml_list = glob.glob(os.path.join(pmc_dir, '*.nxml'))
            if not nxml_list:
                noneng_f.write(pmcid + "\n")
                continue

            nxml_file = nxml_list[0]
            try:
                # parse all paragraphs and join
                list_of_paragraphs = pp.parse_pubmed_paragraph(nxml_file, all_paragraph=True)
                abstract = "\n".join(p.get('text', '') for p in list_of_paragraphs).strip()
            except (XMLSyntaxError, ET.ParseError) as e:
                logging.error(f"Error parsing abstract in {nxml_file}: {e}")
                noneng_f.write(pmcid + "\n")
                continue
            except Exception as e:
                logging.error(f"Unexpected error on {nxml_file}: {e}")
                noneng_f.write(pmcid + "\n")
                continue

            if not abstract:
                noneng_f.write(pmcid + "\n")
                continue

            if not is_english(abstract):
                noneng_f.write(pmcid + "\n")

            if idx % 500 == 0:
                print(f"Processed {idx} papers")


if __name__ == "__main__":
    main()

