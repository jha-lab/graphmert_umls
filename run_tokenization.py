import argparse
import logging
import sys

from utils import tokenization_utils


logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def main(args):
	"""Run pretraining using the MLM objective"""
	tokenization_utils.main(yaml_file=args.yaml_file)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Input parameters for MLM pretraining',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	
	parser.add_argument('--yaml_file',
		metavar='',
		type=str,
		help='path to the yaml file with all training parameters',
		default=None)
	parser.set_defaults(do_train=False, smoke_test=False)

	args = parser.parse_args()
	main(args)
