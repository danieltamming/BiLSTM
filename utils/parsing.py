import argparse
import json

from easydict import EasyDict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config', default='crossval.json', type=str,
                        help='Config file name?')
    return parser.parse_args()

def get_config():
	args = get_args()
	with open('configs/'+args.config) as f:
		return EasyDict(json.load(f))