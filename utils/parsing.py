import argparse
import json

from easydict import EasyDict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config', default='crossval_sr.json', type=str,
                        help='Config file name?')
    parser.add_argument('-wkrs', '--num_workers', default=-1, type=int,
    					help='Number of workers for dataloaders?')
    parser.add_argument('-bsize', '--batch_size', default=-1, type=int,
    					help='Batch size for datloaders?')
    return parser.parse_args()

def get_config():
    args = get_args()
    with open('configs/'+args.config) as f:
        config = EasyDict(json.load(f))

    if args.num_workers != -1: 
        config.num_workers = args.num_workers
    if args.batch_size != -1: 
        config.batch_size = args.batch_size

    return config
