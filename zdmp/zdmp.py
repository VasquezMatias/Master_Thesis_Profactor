import os
import sys
import torch
import argparse
import utils.casting_dataset as ds

from utils.luna_model import Luna
from utils.lightning_classifier import Classifier
import utils.casting_dataset as cds


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name',
        help="Name of the model used and how it will be saved. (Luna/ViT)",
        default="Luna",
        type=str
    )
    parser.add_argument('--ds-dir',
        help="Give a new directory where the dataset is saved.",
        default=False,
        action='store_true'
    )
    return parser

def is_ipynb():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def get_luna(pretrained=True):
    return Luna(pretrained=pretrained)

def get_luna_trainer():
    model = Luna(pretrained=False)
    return Classifier(model)
    
def get_train_data():
    return cds.get_train_data()

def get_test_data():
    return cds.get_test_data()

def main():
    parser = parse_arguments()
    args, _ = parser.parse_known_args()
    data_path = ds.dataset_path(new_dir=args.ds_dir, is_notebook=is_ipynb())
    
    luna_model = Luna()

if __name__ == '__main__':
    main()