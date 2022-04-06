import os
import sys
import torch
import argparse
import warnings
import utils.casting_dataset as ds

from utils.luna_model import Luna
from utils.external_utils.vision_transformer import VitGenerator, VisionTransformer
from utils.external_utils.preprocess import visualize_attention
from utils.lightning_classifier import Classifier
import utils.casting_dataset as cds

from functools import partial

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

def get_vit(path="/content/drive/MyDrive/Master_Thesis_Profactor/zdmp/pretrained_weights/vit.pth.tar", num_classes=2):
    if os.path.isfile(path):
        patch_size=8
        model = VisionTransformer(patch_size=patch_size, num_classes=num_classes, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
                        qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))
        model.load_state_dict(torch.load())
        return model
    else:
        warnings.warn(f"Pretrained weights for ViT not found in following path:\n\n{path}\n\n")
        return get_vit_finetune()

def get_vit_finetune(
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ):
    name = 'vit_small'
    patch_size = 8
    model = VitGenerator(
      name, 
      patch_size, 
      device, 
      evaluate=False, 
      random=False,
      verbose=True,
      num_classes=2
    )
    return model
    
def get_train_data(vit=False, mean=0.5642, std=0.2386, calc_mean_std=False):
    return cds.get_train_data(vit=vit, mean=mean, std=std, calc_mean_std=calc_mean_std)

def get_test_data(vit=False, mean=0.5642, std=0.2386):
    print(f"zdmp - ViT size - {vit}")
    return cds.get_test_data(vit=vit, mean=mean, std=std)

def main():
    parser = parse_arguments()
    args, _ = parser.parse_known_args()
    data_path = ds.dataset_path(new_dir=args.ds_dir, is_notebook=is_ipynb())
    
    luna_model = Luna()

if __name__ == '__main__':
    main()