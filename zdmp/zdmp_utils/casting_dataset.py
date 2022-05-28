import os
import random
import torch
import torchvision
import torchvision.transforms as T

from torch.utils.data import random_split


def path_dialog():
    import tkinter
    from tkinter import filedialog
    root = tkinter.Tk()
    root.withdraw()
    return filedialog.askdirectory()

def check_extension(path):
    ext = str(path).split(".")[-1]
    if ext == "jpg" or ext == "jpeg":
        return True
    else:
        return False

def dataset_path(new_dir:str=None):
    txt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            r"casting_dataset_path.txt" if new_dir is None else new_dir
        )
    if os.path.isfile(txt_path) and not new_dir:
        with open(txt_path, "r") as text_file:
            path = text_file.readlines()[0]
    else:
        path = path_dialog()
        with open(txt_path, "w") as text_file:
            text_file.write(path)
    return path

def transform_train(vit=False, mean=0.5643, std=0.2386):
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomAffine(
            degrees=2.5,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            hue=0.2
        ),
        T.ToTensor(),
        T.Resize(224) if vit else T.Resize(300),
        T.Normalize(
            mean=torch.tensor(mean),
            std=torch.tensor(std)
        )
    ])

def transform_test(vit=False, mean=0.5643, std=0.2386):
    print(f"transform - ViT size - {vit}")
    return T.Compose([
        T.ToTensor(),
        T.Resize(224) if vit else T.Resize(300),
        T.Normalize(
            mean=torch.tensor(mean),
            std=torch.tensor(std)
        )
    ])

def get_train_data(rand:bool=True, new_dir:str=None, vit:bool=False, validate=True, mean=0.5643, std=0.2386, calc_mean_std=False):
    img_fldr = torchvision.datasets.ImageFolder(
        root=os.path.join(dataset_path(new_dir=new_dir), "train"),
        transform=transform_train(vit=vit, mean=mean, std=std) if not calc_mean_std else T.ToTensor(),
        is_valid_file=check_extension
    )
    if rand:
        img_fldr.samples = sorted(img_fldr.samples, key=lambda k:random.random())
    if validate:
        size = len(img_fldr)
        size_t = round(size*0.9)
        size_v = size-size_t
        train, val = random_split(img_fldr, [size_t, size_v], generator=torch.Generator().manual_seed(42))
        return train, val
    else:
        return img_fldr


def get_test_data(rand:bool=True, new_dir:str=None, vit:bool=False, mean=0.5643, std=0.2386):
    print(f"get_data - ViT size - {vit}")
    img_fldr = torchvision.datasets.ImageFolder(
        root=os.path.join(dataset_path(new_dir=new_dir), "test"),
        transform=transform_test(vit=vit, mean=mean, std=std),
        is_valid_file=check_extension
    )
    if rand:
        img_fldr.samples = sorted(img_fldr.samples, key=lambda k:random.random())
    return img_fldr