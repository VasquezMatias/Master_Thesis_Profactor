import os
import random
import torch
import torchvision
import torchvision.transforms as T


def path_dialog():
    import tkinter
    from tkinter import filedialog
    root = tkinter.Tk()
    root.withdraw()
    return filedialog.askdirectory()

def dataset_path(new_dir:bool=False, is_notebook:bool=False):
    print(f"NOTEBOOK: {is_notebook}!")
    txt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            r"casting_dataset_path.txt"
        )
    if os.path.isfile(txt_path) and not new_dir:
        with open(txt_path, "r") as text_file:
            path = text_file.readlines()
    else:
        path = path_dialog()
        with open(txt_path, "w") as text_file:
            text_file.write(path)
    return path

def transform_train(vit=False, mean=torch.tensor([0.3460, 0.4832, 0.7033]), std=torch.tensor([1.0419, 1.0652, 1.0605])):
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomAffine(
            degrees=45,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            hue=0.2
        ),
        T.ToTensor(),
        T.Resize(224) if vit else torch.nn.Identity,
        T.Normalize(
            mean=mean,
            std=std
        )
    ])

def transform_test(vit=False, mean=torch.tensor([0.3460, 0.4832, 0.7033]), std=torch.tensor([1.0419, 1.0652, 1.0605])):
    return T.Compose([
        T.ToTensor(),
        T.Resize(224) if vit else torch.nn.Identity,
        T.Normalize(
            mean=mean,
            std=std
        )
    ])

def get_train_data(rand:bool=True, new_dir:bool=False, vit:bool=False):
    img_fldr = torchvision.datasets.ImageFolder(
        root=os.path.join(dataset_path(new_dir=new_dir), "train"),
        transform=transform_train(vit=vit)
    )
    if rand:
        img_fldr.samples = sorted(img_fldr.samples, key=lambda k:random.random())
    return img_fldr


def get_test_data(rand:bool=True, new_dir:bool=False, vit:bool=False):
    img_fldr = torchvision.datasets.ImageFolder(
        root=os.path.join(dataset_path(new_dir=new_dir), "test"),
        transform=transform_test(vit=vit)
    )
    if rand:
        img_fldr.samples = sorted(img_fldr.samples, key=lambda k:random.random())
    return img_fldr