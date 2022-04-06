# Machine Learning for Zero Defect Manufacturing Platform

## Installation

Download this [dataset](https://www.kaggle.com/ravirajsinh45/real-life-industrial-dataset-of-casting-product) from [kaggle](https://www.kaggle.com).


## More details

The thesis is being written with LaTex, using overleaf. 
* The latest version can be visited through this [link](https://www.overleaf.com/read/bdkddjsdwmwg).

The topics that are done or almost done are:
* Dataset
* Luna Model
* Transformer (NLP)
* Vision transformer

Some of the other topics have been started just to have a guide of what comes in that position.




## Requirements

* torch 1.11.0+cu102
* torchvision 0.12.0+cu102
* timm 0.5.4
* pytorch-lighning 1.6.0

### Install requirements through pip

```python
pip install --upgrade torch
pip install --upgrade torchtext     # This needs to be updated for pytorch-lightning to work properly
pip install --upgrade torchvision
pip install pytorch-lightning
pip install timm
```

## Usage

Use one of the following codes to load the pretrained weights depending on the model to be used.

### [LUNA](zdmp/utils/luna_model.py)
```python
luna_model = torch.utils.model_zoo.load_url("https://drive.google.com/uc?export=download&id=1-4k-6UBn4kH1ueanXB39Z375FaRmYCYS&confirm=t")
```

### ALEXNET
```python
alexnet_model = torch.utils.model_zoo.load_url("https://drive.google.com/uc?export=download&id=1TjgEAfjbQ9NTzF0Z0S4SiEF9uvmRYO_3&confirm=t")
```

### RESNET18
```python
resnet18_model = torch.utils.model_zoo.load_url("https://drive.google.com/uc?export=download&id=1-Dh-hAa6OIQaU6PMlNXmy9NCAWLrgY7U&confirm=t")
```

### EFFICIENTNETV2
```python
effnetv2_model = torch.utils.model_zoo.load_url("https://drive.google.com/uc?export=download&id=1-EUXk4XO-4kqFlv_MXw0LMDHxTDwgSNg&confirm=t")
```

### CONVNEXT
```python
convnext_model = torch.utils.model_zoo.load_url("https://drive.google.com/uc?export=download&id=1-ExFlJq02ormAJrslXOR2cAuLth3Td2O&confirm=t")
```

### ViT
```python
vit_model = torch.utils.model_zoo.load_url("https://drive.google.com/uc?export=download&id=1CLAj3L9iz8y9saojWLzjPwmjW33MLuSY&confirm=t")
```

### SWIN-T
``` python
swin_t_model = torch.utils.model_zoo.load_url("https://drive.google.com/uc?export=download&id=1-4MKJzlIWcWQeb5D8Ly7aJhfk29fz9-c&confirm=t")
```
## Examples

The following notebooks show how the models where trained and how to use them with the casting dataset. 

* [Training the models](notebooks/train_models_example.ipynb)
* [Loading the models](notebooks/load_models_example.ipynb)

This notebook shows the comparisson between the models

* [Comparisson](notebooks/comparisson.ipynb)

## Author
Matias Vasquez (e11742193@student.tuwien.ac.at)

## References

