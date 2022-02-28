# Machine Learning for Zero Defect Manufacturing Platform

## Installation

Download this [dataset](https://www.kaggle.com/ravirajsinh45/real-life-industrial-dataset-of-casting-product) from [kaggle](https://www.kaggle.com).


## More details

## Requirements

* torch
* timm

## Usage

Use one of the following codes to load the pretrained weights depending on the model to be used.

### [LUNA](zdmp/utils/luna_model.py)
```python
model = Luna()
state_dict = torch.hub.load_state_dict_from_url("https://drive.google.com/uc?export=download&id=18dUsc1hP8ouSdc23BTu5BUiFgfrXr0-Z&confirm=t")
model.load_state_dict(state_dict)
```

### ALEXNET
```python
model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet')
model.classifier[6] = torch.nn.Linear(4096, 2) 
state_dict = torch.hub.load_state_dict_from_url("https://drive.google.com/uc?export=download&id=1WVbXJxI1so6TAh7sM4GXIzNQl-tWctwC&confirm=t")
model.load_state_dict(state_dict)
```

### RESNET18
```python
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')
model.fc = torch.nn.Linear(512, 2)
state_dict = torch.hub.load_state_dict_from_url("https://drive.google.com/uc?export=download&id=1nxGbmtyBDKlCOTOYLNz_8-GmFbJ9ML6V&confirm=t")
model.load_state_dict(state_dict)
```

### EFFICIENTNETV2
```python
model = timm.models.efficientnetv2_rw_s()
model.classifier = torch.nn.Linear(1792, 2)
state_dict = torch.hub.load_state_dict_from_url("https://drive.google.com/uc?export=download&id=1--kslrtZ8ARe1o6EE7EOlcscPSgZTJ55&confirm=t")
model.load_state_dict(state_dict)
```

### CONVNEXT
```python
model = timm.models.convnext.convnext_tiny()
model.head.fc = torch.nn.Linear(768, 2)
state_dict = torch.hub.load_state_dict_from_url("https://drive.google.com/uc?export=download&id=1eTbkcFYrxbSxJ4Jj0BpH1_mbLqJQ0LmP&confirm=t")
model.load_state_dict(state_dict)
```

### ViT
```python
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
state_dict = torch.hub.load_state_dict_from_url("https://drive.google.com/uc?export=download&id=1-7kjESSsw5wKtHUnNx8C1q2lpPbL45jj&confirm=t")
model.load_state_dict(state_dict)
```

### SWIN-T
``` python
model = timm.models.swin_tiny_patch4_window7_224()
model.head = torch.nn.Linear(768, 2)
state_dict = torch.hub.load_state_dict_from_url("https://drive.google.com/uc?export=download&id=1ttiqssbh-nyjCSFQxhsRvG0S66SkrBs6&confirm=t")
model.load_state_dict(state_dict)
```
## Examples

The following notebooks show how the models where trained and how to use them with the casting dataset. 

* [Models with CNN](notebooks/luna_example.ipynb)
* [Transformer models](notebooks/vit_example.ipynb)

This notebook shows the comparisson between the models

* [Comparisson](notebooks/comparisson.ipynb)

## Author
Matias Vasquez (e11742193@student.tuwien.ac.at)

## References

