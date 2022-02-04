import os
import torch
import warnings
from torch import nn as nn
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(title=model.__class__.__name__, field_names=["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    return table, total_params

def Luna(pretrained=True):
    model = LunaModel()
    weights_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..",
                    r"pretrained_weights\Luna.pth.tar"
                )
    if pretrained:
        if os.path.isdir(weights_path):
            model.load_state_dict(torch.load(weights_path))
        else:
            warnings.warn(f"""\nCould not find pretrained weights in following directory: 

{weights_path}

The model was created without pretrained weights.

""")
    model.eval()
    return model

class LunaModel(nn.Module):
    def __init__(self, in_channels=3, conv_channels=8):
        super(LunaModel, self).__init__()

        self.tail_batchnorm = nn.BatchNorm2d(in_channels)

        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)

        self.head_linear = nn.Linear(20736, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self.linear_output = None

    def get_linear_output(self):
        assert self.linear_output != None, "Linear output not defined!"
        return self.linear_output

    def forward(self, input_batch):
        bn_output = self.tail_batchnorm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)

        conv_flat = block_out.view(
            block_out.size(0),
            -1,
        )
        linear_output = self.head_linear(conv_flat)
        self.linear_output = linear_output

        #return linear_output, self.head_softmax(linear_output)
        return self.head_softmax(linear_output)



class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super(LunaBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, conv_channels, kernel_size=3, padding=1, bias=True,
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            conv_channels, conv_channels, kernel_size=3, padding=1, bias=True,
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.maxpool(block_out)