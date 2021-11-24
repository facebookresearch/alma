import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

import time
import itertools
from copy import deepcopy
from pydoc import locate
from fvcore.nn import FlopCountAnalysis as FCA

from torchvision.models import *

from . import sp

# ----------------------------------------------------------------
#                       Models
# ----------------------------------------------------------------

class module_list_wrapper(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer(x)
        return out

    def __getitem__(self, i):
        return self.layer[i]

    def __len__(self):
        return len(self.layer)


def sp_vgg(model, n_classes=10, dimh=16, method='none'):
    cfgs = {
        'vgg11': [1, 'M', 2, 'M', 4, 4, 'M', 8, 8, 'M', 8, 8, 'M'],
        'vgg14': [1, 1, 'M', 2, 2, 'M', 4, 4, 'M', 8, 8, 'M', 8, 8, 'M'],
        'vgg16': [1, 1, 'M', 2, 2, 'M', 4, 4, 4, 'M', 8, 8, 8, 'M', 8, 8, 8, 'M'],
        'vgg19': [1, 1, 'M', 2, 2, 'M', 4, 4, 4, 4, 'M', 8, 8, 8, 8, 'M', 8, 8, 8, 8, 'M'],
    }
    cfg = cfgs[model]
    next_layers = {}
    prev_idx = -1
    in_channels = 3
    net = []
    n = len(cfg)
    for i, x in enumerate(cfg):
        if x == 'M':
            net.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif x == 'A':
            net.append(nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            if method == 'none':
                net.append(sp.Conv2d(in_channels, 64*x, kernel_size=3, padding=1, actv_fn='relu', has_bn=True))
                in_channels = 64*x
            else:
                net.append(sp.Conv2d(in_channels, dimh, kernel_size=3, padding=1, actv_fn='relu', has_bn=True))
                in_channels = dimh
            if prev_idx >= 0: next_layers[prev_idx] = [i]
            prev_idx = i
    net.append(sp.Conv2d(in_channels, n_classes, kernel_size=1, padding=0, actv_fn='none', can_split=False))
    net.append(nn.Flatten())

    net = module_list_wrapper(net)
    old_fwd = net.forward

    next_layers[prev_idx] = [n]
    layer2split = list(next_layers.keys())


    return net, next_layers, layer2split
