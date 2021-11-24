# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import numpy as np

from copy import deepcopy
from crlapi.sl.architectures.mixture_model import (
    HardSoftMaxGateModule,
    SoftMaxGateModule,
    MixtureLayer,
    MoE_RandomGrow,
    MoE_UsageGrow,
    Gate,
    MoE,
)

# -- Gates

class SoftGate(Gate):
    def __init__(self, input_shape, n_experts, prepro_fn=None):
        super().__init__(input_shape, n_experts)

        gate_fn = nn.Linear(input_shape, n_experts)
        if prepro_fn is not None:
            self.prepro_fn = prepro_fn
            gate_fn = nn.Sequential(prepro_fn, gate_fn)

        self.module = SoftMaxGateModule(gate_fn)

    def forward(self,x):
        return self.module(x)

class HardGate(Gate):
    def __init__(self, input_shape, n_experts, prepro_fn=None):
        super().__init__(input_shape, n_experts)

        gate_fn = nn.Linear(input_shape, n_experts)
        if prepro_fn is not None:
            gate_fn = nn.Sequential(prepro_fn, gate_fn)

        self.module = HardSoftMaxGateModule(gate_fn)

    def forward(self,x):
        return self.module(x)


# -- Layers

def _make_layers(array, in_channels):
    layers = []
    for x in array:

        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                       nn.BatchNorm2d(x),
                       nn.ReLU(inplace=True)]
            in_channels = x

    return in_channels, nn.Sequential(*layers)


def VGG(task, n_channels):
    vgg_parts = [ [64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 256, 'M'], [512, 512, 512, 512, 'M'], [512, 512, 512, 512, 'M'] ]

    if n_channels > 0:
        vgg_parts = [[n_channels if type(x) == int else x for x in block] for block in vgg_parts]

    in_channels, block0   = _make_layers(vgg_parts[0], 3)
    in_channels, block1 = _make_layers(vgg_parts[1], in_channels)
    in_channels, block2 = _make_layers(vgg_parts[2], in_channels)
    in_channels, block3 = _make_layers(vgg_parts[3], in_channels)
    in_channels, block4 = _make_layers(vgg_parts[4], in_channels)

    return nn.Sequential(
        block0,
        block1,
        block2,
        block3,
        block4,
        nn.Flatten(),
        nn.Linear(in_channels, task.n_classes)
    )


def MoE_VGG(task, n_channels, n_adaptivepooling, n_experts, is_hard):
    vgg_parts = [ [64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 256, 'M'], [512, 512, 512, 512, 'M'], [512, 512, 512, 512, 'M'] ]

    input_shape = task.input_shape
    gate = HardGate if is_hard else SoftGate

    if n_channels > 0:
        vgg_parts = [[n_channels if type(x) == int else x for x in block] for block in vgg_parts]

    in_channels, head   = _make_layers(vgg_parts[0], 3)
    in_channels, block1 = _make_layers(vgg_parts[1], in_channels)
    in_channels, block2 = _make_layers(vgg_parts[2], in_channels)
    in_channels, block3 = _make_layers(vgg_parts[3], in_channels)
    in_channels, block4 = _make_layers(vgg_parts[4], in_channels)
    blocks  = [block1, block2, block3, block4]

    dim_gates = []
    x = torch.randn(1,3,32,32)
    for layer in [head] + blocks:
        x = layer(x)
        dim_gates += [x.shape[1:]]

    # Build Layers
    layers = [head]

    for i, (block, dim_gate) in enumerate(zip(blocks, dim_gates[:-1])):
        # build adaptive pooling gate
        input_size = dim_gate[0] * n_adaptivepooling ** 2
        gate_fn = nn.Sequential(
                nn.AdaptiveAvgPool2d(n_adaptivepooling),
                nn.Flatten(),
        )
        experts = [deepcopy(block) for _ in range(n_experts)]
        layers += [MixtureLayer(gate(input_size, n_experts, gate_fn), experts)]

    linear  = nn.Linear(np.prod(dim_gates[-1]), task.n_classes)
    layers += nn.Sequential(nn.Flatten(), linear)

    model = MoE(layers)
    return model


def MoE_VGG_RandomGrow(task, n_channels, n_adaptivepooling, n_experts, is_hard, n_experts_to_split):
    moe = MoE_VGG(task, n_channels, n_adaptivepooling, n_experts, is_hard)
    return MoE_RandomGrow(moe.layers,n_experts_to_split)


def MoE_VGG_UsageGrow(task, n_channels, n_adaptivepooling, n_experts, is_hard, n_experts_to_split):
    moe = MoE_VGG(task, n_channels, n_adaptivepooling, n_experts, is_hard)
    return MoE_UsageGrow(moe.layers,n_experts_to_split)
