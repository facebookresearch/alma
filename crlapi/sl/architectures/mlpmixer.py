# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
import torch.nn as nn
from crlapi.sl.architectures.mixture_model import MixtureLayer,SoftMaxGateModule,HardSoftMaxGateModule,Gate,MoE,MoE_RandomGrow,MoE_UsageGrow

class LinearSoftGate(Gate):
    def __init__(self,input_shape, n_experts, prepro_fn=None):
        super().__init__(input_shape,n_experts)
        assert len(input_shape)==1
        self.module=SoftMaxGateModule(nn.Linear(input_shape[0],n_experts))

    def forward(self,x):
        print(x.size())
        return self.module(x)

class LinearHardGate(Gate):
    def __init__(self,input_shape, n_experts, prepro_fn=None):
        super().__init__(input_shape,n_experts)
        assert len(input_shape)==1
        self.module=HardSoftMaxGateModule(nn.Linear(input_shape[0],n_experts))

    def forward(self,x):
        return self.module(x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        PrintModule("After dense"),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

class PrintModule(nn.Module):
    def __init__(self,msg=""):
        super().__init__()
        self.msg=msg

    def forward(self,x):
        print(self.msg," : ",x.size())
        return x


def MLPMixer(task,  patch_size, dim, depth, expansion_factor = 4, dropout = 0.):
    image_size= task.input_shape[1]
    assert image_size==task.input_shape[2]
    channels=task.input_shape[0]
    num_classes=task.n_classes

    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_size // patch_size) ** 2
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        PrintModule("L1"),
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        PrintModule("L2"),
        nn.Linear((patch_size ** 2) * channels, dim),
        PrintModule("L3"),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last)),
            PrintModule("L."),
        ) for _ in range(depth)],
        PrintModule("L4"),
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )
