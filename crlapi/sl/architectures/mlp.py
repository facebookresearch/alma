# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from crlapi.sl.architectures.mixture_model import MixtureLayer,SoftMaxGateModule,HardSoftMaxGateModule,Gate,MoE,MoE_RandomGrow,MoE_UsageGrow

class MLP(nn.Module):
    def __init__(self,task,**args):
        super().__init__()
        input_shape=task.input_shape
        d=1
        for k in input_shape:
            d*=k
        input_dim=d
        output_dim=task.n_classes
        sizes=[input_dim]+[args["size_layers"] for k in range(args["n_layers"])]+[output_dim]
        print(sizes)

        layers=[]
        for k in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[k],sizes[k+1]))
            if not k==len(sizes)-2:
                layers.append(nn.ReLU())
        self.model=nn.Sequential(*layers)

    def forward(self,x):
        x=torch.flatten(x,start_dim=1)
        return self.model(x)


class LinearSoftGate(Gate):
    def __init__(self,input_shape, n_experts, prepro_fn=None):
        super().__init__(input_shape,n_experts)
        assert len(input_shape)==1
        self.module=SoftMaxGateModule(nn.Linear(input_shape[0],n_experts))

    def forward(self,x):
        return self.module(x)

class LinearHardGate(Gate):
    def __init__(self,input_shape, n_experts, prepro_fn=None):
        super().__init__(input_shape,n_experts)
        assert len(input_shape)==1
        self.module=HardSoftMaxGateModule(nn.Linear(input_shape[0],n_experts))

    def forward(self,x):
        return self.module(x)

def mlp_layers(task,size_layers,n_layers,n_experts,is_hard):
        input_shape=task.input_shape
        d=1
        for k in input_shape:
            d*=k
        input_dim=d
        output_dim=task.n_classes

        sizes=[input_dim]+[size_layers for k in range(n_layers)]+[output_dim]

        layers=[nn.Flatten(start_dim=1)]
        for k in range(len(sizes)-2):
            if is_hard:
                gate=LinearHardGate([sizes[k]],n_experts)
            else:
                gate=LinearSoftGate([sizes[k]],n_experts)
            experts=[nn.Sequential(nn.Linear(sizes[k],sizes[k+1]),nn.ReLU()) for _ in range(n_experts)]
            layer=MixtureLayer(gate,experts)
            layers.append(layer)

        layers.append(nn.Linear(sizes[-2],sizes[-1]))
        return layers

def MoE_MLP(task,size_layers,n_layers,n_experts,is_hard):
        return MoE(mlp_layers(task,size_layers,n_layers,n_experts,is_hard))

def MoE_MLP_RandomGrow(task,size_layers,n_layers,n_experts,is_hard,n_experts_to_split):
        return MoE_RandomGrow(mlp_layers(task,size_layers,n_layers,n_experts,is_hard),n_experts_to_split)

def MoE_MLP_UsageGrow(task,size_layers,n_layers,n_experts,is_hard,n_experts_to_split):
        return MoE_UsageGrow(mlp_layers(task,size_layers,n_layers,n_experts,is_hard),n_experts_to_split)
