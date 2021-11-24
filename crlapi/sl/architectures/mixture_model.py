import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import numpy as np

class SoftMaxGateModule(nn.Module):
    def __init__(self,module):
        super().__init__()
        self.module=module

    def forward(self,x):
        y=self.module(x)
        return torch.softmax(y,dim=1)

class HardSoftMaxGateModule(nn.Module):
    def __init__(self,module):
        super().__init__()
        self.module=module

    def forward(self,x):
        y=self.module(x)
        if self.training:
            dist=torch.distributions.Categorical(y)
            sampled_y=dist.sample()
            oh=F.one_hot(sampled_y,num_classes=p.size()[1])
            return oh+(y-y.detach())
        else:
            _max=y.max(1)[1]
            oh=F.one_hot(_max,num_classes=y.size()[1])
            return oh

class Gate(nn.Module):
    def __init__(self,input_shape,n_experts, prepro_fn=None):
        self.input_shape=input_shape
        super().__init__()


def _weight(output,score):
    s=output.size()
    while(len(score.size())<len(s)):
        score=score.unsqueeze(-1)
    score=score.repeat(1,*s[1:])
    return output*score

class MixtureLayer(nn.Module):
    def __init__(self,gate_module,experts):
        super().__init__()
        assert isinstance(gate_module,Gate)
        self.gate=gate_module
        self.experts=nn.ModuleList(experts)

    def forward(self,x):
        out=0.0
        scores=self.gate(x)
        gate_scores=[]
        for k,e in enumerate(self.experts):
            score=scores[:,k]
            if isinstance(e,MixtureLayer):
                y,g=e(x)
                for kk,vv in enumerate(g):
                    gate_scores.append(([k]+vv[0],vv[1]*score))
            else:
                y=e(x)
                gate_scores.append(([k],score))
            y=_weight(y,score)
            out=out+y

        return out,gate_scores

class MoE(nn.Module):
    def __init__(self,layers):
        super().__init__()
        self.layers=nn.ModuleList(layers)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self,x,with_gate_scores=False):
        gate_scores=[]
        for l in self.layers:
            if isinstance(l,MixtureLayer):
                x,g=l(x)
                gate_scores.append(g)
            else:
                x=l(x)

        if with_gate_scores:
            return x,gate_scores
        else:
            return x

class MoE_RandomGrow(MoE):
    def __init__(self,layers,n_experts_split):
        super().__init__(layers)
        self.n_experts_split=n_experts_split

    def _list_experts(self,layer):
        assert isinstance(layer,MixtureLayer)
        experts_url=[]
        for k,e in enumerate(layer.experts):
            if not isinstance(e,MixtureLayer):
                experts_url.append([k])
            else:
                le=self._list_experts(e)
                for v in le:
                    experts_url.append([k]+v)
        return experts_url


    def _generate_splitting(self,layer,url_to_split):
        idx_split=url_to_split[0]
        gate=copy.deepcopy(layer.gate)
        experts=[]
        for k,e in enumerate(layer.experts):
            if k!=idx_split:
                experts.append(copy.deepcopy(e))
            elif len(url_to_split)>1:
                experts.append(self._generate_splitting(e,url_to_split[1:]))
            else:
                n_experts=[copy.deepcopy(e) for _ in range(self.n_experts_split)]
                n_gate=layer.gate.__class__(layer.gate.input_shape, self.n_experts_split, getattr(layer.gate, 'prepro_fn', None))
                experts.append(MixtureLayer(n_gate,n_experts))

        return MixtureLayer(gate,experts)

    def _grow_layer(self,layer):
        assert isinstance(layer,MixtureLayer)

        #First, we list all the experts
        experts_urls=self._list_experts(layer)
        print("\tList of experts: ",experts_urls)
        #Choose one expert at random
        expert_to_split=random.choice(experts_urls)
        print("\t\tSplitting expert: "+str(expert_to_split))
        new_module=self._generate_splitting(layer,expert_to_split)
        experts_urls=self._list_experts(new_module)
        print("\t\tNew list of experts = ",experts_urls)
        return new_module

    def grow(self,dataset_loader,**args):
        if self.n_experts_split==0:
            return self

        self.zero_grad()
        new_layers=[]
        for l in self.layers:
            if isinstance(l,MixtureLayer):
                new_layers.append(self._grow_layer(l))
            else:
                new_layers.append(copy.deepcopy(l))
        return MoE_RandomGrow(new_layers,self.n_experts_split)

class MoE_UsageGrow(MoE):
    def __init__(self,layers,n_experts_split):
        super().__init__(layers)
        self.n_experts_split=n_experts_split

    def _list_experts(self,layer):
        assert isinstance(layer,MixtureLayer)
        experts_url=[]
        for k,e in enumerate(layer.experts):
            if not isinstance(e,MixtureLayer):
                experts_url.append([k])
            else:
                le=self._list_experts(e)
                for v in le:
                    experts_url.append([k]+v)
        return experts_url


    def _generate_splitting(self,layer,url_to_split):
        idx_split=url_to_split[0]
        gate=copy.deepcopy(layer.gate)
        experts=[]
        for k,e in enumerate(layer.experts):
            if k!=idx_split:
                experts.append(copy.deepcopy(e))
            elif len(url_to_split)>1:
                experts.append(self._generate_splitting(e,url_to_split[1:]))
            else:
                n_experts=[copy.deepcopy(e) for _ in range(self.n_experts_split)]
                n_gate=layer.gate.__class__(layer.gate.input_shape, self.n_experts_split, getattr(layer.gate, 'prepro_fn', None))
                experts.append(MixtureLayer(n_gate,n_experts))

        return MixtureLayer(gate,experts)

    def _grow_layer(self,layer,to_split_expert):
        assert isinstance(layer,MixtureLayer)

        #First, we list all the experts
        experts_urls=self._list_experts(layer)
        print("\tList of experts: ",experts_urls)
        print("\t To split: ",to_split_expert)
        assert to_split_expert in experts_urls
        new_module=self._generate_splitting(layer,to_split_expert)
        experts_urls=self._list_experts(new_module)
        print("\t\tNew list of experts = ",experts_urls)
        return new_module

    def grow(self,dataset_loader,**args):
        if self.n_experts_split==0:
            return self

        with torch.no_grad():
            usage=None
            n=0
            for x,y in dataset_loader:
                x, y = x.to(self.device), y.to(self.device)
                out,gate_scores=self(x,with_gate_scores=True)
                loss=F.cross_entropy(out,y,reduction='none')
                gate_scores=[[(gg[0],gg[1].sum(0)) for gg in g] for g in gate_scores]
                n+=x.size()[0]
                if usage is None:
                    usage=gate_scores
                else:
                    for k,g in enumerate(gate_scores):
                        for kk,gg in enumerate(g):
                            assert gg[0]==usage[k][kk][0]
                            usage[k][kk]=(gg[0],gg[1]+usage[k][kk][1])

        self.zero_grad()
        new_layers=[]
        p=0
        for k,l in enumerate(self.layers):
            if isinstance(l,MixtureLayer):
                u=usage[p]
                us=[uu[1].item() for uu in u]
                idx=np.argmax(us)
                print("Expert usage at layer ",k," is ",{str(uu[0]):uu[1].item() for uu in u})
                max_expert=u[idx][0]
                print("\tSplitting expert ",max_expert)
                new_layers.append(self._grow_layer(l,max_expert))
                p+=1
            else:
                new_layers.append(copy.deepcopy(l))
        return MoE_UsageGrow(new_layers,self.n_experts_split)
