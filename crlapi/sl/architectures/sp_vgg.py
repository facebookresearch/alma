import math
import copy
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

def percentile(t, qq):
    k = int(qq * t.numel()) #1 + round(float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()

class GetSubnet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, zeros, ones, sparsity):
        k_val = percentile(scores, sparsity)
        return torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


# Not learning weights, finding subnet
class SubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        self.register_buffer('ones', torch.ones_like(self.scores.data))
        self.register_buffer('zeros', torch.zeros_like(self.scores.data))

        # hardcoded for now
        self.prune_rate = 0.5

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        return self.scores.abs()

    def forward(self, x):
        subnet = GetSubnet.apply(self.clamped_scores, self.zeros, self.ones, self.prune_rate)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


# -- Layers

def _make_layers(array, in_channels):
    layers = []
    for x in array:

        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [SubnetConv(in_channels, x, kernel_size=3, padding=1),
                       nn.BatchNorm2d(x, affine=False),
                       nn.ReLU(inplace=True)]
            in_channels = x

    return in_channels, layers


class SubnetVGG(nn.Module):
    def __init__(self, task, n_channels, grow_n_units):
        super().__init__()

        self.grow_n_units = grow_n_units
        vgg_parts = [ 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M' ]

        if n_channels > 0:
            vgg_parts = [n_channels if type(x) == int else x for x in vgg_parts]

        out_channels, base = _make_layers(vgg_parts, 3)
        self.net = nn.Sequential(
                        *base,
                        SubnetConv(out_channels, task.n_classes, kernel_size=1, padding=0),
                        nn.Flatten(),
        )


    def forward(self, x):
        return self.net(x)


    def grow(self, valid_loader, **args):

        x = torch.FloatTensor(64, 3, 32, 32).normal_()
        new_layers = []
        for i, layer in enumerate(self.net):
            if isinstance(layer, SubnetConv):
                # input size
                in_c = 3 if i == 0 else last_output_channels

                # output size
                out_c = layer.out_channels + (self.grow_n_units if i < len(self.net) - 2 else 0)

                # what is the minimal score to be selected ?
                max_val = percentile(layer.scores.abs(), layer.prune_rate)
                min_val = layer.scores.abs().min().item()
                # init new layer
                new_layer = SubnetConv(in_c, out_c, kernel_size=layer.kernel_size, padding=layer.padding)
                new_layer.scores.data.uniform_(min_val, max_val)

                # adjust the prune rate so that the same amount of points get selected
                new_layer.prune_rate = 1 - (1 - layer.prune_rate) * layer.weight.numel() / new_layer.weight.numel()

                # copy the old params
                a, b, c, d = layer.scores.size()
                new_layer.weight[:a, :b, :c, :d].data.copy_(layer.weight.data)
                new_layer.scores[:a, :b, :c, :d].data.copy_(layer.scores.data)
                new_layer.bias.data.fill_(0)
                new_layer.bias[:a].data.copy_(layer.bias)
                last_output_channels = out_c
                new_layers += [new_layer]

                new_sub = torch.where(new_layer.clamped_scores < percentile(new_layer.clamped_scores, new_layer.prune_rate), new_layer.zeros, new_layer.ones)
                import pdb
                # assert torch.allclose(layer(x[:, :b]), new_layer(x)[:, :a]), pdb.set_trace()

            elif isinstance(layer, nn.BatchNorm2d):
                new_bn = nn.BatchNorm2d(last_output_channels, affine=False)
                c = layer.running_mean.size(0)
                new_bn.running_mean[:c].data.copy_(layer.running_mean.data)
                new_bn.running_var[:c].data.copy_(layer.running_var.data)
                new_layers += [new_bn]

                new_bn.training = layer.training

                # assert torch.allclose(layer(x[:, :c]), new_bn(x)[:, :c], atol=1e-7)
            else:
                new_layers += [copy.deepcopy(layer)]

            x = new_layers[-1](x)

        net  =  nn.Sequential(*new_layers)

        copy_self = copy.deepcopy(self)
        copy_self.net = net
        print(net)

        return copy_self
