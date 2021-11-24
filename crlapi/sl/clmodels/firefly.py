# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import copy
import time
from pydoc import locate
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from crlapi.core import CLModel
from crlapi.sl.clmodels.core import SupervisedCLModel
from crlapi.sl.architectures.firefly_vgg import sp
from crlapi.sl.architectures.firefly_vgg.models import sp_vgg

def _state_dict(model, device):
    sd = model.state_dict()
    for k, v in sd.items():
        sd[k] = v.to(device)
    return sd

# --- Firefly Implementation. Since we do not plan to extend this method,
# --- everything is self-contained here.

class Firefly(SupervisedCLModel):
    def __init__(self, stream, clmodel_args):
        super().__init__()
        self.models = []
        self.config = clmodel_args
        self.verbose = True

    def build_initial_net(self, task, **model_args):
        # only support the custom VGG backbone for now
        model, next_layers, layers_to_split = \
            sp_vgg('vgg19',
                   n_classes=task.n_classes,
                   dimh=model_args['n_channels'],
                   method='fireflyn')

        # Hacky AF
        model.next_layers = next_layers
        model.layers_to_split = layers_to_split

        return model

    def get_prediction_net(self,task):
        if task.task_descriptor() is None:
            model = self.models[-1]
        else:
            model = self.models[task.task_descriptor]

        model.eval()
        return model

    def update(self, task, logger):
        assert isinstance(task.task_descriptor(),int)

        logger.message("Building training dataset")
        training_dataset = task.task_resources().make()

        # Creating datasets and loaders
        training_loader,validation_loader = self.get_train_and_validation_loaders(training_dataset)


        if len(self.models)==0:
            model_args=self.config.model
            model=self.build_initial_net(task,**model_args)

        elif (task.task_descriptor() % self.config.grow_every) == 0:
            model=copy.deepcopy(self.models[task.task_descriptor()-1])
            print('growing')
            base_gr = self.config.model.grow_ratio
            grow_ratio = (base_gr * task.task_descriptor() + 1) / (base_gr * (task.task_descriptor() - 1) + 1) - 1
            n_pre  = sum(np.prod(x.shape) for x in model.parameters())
            added  = self.split(model, training_loader, grow_ratio)
            n_post = sum(np.prod(x.shape) for x in model.parameters())
            assert n_post > n_pre
            print(f'from {n_pre} to {n_post}')
        else:
            model=copy.deepcopy(self.models[task.task_descriptor()-1])

        flops_per_input = self.count_flops(task, model)
        best_model=copy.deepcopy(model)
        best_loss, best_acc = None, None

        # Optionally create GPU training augmentations
        train_aug = self.get_train_augs()

        # Optinally use patience :)
        patience = self.config.patience
        patience_delta = self.config.patience_delta
        patience_count = 0

        device=self.config["device"]
        model.to(device)
        optimizer = self.get_optimizer(model.parameters())

        #Launching training procedure
        logger.message("Start training for "+str(self.config["max_epochs"])+" epochs")
        iteration, n_fwd_samples = 0, 0
        for epoch in range(self.config["max_epochs"]):

            # Make sure model is ready for train
            model.train()

            #Training loop
            for i, (raw_x, y) in enumerate(training_loader):
                raw_x, y = raw_x.to(device), y.to(device)

                # apply transformations
                x = train_aug(raw_x)

                predicted=model(x)
                loss=F.cross_entropy(predicted,y)
                nb_ok=predicted.max(1)[1].eq(y).float().sum().item()
                accuracy=nb_ok/x.size()[0]
                logger.add_scalar("train/loss",loss.item(),iteration)
                logger.add_scalar("train/accuracy",accuracy,iteration)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iteration += 1
                n_fwd_samples += x.size(0)

            #Validation
            out=self._validation_loop(model,device,validation_loader)
            validation_loss,validation_accuracy=out["loss"],out["accuracy"]

            logger.add_scalar("validation/loss",validation_loss,epoch)
            logger.add_scalar("validation/accuracy",validation_accuracy,epoch)

            # Right now CV against accuracy
            # if best_loss is None or validation_loss < (best_loss - patience_delta):
            if best_acc is None or validation_accuracy > (best_acc + patience_delta):
                print("\tFound best model at epoch ",epoch)
                best_model.load_state_dict(_state_dict(model,"cpu"))
                best_loss = validation_loss
                best_acc  = validation_accuracy
                patience_count = 0
            else:
                patience_count += 1

            logger.message(f"Validation Acc {validation_accuracy:.4f}\t Loss {validation_loss:.4}")

            if patience_count == patience:
                break

        self.models.append(best_model)
        logger.message("Training Done...")
        logger.add_scalar('train/model_params', sum([np.prod(x.shape) for x in model.parameters()]), 0)
        logger.add_scalar('train/one_sample_megaflop', flops_per_input / 1e6, 0)
        logger.add_scalar('train/total_megaflops', n_fwd_samples * flops_per_input / 1e6, 0)
        return self


    # --------------------------------------------
    # Firefly specific methods (from Classifier)
    # --------------------------------------------
    def spffn_forward(self, net, x, alpha):
        for layer in net:
            #if isinstance(layer, sp.SpModule) and layer.can_split:
            prev_x = x.cpu().data.numpy()
            if isinstance(layer, sp.SpModule):
                x = layer.spffn_forward(x, alpha=alpha)
            else:
                x = layer(x)

        return x.view(x.shape[0], -1)

    def spffn_loss_fn(self, net, x, y, alpha=-1):
        scores = self.spffn_forward(net, x, alpha=alpha)
        loss = F.cross_entropy(scores, y)

        return loss

    ## -- firefly new split -- ##
    def spffn(self, net, loader, n_batches):
        v_params = []
        for i, layer in enumerate(net):
            if isinstance(layer, sp.SpModule):
                enlarge_in = (i > 0)
                enlarge_out = (i < len(net)-1)
                net[i].spffn_add_new(enlarge_in=enlarge_in, enlarge_out=enlarge_out)
                net[i].spffn_reset()
                if layer.can_split:
                    v_params += [net[i].v]
                if enlarge_in:
                    v_params += [net[i].vni]
                if enlarge_out:
                    v_params += [net[i].vno]

        opt_v = torch.optim.RMSprop(nn.ParameterList(v_params), lr=1e-3, momentum=0.1, alpha=0.9)

        self.device = next(iter(net.parameters())).device

        torch.cuda.empty_cache()

        n_batches = 0
        for i, (x, y) in enumerate(loader):
            n_batches += 1

            x, y = x.to(self.device), y.to(self.device)
            loss = self.spffn_loss_fn(net, x, y)
            opt_v.zero_grad()
            loss.backward()
            for layer in net:
                if isinstance(layer, sp.SpModule):
                    layer.spffn_penalty()
            opt_v.step()

        self.config.model.granularity = 1
        alphas = np.linspace(0, 1, self.config.model.granularity*2+1)
        for alpha in alphas[1::2]:
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = self.spffn_loss_fn(net, x, y, alpha=1.0)
                opt_v.zero_grad()
                loss.backward()
                # for i in self.layers_to_split:
                for i in net.layers_to_split:
                    net[i].spffn_update_w(self.config.model.granularity * n_batches, output = False)

    # --------------------------------------------
    # Firefly specific methods (from SpNet)
    # --------------------------------------------

    def clear(self, net):
        for layer in net:
            if isinstance(layer, sp.SpModule):
                layer.clear()

    def get_num_elites(self, net, grow_ratio):
        n = 0
        # for i in self.layers_to_split:
        for i in net.layers_to_split:
            n += net[i].module.weight.shape[0]
        self.n_elites = int(n * grow_ratio)

    def get_num_elites_group(self, net, group_num, grow_ratio):
        for g in range(group_num):
            n = 0
            for i in self.layers_to_split_group[g]:
                n += net[i].module.weight.shape[0]
            try:
                self.n_elites_group[g] = int(n * grow_ratio)
            except:
                self.n_elites_group = {}
                self.n_elites_group[g] = int(n * grow_ratio)

    def sp_threshold(self, net):
        # ws, wi = torch.sort(torch.cat([net[i].w for i in self.layers_to_split]).reshape(-1))
        ws, wi = torch.sort(torch.cat([net[i].w for i in net.layers_to_split]).reshape(-1))
        total= ws.shape[0]
        threshold = ws[self.n_elites]
        return threshold

    def sp_threshold_group(self, net, group_num):
        # ws, wi = torch.sort(torch.cat([net[i].w for i in self.layers_to_split_group[group_num]]).reshape(-1))
        ws, wi = torch.sort(torch.cat([net[i].w for i in net.layers_to_split_group[group_num]]).reshape(-1))
        total= ws.shape[0]
        threshold = ws[self.n_elites_group[group_num]]
        return threshold


    def split(self, net, loader, grow_ratio, n_batches=-1, split_method='fireflyn'):

        self.num_group = 1# if self.config.backbne != 'mobile' else 2
        if split_method not in ['random', 'exact', 'fast', 'firefly', 'fireflyn']:
            raise NotImplementedError

        if self.verbose:
            print('[INFO] start splitting ...')

        start_time = time.time()
        net.eval()

        if self.num_group == 1:
            self.get_num_elites(net, grow_ratio)
        else:
            self.get_num_elites_group(net, grow_ratio, self.num_group)
        split_fn = {
            #'exact': self.spe,
            #'fast': self.spf,
            #'firefly': self.spff,
            'fireflyn': self.spffn,
        }

        if split_method != 'random':
            split_fn[split_method](net, loader, n_batches)

        n_neurons_added = {}

        if split_method == 'random':
            # n_layers = len(self.layers_to_split)
            n_layers = len(net.layers_to_split)
            n_total_neurons = 0
            threshold = 0.
            # for l in self.layers_to_split:
            for l in net.layers_to_split:
                n_total_neurons += net[l].get_n_neurons()
            n_grow = int(n_total_neurons * grow_ratio)
            n_new1 = np.random.choice(n_grow, n_layers, replace=False)
            n_new1 = np.sort(n_new1)
            n_news = []
            for i in range(len(n_new1) - 1):
                if i == 0:
                    n_news.append(n_new1[i])
                    n_news.append(n_new1[i + 1] - n_new1[i])
                else:
                    n_news.append(n_new1[i + 1] - n_new1[i])
                    n_news[-1] += 1
            # for i, n_new_ in zip(reversed(self.layers_to_split), n_news):
            for i, n_new_ in zip(reversed(net.layers_to_split), n_news):
                if isinstance(net[i], sp.SpModule) and net[i].can_split:
                    n_new, idx = net[i].random_split(n_new_)
                    n_neurons_added[i] = n_new
                    if n_new > 0: # we have indeed splitted this layer
                        # for j in self.next_layers[i]:
                        for j in net.next_layers[i]:
                            net[j].passive_split(idx)
        elif split_method == 'fireflyn':
            if self.num_group == 1:
                threshold = self.sp_threshold(net)
            # for i in reversed(self.layers_to_split):
            for i in reversed(net.layers_to_split):
                if isinstance(net[i], sp.SpModule) and net[i].can_split:
                    if self.num_group != 1:
                        group = self.total_group[i]
                        threshold = self.sp_threshold_group(net, group)
                    n_new, split_idx, new_idx = net[i].spffn_active_grow(threshold)
                    sp_new = split_idx.shape[0] if split_idx is not None else 0
                    n_neurons_added[i] = (sp_new, n_new-sp_new)
                    if net[i].kh == 1:
                        isfirst = True
                    else:
                        isfirst = False
                    # for j in self.next_layers[i]:
                    for j in net.next_layers[i]:
                        print('passive', net[j].module.weight.shape)
                        net[j].spffn_passive_grow(split_idx, new_idx)

        else:
            threshold= self.sp_threshold()
            # actual splitting
            # for i in reversed(self.layers_to_split):
            for i in reversed(net.layers_to_split):
                if isinstance(net[i], sp.SpModule) and net[i].can_split:
                    n_new, idx = net[i].active_split(threshold)
                    n_neurons_added[i] = n_new
                    if n_new > 0: # we have indeed splitted this layer
                        # for j in self.next_layers[i]:
                        for j in net.next_layers[i]:
                            net[j].passive_split(idx)

        net.train()
        self.clear(net) # cleanup auxiliaries

        end_time = time.time()
        if self.verbose:
            print('[INFO] splitting takes %10.4f sec. Threshold value is %10.9f' % (
                end_time - start_time, threshold))
            if split_method == 'fireflyn':
                print('[INFO] number of added neurons: \n%s\n' % \
                        '\n'.join(['-- %d grows (sp %d | new %d)' % (x, y1, y2) for x, (y1, y2) in n_neurons_added.items()]))
            else:
                print('[INFO] number of added neurons: \n%s\n' % \
                        '\n'.join(['-- %d grows %d neurons' % (x, y) for x, y in n_neurons_added.items()]))
        return n_neurons_added
