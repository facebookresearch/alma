# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import numpy as np
from pydoc import locate
from random import shuffle
from crlapi.core import CLModel
from fvcore.nn import FlopCountAnalysis as FCA

import torch
import torch.nn as nn
import torch.nn.functional as F
from crlapi import instantiate_class,get_class,get_arguments
import torch.utils.data

class SupervisedCLModel(CLModel, nn.Module):
    """ A CLmodel based on a pytorch model, for supervised task over dataset

    Args:
        CLModel ([type]): [description]
    """
    def __init__(self):
        nn.Module.__init__(self)
        self.memory_training_set=None
        self.memory_validation_set=None

    def update(self, task, logger):
        raise NotImplementedError

    def get_prediction_net(self,task):
        raise NotImplementedError

    def count_flops(self, task, model=None):
        if model is None:
            model = self.get_prediction_model(task)

        # don't mess up BN stats!
        model = model.eval()

        input = torch.FloatTensor(size=(1, *task.input_shape)).to(self.config['device']).normal_()
        model = model.to(self.config['device'])
        flops = FCA(model, input).total()
        return flops

    def _validation_loop(self,net,device,dataloader):
        net = net.eval()
        net.to(device)

        with torch.no_grad():
            loss_values=[]
            nb_ok=0
            nb_total=0
            for x,y in dataloader:
                x,y=x.to(device),y.to(device)
                predicted=net(x)
                loss=F.cross_entropy(predicted,y)
                loss_values.append(loss.item())
                nb_ok+=predicted.max(1)[1].eq(y).float().sum().item()
                nb_total+=x.size()[0]

            loss=np.mean(loss_values)
            accuracy=nb_ok/nb_total

        net = net.train()
        return {"loss":loss,"accuracy":accuracy}

    def evaluate(self,task,logger,evaluation_args):
        logger.message("Evaluating...")

        evaluation_dataset = task.task_resources().make()

        #Building dataloader for both
        evaluation_loader = torch.utils.data.DataLoader(
            evaluation_dataset,
            batch_size=evaluation_args["batch_size"],
            num_workers=evaluation_args["num_workers"],
        )

        # TODO: is deepcopy here necessary ?
        evaluation_model=copy.deepcopy(self.get_prediction_net(task))

        evaluation_model.eval()

        device=evaluation_args["device"]
        evaluation_model.to(device)

        with torch.no_grad():
            loss_values=[]
            nb_ok=0
            nb_total=0
            for x,y in evaluation_loader:
                x,y=x.to(device),y.to(device)
                predicted=evaluation_model(x)
                loss=F.cross_entropy(predicted,y).item()
                nb_ok+=predicted.max(1)[1].eq(y).float().sum().item()
                nb_total+=x.size()[0]
                loss_values.append(loss)

            evaluation_loss=np.mean(loss_values)
            accuracy=nb_ok/nb_total
        r={"loss":evaluation_loss,"accuracy":accuracy}
        logger.debug(str(r))
        return r

    def build_initial_net(self,task,**model_args):
        from importlib import import_module

        classname=model_args["class_name"]
        del model_args["class_name"]
        module_path, class_name = classname.rsplit(".", 1)
        module = import_module(module_path)
        c = getattr(module, class_name)
        return c(task, **model_args)

    # -- Helpers
    def get_train_and_validation_loaders(self, dataset):
        val_size = int(len(dataset) * self.config.validation_proportion)
        tr_size  = len(dataset) - val_size
        training_dataset, validation_dataset = torch.utils.data.random_split(dataset, [tr_size, val_size])

        if self.config.train_replay_proportion>0.0:
            if not self.memory_training_set is None:
                l=int(len(self.memory_training_set)*self.config.train_replay_proportion)
                m,_= torch.utils.data.random_split(self.memory_training_set,[l,len(self.memory_training_set)-l])
                training_dataset=torch.utils.data.ConcatDataset([training_dataset,m])

        if self.config.validation_replay_proportion>0.0:
            if not self.memory_validation_set is None:
                l=int(len(self.memory_validation_set)*self.config.validation_replay_proportion)
                m,_= torch.utils.data.random_split(self.memory_validation_set,[l,len(self.memory_validation_set)-l])
                validation_dataset=torch.utils.data.ConcatDataset([validation_dataset,m])

        print("Training set size = ",len(training_dataset))
        print("Validation set size = ",len(validation_dataset))

        self.memory_training_set=training_dataset
        self.memory_validation_set=validation_dataset

        training_loader = torch.utils.data.DataLoader(
                training_dataset,
                batch_size=self.config.training_batch_size,
                num_workers=self.config.training_num_workers,
                persistent_workers=self.config.training_num_workers>0,
                shuffle=True,
                # pin_memory=self.config['device'] != 'cpu'
            )
        validation_loader = torch.utils.data.DataLoader(
                validation_dataset,
                batch_size=self.config.validation_batch_size,
                num_workers=self.config.validation_num_workers,
                persistent_workers=self.config.validation_num_workers>0,
                shuffle=False,
                # pin_memory=self.config['device'] != 'cpu'
            )

        return training_loader,validation_loader

    def get_optimizer(self, model_params):
        c=get_class(self.config.optim)
        args=get_arguments(self.config.optim)
        return c(model_params,**args)

    def get_train_augs(self):

        if self.config.get('kornia_augs', None) is not None:
            tfs = []
            import kornia

            for tf_cfg in self.config['kornia_augs']:
                tf = locate(f'kornia.augmentation.{tf_cfg.name}')
                args = dict(tf_cfg)
                args.pop('name')
                tfs += [tf(**args)]

            tfs = nn.Sequential(*tfs)
        else:
            tfs = nn.Identity()

        tfs = tfs.to(self.config['device'])

        return tfs
