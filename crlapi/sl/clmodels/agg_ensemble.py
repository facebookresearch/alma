# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from crlapi.core import CLModel
from crlapi.sl.clmodels.finetune import Finetune

import copy
import numpy as np
from pydoc import locate
from itertools import chain

def _state_dict(model, device):
    sd = model.state_dict()
    for k, v in sd.items():
        sd[k] = v.to(device)
    return sd


class AggEnsemble(Finetune):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_print = []

        print(f'voting {self.config.vote}')

    def get_prediction_net(self,task):
        for model in self.models:
            model.eval()

        return self

    def forward(self, x):
        outs = []
        for model in self.models:
            outs += [model(x)]

        out = torch.stack(outs).sum(0)
        return out

    def _validation_loop(self,nets,device,dataloader):
        [net.eval() for net in nets]
        [net.to(device) for net in nets]

        with torch.no_grad():
            loss_values=[]
            nb_ok=0
            nb_total=0
            for x,y in dataloader:
                x,y=x.to(device),y.to(device)
                predicted=0

                for net in nets:
                    predicted += net(x)

                loss=F.cross_entropy(predicted,y)
                loss_values.append(loss.item())
                nb_ok+=predicted.max(1)[1].eq(y).float().sum().item()
                nb_total+=x.size()[0]

            loss=np.mean(loss_values)
            accuracy=nb_ok/nb_total

        net = net.train()
        return {"loss":loss,"accuracy":accuracy}


    def update(self, task, logger):
        assert isinstance(task.task_descriptor(),int)

        if len(self.models)==0 or getattr(self.config, 'init_from_scratch', False):
            model_args=self.config.model
            models = [self.build_initial_net(task,**model_args) for _ in range(self.config.k)]
            n_params = sum(np.prod(x.shape) for x in models[0].parameters())
            print(f'new model has {n_params} params')
        else:
            # get the last k models
            models = [copy.deepcopy(model) for model in self.models]

        logger.message("Building training dataset")
        training_dataset = task.task_resources().make()
        flops_per_input  = self.count_flops(task, models[0]) * self.config.k

        # Creating datasets and loaders
        training_loader, validation_loader = self.get_train_and_validation_loaders(training_dataset)

        best_models = [copy.deepcopy(model) for model in models]
        best_loss, best_acc = 1e10, None

        # Optionally create GPU training augmentations
        train_aug = self.get_train_augs()

        # Optinally use patience :)
        patience = self.config.patience
        patience_delta = self.config.patience_delta
        patience_count = 0

        device=self.config.device

        models = [model.to(device) for model in models]
        optimizer = self.get_optimizer(chain(*[model.parameters() for model in models]))

        #Launching training procedure
        logger.message("Start training for " + str(self.config.max_epochs) + " epochs")

        iteration, n_fwd_samples = 0, 0
        for epoch in range(self.config.max_epochs):

            # Make sure model is ready for train
            [model.train() for model in models]

            # Keep a single track of these for now
            training_loss=0.0
            training_accuracy=0.0
            n=0

            for i, (raw_x, y) in enumerate(training_loader):
                raw_x, y = raw_x.to(device), y.to(device)
                # apply transformations
                x = train_aug(raw_x)

                predicted = 0.

                for model in models:
                    predicted += model(x)

                loss  = F.cross_entropy(predicted,y)
                nb_ok = predicted.max(1)[1].eq(y).float().sum().item()
                acc   = nb_ok/x.size()[0]

                accuracy = acc
                loss_    = loss.item()
                training_accuracy += accuracy
                training_loss     += loss_

                n += x.size(0)
                n_fwd_samples += x.size(0)

                logger.add_scalar("train/loss",loss_,iteration)
                logger.add_scalar("train/accuracy",accuracy,iteration)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iteration += 1

            #Validation
            training_accuracy /= i
            training_loss /= i
            out=self._validation_loop(models,device,validation_loader)
            validation_loss, validation_accuracy = out["loss"], out["accuracy"]

            logger.add_scalar("validation/loss",validation_loss,epoch)
            logger.add_scalar("validation/accuracy",validation_accuracy,epoch)

            if best_acc is None or validation_accuracy > (best_acc):
                best_acc = validation_accuracy
                for model_idx in range(self.config.k):
                    best_models[model_idx].load_state_dict(_state_dict(models[model_idx],"cpu"))

                patience_count = 0
            else:
                patience_count += 1

            logger.message(f"Validation Acc {validation_accuracy:.4f}\t Validation Loss {validation_loss:.4f}")
            logger.message(f"Training Acc {training_accuracy:.4f}\t Training Loss {training_loss:.4f}")

            if patience_count == patience:
                break

        # overwrite the best models
        self.models = nn.ModuleList(best_models)

        logger.message("Training Done...")
        logger.add_scalar('train/model_params', len(self.models) * sum([np.prod(x.shape) for x in model.parameters()]), 0)
        logger.add_scalar('train/one_sample_megaflop', flops_per_input / 1e6 * len(self.models), 0)
        logger.add_scalar('train/total_megaflops', n_fwd_samples * flops_per_input / 1e6, 0)
        logger.add_scalar('train/best_validation_accuracy', best_acc, 0)
        return self

