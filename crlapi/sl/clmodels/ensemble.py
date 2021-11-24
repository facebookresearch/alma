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

def _state_dict(model, device):
    sd = model.state_dict()
    for k, v in sd.items():
        sd[k] = v.to(device)
    return sd

class Ensemble(Finetune):
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

        out = torch.stack(outs)

        if self.config.vote:
            votes = out.argmax(-1)
            oh_votes = F.one_hot(votes, dim=-1, num_classes=out.size(-1))
            vote_count = oh_votes.sum(0).float()
            most_confident = out.max(0)[0].max(-1)[1]

            # Break ties
            vote_count[torch.arange(out.size(0)), most_confident] += 0.1
            out = vote_count

        else:
            out = out.mean(0)


        return out

    def _all_validation_loop(self, device, dataloader,task):
        """ weight loss and accuracy using sample specific weights """

        self.get_prediction_net(task)

        ds_len = len(dataloader.dataset)
        acc = 0

        with torch.no_grad():
            loss_values=[]
            acc = 0

            for i, (x,y) in enumerate(dataloader):

                x, y= x.to(device),y.to(device)

                out = []
                for model in self.models:
                    out += [model(x)]

                out = torch.stack(out).argmax(-1)
                acc += (out == y.view(1,-1)).int().max(0)[0].float().sum().item()

        return acc / ds_len

    def update(self, task, logger):
        assert isinstance(task.task_descriptor(),int)

        if len(self.models)==0 or getattr(self.config, 'init_from_scratch', False):
            model_args=self.config.model
            model=self.build_initial_net(task,**model_args)
            n_params = sum(np.prod(x.shape) for x in model.parameters())
            print(f'new model has {n_params} params')
        else:
            model=copy.deepcopy(self.models[task.task_descriptor()-1])

        logger.message("Building training dataset")
        training_dataset = task.task_resources().make()
        flops_per_input  = self.count_flops(task, model)

        # Creating datasets and loaders
        training_loader,validation_loader = self.get_train_and_validation_loaders(training_dataset)

        best_model=copy.deepcopy(model)
        best_loss, best_acc = None, None

        # Optionally create GPU training augmentations
        train_aug = self.get_train_augs()

        # Optinally use patience :)
        patience = self.config.patience
        patience_delta = self.config.patience_delta
        patience_count = 0

        device=self.config.device
        model.to(device)
        optimizer = self.get_optimizer(model.parameters())

        #Launching training procedure
        logger.message("Start training for "+str(self.config.max_epochs)+" epochs")
        iteration, n_fwd_samples = 0, 0
        for epoch in range(self.config.max_epochs):

            # Make sure model is ready for train
            model.train()

            #Training loop
            training_loss=0.0
            training_accuracy=0.0
            n=0
            for i, (raw_x, y) in enumerate(training_loader):
                raw_x, y = raw_x.to(device), y.to(device)
                n+=raw_x.size()[0]
                # apply transformations
                x = train_aug(raw_x)

                predicted=model(x)
                loss=F.cross_entropy(predicted,y)
                nb_ok=predicted.max(1)[1].eq(y).float().sum().item()
                accuracy=nb_ok/x.size()[0]
                training_accuracy+=nb_ok
                training_loss+=loss.item()
                logger.add_scalar("train/loss",loss.item(),iteration)
                logger.add_scalar("train/accuracy",accuracy,iteration)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iteration += 1
                n_fwd_samples += x.size(0)

            #Validation
            training_accuracy/=n
            training_loss/=n
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

            logger.message(f"Validation Acc {validation_accuracy:.4f}\t Validation Loss {validation_loss:.4f}")
            logger.message(f"Training Acc {training_accuracy:.4f}\t Training Loss {training_loss:.4f}")

            if patience_count == patience:
                break

        self.models.append(best_model)

        # Evaluate each model individually :
        accs = []
        for model in self.models:
            accs += [self._validation_loop(model, device, validation_loader)['accuracy']]

        self.prog_pred_stats = []
        ensemble = self._validation_loop(self, device, validation_loader)['accuracy']

        best=self._all_validation_loop(device,validation_loader,task)
        print('among best ', best)

        fill = lambda x : str(x) + (100 - len(str(x))) * ' '
        self.to_print += [fill(accs)  + '\t' + str(ensemble)]
        for item in self.to_print: print(item)


        logger.message("Training Done...")
        logger.add_scalar('train/model_params', len(self.models) * sum([np.prod(x.shape) for x in model.parameters()]), 0)
        logger.add_scalar('train/one_sample_megaflop', flops_per_input / 1e6 * len(self.models), 0)
        logger.add_scalar('train/total_megaflops', n_fwd_samples * flops_per_input / 1e6, 0)
        logger.add_scalar('train/best_validation_accuracy', best_acc, 0)
        return self

