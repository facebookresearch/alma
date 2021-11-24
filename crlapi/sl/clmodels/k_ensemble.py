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


class KEnsemble(Finetune):
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
        out = F.softmax(out, dim=-1)

        if self.config.vote:
            votes = out.argmax(-1)
            oh_votes = F.one_hot(votes, num_classes=out.size(-1))
            vote_count = oh_votes.sum(0).float()
            most_confident = out.max(0)[0].max(-1)[1]

            # Break ties
            vote_count[torch.arange(vote_count.size(0)), most_confident] += 0.1
            out = vote_count

        else:
            out = out.mean(0)

        return out


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

        training_loaders = []
        for i in range(self.config.k):
            training_loaders += [torch.utils.data.DataLoader(
                training_loader.dataset,
                batch_size=training_loader.batch_size,
                shuffle=True
            )]

        best_models = [copy.deepcopy(model) for model in models]
        best_losses, best_accs = [1e10] * self.config.k, [0] * self.config.k

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

            for i, items in enumerate(zip(*training_loaders)):
                xs, ys = [], []
                for item in items:
                    x, y = item
                    x, y = x.to(device), y.to(device)
                    xs += [train_aug(x)]
                    ys += [y]

                xs = torch.stack(xs)
                ys = torch.stack(ys)

                loss, acc = 0, 0
                for model_idx in range(self.config.k):
                    model, x, y = models[model_idx], xs[model_idx], ys[model_idx]

                    predicted = model(x)
                    loss += F.cross_entropy(predicted,y)
                    nb_ok = predicted.max(1)[1].eq(y).float().sum().item()
                    acc  += nb_ok/x.size()[0]

                accuracy = acc / self.config.k
                loss_    = loss.item() / self.config.k
                training_accuracy += accuracy
                training_loss     += loss_

                n += xs.size(1)
                n_fwd_samples += xs.size(1)

                logger.add_scalar("train/loss",loss_,iteration)
                logger.add_scalar("train/accuracy",accuracy,iteration)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iteration += 1

            #Validation
            training_accuracy /= i
            training_loss /= i
            outs = [self._validation_loop(model,device,validation_loader) for model in models]
            validation_losses = [x['loss'] for x in outs]
            validation_accuracies = [x['accuracy'] for x in outs]

            validation_loss, validation_accuracy = np.mean(validation_losses), np.mean(validation_accuracies)

            logger.add_scalar("validation/loss",validation_loss,epoch)
            logger.add_scalar("validation/accuracy",validation_accuracy,epoch)

            found_best = False
            for model_idx in range(self.config.k):
                if validation_accuracies[model_idx] > best_accs[model_idx]:
                    print("\tFound best model at epoch ",epoch, '\t', model_idx)

                    best_models[model_idx].load_state_dict(_state_dict(models[model_idx],"cpu"))
                    best_accs[model_idx]  = validation_accuracies[model_idx]
                    found_best = True

            logger.message(f"Validation Acc {validation_accuracy:.4f}\t Validation Loss {validation_loss:.4f}")
            logger.message(f"Training Acc {training_accuracy:.4f}\t Training Loss {training_loss:.4f}")

            if found_best:
                patience_count = 0
            else:
                patience_count += 1

            if patience_count == patience:
                break

        # overwrite the best models
        self.models = nn.ModuleList(best_models)

        # Evaluate each model individually :
        accs = []
        for model in self.models:
            accs += [self._validation_loop(model, device, validation_loader)['accuracy']]

        self.prog_pred_stats = []
        ensemble = self._validation_loop(self, device, validation_loader)['accuracy']

        fill = lambda x : str(x) + (100 - len(str(x))) * ' '
        self.to_print += [fill(accs)  + '\t' + str(ensemble)]
        for item in self.to_print: print(item)

        logger.message("Training Done...")
        logger.add_scalar('train/model_params', len(self.models) * sum([np.prod(x.shape) for x in model.parameters()]), 0)

        # TODO: FIX! this is wrong
        logger.add_scalar('train/one_sample_megaflop', flops_per_input / 1e6 * len(self.models), 0)
        logger.add_scalar('train/total_megaflops', n_fwd_samples * flops_per_input / 1e6, 0)
        logger.add_scalar('train/best_validation_accuracy', np.mean(best_accs), 0)
        return self

