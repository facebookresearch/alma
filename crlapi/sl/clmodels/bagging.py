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


class BaggingSampler(torch.utils.data.Sampler):
    """ Simulate a Dataset Sampled with Replacement """

    def __init__(self, indices, real_ds_size):
        self.size = real_ds_size
        self.indices = indices

        weights = torch.zeros(size=(self.size,)).float()
        weights[self.indices] = 1
        self.weights = weights

        # do this here so that each epoch sees same sample dist
        samples = torch.multinomial(weights, self.size, replacement=True)
        self.samples = samples

        unique_samples = samples.unique()
        counts = samples.bincount().bincount()
        assert (counts * torch.arange(counts.size(0)))[1:].sum().item() == self.size
        print(counts, unique_samples.size(0), self.indices.size(0))

        for ss in unique_samples:
            assert (ss == unique_samples).sum() > 0


    def __iter__(self):
        samples = self.samples
        samples = samples[torch.randperm(samples.size(0))]

        # RESAMPLING
        samples = torch.multinomial(self.weights, self.size, replacement=True)

        for sample in samples:
            yield sample.item()


class Bagging(Finetune):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_print = []

    def get_prediction_net(self,task):
        for model in self.models:
            model.eval()

        return self

    def forward(self, x):
        outs = []
        for model in self.models:
            outs += [model(x)]

        out = torch.stack(outs).mean(0)

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
            models = [copy.deepcopy(self.models[-i]) for i in range(self.config.k)]

        logger.message("Building training dataset")
        training_dataset = task.task_resources().make()
        flops_per_input  = self.count_flops(task, models[0]) * self.config.k

        # Creating datasets and loaders
        training_loader, validation_loader = self.get_train_and_validation_loaders(training_dataset)
        ds_len = len(training_loader.dataset)

        training_loaders = []
        # build boosted loaders
        for _ in range(self.config.k):
            all_idx = torch.arange(ds_len)

            idx = torch.multinomial(
                    torch.ones_like(all_idx).float(),
                    int(self.config.subsample_p * ds_len),
                    replacement=False
            )

            sampler = BaggingSampler(idx, ds_len)
            loader  = torch.utils.data.DataLoader(
                    training_loader.dataset,
                    batch_size=training_loader.batch_size,
                    sampler=sampler
            )

            training_loaders += [loader]

        best_models = [copy.deepcopy(model) for model in models]
        best_losses, best_accs = [1e10] * self.config.k, [0] * self.config.k

        # Optionally create GPU training augmentations
        train_aug = self.get_train_augs()

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

            for model_idx in range(self.config.k):
                if validation_accuracies[model_idx] > best_accs[model_idx]:
                    print("\tFound best model at epoch ",epoch, '\t', model_idx)

                    best_models[model_idx].load_state_dict(_state_dict(models[model_idx],"cpu"))
                    best_accs[model_idx]  = validation_accuracies[model_idx]

            logger.message(f"Validation Acc {validation_accuracy:.4f}\t Validation Loss {validation_loss:.4f}")
            logger.message(f"Training Acc {training_accuracy:.4f}\t Training Loss {training_loss:.4f}")

        for best_model in best_models:
            self.models.append(best_model)

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
        logger.add_scalar('train/one_sample_megaflop', flops_per_input / 1e6 * len(self.models), 0)
        logger.add_scalar('train/total_megaflops', n_fwd_samples * flops_per_input / 1e6, 0)
        logger.add_scalar('train/best_validation_accuracy', np.mean(best_accs), 0)
        return self

