# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy

import torch
import numpy as np
import torch.nn.functional as F
from crlapi.core import CLModel
from crlapi.sl.clmodels.core import SupervisedCLModel
from pydoc import locate

def _state_dict(model, device):
    sd = model.state_dict()
    for k, v in sd.items():
        sd[k] = v.to(device)
    return sd

class Finetune_Grow(SupervisedCLModel):
    def __init__(self, stream, clmodel_args):
        super().__init__()
        self.models=[]
        self.config=clmodel_args

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
        elif (task.task_descriptor() % self.config['grow_every']) == 0:
            print('growing')
            model=copy.deepcopy(self.models[task.task_descriptor()-1])
            model=model.grow(validation_loader,**self.config)
        else:
            model=copy.deepcopy(self.models[task.task_descriptor()-1])

        if getattr(self.config, 'init_from_scratch', False):
            print('re-initializing the model')
            def weight_reset(m):
                try:    m.reset_parameters()
                except: pass
            model.apply(weight_reset)

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
        flops_per_input  = self.count_flops(task, model)
        optimizer = self.get_optimizer(model.parameters())

        #Launching training procedure
        logger.message("Start training for "+str(self.config.max_epochs)+" epochs")
        iteration, n_fwd_samples = 0, 0
        for epoch in range(self.config.max_epochs):

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

            logger.message(f"Validation Acc {validation_accuracy:.4f}\t Loss {validation_loss:.4f}")

            if patience_count == patience:
                break

        self.models.append(best_model)
        logger.message("Training Done...")
        logger.add_scalar('train/model_params', sum([np.prod(x.shape) for x in model.parameters()]), 0)
        logger.add_scalar('train/one_sample_megaflop', flops_per_input / 1e6, 0)
        logger.add_scalar('train/total_megaflops', n_fwd_samples * flops_per_input / 1e6, 0)
        logger.add_scalar('train/best_validation_accuracy', best_acc, 0)
        logger.message("Training Done...")
        return self
