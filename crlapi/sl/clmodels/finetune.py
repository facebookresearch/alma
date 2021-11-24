import torch
import torch.nn as nn
import torch.nn.functional as F
from crlapi.core import CLModel
from crlapi.sl.clmodels.core import SupervisedCLModel

import time
import copy
import numpy as np
from pydoc import locate

def _state_dict(model, device):
    sd = model.state_dict()
    for k, v in sd.items():
        sd[k] = v.to(device)
    return sd

class Finetune(SupervisedCLModel):
    def __init__(self, stream, clmodel_args):
        super().__init__()
        self.models = nn.ModuleList()
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

        if len(self.models)==0 or getattr(self.config, 'init_from_scratch', False):
            model_args=self.config.model
            model=self.build_initial_net(task,**model_args)
            n_params = sum(np.prod(x.shape) for x in model.parameters())
            print(model)
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
            start = time.time()
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
            epoch_time = time.time() - start
            training_accuracy/=n
            training_loss/=n
            out=self._validation_loop(model,device,validation_loader)
            validation_loss,validation_accuracy=out["loss"],out["accuracy"]

            logger.add_scalar('training/one_epoch_time', epoch_time, epoch)
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
            logger.message(f"Training Acc {training_accuracy:.4f}\t Training Loss {training_loss:.4f}\t Time {epoch_time:.4f}")

            if patience_count == patience:
                break

        self.models.append(best_model)
        logger.message("Training Done...")
        logger.add_scalar('train/model_params', sum([np.prod(x.shape) for x in model.parameters()]), 0)
        logger.add_scalar('train/one_sample_megaflop', flops_per_input / 1e6, 0)
        logger.add_scalar('train/total_megaflops', n_fwd_samples * flops_per_input / 1e6, 0)
        logger.add_scalar('train/best_validation_accuracy', best_acc, 0)
        return self
