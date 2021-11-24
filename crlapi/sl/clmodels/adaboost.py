import torch
import torch.nn as nn
import torch.nn.functional as F
from crlapi.core import CLModel
from crlapi.sl.clmodels.finetune import Finetune

import copy
import numpy as np
from pydoc import locate


class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, og_dataset):
        self.og_dataset = og_dataset

    def __getitem__(self, index):
        data, target = self.og_dataset[index]

        return data, target, index

    def __len__(self):
        return len(self.og_dataset)

def _state_dict(model, device):
    sd = model.state_dict()
    for k, v in sd.items():
        sd[k] = v.to(device)
    return sd

class AdaBoost(Finetune):

    def get_prediction_net(self,task):
        for i, model in enumerate(self.models):
            model.eval()
            self.models[i] = model.to(self.config.device)

        return self

    def forward(self, x):
        outs = []
        for model in self.models:
            outs += [model(x)]

        out = torch.stack(outs) * self.model_weights.reshape((-1,) + (1,) * outs[0].ndim)
        out = out.mean(0)

        return out

    def compute_errors(self, loader, model):
        unshuffled_loader = torch.utils.data.DataLoader(
                loader.dataset, batch_size=loader.batch_size, drop_last=False, shuffle=False)
        device=self.config.device

        model.to(device)
        model.eval()

        # --- Upweighting
        err = []

        # eval mode
        with torch.no_grad():
            for x, y in unshuffled_loader:
                x, y = x.to(device), y.to(device)

                err += [~model(x).argmax(1).eq(y)]

        err = torch.cat(err).float() # (DS, )

        return err


    def update(self, task, logger):
        assert isinstance(task.task_descriptor(),int)

        logger.message("Building training dataset")
        training_dataset = task.task_resources().make()

        # Creating datasets and loaders
        og_training_loader, validation_loader = self.get_train_and_validation_loaders(training_dataset)

        training_loader = torch.utils.data.DataLoader(
                IndexDataset(og_training_loader.dataset),
                batch_size=og_training_loader.batch_size,
                shuffle=True
        )

        to_print = []

        # --- step 1 : Initialize the observation weights uniformly
        ds_len = len(og_training_loader.dataset)
        sample_weights = torch.zeros(size=(ds_len,)).fill_(1. / ds_len).to(self.config.device)

        for around in range(self.config.n_rounds):

            # --- 2. a) Fit new classifier on weighted data

            # init model
            model_args=self.config.model
            model, best_model = [self.build_initial_net(task,**model_args) for _ in range(2)]
            flops_per_input  = self.count_flops(task, model)
            n_params = sum(np.prod(x.shape) for x in model.parameters())
            print(f'new model has {n_params} params')

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
            logger.message("Start training for "+str(self.config.max_epochs) + " epochs")
            iteration, n_fwd_samples = 0, 0

            epoch = 0
            while True:     # Run until convergence
                epoch += 1

                # Make sure model is ready for train
                model.train()

                # Training loop
                training_loss=0.0
                training_accuracy=0.0
                n=0

                for i, (raw_x, y, idx) in enumerate(training_loader):
                    raw_x, y = raw_x.to(device), y.to(device)
                    weight_x = sample_weights[idx]

                    n += y.size(0)

                    # apply transformations
                    x = train_aug(raw_x)

                    predicted = model(x)
                    loss = F.cross_entropy(predicted, y, reduction='none')
                    loss = (loss * weight_x).mean() * ds_len

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

                # Validation
                training_accuracy/=n
                training_loss/=n
                out=self._validation_loop(model,device,validation_loader)
                validation_loss,validation_accuracy=out["loss"],out["accuracy"]

                logger.add_scalar("validation/loss",validation_loss,epoch)
                logger.add_scalar("validation/accuracy",validation_accuracy,epoch)

                # Right now CV against accuracy
                if best_acc is None or validation_accuracy > (best_acc + patience_delta):
                    print(f"\t Round {around}. Found best model at epoch ",epoch)
                    best_model.load_state_dict(_state_dict(model,"cpu"))
                    best_loss = validation_loss
                    best_acc  = validation_accuracy
                    patience_count = 0
                else:
                    patience_count += 1

                logger.message(f"Validation Acc {validation_accuracy:.4f}\t Validation Loss {validation_loss:.4f}")
                logger.message(f"Training Acc {training_accuracy:.4f}\t Training Loss {training_loss:.4f}")

                if patience_count == patience or epoch == self.config.max_epochs:
                    break

            del model

            # --- Step 2 b) Compute the new classifier errors
            all_errs = self.compute_errors(og_training_loader, best_model) # 1, ds_size
            assert all_errs.shape == (ds_len, )

            cls_err = (all_errs * sample_weights).sum() / sample_weights.sum()

            # --- Step 2 c) Compute the new classifier weight
            K = task.n_classes
            cls_alpha = torch.log((K - 1) * (1 - cls_err) / cls_err)

            # --- Step 2 d) Update the sample weights
            sample_weights = sample_weights * torch.exp(cls_alpha * all_errs)
            sample_weights /= sample_weights.sum()

            print(f'sample weights min {sample_weights.min():.6f}\t max {sample_weights.max():.6f} \t median {sample_weights.median():.6f}')
            print(torch.multinomial(sample_weights, ds_len, replacement=True).bincount().bincount())

            # store best model
            self.models.append(best_model)

            cls_alpha = cls_alpha.reshape(1)
            # store classifier weights
            if not hasattr(self, 'model_weights'):
                self.model_weights = cls_alpha
            else:
                self.model_weights = torch.cat((self.model_weights, cls_alpha))
            print(self.model_weights)

            # Evaluate each model individually :
            accs = []
            for model in self.models:
                accs += [self._validation_loop(model, device, validation_loader)['accuracy']]

            ensemble = self._validation_loop(self, device, validation_loader)['accuracy']

            fill = lambda x : str(x) + (100 - len(str(x))) * ' '
            to_print += [fill(accs)  + '\t' + str(ensemble)]
            for item in to_print: print(item)

            logger.message("Training Done...")
            logger.add_scalar('train/model_params', len(self.models) * sum([np.prod(x.shape) for x in model.parameters()]), 0)
            logger.add_scalar('train/one_sample_megaflop', len(self.models) * flops_per_input / 1e6 * len(self.models), 0)
            logger.add_scalar('train/total_megaflops', n_fwd_samples * flops_per_input / 1e6, 0)
            logger.add_scalar('train/best_validation_accuracy', best_acc, 0)

        return self

