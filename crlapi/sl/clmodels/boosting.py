import torch
import torch.nn as nn
import torch.nn.functional as F
from crlapi.core import CLModel
from crlapi.sl.clmodels.finetune import Finetune

import time
import copy
import numpy as np
from pydoc import locate


class IndexDataset(torch.utils.data.Dataset):
    """ Wrapper that additionally returns the index for each sample """

    def __init__(self, og_dataset):
        self.og_dataset = og_dataset

    def __getitem__(self, index):
        data, target = self.og_dataset[index]

        return data, target, index

    def __len__(self):
        return len(self.og_dataset)


class BoostingSampler(torch.utils.data.Sampler):
    """ Upsample points based on sample weight """

    def __init__(self, weights):
        self.weights = weights

    def __iter__(self):

        assert -1e-5 < self.weights.sum().item() - 1 < 1e-5
        samples = torch.multinomial(self.weights, self.weights.size(0), replacement=True)

        if not hasattr(self, 'epoch'):
            print('sampling with replacement counts', samples.bincount().bincount())
            self.epoch = 0
        else:
            self.epoch += 1

        for sample in samples:
            yield sample.item()


def _state_dict(model, device):
    sd = model.state_dict()
    for k, v in sd.items():
        sd[k] = v.to(device)
    return sd


class AdaBoost(Finetune):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_print = []
        self.prog_pred_stats = []

    def get_prediction_net(self,task):
        for i, model in enumerate(self.models):
            model.eval()
            self.models[i] = model.to(self.config.device)

        return self

    def forward(self, x):
        outs = []
        for model in self.models:
            outs += [model(x)]

        out = torch.stack(outs) * self.model_alphas.reshape((-1,) + (1,) * outs[0].ndim)
        u_out = torch.stack(outs)

        print('diff weighted / unw', (out.sum(0).argmax(-1) != u_out.sum(0).argmax(-1)).float().mean())

        prog_pred = out.cumsum(0).argmax(-1)

        diff_from_last = prog_pred[-1, :] != prog_pred
        diff_from_last = torch.flip(diff_from_last, dims=(0,)) # n_models, bs : with oldest model as idx = 0

        last_conseq_steps_with_same_pred = (diff_from_last.int().cumsum(0) == 0).int().sum(0)
        useful_steps = len(self.models) - last_conseq_steps_with_same_pred + 1

        self.prog_pred_stats += [useful_steps.bincount(minlength=len(self.models) + 1)]

        return out.sum(0)

        """
        # --- actually let's pick the most confident model
        out = u_out

        max_prob = F.softmax(out,-1).max(-1)[0] # n_models, BS
        model_idx = max_prob.argmax(0) #BS,

        N_CLS = out.size(-1)
        idx   = torch.arange(model_idx.size(0)).cuda() * len(self.models) + model_idx
        out   = out.transpose(1,0) # BS, n_models, C
        out   = out.reshape(-1,N_CLS)[idx].reshape(-1,N_CLS)
        return out

        # ----

        #return u_out.sum(0)
        #return out.sum(0)
        """


    def weighted_validation_loop(self,net,device, dataloader, weights):
        """ weight loss and accuracy using sample specific weights """

        net = net.eval()

        # Return indices for the dataset
        loader = torch.utils.data.DataLoader(
                IndexDataset(dataloader.dataset),
                batch_size=dataloader.batch_size,
                shuffle=False
        )

        ds_len = len(dataloader.dataset)

        with torch.no_grad():
            loss_values=[]
            acc = 0

            for i, (x,y,idx) in enumerate(loader):

                x, y, idx = x.to(device),y.to(device), idx.to(device)
                weight_x  = weights[idx]

                predicted=net(x)

                loss = F.cross_entropy(predicted,y, reduction='none')
                loss = (loss * weight_x).mean() * ds_len
                loss_values.append(loss.item())

                acc += (predicted.argmax(1).eq(y).float() * weight_x).sum()

            loss=np.mean(loss_values)

        net = net.train()
        return {"loss":loss,"accuracy":acc.item()}


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


    def compute_errors(self, loader, models):
        """ given a loader and a list of models, returns a per_model x per_sample error matrix """

        unshuffled_loader = torch.utils.data.DataLoader(
                loader.dataset, batch_size=loader.batch_size, drop_last=False, shuffle=False)
        device=self.config.device

        # --- Upweighting
        all_errs = []

        # eval mode
        [x.eval() for x in models]

        with torch.no_grad():
            for x, y in unshuffled_loader:
                x, y = x.to(device), y.to(device)

                for i, model in enumerate(models):
                    if i == 0:
                        err  = [~model(x).argmax(1).eq(y)]
                    else:
                        err += [~model(x).argmax(1).eq(y)]

                err = torch.stack(err) # n_models, bs

                all_errs += [err]

        all_errs  = torch.cat(all_errs, dim=1).float() # n_models, DS

        return all_errs


    def compute_model_and_sample_weights(self, err_matrix, task):
        """ compound sample and model models w.r.t to each model's performance """

        n_models, ds_len = err_matrix.size()

        sample_weights = torch.zeros(size=(ds_len,)).fill_(1. / ds_len).to(self.config.device)
        model_alphas   = []

        for model_idx in range(n_models):
            model_err = err_matrix[model_idx]
            weighted_model_err = (sample_weights * model_err).sum() / sample_weights.sum()

            model_alpha = torch.log((1 - weighted_model_err) / weighted_model_err) + np.log(task.n_classes - 1)
            model_alphas += [model_alpha.reshape(1)]

            sample_weights = sample_weights * torch.exp(model_alpha * model_err)

            sample_weights /= sample_weights.sum()

        return sample_weights, model_alphas


    def update(self, task, logger):
        """ train model on new MB """

        task_id = task.task_descriptor()
        assert isinstance(task_id, int)
        self.validation_outputs = None

        if task_id == 0 or self.config.init == 'scratch':
            # create model
            model_args = self.config.model
            model      = self.build_initial_net(task,**model_args)
        elif self.config.init == 'last':
            model      = copy.deepcopy(self.models[-1])
        elif self.config.init == 'first':
            model      = copy.deepcopy(self.models[0])

        # Creating datasets and loaders
        logger.message("Building training dataset")
        training_dataset = task.task_resources().make()
        flops_per_input  = self.count_flops(task, model)
        og_training_loader, validation_loader = self.get_train_and_validation_loaders(training_dataset)
        ds_len = len(og_training_loader.dataset)

        # --- get per sample weights
        if task_id == 0:
            n_params = sum(np.prod(x.shape) for x in model.parameters())
            print(model)
            print(f'new model has {n_params} params')
            sample_weights = torch.zeros(size=(ds_len,)).fill_(1. / ds_len).to(self.config.device)
            model_alphas = []
            err_matrix = val_err_matrix = None
        else:
            err_matrix = self.compute_errors(og_training_loader, self.models)
            sample_weights, model_alphas = self.compute_model_and_sample_weights(err_matrix, task)
            val_err_matrix = self.compute_errors(validation_loader, self.models)
            val_sample_weights, val_model_alphas = self.compute_model_and_sample_weights(val_err_matrix, task)
            print('tr sample weights',torch.multinomial(sample_weights, sample_weights.size(0), replacement=True).bincount().bincount())
            print('val sample weights',torch.multinomial(val_sample_weights, val_sample_weights.size(0), replacement=True).bincount().bincount())

            if self.config.compute_model_weights_on_val:
                model_alphas = val_model_alphas

        if self.config.boosting == 'weighting' or task_id == 0:
            # sample normally, but weight each point
            sampler = None
            training_weights = sample_weights

            #like ensembleat thispoint
            #print('UNIFORM WEIGHTS')
            #training_weights = torch.zeros(size=(ds_len,)).fill_(1. / ds_len).to(self.config.device)

        elif self.config.boosting == 'sampling':
            # oversample points with high weight --> no need to upweight them
            # print('UNIFORM WEIGHTS')
            #sample_weights = torch.zeros(size=(ds_len,)).fill_(1. / ds_len).to(self.config.device)

            sampler = BoostingSampler(sample_weights)
            training_weights = torch.zeros(size=(ds_len,)).fill_(1. / ds_len).to(self.config.device)
        else:
            raise ValueError

        # Return indices for the dataset
        training_loader = torch.utils.data.DataLoader(
                IndexDataset(og_training_loader.dataset),
                batch_size=og_training_loader.batch_size,
                shuffle=sampler is None,
                sampler=sampler
        )

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

        # Launching training procedure
        logger.message("Start training for "+str(self.config.max_epochs)+" epochs")

        iteration = 0
        for epoch in range(self.config.max_epochs):

            # Make sure model is ready for train
            model.train()

            # Training loop
            training_loss=0.0
            training_accuracy=0.0
            n=0
            start = time.time()
            for i, (raw_x, y, idx) in enumerate(training_loader):
                raw_x, y = raw_x.to(device), y.to(device)
                weight_x = training_weights[idx]

                n += y.size(0)

                # apply transformations
                x = train_aug(raw_x)

                predicted = model(x)

                loss = F.cross_entropy(predicted, y, reduction='none')
                loss = (loss * weight_x).mean() * ds_len

                nb_ok = predicted.argmax(1).eq(y).sum().item()
                accuracy = nb_ok/x.size(0)
                training_accuracy += nb_ok
                training_loss += loss.item()
                logger.add_scalar("train/loss",loss.item(),  iteration)
                logger.add_scalar("train/accuracy",accuracy, iteration)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iteration += 1

            # Validation
            epoch_time = time.time() - start
            training_accuracy /= n
            training_loss /= n

            if task_id == 0 or self.config.validation == 'normal':
                out=self._validation_loop(model,device,validation_loader)
            elif self.config.validation == 'weighted':
                out=self.weighted_validation_loop(model,device,validation_loader, val_sample_weights)
            else:
                raise ValueError

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

        # Store best model
        self.models.append(best_model)

        # Before making predictions, we need to calculate the weight of the new model
        if self.config.compute_model_weights_on_val:
            final_loader, err_mat = validation_loader, val_err_matrix
        else:
            final_loader, err_mat = og_training_loader, err_matrix

        new_model_err  = self.compute_errors(final_loader, [best_model]) # (1, DS)
        if err_mat is not None:
            err_mat = torch.cat((err_mat, new_model_err))
        else:
            err_mat = new_model_err

        _, model_alphas = self.compute_model_and_sample_weights(err_mat, task)
        self.model_alphas = torch.cat(model_alphas)

        # Evaluate each model individually :
        accs = []
        for model in self.models:
            accs += [self._validation_loop(model, device, validation_loader)['accuracy']]

        self.prog_pred_stats = []
        ensemble = self._validation_loop(self, device, validation_loader)['accuracy']

        best=self._all_validation_loop(device,validation_loader,task)
        print('among best ', best)

        pred_stats = torch.stack(self.prog_pred_stats).sum(0).float()
        pred_stats /= pred_stats.sum()

        print('model weights', self.model_alphas)
        print('pred stats', pred_stats)

        for i in range(pred_stats.size(0)):
            logger.add_scalar('model/prediction_depth', pred_stats[i].item(), i)

        fill = lambda x : str(x) + (100 - len(str(x))) * ' '
        self.to_print += [fill(accs)  + '\t' + str(ensemble)]
        for item in self.to_print: print(item)

        logger.message("Training Done...")
        logger.add_scalar('train/model_params', sum([np.prod(x.shape) for x in model.parameters()]), 0)
        logger.add_scalar('train/one_sample_megaflop', flops_per_input / 1e6, 0)
        logger.add_scalar('train/total_megaflops', n * flops_per_input / 1e6, 0)
        logger.add_scalar('train/best_validation_accuracy', best_acc, 0)
        return self
