# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from crlapi.core import TaskResources, Stream, Task
import torchvision.datasets
import torchvision.transforms
import numpy.random
import numpy
import torch.utils.data
import torch


class CachedMNIST(torchvision.datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.targets = self.targets.numpy()
        self.data    = (self.data / 255. - .1307) / .3081


    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]

        if self.transform is not None:
            x = self.transform(x)

        return x, y


class ClassificationResources(TaskResources):
    def __init__(self):
        pass

class MNISTResources(ClassificationResources):
    def __init__(self, idx_batch, n_total_batches, seed,train,directory):
        self.idx_batch=idx_batch
        self.n_total_batches=n_total_batches
        self.train=train
        self.seed=seed
        self.directory=directory

    def make(self):
        dataset=CachedMNIST(self.directory, train=self.train, download=True)
        if self.n_total_batches==1:
            return dataset
        numpy.random.seed(self.seed)
        indices=numpy.arange(len(dataset))
        indices=numpy.random.permutation(indices)
        _indices=numpy.array_split(indices,self.n_total_batches)
        indices=list(_indices[self.idx_batch])
        _set=torch.utils.data.Subset(dataset,indices)
        return _set

class MNISTTask(Task):
    def __init__(self,task_descriptor,resources):
        self._task_descriptor=task_descriptor
        self._resources=resources
        self.input_shape=(1,28,28)
        self.n_classes=10

    def task_descriptor(self):
        return self._task_descriptor

    def task_resources(self):
        return self._resources


class MNISTEvaluationAnytimeStream(Stream):
    def __init__(self,n_megabatches,seed,directory):
        self.tasks=[]
        evaluation_resources=MNISTResources(0,1,seed,False,directory)
        self.tasks.append(MNISTTask(None,evaluation_resources))

    def __len__(self):
        return len(self.tasks)

    def __iter__(self):
        return self.tasks.__iter__()

    def __getitem__(self,k):
        return self.tasks[k]


class MNISTTrainAnytimeStream(Stream):
    def __init__(self,n_megabatches,seed,directory):
        self.tasks=[]
        for k in range(n_megabatches):
            resources=MNISTResources(k,n_megabatches,seed,True,directory)
            self.tasks.append(MNISTTask(k,resources))

    def __len__(self):
        return len(self.tasks)

    def __iter__(self):
        return self.tasks.__iter__()

    def __getitem__(self,k):
        return self.tasks[k]
