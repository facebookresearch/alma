from crlapi.core import TaskResources, Stream, Task
import torchvision.datasets
import torchvision.transforms
import numpy.random
import numpy
import torch.utils.data
import torch

class CachedCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = torch.from_numpy(self.data).float().permute(0, 3, 1, 2)
        self.targets = numpy.array(self.targets)

        self.data = self.data / 255.

        # normalize
        mu  = torch.Tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        var = torch.Tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

        self.data = (self.data - mu) / var


    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]

        if self.transform is not None:
            x = self.transform(x)

        return x, y


class ClassificationResources(TaskResources):
    def __init__(self):
        pass

class CIFAR10Resources(ClassificationResources):
    def __init__(self, idx_batch, n_total_batches, seed,train,directory):
        self.idx_batch=idx_batch
        self.n_total_batches=n_total_batches
        self.train=train
        self.seed=seed
        self.directory=directory

    def make(self):
        dataset=CachedCIFAR10(self.directory, train=self.train, download=True)
        if self.n_total_batches==1:
            return dataset

        numpy.random.seed(self.seed)
        indices=numpy.arange(len(dataset))
        indices=numpy.random.permutation(indices)
        _indices=numpy.array_split(indices,self.n_total_batches)
        indices=list(_indices[self.idx_batch])
        _set=torch.utils.data.Subset(dataset,indices)
        return _set

class CIFAR10Task(Task):
    def __init__(self,task_descriptor,resources):
        self._task_descriptor=task_descriptor
        self._resources=resources
        self.input_shape=(3, 32, 32)
        self.n_classes=10

    def task_descriptor(self):
        return self._task_descriptor

    def task_resources(self):
        return self._resources


class CIFAR10EvaluationAnytimeStream(Stream):
    def __init__(self,n_megabatches,seed,directory):
        self.tasks=[]
        evaluation_resources=CIFAR10Resources(0,1,seed,False,directory)
        self.tasks.append(CIFAR10Task(None,evaluation_resources))

    def __len__(self):
        return len(self.tasks)

    def __iter__(self):
        return self.tasks.__iter__()

    def __getitem__(self,k):
        return self.tasks[k]


class CIFAR10TrainAnytimeStream(Stream):
    def __init__(self,n_megabatches,seed,directory):
        self.tasks=[]
        for k in range(n_megabatches):
            resources=CIFAR10Resources(k,n_megabatches,seed,True,directory)
            self.tasks.append(CIFAR10Task(k,resources))

    def __len__(self):
        return len(self.tasks)

    def __iter__(self):
        return self.tasks.__iter__()

    def __getitem__(self,k):
        return self.tasks[k]
