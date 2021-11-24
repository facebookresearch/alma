from crlapi.core import TaskResources, Stream, Task
import torchvision.datasets
import torchvision.transforms
import numpy.random
import numpy
import torch.utils.data
import torch


# TODO: did not verify this dataset
class CachedEMNIST(torchvision.datasets.EMNIST):
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

class EMNISTResources(ClassificationResources):
    def __init__(self, idx_batch, n_total_batches, seed,train,split,directory):
        self.idx_batch=idx_batch
        self.split=split
        self.n_total_batches=n_total_batches
        self.train=train
        self.seed=seed
        self.directory=directory

    def n_classes(self):
        print("Compputing n classes...")
        dataset=CachedEMNIST(self.directory, split=self.split,train=self.train, download=True)
        n=len(dataset.classes_split_dict[self.split])
        return n

    def make(self):
        dataset=torchvision.datasets.EMNIST(self.directory, split=self.split,train=self.train, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
        if self.n_total_batches==1:
            return dataset
        numpy.random.seed(self.seed)
        indices=numpy.arange(len(dataset))
        indices=numpy.random.permutation(indices)
        _indices=numpy.array_split(indices,self.n_total_batches)
        indices=list(_indices[self.idx_batch])
        _set=torch.utils.data.Subset(dataset,indices)
        return _set

class EMNISTTask(Task):
    def __init__(self,task_descriptor,resources):
        self._task_descriptor=task_descriptor
        self._resources=resources
        self.input_shape=(1,28,28)
        self.n_classes=self._resources.n_classes()

    def task_descriptor(self):
        return self._task_descriptor

    def task_resources(self):
        return self._resources


class EMNISTEvaluationAnytimeStream(Stream):
    def __init__(self,n_megabatches,seed,split,directory):
        self.tasks=[]
        evaluation_resources=EMNISTResources(0,1,seed,False,split,directory)
        self.tasks.append(EMNISTTask(None,evaluation_resources))

    def __len__(self):
        return len(self.tasks)

    def __iter__(self):
        return self.tasks.__iter__()

    def __getitem__(self,k):
        return self.tasks[k]


class EMNISTTrainAnytimeStream(Stream):
    def __init__(self,n_megabatches,seed,split,directory):
        self.tasks=[]
        for k in range(n_megabatches):
            resources=EMNISTResources(k,n_megabatches,seed,True,split,directory)
            self.tasks.append(EMNISTTask(k,resources))

    def __len__(self):
        return len(self.tasks)

    def __iter__(self):
        return self.tasks.__iter__()

    def __getitem__(self,k):
        return self.tasks[k]
