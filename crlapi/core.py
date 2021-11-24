import copy


class TaskResources:
    """ Describe resources for a task (e.g a dataset, and environments, etc...)
    """
    def make(self):
        raise NotImplementedError


class Task:
    """ Describe a task to solve with a task descriptor, and associated ressources
    """
    def task_descriptor(self):
        raise NotImplementedError

    def task_resources(self):
        raise NotImplementedError


class CLModel:
    """ A continual learning model that is updated on different tasks. Such a model can evaluate itself on a particular task
    """
    def __init__(self, config):
        self.config = config

    def update(self, task, logger):
        # return a clmodel
        raise NotImplementedError

    def evaluate(self, task,logger,**evaluation_args):
        raise NotImplementedError

class Stream:
    """ A stream of tasks
    """
    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __getitem__(self,k):
        raise NotImplementedError
