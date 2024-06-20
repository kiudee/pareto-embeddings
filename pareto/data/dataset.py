from abc import ABCMeta, abstractmethod

from pareto.util import check_random_state


class AbstractDataset(object, metaclass=ABCMeta):
    def __init__(self, *args, random_state=None, **kwargs):
        self.random_state = check_random_state(random_state)

    @abstractmethod
    def get_xy(self):
        raise NotImplementedError


class AbstractGenerator(AbstractDataset, metaclass=ABCMeta):
    def __init__(
        self, *args, n_features=None, n_instances=None, n_objects=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.n_features = n_features
        self.n_instances = n_instances
        self.n_objects = n_objects

    @abstractmethod
    def get_xy(self, **kwargs):
        raise NotImplementedError
