import abc
from abc import ABC


class BaseModel(ABC):
    @abc.abstractmethod
    def fit_transform(self, graph):
        pass
