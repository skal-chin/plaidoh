from abc import ABC, abstractmethod

class AbstractHyperparametersFactory(ABC):

    @abstractmethod
    def get_hyperparameters(self):
        pass