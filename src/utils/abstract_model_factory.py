from abc import ABC, abstractmethod

class AbstractModelFactory(ABC):

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def get_model_params(self):
        pass