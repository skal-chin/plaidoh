from abc import ABC, abstractmethod

class AbstractPreprocessor(ABC):

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def get_preprocessed_data(self):
        pass