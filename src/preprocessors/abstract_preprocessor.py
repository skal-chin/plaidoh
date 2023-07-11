from abc import ABC, abstractmethod

class AbstractPreprocessor(ABC):

    @abstractmethod
    def preprocess(self):
        pass