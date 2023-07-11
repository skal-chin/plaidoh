from abc import ABC, abstractmethod

class AbstractOptimizer(ABC):

    @abstractmethod
    def optimize(self):
        pass