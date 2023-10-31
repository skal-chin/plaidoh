import numpy as np
import pandas as pd

from abstract_optimizer import (
    AbstractOptimizer,
)

from typing import (
    List,
    Dict,
    Tuple,
)

from sklearn.model_selection import (
    RandomizedSearchCV,
    ParameterGrid,
)

class ClassicalOptimizer(AbstractOptimizer):
    def __init__(self,
                 data : pd.DataFrame = None,
                 target_column : str = None,
                 models : List = [],
                 optimized_models : Dict = {},
                 ):
        pass

    def optimize(self):
        """
        performs the optimezation of the models.
        """
        pass
