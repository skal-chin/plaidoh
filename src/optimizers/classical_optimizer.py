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

from utils.model_utils import (
    get_model,
    get_model_params,
)

from sklearn.model_selection import (
    RandomizedSearchCV,
    ParameterGrid,
)

class ClassicalOptimizer(AbstractOptimizer):

    __data : pd.DataFrame = None
    __target_column : str = None
    __models : List = []
    __optimized_models : Dict = {}
    __best_params : Dict = {}

    def __init__(self,
                 data : pd.DataFrame = None,
                 target_column : str = None,
                 models : List = [],
                 ):
        
        if data is None:
            raise Exception("data is None")
        
        if target_column is None:
            raise Exception("target_column is None")
        
        self.__data = data
        self.__target_column = target_column
        self.__models = models        

    def optimize(self):
        """
        performs the optimization of the models.
        """
        self.__optimized_models = {}
        self.__best_params = {}

        for model in self.__models:
            self.__optimized_models[model] = self.__optimize_model(model)
            self.__best_params[model] = self.__optimized_models[model].best_params_

    def __optimize_model(self, model : str) -> RandomizedSearchCV:

        if model not in self.__models:
            raise Exception("model not in models")
        
        model = get_model(model)

        X = self.__data.drop(self.__target_column, axis=1)
        y = self.__data[self.__target_column]

        param_grid = get_model_params(model)
        param_grid = ParameterGrid(param_grid)

        random_search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=10,
            scoring='accuracy',
            n_jobs=-1,
            cv=5,
            verbose=3,
            random_state=42,
        )

        random_search.fit(X, y)

        return random_search

    def get_best_params(self, model : str = None) -> Dict:
        """
        returns the best parameters for the model.
        """
        if model is None or model not in self.__models:
            print(f'{model} is not in the given models. Returning all models')
            for m in self.models:
                print(f'{m}')
            return self.__best_params
        
        return self.__best_params[model]
    
    def get_optimized_model(self, model : str = None) -> RandomizedSearchCV:
        """
        returns the optimized model.
        """
        if model is None or model not in self.__models:
            print(f'{model} is not in the given models. Returning all models')
            for m in self.models:
                print(f'{m}')

            return self.__optimized_models
        
        return self.__optimized_models[model]