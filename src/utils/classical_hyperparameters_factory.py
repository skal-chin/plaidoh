from abstract_hyperparameters_factory import AbstractHyperparametersFactory

from typing import (
    List,
    Dict,
)

class ClassicalHyperparametersFactory(AbstractHyperparametersFactory):
    def __init_(self):
        pass

    def get_hyperparameters(self, models : List or str = None) -> Dict:
        """
        retrieves the hyperparameters for the given models

        Parameters
        ----------
        models : List or str
            List of models in string form or a singular string model.

        Returns
        -------
        hyperparameters : Dict
            Dictionary of hyperparameters for the given models.
        """
        hyperparameters = {}

        if models is None:
            return hyperparameters
        
        if isinstance(models, str):
            model = models
            model_hp = self.__get_model_hyperparameters(model)
            hyperparameters[model] = model_hp

            return hyperparameters
        
        for model in models:
            model_hp = self.__get_model_hyperparameters(model)
            hyperparameters[model] = model_hp

        return hyperparameters


    def __get_model_hyperparameters(self, model : str) -> Dict:
        """
        retrieves the hyperparameters for the given model

        Parameters
        ----------
        model : str
            the model in string form

        Returns
        -------
        hyperparameters : Dict
            Dictionary of hyperparameters for the given model.
        """

        if model == 'random_forest_classifier':
            return {
                'n_estimators': [10, 50, 100, 200, 500],
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 5, 10, 20, 50, 100],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2'],
                'bootstrap': [True, False],
            }
        elif model == 'random_forest_regressor':
            return {
                'n_estimators': [10, 50, 100, 200, 500],
                'criterion': ['mse', 'mae'],
                'max_depth': [None, 5, 10, 20, 50, 100],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2'],
                'bootstrap': [True, False],
            }
        elif model == 'gradient_boosting_classifier':
            return {
                'loss': ['deviance', 'exponential'],
                'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
                'n_estimators': [10, 50, 100, 200, 500],
                'criterion': ['friedman_mse', 'mse', 'mae'],
                'max_depth': [None, 5, 10, 20, 50, 100],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2'],
                'warm_start': [True, False],
            }
        elif model == 'gradient_boosting_regressor':
            return {
                'loss': ['ls', 'lad', 'huber', 'quantile'],
                'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
                'n_estimators': [10, 50, 100, 200, 500],
                'criterion': ['friedman_mse', 'mse', 'mae'],
                'max_depth': [None, 5, 10, 20, 50, 100],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2'],
                'warm_start': [True, False],
            }
        elif model == 'linear_regression':
            return {
                'fit_intercept': [True, False],
                'normalize': [True, False],
                'copy_X': [True, False],
            }
        elif model == 'logistic_regression':
            return {
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'C': [0.001, 0.01, 0.1, 0.2, 0.3],
                'fit_intercept': [True, False],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'max_iter': [100, 200, 500, 1000, 2000],
            }
        elif model == 'svc':
            return {
                'C': [0.001, 0.01, 0.1, 0.2, 0.3],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                'degree': [1, 2, 3, 4, 5],
                'gamma': ['scale', 'auto'],
                'coef0': [0.0, 0.1, 0.2, 0.3],
                'shrinking': [True, False],
                'probability': [True, False],
                'tol': [0.0001, 0.001, 0.01, 0.1],
                'max_iter': [100, 200, 500, 1000, 2000],
            }
        elif model == 'svr':
            return {
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                'degree': [1, 2, 3, 4, 5],
                'gamma': ['scale', 'auto'],
                'coef0': [0.0, 0.1, 0.2, 0.3],
                'shrinking': [True, False],
                'tol': [0.0001, 0.001, 0.01, 0.1],
                'max_iter': [100, 200, 500, 1000, 2000],
            }
        elif model == 'k_neighbors_classifier':
            return {
                'n_neighbors': [3, 5, 10, 20, 50],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [10, 20, 30, 40, 50],
                'p': [1, 2],
            }
        elif model == 'decision_tree_classifier':
            return {
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'max_depth': [None, 5, 10, 20, 50, 100],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2'],
            }
        elif model == 'decision_tree_regressor':
            return {
                'criterion': ['mse', 'friedman_mse', 'mae'],
                'splitter': ['best', 'random'],
                'max_depth': [None, 5, 10, 20, 50, 100],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2'],
            }
        else:
            raise ValueError('Invalid model name given.')