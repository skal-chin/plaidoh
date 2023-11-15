from abstract_model_factory import AbstractModelFactory

from typing import (
    List,
    Dict,
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)

from sklearn.svm import (
    SVC,
    SVR,
)

from sklearn.neighbors import (
    KNeighborsClassifier,
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

class ClassicalModelFactory(AbstractModelFactory):
    def __init_(self):
        pass

    def get_model(self, model : str = None):
        """
        retrieves the model for the given model name

        Parameters
        ----------
        model : str
            the model in string form

        Returns
        -------
        model : sklearn model
            the model
        """
        if model == 'random_forest_classifier':
            return RandomForestClassifier()
        elif model == 'random_forest_regressor':
            return RandomForestRegressor()
        elif model == 'gradient_boosting_classifier':
            return GradientBoostingClassifier()
        elif model == 'gradient_boosting_regressor':
            return GradientBoostingRegressor()
        elif model == 'linear_regression':
            return LinearRegression()
        elif model == 'logistic_regression':
            return LogisticRegression()
        elif model == 'svc':
            return SVC()
        elif model == 'svr':
            return SVR()
        elif model == 'k_neighbors_classifier':
            return KNeighborsClassifier()
        elif model == 'decision_tree_classifier':
            return DecisionTreeClassifier()
        elif model == 'decision_tree_regressor':
            return DecisionTreeRegressor()
        else:
            raise ValueError('Invalid model name given.')

    def get_model_params(self, model : str or List) -> Dict:
        """
        retrieves the hyperparameters for the given model

        Parameters
        ----------
        model : str or List
            the model in string form
            if given a list, it will return a dictionary of hyperparameters

        Returns
        -------
        hyperparameters : Dict
            Dictionary of hyperparameters for the given model.
        """

        if isinstance(model, List):
            model_params = {}

            for m in model:
                model_params[m] = self.get_model_params(m)

            return model_params

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
        
