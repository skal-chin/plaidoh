import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessors.abstract_preprocessor import (
    AbstractPreprocessor,
)

from optimizers.abstract_optimizer import (
    AbstractOptimizer,
)

from typing import (
    List,
    Dict,
    Tuple,
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

import mysql.connector

class Plaidoh:
    def __init__(
            self,
            data : pd.DataFrame or str = None,
            model_type : str = 'classification',
            models : List = [],
            model_metrics : List = [],
            # analyses : List = [],
            # visualizations : List = [],
            optimize : bool = False,
            preprocess_param : Dict = {},
            column_dtypes : Dict = {},
            train_size : float = None,
            target_name : str = None,
            optimizer : AbstractOptimizer = None,
            preprocessor : AbstractPreprocessor = None,
            ):
        pass


    def score(self, models : List = []) -> pd.DataFrame:
        """
        scores the models based on the model metrics. If the metrics given in the list are not
        available for the model, it will be ignored. If no metrics are given, the default metrics for
        the model will be used.

        Parameters
        ----------
        models : List
            List of models to score. If no models are given, all models will be scored.

        Returns
        -------
        scores : pd.DataFrame
            DataFrame containing the scores for each model.

        @skal-chin
        """
        pass

    def test_train_split(self, train_size : float = None) -> None:
        """
        splits the data into training and testing sets.

        Parameters
        ----------
        train_size : float
            The size of the training set. If no size is given, the default size will be used. Sets
            the split data to the Plaidoh object.

        Returns
        -------
        None

        @skal-chin
        """
        pass

    
    def train(self, models : List = []) -> None:
        """
        trains the models on the training data.

        Parameters
        ----------
        models : List
            List of models to train. If no models are given, all models will be trained.

        Returns
        -------
        None
            The models will be set to the Plaidoh object.

        @skal-chin
        """
        pass

    
    def predict(self, models : List = []) -> Dict:
        """
        predicts the target values for the testing data.

        Parameters
        ----------
        models : List
            List of models to predict. If no models are given, all models will be predicted.

        Returns
        -------
        predictions : Dict
            Dictionary containing the predictions for each model.

        @skal-chin
        """
        pass

    def optimize(self, models : List = []) -> None:
        """
        optimizes the models based using random search. This is called during the Plaidoh
        initializer if optimize is set to true. 

        Parameters
        ----------
        models : List
            List of models to optimize. If no models are given, all models will be optimized.

        Returns
        -------
        None
            The models will be set to the Plaidoh object.

        @skal-chin
        """
        pass

    def visualize(self, models : List = [], visualizations : Dict = {}):
        """
        visualizes the models using the given visualizations. If no visualizations are
        given, none will be created. If the viusalization is not available for the model, it will be
        ignored.

        Parameters
        ----------
        models : List
            List of models to visualize. If no models are given, all models will be visualized.
        visualizations : Dict
            Dictionary of visualizations to create. If no visualizations are given, none will be
            created.

        Returns
        -------
        None
            The visualizations will be saved to the Plaidoh object.

        @skal-chin
        """
        pass

    def export_data(self, export_path : str = None) -> None:
        """
		exports the preprocessed data to a csv file.

		Parameters
		----------
		export_path : str
			the path to export the data to

		Returns
		-------
		None
		"""
        data = self.preprocessor.get_preprocessed_data()
        
        if data is None:
            raise ValueError("No data to export.")
        
        data.to_csv(export_path, index=False)
        return

    def export_data_to_db(self, db_config : Dict = {}) -> None:
        """
		exports the data to a database.

		Parameters
		----------
		db_config : Dict
			the configuration for the database as shown
			{host : HOST_NAME,
			user : USER_NAME,
			password : PASSWORD,
			db : DATABASE_NAME}

		Returns
		-------
		None
		"""
        data = self.preprocessor.get_preprocessed_data()

        if data is None:
            raise ValueError("No data to export.")
        
        mydb = mysql.connector.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password']
        )

        mycursor = mydb.cursor()

        mycursor.execute("CREATE DATABASE IF NOT EXISTS " + db_config['db'])

        mydb = mysql.connector.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['db']
        )

        column_names = data.columns.values.tolist()
        column_names = [column_name.replace(' ', '_') for column_name in column_names]
        column_names = [column_name.replace('-', '_') for column_name in column_names]

        column_types = data.dtypes.values.tolist()

        for i in range(len(column_types)):
            if column_types[i] == 'int64':
                column_types[i] = 'INT'
            elif column_types[i] == 'float64':
                column_types[i] = 'FLOAT'
            else:
                column_types[i] = 'VARCHAR(255)'

        column_names_and_types = [column_names[i] + ' ' + column_types[i] for i in range(len(column_names))]
        column_names_and_types = ', '.join(column_names_and_types)

        mycursor.execute("CREATE TABLE IF NOT EXISTS " + db_config['db'] + ".data (" + column_names_and_types + ")")

        for i in range(len(data)):
            row = data.iloc[i].values.tolist()
            row = [str(value) for value in row]
            row = ', '.join(row)
            mycursor.execute("INSERT INTO " + db_config['db'] + ".data VALUES (" + row + ")")

        mydb.commit()

        return

    def export_models(self, models : List = [], path : str = None) -> None:
        """
        exports the models to the given path.

        Parameters
        ----------
        models : List
            List of models to export. If no models are given, all models will be exported.
        path : str
            Path to export the models to. If no path is given, the models will be exported to the
            working directory.

        Returns
        -------
        None
            The models will be exported to the given path.

        @skal-chin
        """
        pass

    
        

