import pandas as pd
import numpy as np

from abstract_preprocessor import (
    AbstractPreprocessor,
)

from typing import (
    List,
    Dict,
    Tuple,
)

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
)

from sklearn.impute import (
    SimpleImputer,
    IterativeImputer,
)

from sklearn.pipeline import (
    make_pipeline,
)

from sklearn.compose import (
    make_column_transformer,
)

class ClassicalPreprocessor(AbstractPreprocessor):
    def __init__(self,
                 data : pd.DataFrame = None,
                 target_column : str = None,
                 preprocess_param : Dict = {},
                 ):
        pass

    def preprocess(self, data : pd.DataFrame = None) -> pd.DataFrame:
        """
        encapsulates all of the preprocessing steps for classical preprocessing.

        Parameters
        ----------
        data : pd.DataFrame
            the data to be preprocessed

        Returns
        -------
        preprocessed_data : pd.DataFrame
            the preprocessed data
        """
        pass

    def clean_missing_values(self, data : pd.DataFrame = None) -> pd.DataFrame:
        """
        cleans missing values using different methods. The method chosen is based on the type of data, 
        the amount of missing data, and a given method by the user. 

        Parameters
        ----------
        data : pd.DataFrame
            the data to be cleaned

        Returns
        -------
        cleaned_data : pd.DataFrame
            the cleaned data
        """
        pass

    def clean_outliers(self, data : pd.DataFrame = None) -> pd.DataFrame:
        """
        cleans up the outliers in the data. The method chosen is either specified by the user or
        chosen based on how many outliers are in the data and the size difference between the outliers
        and the rest of the data.

        Parameters
        ----------
        data : pd.DataFrame
            the data to be cleaned

        Returns
        -------
            cleaned_data : pd.DataFrame
                the cleaned data
        """
        pass

    def clean_duplicates(self, data : pd.DataFrame = None) -> pd.DataFrame:
        """
        cleans up the duplicates in the data by removal.

        Parameters
        ----------
        data : pd.DataFrame
            the data to be cleaned

        Returns
        -------
        cleaned_data : pd.DataFrame
            the cleaned data
        """
        pass

    def process_pipeline(self, data : pd.DataFrame = None) -> pd.DataFrame:
        """
        processes the user defined pipeline for the data.

        Parameters
        ----------
        data : pd.DataFrame
            the data to be processed

        Returns
        -------
        processed_data : pd.DataFrame
            the processed data
        """
        pass

    def export_data(self, data : pd.DataFrame = None, export_path : str = None) -> None:
        """
        exports the data to a csv file.

        Parameters
        ----------
        data : pd.DataFrame
            the data to be exported
        export_path : str
            the path to export the data to

        Returns
        -------
        None
        """
        pass

    def export_to_db(self, db_config : Dict = {}) -> None:
        """
        exports the data to a database.

        Parameters
        ----------
        db_config : Dict
            the configuration for the database

        Returns
        -------
        None
        """
        pass