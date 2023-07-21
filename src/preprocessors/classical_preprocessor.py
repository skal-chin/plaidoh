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
    Pipeline,
    make_pipeline,
)

from sklearn.compose import (
    ColumnTransformer,
    make_column_transformer,
)

class ClassicalPreprocessor(AbstractPreprocessor):

    __data = None
    __target_column = None
    __column_dtypes = {}
    __preprocess_param = {}
    __transformer = ColumnTransformer
    __pipeline = Pipeline
    __preprocessed_data = None


    def __init__(
            self,
            data : pd.DataFrame = None,
            target_column : str = None,
            column_dtypes : Dict = {},
            preprocess_param : Dict = {},
            ):
        
        if data is None:
            raise Exception('data cannot be None')
        
        if data is not type(pd.DataFrame):
            raise TypeError('data must be of type pd.DataFrame')
        
        if column_dtypes is None or preprocess_param is None:
            raise Exception('column_dtypes or preprocess_param must be populated')
        
        self.__data = data
        self.__target_column = target_column
        self.__column_dtypes = column_dtypes
        self.__preprocess_param = preprocess_param

        self.__transformer = self.__build_transformer(self.__data, self.__column_dtypes, self.__preprocess_param)
        self.__pipeline = self.__create_pipeline(self.__data, self.__transformer)
        self.__preprocessed_data = self.__process_pipeline(self.__data, self.__pipeline)

    def __build_transformer(self, data : pd.DataFrame = None, column_dtypes : Dict = {}, preprocess_param : Dict = {}) -> ColumnTransformer:
        """
        builds the transformers for the given data. If both are not none, the function will use the 
        preprocess_param first and then the column_dtypes. If a single column is used multiple times, 
        the transformers will be applied in the order they are given.

        Parameters
        ----------
        data : pd.DataFrame
            the data to be processed
        column_dtypes : Dict
            the datatypes for each column, where the key is the column name and
            the value is the column type in the form of nominal, ordinal, ratio, or interval
            {COLUMN_NAME1 : 'nominal', ..., COLUMN_NAMEn : 'interval'}
        preprocess_param : Dict
            the preprocessing parameters defined by the user, where the key is the column name and 
            the value is a list of transformers to be used on the column in the form of a function or a string
            {COLUMN_NAME1 : [TRANSFORMER1(), ..., 'TRANSFORMERn'], ..., COLUMN_NAMEn : [TRANSFORMER1(), ..., 'TRANSFORMERn']}

        Returns
        -------
        transformers : ColumnTransformer
            the transformers for the data
        """
        if column_dtypes is None and preprocess_param is None:
            print('''Could not create a transformer for the data. \n
                    Please provide either a column_dtypes as such: \n
                    {\'column_name\' : \'column_type\'} \n
                    using nominal, ordinal, or continuous as the column_type \n
                    or a preprocess_param as such: \n
                    {\'column_name\' : [TRANSFORMER1(), \'TRANSFORMER2\']} \n
                    using preprocessors in the form of a function or a string \n

                ''')
            return None
        

        pass
    
    def __create_pipeline(self, data : pd.DataFrame = None, transformer : ColumnTransformer = None) -> Pipeline:
        """
        creates a processing pipeline for the given data from a given transformer.

        Parameters
        ----------
        data : pd.DataFrame
            the data to be processed
        transformer : ColumnTransformer
            the transformer to be used


        Returns
        -------
        pipeline : Pipeline
            the pipeline for the data
        """
        pass

    def __process_pipeline(self, data : pd.DataFrame = None, pipeline : Dict = {}) -> pd.DataFrame:
        """
        processes the defined pipeline for the data.

        Parameters
        ----------
        data : pd.DataFrame
            the data to be processed
        pipeline : Dict
            the pipeline to be used

        Returns
        -------
        processed_data : pd.DataFrame
            the processed data
        """
        pass

    def __detect_type(self, data : pd.DataFrame = None, column_name : str = None) -> str:
        """
        detects the type of data within the given column

        Parameters
        ----------
        data : pd.DataFrame
            the data to be processed
        column_name : str
            the name of the column

        Returns
        -------
        column_type : str
            the type of the column
        """
        return 

    def __clean_missing_values(self, data : pd.DataFrame = None) -> pd.DataFrame:
        """
        cleans missing values using different methods. The method chosen is based on the type of data and 
        the amount of missing data. 

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

    def __clean_outliers(self, data : pd.DataFrame = None) -> pd.DataFrame:
        """
        cleans up the outliers in the data. The method chosen is based on how many outliers are in the data and the size difference between the outliers
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

    def __clean_duplicates(self, data : pd.DataFrame = None) -> pd.DataFrame:
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

    def get_data(self) -> pd.DataFrame:
        return self.__data
    
    def get_column_dtypes(self) -> Dict:
        return self.__column_dtypes
    
    def get_preprocess_param(self) -> Dict:
        return self.__preprocess_param

    def get_preprocessed_data(self) -> pd.DataFrame:
        return self.__preprocessed_data

    def get_target_column(self) -> str:
        return self.__target_column
    
    def get_pipeline(self) -> Dict:
        return self.__pipeline