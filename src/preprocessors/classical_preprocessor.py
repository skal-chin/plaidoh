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

from types import (
	FunctionType,
)

from sklearn.preprocessing import (
	StandardScaler,
	MinMaxScaler,
	RobustScaler,
	MaxAbsScaler,
	Normalizer,
	OneHotEncoder,
	OrdinalEncoder,
	KBinsDiscretizer
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
		self.__preprocessed_data = self.__process_data(self.__data, self.__transformer)

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
			{COLUMN_NAME1 : ['nominal'], ..., COLUMN_NAMEn : ['interval']}.
			If the column is labeled nominal, an ignore tag can be included to ignore the column such as
				{COLUMN_NAME1 : ['nominal', 'ignore']}.
			If the column is labeled ordinal, the order of the categories can be specified such as
				{COLUMN_NAME1 : ['ordinal', ['low', 'medium', 'high']]}. Other wise it will default to ignore.
			If the column is labeled ratio or interval, number of bins can be suggested and the data will be binned equally such as
				{COLUMN_NAME1 : ['ratio', 5]}. Or bins can be specified such as
				{COLUMN_NAME1 : ['ratio', [0, 10, 20, 30, 40, 50]]}. If the number is binned, they will be treated to
				an ordinal encoding method. Otherwise, the column will be treated as continuous and default to StandardScaler.
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
			raise Exception('Could not build a transformer. Provide either column_dtypes or preprocess_param')
		
		generated_transformers = []


		if preprocess_param is not None:
			for column_name, transformers in preprocess_param.items():

				if column_name not in data.columns:
					continue

				for transformer in transformers:

					if isinstance(transformer, str):

						#TODO: add get transformer module
						new_transformer = get_transformer(transformer)

					else:
						new_transformer = transformer

					name = self.__generate_transformer_name(column_name, new_transformer)

					generated_transformers.append((name, new_transformer, [column_name]))

		if column_dtypes is not None:
			for column_name, column_type in column_dtypes.items():

				if column_name not in data.columns:
					continue

				if isinstance(column_type, list) is False:
					raise TypeError('column_type must be a list')
				
				if column_type[0] == 'nominal':

					if len(column_type) == 2 and column_type[1] == 'ignore':
						continue

					new_transformer = OneHotEncoder(sparse=False, handle_unknown='ignore')
					name = self.__generate_transformer_name(column_name, new_transformer)

					generated_transformers.append((name, new_transformer, [column_name]))

				elif column_type[0] == 'ordinal':

					if len(column_type) == 1:
						continue

					new_transformer = OrdinalEncoder(categories=[column_type[1]])
					name = self.__generate_transformer_name(column_name, new_transformer)

					generated_transformers.append((name, new_transformer, [column_name]))

				elif column_type[0] == 'ratio' or column_type[0] == 'interval':

					if len(column_type) == 1:

						new_transformer = StandardScaler()
						name = self.__generate_transformer_name(column_name, new_transformer)

						generated_transformers.append((name, new_transformer, [column_name]))

					elif isinstance(column_type[1], int):

						new_transformer = KBinsDiscretizer(n_bins=column_type[1], encode='ordinal', strategy='uniform')
						name = self.__generate_transformer_name(column_name, new_transformer)

						generated_transformers.append((name, new_transformer, [column_name]))

					elif isinstance(column_type[1], np.ndarray) and all(isinstance(x, (int, float)) for x in column_type[1]):

						new_transformer = KBinsDiscretizer(n_bins=column_type[1], encode='ordinal', strategy='uniform')
						name = self.__generate_transformer_name(column_name, new_transformer)

						generated_transformers.append((name, new_transformer, [column_name]))

					else:

						new_transformer = StandardScaler()
						name = self.__generate_transformer_name(column_name, new_transformer)

						generated_transformers.append((name, new_transformer, [column_name]))

		generated_transformers.append((self.__target_column, 'passthrough', [self.__target_column]))
		transformer = ColumnTransformer(generated_transformers, remainder='drop')
		return transformer
				
	def __process_data(self, data : pd.DataFrame = None, transformer : ColumnTransformer = None, clean_data : bool = False) -> pd.DataFrame:
		"""
		processes the data in respect of the created transformer.

		Parameters
		----------
		data : pd.DataFrame
			the data to be processed
		transformer : ColumnTransformer
			the transformer to be used
		clean_data : bool
			whether or not to clean the data

		Returns
		-------
		processed_data : pd.DataFrame
			the processed data
		"""
		working_data = data.copy()
		if clean_data is True:
			working_data = self.__clean_missing_values(working_data).copy()
			working_data = self.__clean_outliers(working_data).copy()
			working_data = self.__clean_duplicates(working_data).copy()

		transformer.set_output('pandas')

		processed_data = transformer.fit_transform(working_data)
		return processed_data

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

	def __generate_transformer_name(self, column_name : str = None, transformer = None) -> str:
		"""
		generates a name for the transformer based on the column name and the transformer.
		
		Parameters
		----------
		column_name : str
			the name of the column
		transformer : None
			the transformer to be used
			The reason why the transfomer parameter does not restrict type is because there are
			many classes that can be used as a transformer.

		Returns
		-------
		name : str
		"""

		if column_name is None or transformer is None:
			raise Exception('column_name or transformer cannot be None')
		
		transformer_name = transformer.__str__().replace('()', '')
		name = column_name + '_' + transformer_name

		return name

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