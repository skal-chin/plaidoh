import pandas as pd
import numpy as np

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

class ClassicalPreprocessor():
    def __init__(self,
                 data : pd.DataFrame = None,
                 preprocess_param : Dict = {},
                 ):
        pass