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

class TransformerFactory():
    """
    Factory class for creating transformers.
    """
    def __init__(self) -> None:
        pass

    def get_transformer(self, name : str) -> object:
        """
        Returns a transformer object based on the name.

        Parameters
        ----------
        name : str
            the name of the transformer to be created

        Returns
        -------
        transformer : object
            the transformer object
        """
        if name == 'standard_scaler':
            return StandardScaler()
        elif name == 'min_max_scaler':
            return MinMaxScaler()
        elif name == 'robust_scaler':
            return RobustScaler()
        elif name == 'max_abs_scaler':
            return MaxAbsScaler()
        elif name == 'normalizer':
            return Normalizer()
        elif name == 'one_hot_encoder':
            return OneHotEncoder()
        elif name == 'ordinal_encoder':
            return OrdinalEncoder()
        elif name == 'simple_imputer':
            return SimpleImputer()
        elif name == 'iterative_imputer':
            return IterativeImputer()
        else:
            raise ValueError('Invalid transformer name given.') 