from loguru import logger
from src.embeddings import GenericNorm

from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    MaxAbsScaler
)


class StandardNorm(GenericNorm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.scaler_obj = StandardScaler(**kwargs)
        except Exception as e:
            print(f"Error: {e}")
            self.scaler_obj = StandardScaler()

    def fit(self, X, y=None):
        self.scaler_obj.fit(X, y)
        return self

    def transform(self, X):
        return self.scaler_obj.transform(X)

    def fit_transform(self, X, y=None):
        return self.scaler_obj.fit_transform(X, y)

    def inverse_transform(self, X, y=None):
        return self.scaler_obj.inverse_transform(X, y)


class RobustNorm(GenericNorm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.scaler_obj = RobustScaler(**kwargs)
        except Exception as e:
            print(f"Error: {e}")
            self.scaler_obj = RobustScaler()

    def fit(self, X, y=None):
        self.scaler_obj.fit(X, y)
        return self

    def transform(self, X):
        return self.scaler_obj.transform(X)

    def fit_transform(self, X, y=None):
        return self.scaler_obj.fit_transform(X, y)

    def inverse_transform(self, X, y=None):
        return self.scaler_obj.inverse_transform(X, y)

class MinMaxNorm(GenericNorm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.scaler_obj = MinMaxScaler(**kwargs)
        except Exception as e:
            print(f"Error: {e}")
            self.scaler_obj = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler_obj.fit(X, y)
        return self

    def transform(self, X):
        return self.scaler_obj.transform(X)

    def fit_transform(self, X, y=None):
        return self.scaler_obj.fit_transform(X, y)

    def inverse_transform(self, X, y=None):
        return self.scaler_obj.inverse_transform(X, y)

class MaxAbsNorm(GenericNorm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.scaler_obj = MaxAbsScaler(**kwargs)
        except Exception as e:
            print(f"Error: {e}")
            self.scaler_obj = MaxAbsScaler()

    def fit(self, X, y=None):
        self.scaler_obj.fit(X, y)
        return self

    def transform(self, X):
        return self.scaler_obj.transform(X)

    def fit_transform(self, X, y=None):
        return self.scaler_obj.fit_transform(X, y)
    
    def inverse_transform(self, X, y=None):
        return self.scaler_obj.inverse_transform(X, y)   
    

class GlobalNorm:
    N_MAP = {
        'maxabs': MaxAbsNorm,
        'minmax': MinMaxNorm,
        'standard': StandardNorm,
        'robust': RobustNorm,
        'identity': GenericNorm
    }
    
    def __init__(self, n_type: str = 'standard', **kwargs):
        try:
            # tag
            norm_tag = n_type.lower() if isinstance(n_type, str) else None
            
            # norm class
            cls_ = self.N_MAP.get(norm_tag, GenericNorm)
            
            # norm instance
            self.scaler_obj = cls_(**kwargs)
        except Exception as e:
            logger.info(f"Error: {e}")
            self.scaler_obj = GenericNorm()

    def fit(self, X, y=None):
        self.scaler_obj.fit(X, y)
        return self

    def transform(self, X):
        return self.scaler_obj.transform(X)

    def fit_transform(self, X, y=None):
        return self.scaler_obj.fit_transform(X, y)   
    
    def inverse_transform(self, X):
        return self.scaler_obj.inverse_transform(X)     
    