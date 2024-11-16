import numpy as np


class GenericEmbedder:
    def __init__(self, **kwargs) -> None:
        pass

    def fit(self, X: np.ndarray):
        pass

    def transform(self, X: np.ndarray):
        return X

    def fit_transform(self, X: np.ndarray):
        # fit
        self.fit(X)

        # transform
        X_new = self.transform(X)

        return X_new


class GenericNorm:
    def __init__(self, **kwargs):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X
    
    def inverse_transform(self, X):
        return X