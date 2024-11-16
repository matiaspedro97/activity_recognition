from typing import Type
from loguru import logger

import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from umap.umap_ import UMAP

from src.embeddings import GenericEmbedder, GenericNorm
from src.embeddings.scale import GlobalNorm


class UMAPEmbedder(GenericEmbedder):
    def __init__(self, scaler: str = None, seed: int = 42, **kwargs) -> None:
        super().__init__(**kwargs)

        # Encoder
        try:
            # try instance with provided params
            self.emb_obj = UMAP(n_jobs=1, **kwargs)
        except Exception as e:
            logger.debug(f"Error:\n{e}")

            # default instance (in case of any error)
            self.emb_obj = UMAP(
                n_neighbors=15,
                n_components=3,
                metric="cosine",
                n_jobs=1,
            )

        # set random state
        self.emb_obj.random_state = seed

        # Scaler
        if isinstance(scaler, str):
            self.scaler_obj = GlobalNorm(n_type=scaler, **kwargs)
        else:
            self.scaler_obj = GenericNorm()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        force_all_finite: bool = "allow-nan",
    ) -> None:
        # fit scaler and scale X
        X_n = self.scaler_obj.fit_transform(X, y)

        # fit encoder with normalized X
        self.emb_obj.fit(X_n, y, force_all_finite)
        return self

    def transform(
        self,
        X: np.ndarray,
        force_all_finite: bool = "allow-nan",
    ) -> np.ndarray:

        # transform
        X_n = self.scaler_obj.transform(X)

        return self.emb_obj.transform(X_n, force_all_finite)

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        force_all_finite: bool = "allow-nan",
    ) -> np.ndarray:

        # fit scaler and transform
        X_n = self.scaler_obj.fit_transform(X, y)

        return self.emb_obj.fit_transform(X_n, y, force_all_finite)


class PCAEmbedder(GenericEmbedder):
    def __init__(self, scaler: str = None, seed: int = 42, **kwargs):
        super().__init__(**kwargs)

        # Encoder
        try:
            self.emb_obj = PCA(**kwargs)
        except Exception as e:
            logger.debug(f"Error: {e}")
            self.emb_obj = PCA()

        # Scaler
        if isinstance(scaler, str):
            self.scaler_obj = GlobalNorm(n_type=scaler, **kwargs)
        else:
            self.scaler_obj = GenericNorm()

        # set random state
        self.emb_obj.random_state = seed

    def fit(self, X, y=None):
        # fit scaler and scale X
        X_n = self.scaler_obj.fit_transform(X, y)

        # fit encoder with normalized X
        self.emb_obj.fit(X_n, y)
        return self

    def transform(self, X):
        X_n = self.scaler_obj.transform(X)
        return self.emb_obj.transform(X_n)

    def fit_transform(self, X, y=None):
        X_n = self.scaler_obj.fit_transform(X, y)
        return self.emb_obj.fit_transform(X_n, y)


class LLEEmbedder(GenericEmbedder):
    def __init__(self, scaler: str = None, seed: int = 42, **kwargs):
        super().__init__(**kwargs)

        # Encoder
        try:
            self.emb_obj = LocallyLinearEmbedding(**kwargs)
        except Exception as e:
            logger.debug(f"Error: {e}")
            self.emb_obj = LocallyLinearEmbedding()

        # Scaler
        if isinstance(scaler, str):
            self.scaler_obj = GlobalNorm(n_type=scaler, **kwargs)
        else:
            self.scaler_obj = GenericNorm()

        # set random state
        self.emb_obj.random_state = seed

    def fit(self, X, y=None):
        # fit scaler and scale X
        X_n = self.scaler_obj.fit_transform(X, y)

        # fit encoder with normalized X
        self.emb_obj.fit(X_n, y)
        return self

    def transform(self, X):
        # transform
        X_n = self.scaler_obj.transform(X)
        return self.emb_obj.transform(X_n)

    def fit_transform(self, X, y=None):
        # fit and transform
        X_n = self.scaler_obj.fit_transform(X, y)
        return self.emb_obj.fit_transform(X_n, y)


class IsomapEmbedder(GenericEmbedder):
    def __init__(self, scaler: str = None, seed: int = 42, **kwargs):
        super().__init__(**kwargs)

        # Encoder
        try:
            self.emb_obj = Isomap(**kwargs)
        except Exception as e:
            print(f"Error: {e}")
            self.emb_obj = Isomap()

        # Scaler
        if isinstance(scaler, str):
            self.scaler_obj = GlobalNorm(n_type=scaler, **kwargs)
        else:
            self.scaler_obj = GenericNorm()

        # set random state
        self.emb_obj.random_state = seed

    def fit(self, X, y=None):
        # fit scaler and scale X
        X_n = self.scaler_obj.fit_transform(X, y)

        # fit encoder with normalized X
        self.emb_obj.fit(X_n, y)
        return self

    def transform(self, X):
        # transform
        X_n = self.scaler_obj.transform(X)
        return self.emb_obj.transform(X_n)

    def fit_transform(self, X, y=None):
        # fit and transform
        X_n = self.scaler_obj.fit_transform(X, y)
        return self.emb_obj.fit_transform(X_n, y)