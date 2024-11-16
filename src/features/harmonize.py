import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

from src.embeddings.scale import GlobalNorm
from src.embeddings.embed import UMAPEmbedder, PCAEmbedder, LLEEmbedder, IsomapEmbedder


class FeatureHarmonizer:
    IMP_MAP = {
        'simple': SimpleImputer,
        'iterative': IterativeImputer,
        'knn': KNNImputer
    }

    ENC_MAP = {
        'umap': UMAPEmbedder,
        'pca': PCAEmbedder,
        'isomap': IsomapEmbedder,
        'lle': LLEEmbedder
    }

    def __init__(
            self, 
            missing_codes: int = None, 
            impute_type: str = None, 
            encoder_type: str = None, 
            norm_type: str = None, 
            imp_kwargs: dict = {},
            enc_kwargs: dict = {}, 
        ) -> None:

        # settings
        self.m_codes = missing_codes

        # get imputer class
        imp_cls = self.IMP_MAP.get(impute_type, None)

        # get encoder class
        enc_cls = self.ENC_MAP.get(encoder_type, None)

        # instance of imputer objects
        if imp_cls is None:
            self.imp = SimpleImputer()
            self.norm = GlobalNorm('identity')
        else:
            self.imp = imp_cls(**imp_kwargs)
            self.norm = GlobalNorm(norm_type)

        # order of columns to be tranformed
        self.imp_cols = None

        # instance of encoder objects
        if enc_cls is None:
            self.enc = UMAPEmbedder(scaler='norm')
        else:
            self.enc = enc_cls(**enc_kwargs)

        # order of columns to be tranformed
        self.enc_cols = None

    def impute_fit_transform(self, X: pd.DataFrame):
        # scale
        X_t = self.norm.fit_transform(X)

        # impute
        X_imp = self.imp.fit_transform(X_t)

        # reverse norm
        X_inv = self.norm.inverse_transform(X_imp)

        # assign columns
        self.imp_cols = list(X.columns)

        # assign to dataframe
        X_inv = pd.DataFrame(X_inv, columns=self.imp_cols)

        return X_inv

    def impute_transform(self, X: pd.DataFrame):
        # checkpoint
        assert list(X.columns) == self.imp_cols, \
        "Columns of X do not match columns of training data"

        # scale
        X_t = self.norm.transform(X)

        # impute
        X_imp = self.imp.transform(X_t)

        # reverse norm
        X_inv = self.norm.inverse_transform(X_imp)

        # assign to dataframe
        X_inv = pd.DataFrame(X_inv, columns=self.imp_cols)

        return X_inv

    def encode_fit_transform(self, X: pd.DataFrame, **kwargs):
        # encode
        X_t = self.enc.fit_transform(X, **kwargs)

        # assign columns
        self.enc_cols = list(X.columns)

        # new columns
        new_cols = [f"emb_{x+1}" for x in range(X_t.shape[-1])]

        # assign to dataframe
        X_t = pd.DataFrame(X_t, columns=new_cols)

        return X_t
    
    def encode_transform(self, X: pd.DataFrame, **kwargs):
        # checkpoint
        assert list(X.columns) == self.enc_cols, \
        "Columns of X do not match columns of training data"

        # encode
        X_t = self.enc.transform(X, **kwargs)

        # new columns
        new_cols = [f"emb_{x+1}" for x in range(X_t.shape[-1])]

        # assign to dataframe
        X_t = pd.DataFrame(X_t, columns=new_cols)

        return X_t
