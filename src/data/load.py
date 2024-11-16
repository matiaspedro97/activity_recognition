import pandas as pd
import numpy as np

from typing import List
from sklearn.model_selection import StratifiedGroupKFold

from src.config import har_dataset_path


class DatasetLoader:
    DATASET_MAP = {
        'har': har_dataset_path
    }

    def __init__(
            self,
            data_path: str,
            feature_cols: List[str], 
            meta_cols: List[str], 
            group_cols: List[str],
            target_col: str,
            test_size: float = 0.2
        ) -> None:
        
        # path to the dataset
        if '.' in data_path:
            self.dset_path = data_path
        else:
            self.dset_path = self.DATASET_MAP.get(data_path.lower(), None)

        # feature columns
        self.ft_cols = feature_cols
        
        # metadata columns
        self.meta_cols = meta_cols
        
        # target columns
        self.target = target_col

        # group
        self.group_cols = group_cols
        
        # need for an interaction term
        self.need_interact = True if isinstance(group_cols, list) else False

        # n folds
        self.n_folds = round(1/test_size)

        # missing value replacer (for spliting)
        self.m_code = 'NaN'

        # data
        self.X = None

        # data maps
        self.gp_map = {}
        self.tgt_map = {}

    def get_col_info(self):
        return self.ft_cols, \
            self.meta_cols, \
            self.group_cols + ['group'] if self.need_interact else [self.group_cols], \
            self.target

    def group_stratify_kfold(
            self,
            X: pd.DataFrame, 
            y: np.ndarray, 
            group: np.ndarray, 
            n_folds: int = 5
        ):

        # train/test pairs
        train_list, test_list = [], []

        # cv instance
        cv = StratifiedGroupKFold(n_folds, random_state=42)

        # X (training set) | y (label set) | groups (subject set)
        split = cv.split(X=X, y=y, groups=group)
        
        # loop over multiple splits
        for _ in range(n_folds):
            # train and test indices for the first split
            train_idx, test_idx = next(split)

            # train and test sets splitted subject and labelwise
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            
            # replace NaN values
            X_train.replace(self.m_code, np.nan, inplace=True)
            X_test.replace(self.m_code, np.nan, inplace=True)

            # assign to sets
            train_list.append(X_train)
            test_list.append(X_test)

        return train_list, test_list
    
    def load_data(self):
        # read data
        self.X = pd.read_csv(self.dset_path)

        if self.need_interact:
            aux_ = self.X[self.group_cols]

            # get group
            self.X['group'] = aux_.apply(
                lambda x: "_".join(x.dropna().astype(str)), axis=1
            )

        else:
            self.X['group'] = self.X[self.group_cols]

        # factorize group
        codes, unq = pd.factorize(self.X['group'])

        # assign codes
        self.X['group'] = codes

        # group map
        self.gp_map = dict(zip(range(len(unq)), unq))
            
        # factorize target
        codes, unq = pd.factorize(self.X[self.target])

        # assign codes
        self.X[self.target] = codes

        # target map
        self.tgt_map = dict(zip(range(len(unq)), unq))
    
    def split_data(self):
        # replace NaNs to split
        self.X = self.X.replace(np.nan, self.m_code)

        # train/test splits
        train, test = self.group_stratify_kfold(
            X=self.X, 
            y=self.X[self.target], 
            group=self.X['group'], 
            n_folds=self.n_folds
        )

        return train, test
    
    def get_data(self, split: bool = True):
        # load data
        self.load_data()

        if split:
            # split data
            train, test = self.split_data()
            
            return train, test
        else:
            return self.X
