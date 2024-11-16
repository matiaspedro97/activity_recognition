import os
import json
import lightgbm
import sklearn.preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from loguru import logger

from sklearn.model_selection import (
    StratifiedKFold, 
    GridSearchCV, 
    train_test_split, 
    StratifiedGroupKFold, 
    GroupKFold
)

from sklearn.feature_selection import (
    RFE, 
    SequentialFeatureSelector, 
    SelectKBest,
    f_classif,
    chi2
)

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline

from src.config import (
    classifier_path,
    selector_path, 
    scaler_path
)


class ModelTrainer:
    def __init__(
            self, 
            model_tag: str = 'RF', 
            norm_tag: str = 'identity', 
            sel_tag: str = 'RFE', 
        ):

        # settings
        ####################
        self.model_tag = model_tag
        self.norm_tag = norm_tag 
        self.sel_tag = sel_tag

        # classifiers
        ####################
        self.cls_model = {}
        self.grid_cls = {}

        self.cls_ = None    

        # feature scalers
        ###################
        self.norm = {}
        self.norm_ = None

        # feature selectors
        ###################
        self.ft_sel = {}
        self.grid_sel = {}

        self.sel_ = None

        # load grid search options
        self.load_options()

        # exec imports
        self.exec_imports()

        # fill in parameters
        self.fill_clf_params()
        self.fill_norm_params()
        self.fill_select_params()

    def load_options(self):
        # classifiers
        ####################
        try:
            self.cls_ = dict(
                json.load(
                    open(classifier_path)
                )
            )
        except Exception as e:
            logger.info(e)
            self.cls_ = {}        

        # feature scalers
        ###################
        try:
            self.norm_ = dict(
                json.load(
                    open(scaler_path)
                )
            )
        except Exception as e:
            logger.info(e)
            self.norm_ = {}

        # feature selectors
        ###################
        try:
            self.sel_ = dict(
                json.load(
                    open(selector_path)
                )
            )
        except Exception as e:
            logger.info(e)
            self.sel_ = {}
    
    def exec_imports(self):
        exec(self.sel_.get(self.sel_tag, {}).get('_imports', ""))
        exec(self.cls_.get(self.model_tag, {}).get('_imports', ""))
        exec(self.norm_.get(self.norm_tag, {}).get('_imports', ""))

    def fill_norm_params(self):
        # import scalers
        self.norm = {
            k: eval(it['_class'])(**it['_params']) 
            for k, it in self.norm_.items() if k == self.norm_tag
        }

    def fill_clf_params(self):
        # import classifiers
        for k, it in self.cls_.items():
            # model tag
            if k == self.model_tag:
                exec(it['_imports'])

                # load model class
                self.cls_model[k] = eval(it['_class'])(**it['_params'])

                # load grid search params
                try:
                    self.grid_cls[k] = {k2: eval(it2) if isinstance(it2, str) else it2
                                        for k2, it2 in it['_grid'].items()}
                except KeyError as e:
                    logger.info(e)
                    self.grid_cls[k] = None

    def fill_select_params(self):
        # import selectors
        for k, it in self.sel_.items():
            if k == self.sel_tag:
                exec(it['_imports'])

                # load feature selector class
                self.ft_sel[k] = eval(it['_class'])(**{k: (eval(it) if '(' in str(it) else it) 
                                                        for k, it in it['_params'].items()})
                
                # load grid search params
                try:
                    self.grid_sel[k] = {k2: eval(it2) if isinstance(it2, str) else it2
                                        for k2, it2 in it['_grid'].items()}
                except KeyError as e:
                    logger.info(e)
                    self.grid_sel[k] = {}

    def gridcv_train(
            self, 
            X: np.ndarray, 
            y: np.ndarray, 
            scoring: str = 'f1_macro', 
            cv: int = 5,
            groups: np.ndarray = None, 
            verbose: int = 2, 
            **kwargs
    ):
        
        # get optimization mappers
        # classifiers
        cls_m, grid_cls = self.cls_model[self.model_tag], self.grid_cls[self.model_tag]
        
        # selector
        sel, grid_sel = self.ft_sel[self.sel_tag], self.grid_sel[self.sel_tag]

        # scaler
        norm = self.norm[self.norm_tag]

        # checkpoint
        #############################################################################################
        assert np.prod(np.array([cls_m, grid_cls, sel, grid_sel, norm]) != None), \
            "Some of your given tags is not defined. Please check that before calling this function."
        #############################################################################################
	
	    # pipeline
        pipe = Pipeline(steps=[('scaler', norm), ('sel', sel), ('clf', cls_m)])
	
	    # define grid maps
        new_grid_clf = {f'clf__{k}': it for k, it in grid_cls.items()}
        new_grid_sel = {f'sel__{k}': it for k, it in grid_sel.items()}

        new_grid = {**new_grid_clf, **new_grid_sel}  # merge both grid params

	    # instantiate grid search object
        clf_grid = GridSearchCV(
            estimator=pipe, 
            param_grid=new_grid, 
            scoring=scoring,
            refit=True, 
            cv=cv, 
            return_train_score=True, 
            n_jobs=-1, 
            verbose=verbose, 
            error_score='raise'
        )
        
        # fit (training)
        clf_grid.fit(X, y, groups=groups)

        return clf_grid
        
        
        
if __name__ == '__main__': 
    # example data
    # features array
    X = np.random.random((500, 20))
    
    # disease array
    y = np.array([0]*250 + [1]*250)
    
    # patient array
    pat = np.array([0]*40 + [1]*60 + [2]*200 + [3]*100 + [4]*50 + [5]*50)

    # trainer instance
    m_gen = ModelTrainer(
        model_tag='RF_comp', 
        norm_tag='identity', 
        sel_tag='identity'
    )

    # splitter 
    cv = StratifiedGroupKFold(5)  # 80 % (train) | 20 % (test)

    # launch training
    grid_cv = m_gen.gridcv_train(X=X, y=y, scoring='f1_macro', cv=cv, groups=pat, verbose=2)

