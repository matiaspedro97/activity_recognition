import json
import pydoc

import pandas as pd

from typing import List
from loguru import logger

from src.pipeline import PipelineEDAGen, PipelineMLGen
from src.models.metric import calculate_metrics, calculate_multiclass_roc_auc

from sklearn.model_selection import StratifiedGroupKFold

class PipelineEDARunner(PipelineEDAGen):
    def __init__(self, config_path: str = None, config_dict: dict = None) -> None:
        # Config path
        self.config_path = config_path

        # load modules
        if isinstance(config_dict, dict):
            config_args = self.load_modules_from_dict(config_dict)
        else:
            config_args = self.load_modules_from_json(config_path)

        # load gen attributes
        super().__init__(**config_args)

    def load_modules_from_json(self, json_path: str):
        config = json.load(open(json_path, 'r'))
        return self.load_modules_from_dict(config)

    def load_modules_from_dict(self, config: dict):    
        # Kwargs
        config_gen_args = {
            k: v 
            for k, v in config.items() 
            if k != 'modules'
        }

        # loading class modules
        for module_name, module in config['modules'].items():
            class_ = pydoc.locate(f"src.{module['class_']}")
            params_ = module['params_']

            try:
                obj = class_(**params_)
                logger.info(f"Module {module_name} successfully loaded")
            except Exception as e:
                logger.info(f"Module {module_name} not loaded correctly." 
                             f"Please check the error:\n{e}")
                obj = None

            # assign to class attribute
            config_gen_args[module_name] = obj
            #exec(f"self.{module_name} = obj")

        return config_gen_args

    def run(self):
        # Train and Test
        data = self.loader.get_data(split=False)

        # Harmonize
        # settings
        feat_cols, meta_cols, gp_cols, tgt_col = self.loader.get_col_info()

        # imputation
        data_imp = self.harmonizer.impute_fit_transform(
            data[feat_cols]
        )
        data_imp_f = pd.concat(
            (data_imp, data[meta_cols + gp_cols + [tgt_col]]), 
            axis=1
        )

        # encoding 
        data_enc = self.harmonizer.encode_fit_transform(
            data_imp
        )
        data_enc_f = pd.concat(
            (data_enc, data[meta_cols + gp_cols + [tgt_col]]),
            axis=1
        )

        # statistical testing
        data_imp_f[tgt_col] = data_imp_f[tgt_col].apply(
            self.loader.tgt_map.get
        )

        tests, p_values, corr_pairs = self.analyzer.run_tests(
            data_imp_f, 
            feat_cols,
            tgt_col,
            gp_cols[0],
            method='pearson',
            threshold=0.75
        )

        # violin plot
        fig = self.analyzer.plot_distributions(
            X=data_imp_f, 
            y_tags=feat_cols, 
            x_tag=tgt_col, 
            plot='violin',
            n_feat = 16
        )

        return fig, (tests, p_values, corr_pairs)

class PipelineMLRunner(PipelineMLGen):
    def __init__(self, config_path: str = None, config_dict: dict = None) -> None:
        # Config path
        self.config_path = config_path

        # load modules
        if isinstance(config_dict, dict):
            config_args = self.load_modules_from_dict(config_dict)
        else:
            config_args = self.load_modules_from_json(config_path)

        # load gen attributes
        super().__init__(**config_args)

    def load_modules_from_json(self, json_path: str):
        config = json.load(open(json_path, 'r'))
        return self.load_modules_from_dict(config)

    def load_modules_from_dict(self, config: dict):    
        # kwargs
        config_gen_args = {
            k: v 
            for k, v in config.items() 
            if k != 'modules'
        }

        # loading class modules
        for module_name, module in config['modules'].items():
            class_ = pydoc.locate(f"src.{module['class_']}")
            params_ = module['params_']

            try:
                obj = class_(**params_)
                logger.info(f"Module {module_name} successfully loaded")
            except Exception as e:
                logger.info(f"Module {module_name} not loaded correctly." 
                             f"Please check the error:\n{e}")
                obj = None

            # assign to class attribute
            config_gen_args[module_name] = obj            

        return config_gen_args

    def run(self):
        # Train and Test
        train, test = self.loader.get_data(split=True)

        # metrics
        metrics = []

        for X_train, X_test in zip(train, test):
            # Harmonize
            # settings
            feat_cols, meta_cols, gp_cols, tgt_col = self.loader.get_col_info()

            # reset_index
            X_train = X_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)

            # imputation
            X_train_imp = self.harmonizer.impute_fit_transform(
                X_train[feat_cols]
            )

            X_train_f = pd.concat(
                (X_train_imp, X_train[meta_cols + gp_cols + [tgt_col]]), 
                axis=1
            )

            # test imputation (only using features)
            X_test_imp = self.harmonizer.impute_transform(
                X_test[feat_cols]
            )

            X_test_f = pd.concat(
                (X_test[meta_cols + gp_cols + [tgt_col]], X_test_imp), 
                axis=1
            )

            # stratified group-kfold
            cv = StratifiedGroupKFold(
                n_splits=self.loader.n_folds,
                shuffle=True, 
                random_state=42
            )

            cv_train = self.trainer.gridcv_train(
                X=X_train_f[feat_cols], 
                y=X_train_f[tgt_col],
                groups=X_train_f['group'],
                scoring='f1_macro', 
                cv=cv, 
                verbose=2
            )

            # test predictions
            y_proba_test = cv_train.predict_proba(X_test_f[feat_cols])
            y_pred_test = y_proba_test.argmax(1)
            y_true = X_test_f[tgt_col].to_numpy()

            # test metrics
            aux_metrics = calculate_metrics(y_true, y_pred_test)

            # assign
            metrics.append(aux_metrics)

        return metrics