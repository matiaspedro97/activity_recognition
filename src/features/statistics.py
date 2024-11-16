import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels as stats
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List
from scipy.stats import shapiro, levene

from src.visualization.visualize import plot_violin, plot_boxplot


class StatisticsAnalyzer:
    def __init__(self) -> None:
        pass

    def check_normality(self, predictor: np.ndarray):
        # Shapiro-Wilk test for normality
        stat, p_value = shapiro(predictor)

        # check condition
        normal = p_value > 0.05

        return {"normal": normal, "p_value": p_value}

    def check_homoscedasticity(self, *groups: np.ndarray):
        # Levene's test for equal variances
        stat, p_value = levene(*groups)

        # check condition
        equal_var = p_value > 0.05

        return {"equal_var": equal_var, "p_value": p_value}

    def _is_parametric(self, X, predictor: str, group_col: str):
        # unique values
        unq_vals = X[group_col].unique()

        # separate in groups
        groups = []
        for unq in unq_vals:
            groups.append(X[X[group_col] == unq][predictor])

        # normality
        norms = []
        for gp in groups:
            norms.append(self.check_normality(gp).get('normal'))
        norm_f = all(norms)
        
        # homoscedasticity
        vars_f = self.check_homoscedasticity(*groups).get('equal_var')

        return norm_f, vars_f

    def get_testing(self, X: pd.DataFrame, predictor: str, target: str, subject: str):
        # determine normality and homoscedasticity
        norm, vars = self._is_parametric(X, predictor, target)

        # only if normal and homoscedasticity is fulfilled
        test = pg.pairwise_tests(
            data=X, 
            dv=predictor, 
            between=target, 
            subject=subject, 
            padjust='bonf',
            parametric=norm*vars
        )

        return test
    
    def get_full_testing(self, X: pd.DataFrame, predictors: List[str], target: str, subject: str):
        # statistical test results
        tests_res = []

        for pred in predictors:
            # test result
            aux_res = self.get_testing(X, pred, target, subject)

            # append
            tests_res.append(aux_res)
        
        return tests_res
    
    def get_correlated_pairs(self, X: pd.DataFrame, method: str = 'pearson', threshold: float = 0.7):
        # correlation matrix
        corr_m = X.corr(method=method)

        # long format
        corr_pairs = corr_m.unstack()
        
        # build dataframe
        corr_df = pd.DataFrame(corr_pairs, columns=['correlation']).reset_index()
        
        # remove self-correlations
        corr_df = corr_df[corr_df['level_0'] != corr_df['level_1']]
        
        # remove duplicate pairs
        corr_df['ordered_pair'] = corr_df.apply(lambda x: tuple(sorted((x['level_0'], x['level_1']))), axis=1)
        corr_df = corr_df.drop_duplicates(subset=['ordered_pair'])
        corr_df = corr_df.drop(columns=['ordered_pair'])
        
        # filter by threshold
        corr_df['abs_correlation'] = corr_df['correlation'].abs()
        high_corr_pairs = corr_df[corr_df['abs_correlation'] > threshold]
        
        return high_corr_pairs
        
    def plot_distributions(
            self, 
            X: pd.DataFrame, 
            y_tags: List[str], 
            x_tag: str, 
            plot: str = 'violin',
            n_feat: int = 9
        ):

        # sample random predictors
        y_tags_rnd = np.random.choice(
            y_tags,
            n_feat, 
            replace=False
        )

        # plot individual distributions over classes
        if plot.lower() == 'violin':
            fig = plot_violin(X, y_tags_rnd, x_tag)
        elif plot.lower() == 'boxplot':
            fig = plot_boxplot(X, y_tags_rnd, x_tag)
        else:
            fig = None

        return fig
    
    def plot_correlation_matrix(self, X: pd.DataFrame):
        # plot correlation matrix
        corr = X.corr()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            corr, 
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm', 
            ax=ax
        )
        
        return fig
    
    def run_tests(
            self, 
            X: pd.DataFrame, 
            predictors: List[str], 
            target: str, 
            subject: str, 
            method: str = 'pearson',
            threshold: float = 0.7
        ):
        # statistical test results
        tests_res = self.get_full_testing(X, predictors, target, subject)

        # get p-values
        p_values = [test['p-unc'][0] for test in tests_res]

        # get significant correlations
        corr_pairs = self.get_correlated_pairs(X[predictors], method, threshold)

        return tests_res, p_values, corr_pairs
        

