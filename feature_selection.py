import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import sklearn
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
from log_code import setup_logging
logger = setup_logging('feature_selection')
from sklearn.feature_selection import chi2
from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold
reg_constant = VarianceThreshold(threshold=0.0)
reg_quasi_constant = VarianceThreshold(threshold=0.1)


class COMPLETE_FEATURE_SELECTION:

    @staticmethod
    def feature_selection(X_train_numeric, X_test_numeric, y_train):
        try:
            logger.info("===================================")
            logger.info("FEATURE SELECTION (NUMERIC ONLY)")
            logger.info("===================================")

            logger.info(f"Train Shape (Before FS): {X_train_numeric.shape}")
            logger.info(f"Test Shape (Before FS): {X_test_numeric.shape}")

            y_train_numeric = np.asarray(y_train).ravel()

            # 1. Constant
            reg_constant.fit(X_train_numeric)
            constant_cols = X_train_numeric.columns[~reg_constant.get_support()]
            logger.info(f'Removed constant: {list(constant_cols)}')

            X_train = pd.DataFrame(
                reg_constant.transform(X_train_numeric),
                columns=X_train_numeric.columns[reg_constant.get_support()]
            )

            X_test = pd.DataFrame(
                reg_constant.transform(X_test_numeric),
                columns=X_test_numeric.columns[reg_constant.get_support()]
            )


            # 2. Quasi-constant
            reg_quasi_constant.fit(X_train)
            quasi_cols = X_train.columns[~reg_quasi_constant.get_support()]
            logger.info(f'Removed quasi-constant: {list(quasi_cols)}')

            X_train = pd.DataFrame(
                reg_quasi_constant.transform(X_train),
                columns=X_train.columns[reg_quasi_constant.get_support()]
            )

            X_test = pd.DataFrame(
                reg_quasi_constant.transform(X_test),
                columns=X_test.columns[reg_quasi_constant.get_support()]
            )
            '''
            # -------------------------
            # 3. Chi-square is used only for categorical columns
            # -------------------------
            if (X_train < 0).any().any():
                logger.warning("Negative values found and it is not categorical -> skipping Chi-square")
            else:
                chi_scores, chi_pvalues = chi2(X_train, y_train_numeric)
                chi_pvalues = pd.Series(chi_pvalues, index=X_train.columns)

                chi_alpha = 0.05
                chi_remove = chi_pvalues[chi_pvalues > chi_alpha].index.tolist()
                logger.info(f'Removed by Chi-square: {chi_remove}')

                X_train = X_train.drop(columns=chi_remove)
                X_test = X_test.drop(columns=chi_remove)
            '''

            # 4. Pearson correlation
            p_values = []
            for col in X_train.columns:
                corr, p_val = pearsonr(X_train[col].values, y_train_numeric)
                p_values.append(p_val)
                logger.info(f"{col} | p-value = {p_val:.6f}")

            p_values = pd.Series(p_values, index=X_train.columns)

            alpha = 0.05
            remove_corr = p_values[p_values > alpha].index.tolist()
            logger.info(f"Removed by correlation: {remove_corr}")

            X_train = X_train.drop(columns=remove_corr)
            X_test = X_test.drop(columns=remove_corr)


            # Final logs
            logger.info("===================================")
            logger.info("FEATURE SELECTION COMPLETED")
            logger.info("===================================")
            logger.info(f"Train Shape (After FS): {X_train.shape}")
            logger.info(f"Test Shape (After FS): {X_test.shape}")
            logger.info(f"Final Numeric Columns:\n{X_train.columns}")

            return X_train, X_test

        except Exception:
            logger.exception("FEATURE SELECTION FAILED")
            return None, None
