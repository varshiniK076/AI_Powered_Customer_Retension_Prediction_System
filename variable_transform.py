import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
import warnings
import os
import sys
from log_code import setup_logging

warnings.filterwarnings("ignore")
logger = setup_logging('variable_transformation')


class VARIABLE_TRANSFORMATION:

    @staticmethod
    def variable_trans(X_train_numeric, X_test_numeric):
        """
        Automatically selects and applies the best transformation
        for each numeric column.
        """
        try:
            plot_path = "plots"

            logger.info("Starting variable transformation (AUTO APPLY MODE)")
            logger.info(f"Numeric columns: {list(X_train_numeric.columns)}")

            best_transform_report = {}

            for col in X_train_numeric.columns:
                try:
                    # Skip binary
                    if X_train_numeric[col].nunique() <= 2:
                        logger.info(f"{col} is binary. Skipping.")
                        continue

                    orig_skew = X_train_numeric[col].skew()
                    orig_kurt = X_train_numeric[col].kurtosis()
                    orig_score = abs(orig_skew) + abs(orig_kurt)

                    logger.info(
                        f"{col} | ORIGINAL -> Skew={orig_skew:.3f}, Kurtosis={orig_kurt:.3f}, Score = {orig_score:.3f}"
                    )

                    # Plot BEFORE
                    plt.figure()
                    sns.kdeplot(X_train_numeric[col], fill=True, bw_adjust=1.5)
                    plt.title(f"Before - {col}")
                    plt.savefig(f"{plot_path}/before_kde_{col}.png")
                    plt.close()

                    transforms = {}

                    # LOG
                    if (X_train_numeric[col] > 0).all():
                        transforms["log"] = (
                            np.log1p(X_train_numeric[col]),
                            np.log1p(X_test_numeric[col]))

                        # BOX-COX
                        pt_bc = PowerTransformer(method="box-cox")
                        transforms["boxcox"] = (
                            pt_bc.fit_transform(X_train_numeric[[col]]).ravel(),
                            pt_bc.transform(X_test_numeric[[col]]).ravel()
                        )

                    # YEO-JOHNSON
                    pt_yj = PowerTransformer(method="yeo-johnson")
                    transforms["yeojohnson"] = (
                        pt_yj.fit_transform(X_train_numeric[[col]]).ravel(),
                        pt_yj.transform(X_test_numeric[[col]]).ravel()
                    )

                    # QUANTILE
                    qt = QuantileTransformer(
                        output_distribution="normal",
                        random_state=42
                    )
                    transforms["quantile"] = (
                        qt.fit_transform(X_train_numeric[[col]]).ravel(),
                        qt.transform(X_test_numeric[[col]]).ravel()
                    )

                    # Find BEST
                    best_name = "original"
                    best_score = orig_score
                    best_train = X_train_numeric[col]
                    best_test = X_test_numeric[col]

                    for name, (tr, te) in transforms.items():
                        skew = pd.Series(tr).skew()
                        kurt = pd.Series(tr).kurtosis()
                        score = abs(skew) + abs(kurt)

                        logger.info(
                            f"{col} | {name.upper()} -> Skew={skew:.3f}, Kurtosis={kurt:.3f}, Score = {score:.3f}"
                        )

                        if score < best_score:
                            best_score = score
                            best_name = name
                            best_train = tr
                            best_test = te

                    # APPLY BEST
                    X_train_numeric[col] = best_train
                    X_test_numeric[col] = best_test

                    final_skew = pd.Series(best_train).skew()
                    final_kurt = pd.Series(best_train).kurtosis()

                    logger.info(
                        f"{col} | SELECTED = {best_name.upper()} | "
                        f"FINAL Skew={final_skew:.3f}, Kurtosis={final_kurt:.3f}, Score = {score:.3f}"
                    )

                    # Plot AFTER
                    plt.figure()
                    sns.kdeplot(X_train_numeric[col], fill=True, bw_adjust=1.5)
                    plt.title(f"After ({best_name}) - {col}")
                    plt.savefig(f"{plot_path}/after_kde_{col}.png")
                    plt.close()

                    best_transform_report[col] = best_name

                except Exception as col_err:
                    etype, emsg, eline = sys.exc_info()
                    logger.error(
                        f"Error transforming {col} | Line {eline.tb_lineno}: {emsg}"
                    )

            logger.info("Variable transformation completed successfully")

            return X_train_numeric, X_test_numeric, best_transform_report

        except Exception as e:
            etype, emsg, eline = sys.exc_info()
            logger.error(
                f"Fatal error in variable_trans | Line {eline.tb_lineno}: {emsg}"
            )
            return X_train_numeric, X_test_numeric, {}
