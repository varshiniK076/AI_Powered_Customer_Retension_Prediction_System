import numpy as np
import pandas as pd
import sys
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from log_code import setup_logging
logger = setup_logging("outlier_handling")

class OUTLIER_HANDLING:

    @staticmethod
    def score_series(s):
        skew = s.skew()
        kurt = s.kurtosis()
        #Outlier_ratio
        out_ratio = np.mean(np.abs((s - s.mean()) / s.std()) > 3)
        score = abs(skew) + abs(kurt) + out_ratio
        return score, skew, kurt, out_ratio

    @staticmethod
    def apply_best(train_df, test_df, plot_path="outlier_plots"):


        numeric_cols = train_df.select_dtypes(include=np.number).columns
        report = {}

        for col in numeric_cols:
            try:
                orig = train_df[col]
                orig_score, orig_skew, orig_kurt, orig_out = \
                    OUTLIER_HANDLING.score_series(orig)

                logger.info(
                    f"{col} | ORIGINAL -> "
                    f"Skew={orig_skew:.3f}, Kurt={orig_kurt:.3f}, "
                    f"Out={orig_out:.3f}, Score={orig_score:.3f}"
                )

                # Plot before
                plt.figure()
                sns.boxplot(x=orig)
                plt.title(f"Before Outlier Handling - {col}")
                plt.savefig(f"{plot_path}/before_{col}.png")
                plt.close()

                methods = {}

                # WINSOR
                p5, p95 = orig.quantile(0.05), orig.quantile(0.95)
                methods["winsor"] = orig.clip(p5, p95)

                # IQR
                Q1, Q3 = orig.quantile(0.25), orig.quantile(0.75)
                IQR = Q3 - Q1
                methods["iqr"] = orig.clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

                # MAD
                med = orig.median()
                mad = np.median(np.abs(orig - med))
                methods["mad"] = orig.clip(med - 3*mad, med + 3*mad)

                # Quantile
                q1, q99 = orig.quantile(0.01), orig.quantile(0.99)
                methods["quantile"] = orig.clip(q1, q99)

                best_name = "original"
                best_score = orig_score
                best_train = orig
                best_test = test_df[col]
                # Score = |skew| + |kurtosis| + outlier_ratio formula for finding best technique
                for name, data in methods.items():
                    score, skew, kurt, out = \
                        OUTLIER_HANDLING.score_series(data)

                    logger.info(
                        f"{col} | {name.upper()} -> "
                        f"Skew={skew:.3f}, Kurt={kurt:.3f}, "
                        f"Out={out:.3f}, Score={score:.3f}"
                    )

                    if score < best_score:
                        best_score = score
                        best_name = name
                        best_train = data
                        best_test = test_df[col].clip(
                            data.min(), data.max()
                        )

                # APPLY BEST
                train_df[col] = best_train
                test_df[col] = best_test

                improvement = round(
                    (orig_score - best_score) / orig_score * 100, 2
                )

                logger.info(
                    f"{col} | SELECTED = {best_name.upper()} | "
                    f"Score {orig_score:.3f} -> {best_score:.3f} "
                    f"({improvement}% improvement)"
                )

                # Plot after
                plt.figure()
                sns.boxplot(x=train_df[col])
                plt.title(f"After ({best_name}) - {col}")
                plt.savefig(f"{plot_path}/after_{col}.png")
                plt.close()

                report[col] = {
                    "method": best_name,
                    "orig_score": round(orig_score,3),
                    "final_score": round(best_score,3),
                    "improvement_%": improvement
                }

            except Exception as e:
                etype, emsg, eline = sys.exc_info()
                logger.error(
                    f"Error handling {col} | Line {eline.tb_lineno}: {emsg}"
                )

        logger.info("Automatic outlier handling completed")
        return train_df, test_df, report
