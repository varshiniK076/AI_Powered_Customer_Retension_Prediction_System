import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

from log_code import setup_logging
logger = setup_logging("missing_values")

# This makes dataset in stable
np.random.seed(42)


class MISSING_VALUES:

    # IMPUTATION FUNCTION
    @staticmethod
    def missing_values(X, strategy="mean"):
        try:
            X = X.copy()

            num_cols = X.select_dtypes(include=["int64", "float64"]).columns
            cat_cols = X.select_dtypes(include=["object"]).columns

            # ---------------- SIMPLE METHODS ----------------
            if strategy in ["mean", "median", "mode"]:
                for col in num_cols:
                    if strategy == "mean":
                        X[col].fillna(X[col].mean(), inplace=True)
                    elif strategy == "median":
                        X[col].fillna(X[col].median(), inplace=True)
                    else:
                        X[col].fillna(X[col].mode()[0], inplace=True)

                for col in cat_cols:
                    X[col].fillna(X[col].mode()[0], inplace=True)

            # ---------------- CONSTANT ----------------
            elif strategy == "constant":
                X[num_cols] = X[num_cols].fillna(-999)
                X[cat_cols] = X[cat_cols].fillna("Unknown")

            # ---------------- ARBITRARY ----------------
            elif strategy == "arbitrary":
                X[num_cols] = X[num_cols].fillna(9999)
                X[cat_cols] = X[cat_cols].fillna("Unknown")

            # ---------------- END OF DISTRIBUTION ----------------
            elif strategy == "end_distribution":
                for col in num_cols:
                    extreme = X[col].mean() + 3 * X[col].std()
                    X[col].fillna(extreme, inplace=True)
            # ---------------- FORWARD FILL ----------------
            elif strategy == "forward_fill":
                    X[num_cols] = X[num_cols].ffill()
                    X[cat_cols] = X[cat_cols].ffill()

            # ---------------- BACKWARD FILL ----------------
            elif strategy == "backward_fill":
                X[num_cols] = X[num_cols].bfill()
                X[cat_cols] = X[cat_cols].bfill()

            # ---------------- INTERPOLATION ----------------
            elif strategy == "interpolation":
                X[num_cols] = X[num_cols].interpolate(method="linear")

                for col in cat_cols:
                    X[col].fillna(X[col].mode()[0], inplace=True)
            # ---------------- RANDOM (PROBABILISTIC) ----------------
            elif strategy == "random":
                for col in num_cols:
                    idx = X[X[col].isnull()].index
                    if len(idx) > 0:
                        sample = X[col].dropna().sample(len(idx), replace=True, random_state=42)
                        sample.index = idx
                        X.loc[idx, col] = sample

                for col in cat_cols:
                    idx = X[X[col].isnull()].index
                    if len(idx) > 0:
                        sample = X[col].dropna().sample(len(idx), replace=True, random_state=42)
                        sample.index = idx
                        X.loc[idx, col] = sample

            # ---------------- SIMPLE IMPUTER ----------------
            elif strategy == "simple":
                num_imp = SimpleImputer(strategy="median")
                cat_imp = SimpleImputer(strategy="most_frequent")

                X[num_cols] = num_imp.fit_transform(X[num_cols])
                X[cat_cols] = cat_imp.fit_transform(X[cat_cols])

            # ---------------- KNN IMPUTER ----------------
            elif strategy == "knn":
                knn = KNNImputer(n_neighbors=5)
                X[num_cols] = knn.fit_transform(X[num_cols])

                for col in cat_cols:
                    X[col].fillna(X[col].mode()[0], inplace=True)

            # ---------------- ITERATIVE IMPUTER (MICE) ----------------
            elif strategy == "mice":
                mice = IterativeImputer(
                    random_state=42,
                    sample_posterior=True
                )
                X[num_cols] = mice.fit_transform(X[num_cols])

                for col in cat_cols:
                    X[col].fillna(X[col].mode()[0], inplace=True)

            # ---------------- BAYESIAN-STYLE SAMPLING ----------------
            elif strategy == "bayesian":
                for col in num_cols:
                    mu = X[col].mean()
                    sigma = X[col].std()
                    idx = X[X[col].isnull()].index

                    if len(idx) > 0:
                        sampled = np.random.normal(mu, sigma, size=len(idx))
                        X.loc[idx, col] = sampled

                for col in cat_cols:
                    X[col].fillna(X[col].mode()[0], inplace=True)

            else:
                raise ValueError("Unknown imputation strategy")

            return X

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(
                f"Error in missing_values at line {error_line.tb_lineno}: {error_msg}"
            )
            return None

    # RUN ALL STRATEGIES
    @staticmethod
    def run_all_strategies(X):
        try:
            strategies = [
                "mean",
                "median",
                "mode",
                "constant",
                "arbitrary",
                "end_distribution",
                "forward_fill",
                "backward_fill",
                "interpolation",
                "random",
                "simple",
                "knn",
                "mice",
                "bayesian"
            ]

            imputed_data = {}

            for strategy in strategies:
                logger.info(f"IMPUTATION TECHNIQUE : {strategy.upper()}")

                X_temp = MISSING_VALUES.missing_values(X, strategy)

                if X_temp is not None:
                    imputed_data[strategy] = X_temp
                    logger.info(
                        f"Remaining missing values: {X_temp.isnull().sum().sum()}"
                    )

            return imputed_data

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(
                f"Error in run_all_strategies at line {error_line.tb_lineno}: {error_msg}"
            )
            return {}

    # EVALUATION with origibal dataset
    @staticmethod
    def distribution_evaluation(X_original, imputed_dict):
        try:
            num_cols = X_original.select_dtypes(include=["int64", "float64"]).columns

            baseline_mean = X_original[num_cols].mean()
            baseline_std = X_original[num_cols].std()

            results = []

            for strategy, X_imp in imputed_dict.items():
                new_mean = X_imp[num_cols].mean()
                new_std = X_imp[num_cols].std()

                mean_diff = (baseline_mean - new_mean).abs().mean()
                std_diff = (baseline_std - new_std).abs().mean()

                score = mean_diff + std_diff

                results.append({
                    "Technique": strategy.upper(),
                    "Mean_Difference": mean_diff,
                    "Std_Difference": std_diff,
                    "Score": score
                })

            result_df = pd.DataFrame(results).sort_values(by="Score")
            warnings.filterwarnings("ignore")

            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

            from log_code import setup_logging
            logger = setup_logging("missing_values")

            # This makes dataset in stable
            np.random.seed(42)

            class MISSING_VALUES:

                # IMPUTATION FUNCTION
                @staticmethod
                def missing_values(X, strategy="mean"):
                    try:
                        X = X.copy()

                        num_cols = X.select_dtypes(include=["int64", "float64"]).columns
                        cat_cols = X.select_dtypes(include=["object"]).columns

                        # ---------------- SIMPLE METHODS ----------------
                        if strategy in ["mean", "median", "mode"]:
                            for col in num_cols:
                                if strategy == "mean":
                                    X[col].fillna(X[col].mean(), inplace=True)
                                elif strategy == "median":
                                    X[col].fillna(X[col].median(), inplace=True)
                                else:
                                    X[col].fillna(X[col].mode()[0], inplace=True)

                            for col in cat_cols:
                                X[col].fillna(X[col].mode()[0], inplace=True)

                        # ---------------- CONSTANT ----------------
                        elif strategy == "constant":
                            X[num_cols] = X[num_cols].fillna(-999)
                            X[cat_cols] = X[cat_cols].fillna("Unknown")

                        # ---------------- ARBITRARY ----------------
                        elif strategy == "arbitrary":
                            X[num_cols] = X[num_cols].fillna(9999)
                            X[cat_cols] = X[cat_cols].fillna("Unknown")

                        # ---------------- END OF DISTRIBUTION ----------------
                        elif strategy == "end_distribution":
                            for col in num_cols:
                                extreme = X[col].mean() + 3 * X[col].std()
                                X[col].fillna(extreme, inplace=True)
                        # ---------------- FORWARD FILL ----------------
                        elif strategy == "forward_fill":
                            X[num_cols] = X[num_cols].ffill()
                            X[cat_cols] = X[cat_cols].ffill()

                        # ---------------- BACKWARD FILL ----------------
                        elif strategy == "backward_fill":
                            X[num_cols] = X[num_cols].bfill()
                            X[cat_cols] = X[cat_cols].bfill()

                        # ---------------- INTERPOLATION ----------------
                        elif strategy == "interpolation":
                            X[num_cols] = X[num_cols].interpolate(method="linear")

                            for col in cat_cols:
                                X[col].fillna(X[col].mode()[0], inplace=True)
                        # ---------------- RANDOM (PROBABILISTIC) ----------------
                        elif strategy == "random":
                            for col in num_cols:
                                idx = X[X[col].isnull()].index
                                if len(idx) > 0:
                                    sample = X[col].dropna().sample(len(idx), replace=True, random_state=42)
                                    sample.index = idx
                                    X.loc[idx, col] = sample

                            for col in cat_cols:
                                idx = X[X[col].isnull()].index
                                if len(idx) > 0:
                                    sample = X[col].dropna().sample(len(idx), replace=True, random_state=42)
                                    sample.index = idx
                                    X.loc[idx, col] = sample

                        # ---------------- SIMPLE IMPUTER ----------------
                        elif strategy == "simple":
                            num_imp = SimpleImputer(strategy="median")
                            cat_imp = SimpleImputer(strategy="most_frequent")

                            X[num_cols] = num_imp.fit_transform(X[num_cols])
                            X[cat_cols] = cat_imp.fit_transform(X[cat_cols])

                        # ---------------- KNN IMPUTER ----------------
                        elif strategy == "knn":
                            knn = KNNImputer(n_neighbors=5)
                            X[num_cols] = knn.fit_transform(X[num_cols])

                            for col in cat_cols:
                                X[col].fillna(X[col].mode()[0], inplace=True)

                        # ---------------- ITERATIVE IMPUTER (MICE) ----------------
                        elif strategy == "mice":
                            mice = IterativeImputer(
                                random_state=42,
                                sample_posterior=True
                            )
                            X[num_cols] = mice.fit_transform(X[num_cols])

                            for col in cat_cols:
                                X[col].fillna(X[col].mode()[0], inplace=True)

                        # ---------------- BAYESIAN-STYLE SAMPLING ----------------
                        elif strategy == "bayesian":
                            for col in num_cols:
                                mu = X[col].mean()
                                sigma = X[col].std()
                                idx = X[X[col].isnull()].index

                                if len(idx) > 0:
                                    sampled = np.random.normal(mu, sigma, size=len(idx))
                                    X.loc[idx, col] = sampled

                            for col in cat_cols:
                                X[col].fillna(X[col].mode()[0], inplace=True)

                        else:
                            raise ValueError("Unknown imputation strategy")

                        return X

                    except Exception as e:
                        error_type, error_msg, error_line = sys.exc_info()
                        logger.error(
                            f"Error in missing_values at line {error_line.tb_lineno}: {error_msg}"
                        )
                        return None

                # RUN ALL STRATEGIES
                @staticmethod
                def run_all_strategies(X):
                    try:
                        strategies = [
                            "mean",
                            "median",
                            "mode",
                            "constant",
                            "arbitrary",
                            "end_distribution",
                            "forward_fill",
                            "backward_fill",
                            "interpolation",
                            "random",
                            "simple",
                            "knn",
                            "mice",
                            "bayesian"
                        ]

                        imputed_data = {}

                        for strategy in strategies:
                            logger.info(f"IMPUTATION TECHNIQUE : {strategy.upper()}")

                            X_temp = MISSING_VALUES.missing_values(X, strategy)

                            if X_temp is not None:
                                imputed_data[strategy] = X_temp
                                logger.info(
                                    f"Remaining missing values: {X_temp.isnull().sum().sum()}"
                                )

                        return imputed_data

                    except Exception as e:
                        error_type, error_msg, error_line = sys.exc_info()
                        logger.error(
                            f"Error in run_all_strategies at line {error_line.tb_lineno}: {error_msg}"
                        )
                        return {}

                # EVALUATION with origibal dataset
                @staticmethod
                def distribution_evaluation(X_original, imputed_dict):
                    try:
                        num_cols = X_original.select_dtypes(include=["int64", "float64"]).columns

                        baseline_mean = X_original[num_cols].mean()
                        baseline_std = X_original[num_cols].std()

                        results = []

                        for strategy, X_imp in imputed_dict.items():
                            new_mean = X_imp[num_cols].mean()
                            new_std = X_imp[num_cols].std()

                            mean_diff = (baseline_mean - new_mean).abs().mean()
                            std_diff = (baseline_std - new_std).abs().mean()

                            score = mean_diff + std_diff

                            results.append({
                                "Technique": strategy.upper(),
                                "Mean_Difference": mean_diff,
                                "Std_Difference": std_diff,
                                "Score": score
                            })

                        result_df = pd.DataFrame(results).sort_values(by="Score")

                        best = result_df.iloc[0]
                        logger.info(
                            f"BEST TECHNIQUE: {best['Technique']} | Score: {best['Score']:.4f}"
                        )

                        return result_df

                    except Exception as e:
                        error_type, error_msg, error_line = sys.exc_info()
                        logger.error(
                            f"Error in distribution_evaluation at line {error_line.tb_lineno}: {error_msg}"
                        )
                        return pd.DataFrame()

            best = result_df.iloc[0]
            logger.info(
                f"BEST TECHNIQUE: {best['Technique']} | Score: {best['Score']:.4f}"
            )

            return result_df

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(
                f"Error in distribution_evaluation at line {error_line.tb_lineno}: {error_msg}"
            )
            return pd.DataFrame()
