import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder
from log_code import setup_logging

logger = setup_logging('cat_to_num')


def cat_to_numeric(X_train_categorical, X_test_categorical):
    try:
        logger.info('===================================================')
        logger.info('Categorical to Numerical Conversion Started')
        logger.info('===================================================')

        logger.info(f'Train Shape (Before): {X_train_categorical.shape}')
        logger.info(f'Test Shape (Before): {X_test_categorical.shape}')

        logger.info(f'Train Columns: {list(X_train_categorical.columns)}')
        logger.info(f'Test Columns: {list(X_test_categorical.columns)}')

        X_train_cat = X_train_categorical.copy()
        X_test_cat = X_test_categorical.copy()

        # Show unique values
        for col in X_train_cat.columns:
            logger.info(f'{col} | Unique (Train): {X_train_cat[col].unique()}')

        n_train = len(X_train_cat)
        n_test = len(X_test_cat)

        # Encoding
        for col in X_train_cat.columns.tolist():

            unique_vals = X_train_cat[col].nunique()

            # Binary
            if unique_vals == 2:
                le = LabelEncoder()
                le.fit(X_train_cat[col])

                X_train_cat[col] = le.transform(X_train_cat[col])
                X_test_cat[col] = le.transform(X_test_cat[col])

                logger.info(f'{col} -> BINARY')

            # One-hot
            elif 2 < unique_vals <= 10:
                combined = pd.concat([X_train_cat[col], X_test_cat[col]])
                dummies = pd.get_dummies(combined, drop_first=True)

                dummies_train = dummies.iloc[:n_train]
                dummies_test = dummies.iloc[n_train:n_train+n_test]

                X_train_cat = pd.concat(
                    [X_train_cat.drop(columns=[col]), dummies_train],
                    axis=1
                )

                X_test_cat = pd.concat(
                    [X_test_cat.drop(columns=[col]), dummies_test],
                    axis=1
                )

                logger.info(f'{col} -> ONE HOT')

            # Frequency
            else:
                freq_map = X_train_cat[col].value_counts(normalize=True)
                X_train_cat[col] = X_train_cat[col].map(freq_map)
                X_test_cat[col] = X_test_cat[col].map(freq_map).fillna(0)
                logger.info(f'{col} -> FREQUENCY')

        logger.info('===================================================')
        logger.info('After Encoding')
        logger.info('===================================================')

        logger.info(f'Final Train Shape: {X_train_cat.shape}')
        logger.info(f'Final Test Shape: {X_test_cat.shape}')

        logger.info(f'Final Train Columns:\n{X_train_cat.columns}')
        logger.info(f'Final Test Columns:\n{X_test_cat.columns}')

        logger.info('===================================================')
        logger.info('Categorical to Numerical Conversion Completed')
        logger.info('===================================================')

        return X_train_cat, X_test_cat


    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.error(
            f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
        return None, None


