import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,MaxAbsScaler
from imblearn.over_sampling import SMOTE
import pickle
from log_code import setup_logging

logger = setup_logging('scaling_balancing')


def score_scaled_df(df):
    #Lower score = better distribution
    skew = df.skew().abs().mean()
    kurt = df.kurtosis().abs().mean()
    return skew + kurt

#y_train for balancing not for scaling
def scale_and_balance(train_data, test_data, y_train, scaler_path="scaler.pkl"):
    try:
        logger.info("===== Starting AUTO Balancing + Scaling =====")

        if not isinstance(y_train, pd.Series):
            y_train = pd.Series(y_train, name="Churn")

        logger.info("Target BEFORE SMOTE:\n" + y_train.value_counts().to_string())

        # -------------------
        # 1. SMOTE FIRST
        # -------------------
        sm = SMOTE(random_state=42)
        X_bal, y_bal = sm.fit_resample(train_data, y_train)

        logger.info("Target AFTER SMOTE:\n" + pd.Series(y_bal).value_counts().to_string())
        logger.info(f"Shape after SMOTE: {X_bal.shape}")

        numeric_cols = X_bal.columns.tolist()

        logger.info("Feature stats BEFORE scaling:\n" +
                    X_bal[numeric_cols].describe().to_string())
        '''
        df.describe() returns

        For every numeric column, it returns:
        
        count	Number of non-missing values
        mean	Average value
        std	Standard deviation (spread of data)
        min	Smallest value
        25%	First quartile (25% of data below this)
        50%	Median (middle value)
        75%	Third quartile (75% of data below this)
        max	Largest value
        '''


        # 2. Try all scalers
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            "maxabs": MaxAbsScaler()
        }

        best_score = np.inf
        best_name = None
        best_scaler = None
        best_X = None
        best_test = None

        for name, scaler in scalers.items():
            X_temp = X_bal.copy()
            T_temp = test_data.copy()

            X_temp[numeric_cols] = scaler.fit_transform(X_temp[numeric_cols])
            T_temp[numeric_cols] = scaler.transform(T_temp[numeric_cols])

            score = score_scaled_df(X_temp[numeric_cols])

            logger.info(f"{name.upper()} scaler score = {score:.4f}")

            if score < best_score:
                best_score = score
                best_name = name
                best_scaler = scaler
                best_X = X_temp
                best_test = T_temp

        # -------------------
        # 3. Apply best scaler
        # -------------------
        logger.info(f"BEST SCALER SELECTED = {best_name.upper()}")

        with open(scaler_path, "wb") as f:
            pickle.dump(best_scaler, f)

        logger.info(f"Scaler saved at {scaler_path}")

        logger.info("Feature stats AFTER scaling:\n" +
                    best_X[numeric_cols].describe().to_string())

        logger.info(f"Final Train Shape: {best_X.shape}")
        logger.info(f"Final Test Shape: {best_test.shape}")

        logger.info("===== AUTO Scaling + Balancing Completed =====")

        return best_X, best_test, pd.Series(y_bal, name="Churn")

    except Exception as e:
        import sys
        error_type, error_msg, error_line = sys.exc_info()
        logger.error(f"Error in Line no : {error_line.tb_lineno}: due to {error_msg}")
        return train_data, test_data, y_train
