from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import sys
from log_code import setup_logging
import pandas as pd

logger = setup_logging('tuning')

def tune_logistic(X_train, y_train, X_test=None, y_test=None):
    """
    Perform hyperparameter tuning for Logistic Regression using GridSearchCV.
    Optionally evaluates test ROC-AUC if X_test and y_test are provided.

    Returns:
        best_model : LogisticRegression : Fitted model with best parameters
        best_params : dict : Best hyperparameters
        best_score  : float : Best CV ROC-AUC score
        test_roc_auc : float : ROC-AUC on test set (if X_test & y_test provided)
    """
    try:
        logger.info("===== Starting Logistic Regression Hyperparameter Tuning =====")

        # Define hyperparameter grid
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'solver': ['liblinear', 'saga'],  # 'saga' supports elasticnet
            'max_iter': [500, 1000, 2000]
        }

        lr = LogisticRegression(random_state=42)

        grid = GridSearchCV(
            estimator=lr,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=5,
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_params = grid.best_params_
        best_score = grid.best_score_

        logger.info(f"Best Logistic Regression Params: {best_params}")
        logger.info(f"Best Logistic Regression CV ROC-AUC: {best_score:.4f}")

        test_roc_auc = None
        if X_test is not None and y_test is not None:
            y_test_pred = best_model.predict_proba(X_test)[:, 1]
            test_roc_auc = roc_auc_score(y_test, y_test_pred)
            logger.info(f"Test ROC-AUC Score: {test_roc_auc:.4f}")

            sample_preds = pd.DataFrame({
                'Actual': y_test,
                'Predicted_Prob': y_test_pred
            }).head(10)
            logger.info(f"Sample Test Predictions:\n{sample_preds}")

        logger.info("===== Logistic Regression Hyperparameter Tuning Completed =====")
        return best_model, best_params, best_score, test_roc_auc

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.error(f"Error in Line no : {error_line.tb_lineno}: due to {error_msg}")
        return None, None, None, None

