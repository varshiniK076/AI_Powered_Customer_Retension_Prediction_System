import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import warnings
warnings.filterwarnings('ignore')
import pickle

from log_code import setup_logging
logger = setup_logging('All_models')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


# Individual model functions
def knn(X_train, y_train, X_test, y_test):
    global knn_reg
    try:
        knn_reg = KNeighborsClassifier(n_neighbors=5)
        knn_reg.fit(X_train, y_train)
        logger.info(f'KNN Test Accuracy : {accuracy_score(y_test, knn_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')


def nb(X_train, y_train, X_test, y_test):
    global naive_reg
    try:
        naive_reg = GaussianNB()
        naive_reg.fit(X_train, y_train)
        logger.info(f'Naive Bayes Test Accuracy : {accuracy_score(y_test, naive_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')


def lr(X_train, y_train, X_test, y_test):
    global lr_reg
    try:
        lr_reg = LogisticRegression(max_iter=1000)
        lr_reg.fit(X_train, y_train)
        y_pred = lr_reg.predict(X_test)
        logger.info(f'LogisticRegression Test Accuracy : {accuracy_score(y_test, y_pred)}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')


def dt(X_train, y_train, X_test, y_test):
    global dt_reg
    try:
        dt_reg = DecisionTreeClassifier(criterion='entropy')
        dt_reg.fit(X_train, y_train)
        logger.info(f'DecisionTreeClassifier Test Accuracy : {accuracy_score(y_test, dt_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')


def rf(X_train, y_train, X_test, y_test):
    global rf_reg
    try:
        rf_reg = RandomForestClassifier(n_estimators=5, criterion='entropy')
        rf_reg.fit(X_train, y_train)
        logger.info(f'RandomForestClassifier Test Accuracy : {accuracy_score(y_test, rf_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')


def ada(X_train, y_train, X_test, y_test):
    global ada_reg
    try:
        t = LogisticRegression()
        ada_reg = AdaBoostClassifier(estimator=t, n_estimators=5)
        ada_reg.fit(X_train, y_train)
        logger.info(f'AdaBoostClassifier Test Accuracy : {accuracy_score(y_test, ada_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')


def gb(X_train, y_train, X_test, y_test):
    global gb_reg
    try:
        gb_reg = GradientBoostingClassifier(n_estimators=5)
        gb_reg.fit(X_train, y_train)
        logger.info(f'GradientBoostingClassifier Test Accuracy : {accuracy_score(y_test, gb_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')


def xgb_(X_train, y_train, X_test, y_test):
    global xg_reg
    try:
        xg_reg = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xg_reg.fit(X_train.values, y_train.values)
        logger.info(f'XGBClassifier Test Accuracy : {accuracy_score(y_test, xg_reg.predict(X_test.values))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

def svm_c(X_train, y_train, X_test, y_test):
    global svm_reg
    try:
        svm_reg = SVC(kernel='rbf', probability=True)
        svm_reg.fit(X_train, y_train)
        logger.info(f'SVM Test Accuracy : {accuracy_score(y_test, svm_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')


def common(X_train, y_train, X_test, y_test):
    try:
        # Train all models
        logger.info('=========KNeighborsClassifier===========')
        knn(X_train, y_train, X_test, y_test)
        logger.info('=========GaussianNB===========')
        nb(X_train, y_train, X_test, y_test)
        logger.info('=========LogisticRegression===========')
        lr(X_train, y_train, X_test, y_test)
        logger.info('=========DecisionTreeClassifier===========')
        dt(X_train, y_train, X_test, y_test)
        logger.info('=========RandomForestClassifier===========')
        rf(X_train, y_train, X_test, y_test)
        logger.info('=========AdaBoostClassifier===========')
        ada(X_train, y_train, X_test, y_test)
        logger.info('=========GradientBoostingClassifier===========')
        gb(X_train, y_train, X_test, y_test)
        logger.info('=========XGBClassifier===========')
        xgb_(X_train, y_train, X_test, y_test)
        logger.info('=========SVM===========')
        svm_c(X_train, y_train, X_test, y_test)

        # Predictions for ROC-AUC
        model_preds = {}

        if knn_reg:
            model_preds["KNN"] = knn_reg.predict_proba(X_test)[:, 1]
        if naive_reg:
            model_preds["Naive Bayes"] = naive_reg.predict_proba(X_test)[:, 1]
        if lr_reg:
            model_preds["Logistic Regression"] = lr_reg.predict_proba(X_test)[:, 1]
        if dt_reg:
            model_preds["Decision Tree"] = dt_reg.predict_proba(X_test)[:, 1]
        if rf_reg:
            model_preds["Random Forest"] = rf_reg.predict_proba(X_test)[:, 1]
        if ada_reg:
            model_preds["AdaBoost"] = ada_reg.predict_proba(X_test)[:, 1]
        if gb_reg:
            model_preds["Gradient Boosting"] = gb_reg.predict_proba(X_test)[:, 1]
        if xg_reg:
            model_preds["XGBoost"] = xg_reg.predict_proba(X_test.values)[:, 1]
        if svm_reg:
            model_preds["SVM"] = svm_reg.predict_proba(X_test)[:, 1]

        # ROC-AUC plot
        plot_dir = 'auc_roc_plotting'
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, 'auc_roc_curve.png')

        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], "k--")  # diagonal

        auc_scores = {}
        for name, pred in model_preds.items():
            fpr, tpr, _ = roc_curve(y_test, pred)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, pred):.2f})")
            auc_scores[name] = roc_auc_score(y_test, pred)

        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("ROC Curve - All Models")
        plt.legend(loc="lower right")

        # Save plot
        plt.savefig(plot_path)
        plt.close()  # prevents popup
        logger.info(f"ROC-AUC plot saved at {plot_path}")

        # Log AUC scores
        for model, score in auc_scores.items():
            logger.info(f"{model} ROC-AUC Score: {score}")

        # Select best model
        best_model_name = max(auc_scores, key=auc_scores.get)
        best_auc = auc_scores[best_model_name]
        logger.info("===================================")
        logger.info(f"BEST MODEL: {best_model_name}")
        logger.info(f"BEST ROC-AUC: {best_auc}")
        logger.info("===================================")

        model_dict = {
            "KNN": knn_reg,
            "Naive Bayes": naive_reg,
            "Logistic Regression": lr_reg,
            "Decision Tree": dt_reg,
            "Random Forest": rf_reg,
            "AdaBoost": ada_reg,
            "Gradient Boosting": gb_reg,
            "XGBoost": xg_reg,
            "SVM": svm_reg
        }

        best_model = model_dict[best_model_name]

        return best_model

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')
