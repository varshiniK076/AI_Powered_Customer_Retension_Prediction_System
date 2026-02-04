'''
In this project implements a ML system that predicts telecommunication customer churn prediction.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import sklearn
import pickle
import warnings
warnings.filterwarnings("ignore")
from log_code import setup_logging
logger = setup_logging('main')
from sklearn.model_selection import train_test_split
from Data_visualize import DATA_VISUALIZE_EDA
from missing_values import MISSING_VALUES
from outlier_handling_overall import OUTLIER_HANDLING
from sklearn.preprocessing import LabelEncoder
from variable_transform import VARIABLE_TRANSFORMATION
from feature_selection import COMPLETE_FEATURE_SELECTION
from categorical_to_numeric import cat_to_numeric
from Scaling_balancing import scale_and_balance
from All_models import common
from tuning import tune_logistic
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


class CUSTOMER_CHURN_PREDICTION:
    try:
        def __init__(self,path):
            self.path = path
            self.df = pd.read_csv(self.path)
            logger.info(f'{self.df.columns}')
            logger.info(f'{self.df.shape}')
            logger.info(f'Total Count of Gender: {self.df['gender'].value_counts()}')
            logger.info(f'Total Count of SeniorCitizen: {self.df['SeniorCitizen'].value_counts()}')
            logger.info(f'Total Count of InternetService: {self.df['InternetService'].value_counts()}')
            logger.info(f'Total Count of Service_Provider: {self.df['Service_Provider'].value_counts()}')
            logger.info(f'Before : {self.df.dtypes}')
            #Changing Totalcharges - object dtype to float dtype
            #TotalCharges column contains blank spaces " ", not numbers to avoid those we use errors='coerce'
            self.df['TotalCharges'] = (self.df['TotalCharges'].replace(" ", np.nan).astype(float))
            #self.df = self.df.dropna(subset=['TotalCharges'])  - when we want remove rows that are Null
            logger.info(f'After : {self.df.dtypes}')
            logger.info(f'=============================================================')
            self.y = self.df['Churn']
            self.X = self.df.drop(columns=['Churn', 'customerID'])
            logger.info(f'{self.X.columns}')
            #logger.info(f'{self.y.info()}')
            logger.info(f'{self.X.shape}')
            logger.info(f'{self.y.shape}')

            # # Split into train/test
            # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)


        def visualize(self):
            try:
                DATA_VISUALIZE_EDA.plot(self.df)
            except Exception as e:
                error_type, error_msg, error_line = sys.exc_info()
                logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

        def handling_missing_values(self):
            try:
                logger.info("Starting missing value evaluation")
                logger.info("==========================================================")

                # BEFORE IMPUTATION
                logger.info("Missing values BEFORE imputation:")
                logger.info(f"\n{self.df.isnull().sum()}")

                #Apply all techniques to dataset
                imputed_data = MISSING_VALUES.run_all_strategies(self.X)
                #This evalutes with all techniques with respect to original dataset
                comparison_df = MISSING_VALUES.distribution_evaluation(self.X, imputed_data)


                logger.info("Imputation technique comparison:")
                logger.info("\n" + comparison_df.to_string(index=False))


                # SELECT BEST TECHNIQUE
                best_row = comparison_df.iloc[0]

                # convert back to lowercase for reuse
                best_technique = best_row["Technique"].lower()
                best_score = best_row["Score"]

                logger.info("==========================================================")
                logger.info(f"BEST MISSING VALUE TECHNIQUE : {best_technique.upper()}")
                logger.info(f"FINAL SCORE (Mean + Std Diff): {best_score:.6f}")

                # APPLY BEST TECHNIQUE
                self.X = MISSING_VALUES.missing_values(self.X,strategy=best_technique)
                # Rebuild dataframe
                self.df = pd.concat([self.X, self.y], axis=1)


                # TRAIN / TEST SPLIT
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42,stratify=self.y)


                # LABEL ENCODING (FOR MODELS & ROC-AUC & Hypothesis testing)
                le = LabelEncoder()
                self.y_train = le.fit_transform(self.y_train)
                self.y_test = le.transform(self.y_test)

                # AFTER IMPUTATION
                logger.info("Missing values AFTER imputation:")
                logger.info(f"\n{self.df.isnull().sum()}")

                logger.info("====== Missing value evaluation completed successfully ======")


            except Exception as e:
                error_type, error_msg, error_line = sys.exc_info()
                logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')


        def var_tranformation(self):
            try:
                self.X_train_numeric = self.X_train.select_dtypes(exclude='object')
                self.X_train_categorical = self.X_train.select_dtypes(include='object')

                self.X_test_numeric = self.X_test.select_dtypes(exclude='object')
                self.X_test_categorical = self.X_test.select_dtypes(include='object')

                (self.X_train_numeric,self.X_test_numeric,self.best_transform_report) = VARIABLE_TRANSFORMATION.variable_trans(self.X_train_numeric,self.X_test_numeric)
                logger.info("========== VARIABLE TRANSFORMATION APPLIED SUCCESSFULLY ==========")
                logger.info("Best transformations per column:")

                for col, method in self.best_transform_report.items():
                    logger.info(f"Variable Transformation Applied | Column: {col} -> Method: {method.upper()}")

                logger.info(f"Transformed train shape: {self.X_train_numeric.shape}")
                logger.info(f"Transformed test shape: {self.X_test_numeric.shape}")

                logger.info("========== VARIABLE TRANSFORMATION STAGE COMPLETED ==========")


            except Exception as e:
                error_type, error_msg, error_line = sys.exc_info()
                logger.error(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')


        def outliers(self):
            try:
                (
                    self.X_train_numeric,
                    self.X_test_numeric,
                    self.outlier_report
                ) = OUTLIER_HANDLING.apply_best(
                    self.X_train_numeric,
                    self.X_test_numeric,
                    plot_path="outlier_plots"
                )


                logger.info("====== OUTLIER HANDLING SUMMARY ======")

                for col, info in self.outlier_report.items():
                    logger.info(
                        f"{col} -> {info['method'].upper()} | "
                        f"{info['orig_score']} -> {info['final_score']} "
                        f"({info['improvement_%']}%)"
                    )
                logger.info('====== Outlier Handling Completed ======')

                logger.info(f'{self.X_train_numeric.isnull().sum()}')
                logger.info(f'{self.X_test_numeric.isnull().sum()}')

            except Exception as e:
                error_type, error_msg, error_line = sys.exc_info()
                logger.error(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

        def encoding(self):
            try:
                logger.info('===================================')
                logger.info('Categorical to Numerical Encoding')
                logger.info('===================================')

                logger.info(f'Categorical Train Columns: {list(self.X_train_categorical.columns)}')
                logger.info(f'Categorical Test Columns: {list(self.X_test_categorical.columns)}')

                logger.info(f'categorical Train Columns : {self.X_train_categorical.shape}')
                logger.info(f'Categorical Test Columns : {self.X_test_categorical.shape}')

                # logger.info(f'{self.X_train.isnull().sum()}')
                # logger.info(f'{self.X_test.isnull().sum()}')

                self.X_train_categorical, self.X_test_categorical = cat_to_numeric(
                    self.X_train_categorical,
                    self.X_test_categorical)


                # AFTER encoding logs
                logger.info('===================================')
                logger.info('After Encoding')
                logger.info('===================================')

                logger.info(f'Final X_train_numeric Shape: {self.X_train_categorical.shape}')
                logger.info(f'Final X_test_numeric Shape: {self.X_test_categorical.shape}')

                logger.info(f'Final Training Columns:\n{self.X_train_categorical.columns}')
                logger.info(f'Final Testing Columns:\n{self.X_test_categorical.columns}')

                # logger.info(f'X_Training_numeric Null Counts:\n{self.X_train_categorical.isnull().sum()}')
                # logger.info(f'X_Testing_numeric Null Counts:\n{self.X_test_categorical.isnull().sum()}')

                logger.info('Encoding step completed successfully')

            except Exception as e:
                error_type, error_msg, error_line = sys.exc_info()
                logger.error(
                    f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
        '''
        def outliers(self):
            try:
                (
                    self.X_training_data,
                    self.X_testing_data,
                    self.outlier_report
                ) = OUTLIER_HANDLING.apply_best(
                    self.X_training_data,
                    self.X_testing_data,
                    plot_path="outlier_plot_encoded"
                )

                logger.info("====== OUTLIER HANDLING SUMMARY ======")

                for col, info in self.outlier_report.items():
                    logger.info(
                        f"{col} -> {info['method'].upper()} | "
                        f"{info['orig_score']} -> {info['final_score']} "
                        f"({info['improvement_%']}%)"
                    )

                logger.info('====== Outlier Handling Completed ======')

                logger.info(f'Train null counts:\n{self.X_training_data.isnull().sum()}')
                logger.info(f'Test null counts:\n{self.X_testing_data.isnull().sum()}')

                logger.info(f"Train shape after outliers: {self.X_training_data.shape}")
                logger.info(f"Test shape after outliers: {self.X_testing_data.shape}")

            except Exception as e:
                import sys
                error_type, error_msg, error_line = sys.exc_info()
                logger.error(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
            '''

        def f_selection(self):
            try:
                logger.info("===================================")
                logger.info("FEATURE SELECTION (NUMERIC ONLY)")
                logger.info("===================================")

                logger.info(f"Numeric Train Shape (Before): {self.X_train_numeric.shape}")
                logger.info(f"Numeric Test Shape  (Before): {self.X_test_numeric.shape}")

                self.X_train_numeric, self.X_test_numeric = COMPLETE_FEATURE_SELECTION.feature_selection(
                    self.X_train_numeric,
                    self.X_test_numeric,
                    self.y_train
                )

                logger.info(f"Numeric Train Shape (After): {self.X_train_numeric.shape}")
                logger.info(f"Numeric Test Shape  (After): {self.X_test_numeric.shape}")

                logger.info(f"Selected Numeric Features: {self.X_train_numeric.shape[1]}")

                logger.info("NUMERIC FEATURE SELECTION DONE")

                logger.info("===================================")
                logger.info("CONCATENATING NUMERIC + CATEGORICAL")
                logger.info("===================================")

                logger.info(f"Numeric Train Shape : {self.X_train_numeric.shape}")
                logger.info(f"Categorical Train Shape : {self.X_train_categorical.shape}")

                logger.info(f"Numeric Test Shape  : {self.X_test_numeric.shape}")
                logger.info(f"Categorical Test Shape  : {self.X_test_categorical.shape}")

                self.X_train_numeric.reset_index(drop=True, inplace=True)
                self.X_train_categorical.reset_index(drop=True, inplace=True)

                self.X_test_numeric.reset_index(drop=True, inplace=True)
                self.X_test_categorical.reset_index(drop=True, inplace=True)

                # Final merge
                self.X_training_data = pd.concat(
                    [self.X_train_numeric, self.X_train_categorical],
                    axis=1
                )

                self.X_testing_data = pd.concat(
                    [self.X_test_numeric, self.X_test_categorical],
                    axis=1
                )

                logger.info("===================================")
                logger.info("FINAL DATASET")
                logger.info("===================================")

                logger.info(f"Final Train Shape: {self.X_training_data.shape}")
                logger.info(f"Final Test Shape : {self.X_testing_data.shape}")

                logger.info(f"Final Columns Count: {len(self.X_training_data.columns)}")

                # logger.info(f"Final Train Nulls:\n{self.X_training_data.isnull().sum()}")
                # logger.info(f"Final Test Nulls:\n{self.X_testing_data.isnull().sum()}")


            except Exception as e:
                error_type, error_msg, error_line = sys.exc_info()
                logger.error(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')



        def scaling_and_balancing(self, scaler_path="scaler.pkl"):
            try:
                logger.info("Starting Scaling and Data Balancing")

                self.X_training_data, self.X_testing_data, self.y_train = scale_and_balance(
                    self.X_training_data,
                    self.X_testing_data,
                    pd.Series(self.y_train, name="Churn"),
                    scaler_path=scaler_path
                )

                logger.info("Scaling and Data Balancing completed successfully")
                logger.info(f"Training shape after SMOTE+Scaling: {self.X_training_data.shape}")
                logger.info(f"Testing shape after Scaling: {self.X_testing_data.shape}")
                logger.info("Churn distribution after balancing:\n" + self.y_train.value_counts().to_string())

            except Exception as e:
                error_type, error_msg, error_line = sys.exc_info()
                logger.error(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

        def train_all_models(self):
            try:
                # Train all classification models and select the best based on ROC-AUC
                common(self.X_training_data, self.y_train, self.X_testing_data, self.y_test)
            except Exception as e:
                error_type, error_msg, error_line = sys.exc_info()
                logger.error(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
       
        def hyperparameter_tuning(self):
            try:
                logger.info("===== Starting Hyperparameter Tuning =====")

                # Pass X_test and y_test to tune_logistic
                best_lr_model, best_lr_params, best_lr_auc, test_roc_auc = tune_logistic(
                    self.X_training_data,
                    self.y_train,
                    self.X_testing_data,
                    self.y_test
                )

                # Log metrics
                y_test_pred_class = best_lr_model.predict(self.X_testing_data)
                test_acc = accuracy_score(self.y_test, y_test_pred_class)

                logger.info(f"Tuned Logistic Regression Test Accuracy: {test_acc:.4f}")
                logger.info(f"Tuned Logistic Regression Test ROC-AUC: {test_roc_auc:.4f}")

                # Save the tuned model
                with open('best_logistic_model.pkl', 'wb') as f:
                    pickle.dump(best_lr_model, f)
                logger.info("Tuned Logistic Regression saved as 'best_logistic_model.pkl'")

                logger.info("===== Hyperparameter Tuning Completed =====")

            except Exception as e:
                error_type, error_msg, error_line = sys.exc_info()
                logger.error(f"Error in Line no : {error_line.tb_lineno}: due to {error_msg}")


    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

if __name__ == '__main__':
    try:
            obj = CUSTOMER_CHURN_PREDICTION('C:\\Users\\VARSHINI\\Downloads\\Telecom_Churn_Prediction\\Final_Dataset.csv')
            #obj.visualize()
            obj.handling_missing_values()
            obj.var_tranformation()
            obj.outliers()
            obj.encoding()
            obj.f_selection()
            obj.scaling_and_balancing()
            obj.train_all_models()
            obj.hyperparameter_tuning()
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
