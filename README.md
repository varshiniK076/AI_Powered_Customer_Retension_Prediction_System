# AI_Powered_Customer_Retension_Prediction_System

## Project Overview

The **AI-Powered Customer Retention System** is designed to help telecommunication companies like **Jio, Vi, Airtel, and BSNL** predict customer churn and proactively retain high-value customers. By analyzing historical customer data—including demographics, subscription details, usage patterns, and service interactions—the system identifies customers at risk of leaving.

This project covers the complete machine learning pipeline, from data preprocessing and feature engineering to model training, evaluation, and deployment through a **Flask web application**. 

## Main Goal

The main goal of the **AI-Powered Customer Retention System** is to **reduce customer churn in the telecommunications industry** by proactively identifying at-risk customers and providing actionable insights to improve retention strategies.

## 📊 Dataset Description

The dataset contains **7,043 unique customer records** with **24 features**, including a mix of categorical (demographics, service details) and numerical (tenure, charges) variables. It captures historical telecom customer information from providers like **Jio, Vi, Airtel, and BSNL**, and is used to predict customer churn.

| Feature            | Description |
|--------------------|-------------|
| customerID         | Unique identifier for each customer |
| gender             | Gender of the customer (Male / Female) |
| SeniorCitizen      | Indicates if the customer is a senior citizen (0 = No, 1 = Yes) |
| Partner            | Whether the customer has a partner (Yes / No) |
| Dependents         | Whether the customer has dependents (Yes / No) |
| tenure             | Number of months the customer has stayed with the company |
| PhoneService       | Whether the customer has phone service (Yes / No) |
| MultipleLines      | Whether the customer has multiple phone lines (Yes / No / No phone service) |
| InternetService    | Type of internet service (DSL, Fiber optic, No) |
| OnlineSecurity     | Whether the customer has online security service (Yes / No / No internet service) |
| OnlineBackup       | Whether the customer has online backup service (Yes / No / No internet service) |
| DeviceProtection   | Whether the customer has device protection (Yes / No / No internet service) |
| TechSupport        | Whether the customer has technical support service (Yes / No / No internet service) |
| StreamingTV        | Whether the customer uses streaming TV (Yes / No / No internet service) |
| StreamingMovies    | Whether the customer uses streaming movies (Yes / No / No internet service) |
| Contract           | Contract type (Month-to-month, One year, Two year) |
| PaperlessBilling   | Whether the customer uses paperless billing (Yes / No) |
| PaymentMethod      | Payment method used by the customer |
| MonthlyCharges     | Monthly amount charged to the customer |
| TotalCharges       | Total amount charged to the customer |
| Churn              | Target variable indicating whether the customer churned (Yes / No) |
| Service_provider   | Telecom provider (Jio, Airtel, Vi, BSNL), derived from internet service |
| Region_Type        | Customer region (Urban, Suburban, Rural) |
| Device_Status      | Indicates whether the customer is using a new or old device |

### Dataset Categories
- **Target Variable:** `Churn` (Yes / No)  
- **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `Region_Type`  
- **Services:** `Service_provider`, `PhoneService`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`  
- **Account Information:** `tenure`, `Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`  

### Summary Statistics
- **Total Customers:** 7,043  
- **Churn Status:**  
  - No (Retained): 5,174 (73.5%)  
  - Yes (Churned): 1,869 (26.5%)  
- **Gender Distribution:**  
  - Male: 3,555  
  - Female: 3,488  
- **Service Providers:**  
  - Jio: 2,858  
  - Airtel: 2,113  
  - Vi: 1,377  
  - BSNL: 695  

This dataset provides a well-rounded view of customer demographics, account usage, and service preferences, which is ideal for building predictive models to identify at-risk customers and improve retention strategies.

---
## ⚙️ Project Structure

```text
AI_Customer_Retention_Prediction_System/
│
├─ data/
│   └─ final_dataset.csv
│
├─ app.py                     # Flask web application entry point
├─ main.py                    # Main script to run ML pipeline
├─ handling_missing_values.py # Script to handle missing data
├─ variable_transformation.py # Feature engineering and transformations
├─ outlier_handling.py        # Handle outliers in numerical features
├─ categorical_to_numeric.py  # Encode categorical variables 
├─ feature_selection.py       # Selecting only necessary features 
├─ Scaling_balancing.py       # Handle class imbalance (SMOTE) and Scaling data
├─ model_training.py          # Train and evaluate ML models using ROC_AUC
├─ tuning_model.py            # Hyperparameter tuning for best model
├─ log_code.py                # Optional logging for pipeline steps
│
├─ models/
│   ├─ best_logistic.pkl
│   ├─ scaler.pkl
│  
│
├─ templates/
│   └─ index.html             # Frontend HTML for Flask app
|
│
├─ requirements.txt           # Required Python packages
└─ README.md                  # Project documentation
├─ EDA images                 # Data visulization
├─  plots                     # normal distribution graphs
└─ outliers_images            # Outliers plotting using box-plot
