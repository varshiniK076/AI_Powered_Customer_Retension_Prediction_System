# 🤖💡 AI_Powered_Customer_Retension_Prediction_System

## 📝 Project Overview

The **AI-Powered Customer Retention System** is designed to help telecommunication companies like **Jio, Vi, Airtel, and BSNL** predict customer churn and proactively retain high-value customers. By analyzing historical customer data—including demographics, subscription details, usage patterns, and service interactions—the system identifies customers at risk of leaving.
This project covers the complete machine learning pipeline, from data preprocessing and feature engineering to model training, evaluation, and deployment through a **Flask web application**. 

## 🎯 Main Goal
The main goal of the **AI-Powered Customer Retention System** is to **reduce customer churn in the telecommunications industry** by proactively identifying at-risk customers and providing actionable insights to improve retention strategies.

## 📁 Dataset Description
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

### 🏷️ Dataset Categories
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
---
## 🗂️ Project Structure

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
```
---


## 🔄 ML - Pipeline

The Telecom Churn Prediction System uses a structured machine learning pipeline to clean, transform, and prepare customer data for modeling. It handles missing values, skewed distributions, outliers, categorical encoding, feature selection, scaling, and class imbalance. After preprocessing, multiple models are trained, the best one is selected and optimized, and finally deployed via a Flask web application for real-time churn prediction. This pipeline ensures robust, accurate, and scalable predictions. for readme file code

### 📊 Data Visualization & Exploration
- **Library:** Matplotlib  
- **Techniques used:** Pie charts, bar charts, stacked/multi-group charts, grids/subplots  
- **Purpose:** Understand churn distribution, customer demographics, and numeric feature patterns  
- **Selected approach:** Matplotlib for full control and interpretability

---

### 🛠️ Feature Engineering

**1. Handling Missing Values**  
- **Techniques tested:** Mean, Median, Mode, Forward/Backward Fill, KNN, MICE  
- **Selected:** Random Imputation – preserves original distribution by filling missing values with random samples from the column  

**2. Data Separation**  
- **Techniques:** Manual separation using `pandas.select_dtypes()`  
- **Selected:** Split into Numerical & Categorical for targeted preprocessing  

**3. Variable Transformation**  
- **Techniques tested:** Log, Box-Cox, Yeo-Johnson, Quantile Transformation  
- **Selected per feature:**  
  - `tenure` → Yeo-Johnson  
  - `MonthlyCharges` → Quantile  
  - `TotalCharges` → Quantile  
- **Reason:** Chosen transformation minimized skewness and kurtosis for more normal-like distributions  

**4. Handling Outliers**  
- **Techniques tested:** IQR, Winsorization, MAD, Quantile Clipping  
- **Selected per feature:**  
  - `SeniorCitizen` → IQR  
  - `tenure` → Original (no action needed)  
  - `MonthlyCharges` → IQR  
  - `TotalCharges` → IQR  
- **Reason:** IQR minimized extreme values while preserving data distribution  

**5. Categorical Encoding**  
- **Techniques tested:** One-Hot Encoding, Label Encoding, Frequency Encoding  
- **Selected:**  
  - Multi-class → One-Hot  
  - Binary → Label Encoding  
- **Reason:** Maintains consistent numeric representation for ML models  

**Outcome:** Fully cleaned, numeric dataset ready for modeling  

---

### 🔍 Feature Selection
- **Techniques tested:** Constant/Quasi-Constant, Pearson Correlation, Chi-Square  
- **Selected:** All features retained  
- **Reason:** All features statistically significant; no variance or correlation issues  

---

### 🔗 Merging Features
- **Technique:** Concatenate numeric and categorical features using `pd.concat(axis=1)`  
- **Outcome:** Final datasets (`X_training_data`, `X_testing_data`) for model training  

---

### ⚖️ Data Balancing
- **Techniques tested:** Undersampling, Oversampling, SMOTE  
- **Selected:** SMOTE applied on training set  
- **Reason:** Creates synthetic minority samples while preserving original distribution, improving model learning  

---

### 📏 Feature Scaling
- **Techniques tested:** StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler  
- **Selected:** StandardScaler  
- **Reason:** Produced lowest combined skewness and kurtosis, ensuring uniform scale and stable learning  

---

### 🏋️ Model Training & Selection
- **Models tested:** KNN, Naive Bayes, Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, SVM  
- **Evaluation metric:** ROC-AUC (preferred over accuracy for imbalanced data)  
- **Selected:** Logistic Regression (ROC-AUC = 0.83)  
- **Reason:** Best balance of performance, interpretability, and efficiency; outputs probability via sigmoid function  

---

### ⚙️ Hyperparameter Tuning
- **Techniques tested:** GridSearchCV, RandomizedSearchCV  
- **Selected:** GridSearchCV (5-fold CV, ROC-AUC metric)  
- **Best parameters:** `C=100`, `penalty=l2`, `solver=saga`, `max_iter=1000`  
- **Performance:** CV ROC-AUC = 0.8611 | Test ROC-AUC = 0.8309  
- **Reason:** Guarantees optimal combination for best model performance  

---

### 🌐 Deployment (Flask Web App)
- **Frontend:** HTML forms with dropdowns for categorical features, real-time prediction display  
- **Backend:** Loads trained model, scaler, encoding mappings; preprocesses input; outputs prediction + probability  
- **Reason:** Ensures consistent preprocessing, real-time predictions, and user-friendly interface for business use

---
## **📦 Install required packages**
> ```
> pip install -r requirements.txt
> ```

## **🚀 Run the Project**
> ```
> python main.py
> python app.py
> ```
---
### **Open your browser and navigate to**
> ```
> http://127.0.0.1:5000/
> ```

---

## **👤 Author**
 ```
 Varadhana Varshini Kolipakula
 Machine Learning & Data Science Enthusiast
 ```

---
