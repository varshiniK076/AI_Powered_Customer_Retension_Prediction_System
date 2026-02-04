import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# --- 1. SETUP PATHS & LOAD MODEL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_logistic_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("Model and Scaler loaded successfully.")
    print(f"Model expects {model.n_features_in_} features.")
except Exception as e:
    print(f"Error loading files: {e}")
    model = None
    scaler = None

# --- 2. EXACT 36 FEATURES FROM YOUR PICKLE FILE ---
EXPECTED_COLUMNS = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender',
    'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Device_Status',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
    'Service_Provider_BSNL', 'Service_Provider_Jio', 'Service_Provider_Vi',
    'Region_Type_Suburban', 'Region_Type_Urban'
]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return render_template('index.html', prediction_text="Error: Model not loaded")

    try:
        form = request.form

        # --- 3. PREPARE INPUT DATAFRAME ---
        # Start with all zeros
        input_df = pd.DataFrame(np.zeros((1, len(EXPECTED_COLUMNS))), columns=EXPECTED_COLUMNS)

        # --- 4. MAP NUMERIC FEATURES ---
        input_df['SeniorCitizen'] = int(form.get('senior', 0))
        input_df['tenure'] = float(form.get('tenure', 0))
        input_df['MonthlyCharges'] = float(form.get('monthly_charges', 0))
        input_df['TotalCharges'] = float(form.get('total_charges', 0))

        # --- 5. MAP LABEL ENCODED FEATURES (0/1) ---
        # Assuming standard encoding: Male=1, Yes=1, New=1
        input_df['gender'] = 1 if form.get('gender') == 'Male' else 0
        input_df['Partner'] = 1 if form.get('partner') == 'Yes' else 0
        input_df['Dependents'] = 1 if form.get('dependents') == 'Yes' else 0
        input_df['PhoneService'] = 1 if form.get('phone_service') == 'Yes' else 0
        input_df['PaperlessBilling'] = 1 if form.get('paperless') == 'Yes' else 0
        input_df['Device_Status'] = 1 if form.get('device_type') == 'New' else 0

        # --- 6. MAP ONE-HOT FEATURES ---

        # Service Provider (The User Selection)
        provider = form.get('service_provider')
        if provider == 'BSNL':
            input_df['Service_Provider_BSNL'] = 1
        elif provider == 'Jio':
            input_df['Service_Provider_Jio'] = 1
        elif provider == 'Vi':
            input_df['Service_Provider_Vi'] = 1
        # If Airtel is selected, all above are 0 (Baseline)

        # Region
        region = form.get('region')
        if region == 'Suburban':
            input_df['Region_Type_Suburban'] = 1
        elif region == 'Urban':
            input_df['Region_Type_Urban'] = 1

        # Internet Service
        internet = form.get('internet_service')
        if internet == 'Fiber optic':
            input_df['InternetService_Fiber optic'] = 1
        elif internet == 'No':
            input_df['InternetService_No'] = 1

        # Multiple Lines
        ml = form.get('multiple_lines')
        if ml == 'Yes':
            input_df['MultipleLines_Yes'] = 1
        elif ml == 'No phone service':
            input_df['MultipleLines_No phone service'] = 1

        # Contract
        contract = form.get('contract')
        if contract == 'One year':
            input_df['Contract_One year'] = 1
        elif contract == 'Two year':
            input_df['Contract_Two year'] = 1

        # Payment Method
        pm = form.get('payment')
        if pm == 'Credit card (automatic)':
            input_df['PaymentMethod_Credit card (automatic)'] = 1
        elif pm == 'Electronic check':
            input_df['PaymentMethod_Electronic check'] = 1
        elif pm == 'Mailed check':
            input_df['PaymentMethod_Mailed check'] = 1

        # Other Services (Loop)
        services_map = {
            'online_security': 'OnlineSecurity',
            'online_backup': 'OnlineBackup',
            'device_protection': 'DeviceProtection',
            'tech_support': 'TechSupport',
            'streaming_tv': 'StreamingTV',
            'streaming_movies': 'StreamingMovies'
        }
        for field, col_prefix in services_map.items():
            val = form.get(field)
            if val == 'Yes':
                input_df[f'{col_prefix}_Yes'] = 1
            elif val == 'No internet service':
                input_df[f'{col_prefix}_No internet service'] = 1

        # --- 7. SCALE & PREDICT ---
        input_scaled = scaler.transform(input_df)
        prob = model.predict_proba(input_scaled)[:, 1][0]

        # DEBUG LOG
        print(f"Provider: {provider}, Tenure: {input_df['tenure'][0]}, Prob: {prob:.4f}")

        # THRESHOLD (Lowered slightly to catch riskier customers)
        THRESHOLD = 0.4
        prediction = 1 if prob > THRESHOLD else 0

        if prediction == 1:
            res_text = "High Churn Risk"
            res_class = "risk"
            confidence = round(prob * 100, 1)
            msg = f"Alert: {confidence}% probability of leaving."
        else:
            res_text = "Likely to Stay"
            res_class = "safe"
            confidence = round((1 - prob) * 100, 1)
            msg = f"Safe: {confidence}% probability of staying."

        return render_template('index.html',
                               prediction_text=res_text,
                               result_class=res_class,
                               message=msg,
                               prob=prob)  # Pass probability for gauge

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)