import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = load_model("heart_model.keras")
scaler = joblib.load("scaler.pkl")  # save your scaler separately

# UI
st.title("Heart Disease Prediction App")
st.markdown("Enter the clinical parameters below to assess the risk of heart disease.")

# User Inputs with Descriptions
cp = st.selectbox(
    "Chest Pain Type",
    options=[0, 1, 2, 3],
    format_func=lambda x: ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"][x],
    help="0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic"
)

thalach = st.slider(
    "Maximum Heart Rate Achieved (thalach)",
    min_value=70, max_value=210, value=150,
    help="Measured in beats per minute (bpm)"
)

ca = st.selectbox(
    "Number of Major Vessels Colored by Fluoroscopy (ca)",
    options=[0, 1, 2, 3],
    help="Ranges from 0 to 3 — more vessels may indicate worse heart condition"
)

thal = st.selectbox(
    "Thalassemia Type",
    options=[1, 2, 3],
    format_func=lambda x: {1: "Fixed defect", 2: "Reversible defect", 3: "Normal"}[x],
    help="1: Fixed, 2: Reversible, 3: Normal"
)

oldpeak = st.slider(
    "ST Depression Induced by Exercise (oldpeak)",
    min_value=0.0, max_value=6.0, value=1.0,
    step=0.1,
    help="ST depression compared to rest; higher values may indicate risk"
)

slope = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    options=[0, 1, 2],
    format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
    help="0: Upsloping, 1: Flat, 2: Downsloping"
)

exang = st.selectbox(
    "Exercise Induced Angina (exang)",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes",
    help="1 = Yes, 0 = No — presence of chest pain during exercise"
)

# Risk Index calculation
risk_index = oldpeak * 0.5 + slope * (-0.3) + exang * (-0.2)

# Prediction
if st.button("Predict"):
    X_input = pd.DataFrame([[cp, thalach, ca, thal, risk_index]], columns=['cp', 'thalach', 'ca', 'thal', 'risk_index'])
    X_scaled = scaler.transform(X_input)
    prediction = model.predict(X_scaled)
    result = "At Risk of Heart Disease" if prediction > 0.5 else "Not At Risk"
    st.subheader(f"Prediction: {result}")
