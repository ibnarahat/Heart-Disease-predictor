import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("model.pkl")

# Page configuration
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below to predict the likelihood of heart disease.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=40)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.number_input("Chest Pain Type (0–3)", min_value=0, max_value=3, value=0)
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
restecg = st.number_input("Resting ECG (0–2)", min_value=0, max_value=2, value=1)
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.number_input("Slope (0–2)", min_value=0, max_value=2, value=1)
ca = st.number_input("Number of Major Vessels (0–3)", min_value=0, max_value=3, value=0)
thal = st.number_input("Thal (0–3)", min_value=0, max_value=3, value=2)

# Convert categorical inputs
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# Prepare data
features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])
input_df = pd.DataFrame(features)

# Prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        result = int(prediction[0])

        if result == 1:
            st.error("⚠️ High Risk: The patient is likely to have heart disease.")
        else:
            st.success("✅ Low Risk: The patient is unlikely to have heart disease.")
    except Exception as e:
        st.warning(f"Error during prediction: {str(e)}")
