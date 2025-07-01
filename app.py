import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model (no scaling used)
model = joblib.load("diabetes_model_simple (1).joblib")

# App title and description
st.title("Diabetes Prediction App")
st.markdown("This app predicts whether a person is likely to have diabetes based on their medical information.")

# Sidebar for user input
st.sidebar.header("Enter Patient Data")

def user_input():
    pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=200, value=120)
    blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=140, value=70)
    skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    diabetes_pedigree = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=33)

    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }

    return pd.DataFrame(data, index=[0])

# Collect user input
input_df = user_input()

# Ensure column order matches the model training
expected_columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
input_data = input_df[expected_columns]

# Show input
st.subheader("Entered Patient Data:")
st.write(input_data)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader("Prediction:")
    st.success("Diabetic" if prediction[0] == 1 else "Not Diabetic")

    st.subheader("Prediction Probability:")
    st.info(f"Probability of being diabetic: {prediction_proba[0][1]:.2f}")
