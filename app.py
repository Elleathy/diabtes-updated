import streamlit as st
import numpy as np
import pandas as pd 
import joblib

st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🧪 Patient Diabetes Prediction")

# Load pre-trained model
try:
    model = joblib.load("./Model/model.joblib")
except FileNotFoundError:
    st.error("❌ Model file 'model.joblib' not found. Please make sure it's in the same folder.")
    st.stop()

# Input form for patient features
df = pd.read_csv("Data/diabetes_cleaned.csv")
with st.form("prediction_form"):
    st.subheader("Enter Patient Information:")

    Pregnancies = st.slider("Pregnancies", int(df["Pregnancies"].min()), int(df["Pregnancies"].max()), step=1)
    Glucose = st.slider("Glucose", float(df["Glucose"].min()), float(df["Glucose"].max()))
    BloodPressure = st.slider("Blood Pressure", int(df["BloodPressure"].min()), int(df["BloodPressure"].max()))
    SkinThickness = st.slider("Skin Thickness", float(df["SkinThickness"].min()), float(df["SkinThickness"].max()))
    Insulin = st.slider("Insulin", int(df["Insulin"].min()), int(df["Insulin"].max()))
    BMI = st.slider("BMI", float(df["BMI"].min()), float(df["BMI"].max()))
    DPF = st.slider("Diabetes Pedigree Function", float(df["DiabetesPedigreeFunction"].min()), float(df["DiabetesPedigreeFunction"].max()))
    Age = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()), step=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DPF, Age]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Prediction: Patient is likely Diabetic (Confidence: {probability:.2f})")
    else:
        st.success(f"✅ Prediction: Patient is likely Not Diabetic (Confidence: {1 - probability:.2f})")
