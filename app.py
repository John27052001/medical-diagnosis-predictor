import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("models/diagnosis_model.pkl")

st.title("🧠 Medical Diagnosis Predictor")
st.write("Enter patient health information to predict diabetes risk.")

# Input fields
preg = st.number_input("Pregnancies", 0, 20, step=1)
glucose = st.number_input("Glucose", 0, 200, step=1)
bp = st.number_input("Blood Pressure", 0, 150, step=1)
skin = st.number_input("Skin Thickness", 0, 100, step=1)
insulin = st.number_input("Insulin", 0, 900, step=1)
bmi = st.number_input("BMI", 0.0, 70.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, step=0.01)
age = st.number_input("Age", 0, 120, step=1)

# When the user clicks Predict
if st.button("Predict"):
    input_data = pd.DataFrame([[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
                               columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                                        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])
    
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.error("⚠️ High Risk: Patient likely has diabetes.")
    else:
        st.success("✅ Low Risk: Patient likely does not have diabetes.")
