import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained CatBoost model
with open("diabetes_cat_model.pkl", "rb") as file:
    cat_model = pickle.load(file)

# Custom CSS for larger fonts and scrollable content
st.markdown(
    """
    <style>
    body {
        font-size: 20px !important;
    }
    .stTextInput, .stNumberInput, .stRadio {
        font-size: 18px !important;
    }
    .stButton>button {
        font-size: 22px !important;
        padding: 10px;
    }
    .main {
        overflow-y: auto;
        height: 80vh; /* Makes it scrollable */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Description
st.title("🩺 Early Stage Diabetes Prediction")
st.write("Enter the patient details below to predict the likelihood of diabetes.")

# Scrollable main content
with st.container():
    age = st.number_input("📅 Age", min_value=16, max_value=90, value=30, step=1)
    gender = st.radio("⚤ Gender", ["Male", "Female"])
    polyuria = st.radio("🚰 Polyuria (Excessive Urination)", ["No", "Yes"])
    polydipsia = st.radio("🥤 Polydipsia (Excessive Thirst)", ["No", "Yes"])
    sudden_weight_loss = st.radio("⚖️ Sudden Weight Loss", ["No", "Yes"])
    weakness = st.radio("💪 Weakness", ["No", "Yes"])
    polyphagia = st.radio("🍔 Polyphagia (Excessive Hunger)", ["No", "Yes"])
    genital_thrush = st.radio("🦠 Genital Thrush", ["No", "Yes"])
    visual_blurring = st.radio("👀 Visual Blurring", ["No", "Yes"])
    itching = st.radio("🤕 Itching", ["No", "Yes"])
    irritability = st.radio("😠 Irritability", ["No", "Yes"])
    delayed_healing = st.radio("⏳ Delayed Healing", ["No", "Yes"])
    partial_paresis = st.radio("🦵 Partial Paresis (Muscle Weakness)", ["No", "Yes"])
    muscle_stiffness = st.radio("🏋️ Muscle Stiffness", ["No", "Yes"])
    alopecia = st.radio("🧑‍🦲 Alopecia (Hair Loss)", ["No", "Yes"])
    obesity = st.radio("⚖️ Obesity", ["No", "Yes"])

# Convert categorical inputs to numerical
def convert_binary(value):
    return 1 if value == "Yes" else 0

gender = 1 if gender == "Male" else 0
features = [
    age, gender, convert_binary(polyuria), convert_binary(polydipsia),
    convert_binary(sudden_weight_loss), convert_binary(weakness),
    convert_binary(polyphagia), convert_binary(genital_thrush),
    convert_binary(visual_blurring), convert_binary(itching),
    convert_binary(irritability), convert_binary(delayed_healing),
    convert_binary(partial_paresis), convert_binary(muscle_stiffness),
    convert_binary(alopecia), convert_binary(obesity)
]

# Centered Predict Button
st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
if st.button("🔎 Predict", help="Click to check the diabetes prediction"):
    input_data = pd.DataFrame([features], columns=[
        "Age", "Gender", "Polyuria", "Polydipsia", "sudden weight loss", "weakness",
        "Polyphagia", "Genital thrush", "visual blurring", "Itching",
        "Irritability", "delayed healing", "partial paresis", "muscle stiffness",
        "Alopecia", "Obesity"
    ])
    
    prediction = cat_model.predict(input_data)
    prediction_proba = cat_model.predict_proba(input_data)

    # Display the results
    st.subheader("🔬 Prediction Result")
    if prediction[0] == 1:
        st.markdown(
            '<p style="color:red; font-size:24px; font-weight:bold;">🚨 High Risk of Diabetes! Please consult a doctor. 🚨</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<p style="color:green; font-size:24px; font-weight:bold;">✅ Low Risk of Diabetes! Keep maintaining a healthy lifestyle. ✅</p>',
            unsafe_allow_html=True
        )

    # Show probability scores
    st.subheader("📈 Prediction Probability")
    st.write(f"🔴 **Chance of having diabetes:** `{prediction_proba[0][1] * 100:.2f}%`")
    st.write(f"🟢 **Chance of not having diabetes:** `{prediction_proba[0][0] * 100:.2f}%`")
