import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model and scaler
model_path = "lr_model.pkl"
scaler_path = "scaler.pkl"


with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit UI
st.title("Ecommerce Customer Spending Prediction")
st.write("Predict the yearly amount a customer is likely to spend based on their usage data.")

# User input fields
avg_session_length = st.number_input("Average Session Length (minutes)", min_value=0.0, format="%.2f")
time_on_app = st.number_input("Time on App (minutes)", min_value=0.0, format="%.2f")
time_on_website = st.number_input("Time on Website (minutes)", min_value=0.0, format="%.2f")
length_of_membership = st.number_input("Length of Membership (years)", min_value=0.0, format="%.2f")

# Prediction button
if st.button("Predict Yearly Spending"):
    input_data = np.array([[avg_session_length, time_on_app, time_on_website, length_of_membership]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    st.success(f"Predicted Yearly Amount Spent: ${prediction[0]:,.2f}")
