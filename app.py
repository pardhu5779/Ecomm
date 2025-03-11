import streamlit as st
import pandas as pd
import numpy as np
import pickle

def load_data():
    data = pd.read_csv("Ecommerce_Customers.csv")
    return data

import pickle
import streamlit as st

def load_pickle_file(filepath):
        with open(filepath, "rb") as file:
            return pickle.load(file)

import streamlit as st
import pickle
import numpy as np

# Function to load pickle files with error handling
def load_pickle_file(filepath):
    try:
        with open(filepath, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"Error: `{filepath}` not found. Ensure it is uploaded to your GitHub repository.")
        st.stop()

# Load Model & Scaler
model = load_pickle_file("lr_model.pkl")
scaler = load_pickle_file("scaler.pkl")

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

