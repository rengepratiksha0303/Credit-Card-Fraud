import streamlit as st
import pickle
import numpy as np

# Load Model
with open("KNN MODEL (1).pkl") as f:
    model = pickle.load(f)

# Load Scaler
with open("SCALER (2).pkl") as f:
    scaler = pickle.load(f)

st.title("Credit Card Fraud Detection")

st.write("Enter Transaction Details")

time = st.number_input("Time", value=0.0)
v1 = st.number_input("V1", value=0.0)
v2 = st.number_input("V2", value=0.0)
v3 = st.number_input("V3", value=0.0)
v4 = st.number_input("V4", value=0.0)
v5 = st.number_input("V5", value=0.0)
amount = st.number_input("Amount", value=0.0)

if st.button("Predict"):

    data = np.array([[time, v1, v2, v3, v4, v5, amount]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    if prediction[0] == 1:
        st.error("Fraudulent Transaction Detected")
    else:
        st.success("Legitimate Transaction")
