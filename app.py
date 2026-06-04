import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Credit Card Fraud Detection")

st.title("Credit Card Fraud Detection")

# Load model and scaler
model = pickle.load(open("knn_model(1).pkl", "rb"))
scaler = pickle.load(open("scaler(1).pkl", "rb"))

st.subheader("Enter Transaction Details")

time = st.number_input("Time")
v1 = st.number_input("V1")
v2 = st.number_input("V2")
v3 = st.number_input("V3")
v4 = st.number_input("V4")
v5 = st.number_input("V5")
amount = st.number_input("Amount")

if st.button("Predict"):

    data = np.array([[time, v1, v2, v3, v4, v5, amount]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    if prediction[0] == 1:
        st.error("Fraudulent Transaction Detected")
    else:
        st.success("Legitimate Transaction")
