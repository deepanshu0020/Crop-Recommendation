import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Crop Recommendation App")

N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=20)
P = st.number_input("Phosphorous (P)", min_value=0, max_value=140, value=30)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=10)
temperature = st.number_input("Temperature (°C)", min_value=0, max_value=40, value=15)
humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=90)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=7.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0, max_value=300, value=100)

if st.button("Predict Crop"):
    input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_features)
    st.success(f"The suggested crop is: {prediction[0]}")