import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('best_random_forest_model.pkl')

# Define the input fields
st.title("Kenya Unemployment Rate Prediction App")
st.write("App by Milan Kumar Behera")
st.write("Email Address: milanbeherazyx@gmail.com")
st.write("This app predicts the Total Unemployment in Kenya based on provided features.")

# Create input fields for the data format you provided
male_labor_participation = st.number_input("Male Labor Participation (%)", min_value=0.0, max_value=100.0, value=50.0)
population_growth = st.number_input("Population Growth Rate (%)", min_value=-5.0, max_value=5.0, value=1.0)
year = st.number_input("Year", min_value=2000, max_value=2021, value=2021)
female_labor_participation = st.number_input("Female Labor Participation (%)", min_value=0.0, max_value=100.0, value=50.0)
real_gdp_ksh = st.number_input("Real GDP (in Ksh)", min_value=0, value=1000000)

# Create a feature vector from user inputs
feature_vector = np.array([male_labor_participation, population_growth, year, female_labor_participation, real_gdp_ksh]).reshape(1, -1)

# Make predictions
if st.button("Predict"):
    prediction = model.predict(feature_vector)
    st.success(f"Predicted Total Unemployment: {prediction[0]:.2f}")

# Display model details and evaluation metrics
st.write("Model Details: Random Forest Regressor")
st.write("Best Hyperparameters: {'bootstrap': True, 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 100}")
st.write("Root Mean Squared Error (RMSE): 2.0310")
st.write("Mean Absolute Error (MAE): 1.9610")
st.write("R2 Score: -13.7480")

# If you want to display additional information or data, you can do so here.
