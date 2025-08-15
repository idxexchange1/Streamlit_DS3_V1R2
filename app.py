# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
import joblib

# --------------------------
# Load saved artifacts
# --------------------------
@st.cache_resource
def load_model():
    import zlib
    import pickle

    with open('model_compressed.pkl', 'rb') as f:
        model_data = zlib.decompress(f.read())
        model = pickle.loads(model_data)
    with open('scaler_v1_r2.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('columns_v1_r2.pkl', 'rb') as f:
        columns = pickle.load(f)
    return model, scaler, columns

model_compressed, scaler, feature_columns = load_model()

# --------------------------
# Streamlit App
# --------------------------
st.title("🏠 Home Price Prediction App")
st.write("Enter property details below to estimate the close price.")

# --------------------------
# User-friendly inputs
# --------------------------
col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input("Bedrooms", min_value=0, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=0, value=2)
    living_area = st.number_input("Living Area (sqft)", min_value=0, value=1500)
    lot_size = st.number_input("Lot Size (sqft)", min_value=0, value=5000)
    stories = st.number_input("Stories", min_value=0, value=1)

with col2:
    building_age = st.number_input("Building Age (years)", min_value=0, value=10)
    parking_total = st.number_input("Total Parking Spaces", min_value=0, value=2)
    main_level_bedrooms = st.number_input("Main Level Bedrooms", min_value=0, value=1)
    latitude = st.number_input("Latitude", value=37.7749)
    longitude = st.number_input("Longitude", value=-122.4194)

county = st.selectbox(
    "County",
    sorted([col.replace("CountyOrParish_", "") for col in feature_columns if col.startswith("CountyOrParish_")])
)

fireplace = st.selectbox("Fireplace", ["No", "Yes"])
school = st.selectbox("Have School Nearby", ["No", "Yes"])
zip_group = st.selectbox("Zip Code Group", [0, 1, 2, 3, 4])

price_year = st.number_input("Price Year", min_value=2000, value=2024)

# --------------------------
# Map inputs to feature vector
# --------------------------
input_data = {col: 0 for col in feature_columns}

# Fill numeric values
input_data['BedroomsTotal'] = bedrooms
input_data['BathroomsTotalInteger'] = bathrooms
input_data['LivingArea'] = living_area
input_data['LotSizeSquareFeet'] = lot_size
input_data['Stories'] = stories
input_data['BuildingAge'] = building_age
input_data['ParkingTotal'] = parking_total
input_data['MainLevelBedrooms'] = main_level_bedrooms
input_data['Latitude'] = latitude
input_data['Longitude'] = longitude
input_data['PriceYear'] = price_year
input_data[f'ZipCodeGroup_{zip_group}'] = 1

# One-hot encode county
county_feature = f"CountyOrParish_{county.lower()}"
if county_feature in input_data:
    input_data[county_feature] = 1

# Boolean fields
input_data['FireplaceYN_True'] = 1 if fireplace == "Yes" else 0
input_data['FireplaceYN_False'] = 1 if fireplace == "No" else 0
input_data['HaveSchoolYN_True'] = 1 if school == "Yes" else 0
input_data['HaveSchoolYN_False'] = 1 if school == "No" else 0

# --------------------------
# Predict button
# --------------------------
if st.button("Predict Price"):
    input_df = pd.DataFrame([input_data])
    numeric_cols = input_df.select_dtypes(include=['int64', 'float64']).columns
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    prediction = model.predict(input_df)[0]
    st.subheader(f"💰 Estimated Close Price: ${prediction:,.2f}")
