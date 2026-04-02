import streamlit as st
import joblib
import os
import gdown
import numpy as np

# -----------------------------
# Load saved files
# -----------------------------
if not os.path.exists("Forestmodel.pkl"):
    url = "https://drive.google.com/file/d/1wwU-BlXKmCLhXbczQCINxPDBAgtcCVa9/view?usp=drive_link"
    gdown.download(url, "model.pkl", quiet=False)
model = joblib.load("Forestmodel.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="House Price Predictor", layout="wide")

# -----------------------------
# Title
# -----------------------------
st.title("🏠 House Price Prediction App")
st.markdown("### Enter property details below")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("ℹ️ About")
st.sidebar.write("Predict house prices using Machine Learning")

# -----------------------------
# Input Section
# -----------------------------
col1, col2 = st.columns(2)

# Left side inputs
# LEFT COLUMN
with col1:
    longitude = st.slider(
        "📍 Longitude",
        min_value=-125.0,
        max_value=-114.0,
        value=-120.0,
        step=0.01
    )

    latitude = st.slider(
        "📍 Latitude",
        min_value=32.0,
        max_value=42.0,
        value=35.0,
        step=0.01
    )

    housing_median_age = st.slider("🏡 House Age (years)", 1, 100, 20)
    total_rooms = st.number_input("🛏 Total Rooms", value=1000)
    total_bedrooms = st.number_input("🛌 Total Bedrooms", value=200)

# Right side inputs
with col2:
    population = st.number_input("👨‍👩‍👧 Population", value=300)
    households = st.number_input("🏠 Households", value=100)
    median_income = st.number_input("💰 Median Income", value=3.0)

    ocean = st.selectbox(
        "🌊 Ocean Proximity",
        ["INLAND", "NEAR OCEAN", "NEAR BAY", "<1H OCEAN", "ISLAND"]
    )

# -----------------------------
# Build input dictionary (IMPORTANT FIX)
# -----------------------------
input_dict = dict.fromkeys(columns, 0)

# Fill numeric features
input_dict["longitude"] = longitude
input_dict["latitude"] = latitude
input_dict["housing_median_age"] = housing_median_age
input_dict["total_rooms"] = total_rooms
input_dict["total_bedrooms"] = total_bedrooms
input_dict["population"] = population
input_dict["households"] = households
input_dict["median_income"] = median_income

# Handle categorical encoding
ocean_col = f"ocean_proximity_{ocean}"
if ocean_col in input_dict:
    input_dict[ocean_col] = 1

# Convert to array
input_array = np.array([list(input_dict.values())])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price 💰"):

    try:
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)

        st.success(f"🏡 Estimated Price: ₹ {prediction[0]:,.2f}")

    except Exception as e:
        st.error("❌ Error in prediction")
        st.write(e)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("Built with ❤️ by Tamanna")