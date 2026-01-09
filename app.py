import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Price Prediction App")
st.write("Predict the **selling price of a used car** or try sample examples below 👇")

st.markdown("---")

# -------------------------------
# Load Model & Features
# -------------------------------
model = pickle.load(open("car_price_model.pkl", "rb"))
model_features = pickle.load(open("model_features.pkl", "rb"))

# -------------------------------
# Example Data
# -------------------------------
examples = {
    "🚗 Budget Petrol Car": {
        "year": 2012,
        "present_price": 3.5,
        "kms": 60000,
        "owner": 1,
        "fuel": "Petrol",
        "seller": "Individual",
        "transmission": "Manual"
    },
    "🚙 Family Diesel Car": {
        "year": 2016,
        "present_price": 7.5,
        "kms": 45000,
        "owner": 0,
        "fuel": "Diesel",
        "seller": "Dealer",
        "transmission": "Manual"
    },
    "🚘 Premium Automatic Car": {
        "year": 2019,
        "present_price": 14.0,
        "kms": 20000,
        "owner": 0,
        "fuel": "Petrol",
        "seller": "Dealer",
        "transmission": "Automatic"
    }
}

# -------------------------------
# Example Selector
# -------------------------------
st.header("🔎 Try Sample Examples")

selected_example = st.selectbox(
    "Choose an example car to preview prediction",
    ["None"] + list(examples.keys())
)

if selected_example != "None":
    sample = examples[selected_example]
else:
    sample = {
        "year": 2015,
        "present_price": 5.0,
        "kms": 30000,
        "owner": 0,
        "fuel": "Petrol",
        "seller": "Dealer",
        "transmission": "Manual"
    }

st.markdown("---")

# -------------------------------
# Input Section
# -------------------------------
st.header("📝 Car Details")

year = st.slider("📅 Year of Purchase", 2000, datetime.now().year, sample["year"])
present_price = st.number_input("💰 Present Price (in lakhs)", 0.1, 50.0, sample["present_price"])
kms_driven = st.number_input("🚘 Kilometers Driven", 0, 500000, sample["kms"])
owner = st.selectbox("👤 Number of Previous Owners", [0, 1, 2, 3], index=sample["owner"])

fuel_type = st.selectbox(
    "⛽ Fuel Type",
    ["Petrol", "Diesel", "CNG"],
    index=["Petrol", "Diesel", "CNG"].index(sample["fuel"])
)

seller_type = st.selectbox(
    "🏪 Seller Type",
    ["Dealer", "Individual"],
    index=["Dealer", "Individual"].index(sample["seller"])
)

transmission = st.selectbox(
    "⚙️ Transmission",
    ["Manual", "Automatic"],
    index=["Manual", "Automatic"].index(sample["transmission"])
)

st.markdown("---")

# -------------------------------
# Feature Engineering
# -------------------------------
current_year = datetime.now().year
car_age = current_year - year

# -------------------------------
# Prepare Input
# -------------------------------
input_dict = {
    "Present_Price": present_price,
    "Kms_Driven": kms_driven,
    "Owner": owner,
    "Car_Age": car_age,
    "Fuel_Type_Diesel": 1 if fuel_type == "Diesel" else 0,
    "Fuel_Type_Petrol": 1 if fuel_type == "Petrol" else 0,
    "Seller_Type_Individual": 1 if seller_type == "Individual" else 0,
    "Transmission_Manual": 1 if transmission == "Manual" else 0
}

input_df = pd.DataFrame([input_dict])

for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[model_features]

# -------------------------------
# Prediction
# -------------------------------
st.header("🔍 Prediction Preview")

if st.button("👀 Preview Prediction"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 **Estimated Selling Price:** ₹ {prediction:.2f} Lakhs")

st.markdown("---")

st.markdown(
    """
    🔹 Try sample cars or enter your own details  
    🔹 Built using **Streamlit + Machine Learning**  
    🔹 Demo-ready project 🚀
    """
)
