import streamlit as st
import pandas as pd
import joblib
import json

# ---------------- LOAD MODEL ---------------- #
model = joblib.load('catboost_model.pkl')

with open('features.json', 'r') as f:
    features = json.load(f)

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Demand Forecasting", layout="centered")

st.title("📦 Demand Forecasting App")
st.markdown("Predict product demand based on business inputs")

st.divider()

# ---------------- DROPDOWN DATA ---------------- #
store_ids = [f"S{str(i).zfill(3)}" for i in range(1, 6)]
product_ids = [f"P{str(i).zfill(4)}" for i in range(1, 21)]

categories = ["Groceries", "Furniture", "Clothing", "Toys", "Electronics"]
regions = ["North", "South", "East", "West"]
weather_options = ["Sunny", "Rainy", "Cloudy", "Snowy"]
promotion_options = ["Yes", "No"]
seasonality_options = ["Winter", "Spring", "Summer", "Autumn"]
epidemic_options = ["Yes", "No"]
years = [2022, 2023, 2024]

# Month mapping
month_map = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}

# Weekday mapping
weekday_map = {
    "Mon": 0, "Tue": 1, "Wed": 2,
    "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6
}

# ---------------- INPUT SECTION ---------------- #
st.subheader("🧾 Input Features")

col1, col2 = st.columns(2)

with col1:
    store_id = st.selectbox("Store ID", store_ids)
    product_id = st.selectbox("Product ID", product_ids)
    category = st.selectbox("Category", categories)
    region = st.selectbox("Region", regions)

with col2:
    inventory = st.number_input("Inventory Level", min_value=0)
    price = st.number_input("Price", min_value=0.0)

    # 👇 USER sees 0–100
    discount_percent = st.slider("Discount (%)", 0, 100, 0)
    
    competitor_price = st.number_input("Competitor Pricing", min_value=0.0)

st.divider()

col3, col4 = st.columns(2)

with col3:
    promotion = st.selectbox("Promotion", promotion_options)
    weather = st.selectbox("Weather Condition", weather_options)

with col4:
    seasonality = st.selectbox("Seasonality", seasonality_options)
    epidemic = st.selectbox("Epidemic", epidemic_options)

st.divider()

# ---------------- TIME FEATURES ---------------- #
st.subheader("📅 Time Features")

col5, col6 = st.columns(2)

with col5:
    year = st.selectbox("Year", years)
    month_name = st.selectbox("Month", list(month_map.keys()))
    month = month_map[month_name]

with col6:
    day = st.slider("Day", 1, 31, 15)
    weekday_name = st.selectbox("Weekday", list(weekday_map.keys()))
    weekday = weekday_map[weekday_name]

# ---------------- DERIVED FEATURES ---------------- #
# Convert discount to 0–1 for model
discount = discount_percent / 100

discounted_price = price * (1 - discount)

# ---------------- PREDICTION ---------------- #
st.divider()

if st.button("🚀 Predict Demand"):

    input_dict = {
        'Store ID': store_id,
        'Product ID': product_id,
        'Category': category,
        'Region': region,
        'Inventory Level': inventory,
        'Price': price,
        'Discount': discount,
        'DiscountedPrice': discounted_price,
        'Promotion': promotion,
        'Weather Condition': weather,
        'Competitor Pricing': competitor_price,
        'Seasonality': seasonality,
        'Epidemic': epidemic,
        'Year': year,
        'Month': month,
        'Day': day,
        'Weekday': weekday
    }

    input_df = pd.DataFrame([input_dict])

    # Ensure correct column order
    input_df = input_df[features]

    prediction = model.predict(input_df)[0]

    # ---------------- OUTPUT ---------------- #
    st.subheader("📊 Prediction Result")

    st.metric("Predicted Demand", f"{round(prediction, 2)} units")

    if prediction > 120:
        st.success("🔥 High demand expected")
    elif prediction > 70:
        st.info("📈 Moderate demand")
    else:
        st.warning("📉 Low demand expected")
        
st.divider()
st.subheader("📌 Dataset Credits")
st.write("Dataset: [Retail Store Inventory and Demand Forecasting](https://www.kaggle.com/datasets/atomicd/retail-store-inventory-and-demand-forecasting)")
st.write("Author: [Wavelet](https://www.kaggle.com/atomicd)")