import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan encoder
model = joblib.load('final_rf_model.pkl')
seasonality_encoder = joblib.load('seasonality_encoder.pkl')

# Judul aplikasi
st.markdown("<h1 style='text-align: center;'>Prediksi Jumlah Penjualan Produk<br>Toko Retail</h1>", unsafe_allow_html=True)


# Input pengguna
inventory_level = st.number_input("Inventory Level", min_value=0)
demand = st.number_input("Demand", min_value=0)
price = st.number_input("Price (IDR)", min_value=0.0)
discount = st.slider("Discount (%)", min_value=0.0, max_value=100.0, step=1.0)
competitor_price = st.number_input("Competitor Pricing", min_value=0.0)

seasonality = st.selectbox("Seasonality", ['Autumn', 'Spring', 'Summer', 'Winter'])
seasonality_encoded = seasonality_encoder.transform([seasonality])[0]

month = st.slider("Month", min_value=1, max_value=12, step=1)

category = st.selectbox("Kategori Produk", ['Clothing', 'Electronics', 'Furniture', 'Groceries', 'Toys'])
category_encoded = {
    'Category_Clothing': 0,
    'Category_Electronics': 0,
    'Category_Furniture': 0,
    'Category_Groceries': 0,
    'Category_Toys': 0
}
category_encoded[f'Category_{category}'] = 1

# Gabungkan semua fitur jadi 1 array
input_data = pd.DataFrame([{
    'Inventory Level': inventory_level,
    'Demand': demand,
    'Price': price,
    'Discount': discount,
    'Competitor Pricing': competitor_price,
    'Seasonality': seasonality_encoded,
    'Month': month,
    **category_encoded
}])

# Prediksi
if st.button("Prediksi Penjualan"):
    pred_log = model.predict(input_data)
    pred = np.expm1(pred_log[0])  # balikkan dari log
    st.success(f"Prediksi Jumlah Penjualan: {pred:.2f} unit")
