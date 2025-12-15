import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Regresi Pemasukan", layout="wide")

st.title("📈 Prediksi Pemasukan (Regresi Linear)")

# load model
model = joblib.load("model_regresi.pkl")

st.sidebar.header("📝 Input Data Transaksi")

harga = st.sidebar.number_input("Harga Produk (Rp)", min_value=0, value=2000)
terjual = st.sidebar.number_input("Jumlah Terjual", min_value=0, value=10)
modal = st.sidebar.number_input("Modal Satuan (Rp)", min_value=0, value=1000)

if st.sidebar.button("🔮 Prediksi Pemasukan"):
    input_df = pd.DataFrame({
        "Harga": [harga],
        "Terjual": [terjual],
        "Modal Satuan": [modal]
    })

    hasil = model.predict(input_df)[0]

    st.success(f"💰 Prediksi Pemasukan: Rp {hasil:,.0f}")
