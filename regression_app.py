import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
st.set_page_config(
    page_title="Regresi Pemasukan (Ensemble)",
    layout="wide"
)

st.title("📈 Prediksi Pemasukan (Bagging + Linear Regression)")

st.write(
    """
Aplikasi ini menggunakan **ensemble method Bagging Regressor**
dengan **Linear Regression** sebagai base estimator
untuk memprediksi **Pemasukan** berdasarkan data transaksi.
"""
)
if not os.path.exists("model_regresi.pkl"):
    st.error("❌ File model_regresi.pkl tidak ditemukan. Pastikan sudah di-upload ke GitHub.")
    st.stop()

model = joblib.load("model_regresi.pkl")

if os.path.exists("catatan.csv"):
    df = pd.read_csv("catatan.csv")
    st.subheader("📄 Contoh Dataset")
    st.dataframe(df.head())
else:
    st.warning("Dataset catatan.csv tidak ditemukan.")

st.subheader("🧮 Input Data Transaksi")

col1, col2, col3 = st.columns(3)

with col1:
    harga = st.number_input(
        "Harga Produk (Rp)",
        min_value=0,
        value=2000,
        step=500
    )

with col2:
    terjual = st.number_input(
        "Jumlah Terjual",
        min_value=0,
        value=10,
        step=1
    )

with col3:
    modal = st.number_input(
        "Modal Satuan (Rp)",
        min_value=0,
        value=1000,
        step=500
    )

if st.button("🔮 Prediksi Pemasukan"):
    input_df = pd.DataFrame({
        "Harga": [harga],
        "Terjual": [terjual],
        "Modal Satuan": [modal]
    })

    prediksi = model.predict(input_df)[0]

    # pembulatan opsional agar realistis (rupiah)
    prediksi_bulat = round(prediksi / 1000) * 1000

    st.success(f"💰 Prediksi Pemasukan: **Rp {prediksi_bulat:,.0f}**")

    st.caption(
        "Catatan: Nilai dibulatkan ke ribuan rupiah agar sesuai konteks mata uang."
    )

st.markdown(
    """
### 📝 Penjelasan Model
- **Model dasar**: Linear Regression  
- **Ensemble method**: Bagging (Bootstrap Aggregating)  
- Model dilatih di Google Colab dan **tidak dilatih ulang di Streamlit**  
- Streamlit hanya digunakan untuk **inferensi / prediksi**
"""
)
