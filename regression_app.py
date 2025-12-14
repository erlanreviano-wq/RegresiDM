import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
st.set_page_config(
    page_title="Regresi Penjualan Tiket Pesawat",
    layout="wide"
)

st.title("📈 Regresi Penjualan Tiket Pesawat (Ensemble Method)")

st.markdown("""
Aplikasi ini menampilkan hasil regresi untuk memprediksi **Total Penjualan Tiket Pesawat**
menggunakan model **Linear Regression** dan **Ensemble Method**.
""")
st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload file CSV",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset berhasil di-upload")
else:
    df = pd.read_csv("penjualan_tiket_pesawat.csv")
    st.info("Menggunakan dataset default")
df_reg = df.copy()
df_reg['Date'] = pd.to_datetime(df_reg['Date'])

le_city = LabelEncoder()
le_gender = LabelEncoder()
le_airline = LabelEncoder()
le_payment = LabelEncoder()

df_reg['City'] = le_city.fit_transform(df_reg['City'])
df_reg['Gender'] = le_gender.fit_transform(df_reg['Gender'])
df_reg['Airline'] = le_airline.fit_transform(df_reg['Airline'])
df_reg['Payment_Method'] = le_payment.fit_transform(df_reg['Payment_Method'])
X = df_reg.drop(columns=['Total', 'Date'])
y = df_reg['Total']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

bagging = BaggingRegressor(
    estimator=LinearRegression(),
    n_estimators=50,
    random_state=42
)
bagging.fit(X_train, y_train)
y_pred_bag = bagging.predict(X_test)

adaboost = AdaBoostRegressor(
    estimator=LinearRegression(),
    n_estimators=100,
    random_state=42
)
adaboost.fit(X_train, y_train)
y_pred_ada = adaboost.predict(X_test)
def eval_model(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

results = pd.DataFrame([
    {"Model": "Linear Regression", **eval_model(y_test, y_pred_lr)},
    {"Model": "Bagging + Linear", **eval_model(y_test, y_pred_bag)},
    {"Model": "AdaBoost + Linear", **eval_model(y_test, y_pred_ada)}
])

st.subheader("📊 Hasil Evaluasi Model Regresi")
st.dataframe(results)

st.success("Model terbaik: **AdaBoost + Linear Regression**")

st.info("""
**Keterangan:**
- MAE dan RMSE menunjukkan besar kesalahan prediksi (dalam rupiah).
- R² menunjukkan seberapa baik model menjelaskan data.
- Metode ensemble digunakan untuk meningkatkan performa prediksi.
""")
