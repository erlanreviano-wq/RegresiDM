import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Regresi Pemasukan (Ensemble Method)",
    layout="wide"
)

st.title("📈 Regresi Pemasukan (Ensemble Method)")
st.write(
    "Aplikasi ini memprediksi **Pemasukan** berdasarkan data transaksi "
    "menggunakan **Linear Regression** dan **Ensemble Method**."
)

st.sidebar.header("📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload file catatan.csv",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Menggunakan dataset upload")
else:
    df = pd.read_csv("catatan.csv")
    st.sidebar.info("Menggunakan dataset default")

st.subheader("📄 Dataset Awal")
st.dataframe(df.head())

df_proc = df.copy()

# Tanggal
df_proc["Tanggal"] = pd.to_datetime(df_proc["Tanggal"])
df_proc["Year"] = df_proc["Tanggal"].dt.year
df_proc["Month"] = df_proc["Tanggal"].dt.month
df_proc["Day"] = df_proc["Tanggal"].dt.day
df_proc.drop(columns=["Tanggal"], inplace=True)

# Encoding kategorikal
for col in df_proc.select_dtypes(include="object").columns:
    df_proc[col] = LabelEncoder().fit_transform(df_proc[col])

# Cleaning
df_proc = df_proc.dropna().drop_duplicates()

target = "Pemasukan"

X = df_proc.drop(columns=[target])
y = df_proc[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

bag = BaggingRegressor(
    estimator=LinearRegression(),
    n_estimators=50,
    random_state=42
)
bag.fit(X_train, y_train)
pred_bag = bag.predict(X_test)

ada = AdaBoostRegressor(
    estimator=LinearRegression(),
    n_estimators=50,
    random_state=42
)
ada.fit(X_train, y_train)
pred_ada = ada.predict(X_test)

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Bagging + Linear",
        "AdaBoost + Linear"
    ],
    "MAE": [
        mean_absolute_error(y_test, pred_lr),
        mean_absolute_error(y_test, pred_bag),
        mean_absolute_error(y_test, pred_ada)
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test, pred_lr)),
        np.sqrt(mean_squared_error(y_test, pred_bag)),
        np.sqrt(mean_squared_error(y_test, pred_ada))
    ],
    "R2": [
        r2_score(y_test, pred_lr),
        r2_score(y_test, pred_bag),
        r2_score(y_test, pred_ada)
    ]
})

st.subheader("📊 Hasil Evaluasi Model")
st.dataframe(results)

st.subheader("📈 Perbandingan R² Model")

fig, ax = plt.subplots()
sns.barplot(
    data=results,
    x="Model",
    y="R2",
    ax=ax
)
ax.set_title("Perbandingan R² Model Regresi")
ax.set_ylabel("R² Score")
ax.set_xlabel("Model")
plt.xticks(rotation=20)

st.pyplot(fig)

st.subheader("📉 Distribusi Error (Actual vs Predicted)")

fig, ax = plt.subplots()
ax.scatter(y_test, pred_ada, alpha=0.6)
ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "--r"
)
ax.set_xlabel("Pemasukan Aktual")
ax.set_ylabel("Pemasukan Prediksi")
ax.set_title("Actual vs Predicted (AdaBoost + Linear)")

st.pyplot(fig)

best_model = results.sort_values("RMSE").iloc[0]["Model"]

st.success(f"✅ Model terbaik berdasarkan RMSE: **{best_model}**")

st.markdown(
    """
### 📝 Keterangan:
- **MAE & RMSE** menunjukkan besar kesalahan prediksi (dalam rupiah).
- **R²** menunjukkan seberapa baik model menjelaskan variasi data.
- **Ensemble Method** digunakan untuk meningkatkan stabilitas dan performa prediksi.
"""
)
