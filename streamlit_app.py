import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import tensorflow as tf
from pathlib import Path

from eda import run_eda_streamlit

DATA_PATH = "data/features_daily.csv"
MODEL_PATH = "run/cnn_lstm_model.h5"
SCALER_PATH = "run/scaler.pkl"
FEATURES_PATH = "run/feature_columns.pkl"

SEQ_LEN = 14

st.set_page_config(page_title="Flood Dashboard", layout="wide")
st.title("🌊 Flood Analytics & Prediction Dashboard (Hourly→Daily)")

if not Path(DATA_PATH).exists():
    st.error("Missing data/features_daily.csv. Run main.py --step all")
    st.stop()


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["location", "date"]).reset_index(drop=True)


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    return model, scaler, feature_cols


df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
loc_opt = st.sidebar.selectbox("Location", ["Both", "Klang", "Petaling"])
if loc_opt == "Both":
    dff = df.copy()
else:
    dff = df[df["location"] == loc_opt].copy()

min_d, max_d = dff["date"].min(), dff["date"].max()
dr = st.sidebar.date_input("Date range", [min_d, max_d])
start_d, end_d = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
dff = dff[dff["date"].between(start_d, end_d)]

with st.expander("📊 EDA", expanded=True):
    run_eda_streamlit(dff)

with st.expander("⚡ Extreme Events (95th percentile)"):
    p = st.slider("Percentile", 80, 99, 95)
    rain_thr = dff["rain_sum_mm"].quantile(p / 100)
    dis_thr = dff["river_discharge_m3s"].quantile(p / 100)

    tmp = dff.copy()
    tmp["extreme_rain"] = (tmp["rain_sum_mm"] >= rain_thr).astype(int)
    tmp["extreme_discharge"] = (tmp["river_discharge_m3s"] >= dis_thr).astype(int)
    tmp["extreme_both"] = ((tmp["extreme_rain"] == 1) & (tmp["extreme_discharge"] == 1)).astype(int)

    fig = px.line(tmp, x="date", y="river_discharge_m3s", color="location")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        tmp[tmp["extreme_both"] == 1][["date", "location", "rain_sum_mm", "river_discharge_m3s"]]
        .sort_values("date")
    )

with st.expander("🤖 Next-day Prediction (CNN-LSTM)"):
    if not (Path(MODEL_PATH).exists() and Path(SCALER_PATH).exists() and Path(FEATURES_PATH).exists()):
        st.warning("Model artifacts missing. Run main.py --step train")
        st.stop()

    model, scaler, feature_cols = load_artifacts()

    # Choose location for prediction
    pred_loc = st.selectbox("Predict for location", ["Klang", "Petaling"])

    g = df[df["location"] == pred_loc].sort_values("date").reset_index(drop=True)
    if len(g) < SEQ_LEN + 1:
        st.error("Not enough history for prediction.")
        st.stop()

    # Choose reference end date
    ref_date = st.date_input("Use history up to date", g["date"].max())
    ref_date = pd.to_datetime(ref_date)

    g = g[g["date"] <= ref_date]
    if len(g) < SEQ_LEN:
        st.error("Not enough rows up to selected date.")
        st.stop()

    seq = g.tail(SEQ_LEN)
    st.caption("Input sequence (last 14 days)")
    st.dataframe(seq[["date"] + feature_cols])

    if st.button("🚀 Predict next day discharge"):
        start = time.time()
        X_raw = seq[feature_cols].values
        X_scaled = scaler.transform(X_raw).reshape(1, SEQ_LEN, len(feature_cols))
        pred = float(model.predict(X_scaled, verbose=0)[0][0])
        latency = round((time.time() - start) * 1000, 2)

        next_day = seq["date"].max() + pd.Timedelta(days=1)
        st.metric("Predicted next-day discharge (m³/s)", round(pred, 3))
        st.write(f"Predicted date: {next_day.date()}  |  Latency: {latency} ms")

        # Plot last actual vs predicted point
        plot_df = pd.DataFrame({
            "date": list(seq["date"]) + [next_day],
            "value": list(seq["river_discharge_m3s"]) + [pred],
            "type": ["Actual"] * len(seq) + ["Predicted"]
        })
        figp = px.line(plot_df, x="date", y="value", title="Last 14 days actual + next-day prediction")
        st.plotly_chart(figp, use_container_width=True)