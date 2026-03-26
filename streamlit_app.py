import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import tensorflow as tf

# ============================================================
# CONFIG / PATHS
# ============================================================
DATA_PATH = Path("data/features_daily.csv")
RUN_DIR = Path("run")

MODEL_PATH = RUN_DIR / "cnn_lstm_model.keras"
SCALER_STATS_PATH = RUN_DIR / "scaler_stats.npz"
FEATURES_PATH = RUN_DIR / "feature_columns.json"
METADATA_PATH = RUN_DIR / "training_metadata.json"

st.set_page_config(page_title="Flood Prediction Dashboard", layout="wide")
st.title("🌊 AI Flood Prediction Dashboard")

# ============================================================
# HELPERS
# ============================================================
def validate_files():
    required = [DATA_PATH, MODEL_PATH, SCALER_STATS_PATH, FEATURES_PATH, METADATA_PATH]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        st.error("❌ Missing required files:")
        for m in missing:
            st.write(f"- {m}")
        st.stop()


def apply_scaler(X, stats):
    X = X.astype(np.float32)
    if stats["with_mean"]:
        X = X - stats["mean_"]
    if stats["with_std"]:
        scale = np.where(stats["scale_"] == 0, 1, stats["scale_"])
        X = X / scale
    return X


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["location"] = df["location"].astype(str)
    return df.sort_values(["location", "date"]).reset_index(drop=True)


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    stats_raw = np.load(SCALER_STATS_PATH)
    scaler_stats = {
        "mean_": stats_raw["mean_"],
        "scale_": stats_raw["scale_"],
        "with_mean": True,
        "with_std": True,
    }

    with open(FEATURES_PATH) as f:
        features = json.load(f)

    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    return model, scaler_stats, features, metadata


# ============================================================
# FORECAST FUNCTION (ENHANCED)
# ============================================================
def forecast_location(df_loc, model, scaler_stats, feature_cols, seq_len, forecast_days):
    df_loc = df_loc.dropna().copy()
    df_loc = df_loc.sort_values("date")

    if len(df_loc) < seq_len:
        return pd.DataFrame()

    window = df_loc.tail(seq_len).copy()
    preds = []

    for i in range(forecast_days):
        X = window[feature_cols].values
        X = apply_scaler(X, scaler_stats)
        X = X.reshape(1, seq_len, len(feature_cols))

        raw_pred = float(model.predict(X, verbose=0)[0][0])

        # Smooth prediction
        prev = window.iloc[-1]["river_discharge_m3s"]
        pred = max(0, 0.7 * raw_pred + 0.3 * prev)

        next_date = window.iloc[-1]["date"] + pd.Timedelta(days=1)

        # Create next row
        new_row = window.iloc[-1].copy()
        new_row["date"] = next_date
        new_row["river_discharge_m3s"] = pred

        # Update discharge features
        new_row["discharge_lag_1"] = pred
        new_row["discharge_delta"] = pred - window.iloc[-1]["river_discharge_m3s"]

        # Update cyclical time
        doy = next_date.dayofyear
        new_row["sin_doy"] = np.sin(2 * np.pi * doy / 365)
        new_row["cos_doy"] = np.cos(2 * np.pi * doy / 365)

        preds.append({"date": next_date, "predicted_discharge": pred})

        window = pd.concat([window.iloc[1:], pd.DataFrame([new_row])])

    return pd.DataFrame(preds)


# ============================================================
# LOAD
# ============================================================
validate_files()
df = load_data()
model, scaler_stats, feature_cols, metadata = load_artifacts()

SEQ_LEN = metadata.get("sequence_length", 14)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Settings")

location = st.sidebar.selectbox(
    "Location",
    ["All"] + sorted(df["location"].unique().tolist())
)

forecast_days = st.sidebar.slider("Forecast Days", 1, 14, 7)

if location == "All":
    df_filtered = df.copy()
else:
    df_filtered = df[df["location"] == location].copy()

# ============================================================
# RUN FORECAST
# ============================================================
if st.button("🚀 Run Prediction"):
    start = time.time()

    all_forecasts = []

    for loc, g in df_filtered.groupby("location"):
        fc = forecast_location(g, model, scaler_stats, feature_cols, SEQ_LEN, forecast_days)
        if not fc.empty:
            fc["location"] = loc
            all_forecasts.append(fc)

    if not all_forecasts:
        st.error("Not enough data for prediction.")
        st.stop()

    forecast_df = pd.concat(all_forecasts)

    latency = round((time.time() - start) * 1000, 2)
    st.success(f"Prediction completed in {latency} ms")

    # ============================================================
    # PLOT
    # ============================================================
    hist = df_filtered.rename(columns={"river_discharge_m3s": "value"})
    hist["type"] = "Actual"

    fc_plot = forecast_df.rename(columns={"predicted_discharge": "value"})
    fc_plot["type"] = "Forecast"

    combined = pd.concat([hist, fc_plot])

    fig = px.line(
        combined,
        x="date",
        y="value",
        color="location",
        line_dash="type",
        title="Actual vs Forecast"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # FLOOD RISK
    # ============================================================
    st.subheader("🚨 Flood Risk")

    latest = forecast_df["predicted_discharge"].iloc[-1]
    q90 = df["river_discharge_m3s"].quantile(0.9)
    q75 = df["river_discharge_m3s"].quantile(0.75)

    if latest >= q90:
        risk = "🔴 HIGH"
    elif latest >= q75:
        risk = "🟠 MEDIUM"
    else:
        risk = "🟢 LOW"

    st.metric("Predicted Discharge", f"{latest:.2f}")
    st.markdown(f"### Risk Level: {risk}")

    # ============================================================
    # MODEL INSIGHT
    # ============================================================
    st.subheader("🧠 Model Insight")
    st.info(
        "Model focuses heavily on recent rainfall trends, discharge momentum, "
        "and seasonal patterns (monsoon cycles)."
    )