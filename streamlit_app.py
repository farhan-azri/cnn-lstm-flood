import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import tensorflow as tf


DATA_PATH = "data/features_daily.csv"
MODEL_PATH = "run/cnn_lstm_model.h5"
SCALER_PATH = "run/scaler.pkl"
FEATURES_PATH = "run/feature_columns.pkl"

# Model was trained using a fixed window (likely 14).
# We'll keep a default and auto-reduce ONLY if not enough data.
MIN_SEQ_LEN = 7  # don't go too low or model becomes unstable


st.set_page_config(page_title="Flood Dashboard", layout="wide")
st.title("🌊 Flood Prediction Dashboard")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["location"] = df["location"].astype(str).str.strip()
    return df.sort_values(["location", "date"]).reset_index(drop=True)


@st.cache_resource
def load_artifacts_safe():
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURES_PATH)

    custom_objects = {
        "mse": tf.keras.losses.MeanSquaredError(),
        "mae": tf.keras.metrics.MeanAbsoluteError(),
        "mean_squared_error": tf.keras.losses.MeanSquaredError(),
        "mean_absolute_error": tf.keras.metrics.MeanAbsoluteError(),
    }

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model, scaler, feature_cols
    except Exception:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
        return model, scaler, feature_cols


# ---- file checks ----
missing = [p for p in [DATA_PATH, MODEL_PATH, SCALER_PATH, FEATURES_PATH] if not Path(p).exists()]
if missing:
    st.error("❌ Missing required files:")
    for m in missing:
        st.write("-", m)
    st.info("Run your pipeline first: python main.py --step all")
    st.stop()

df = load_data()
model, scaler, feature_cols = load_artifacts_safe()


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("🔎 Filters")

location = st.sidebar.selectbox("Location", sorted(df["location"].unique()))
loc_df = df[df["location"] == location].sort_values("date").reset_index(drop=True)

min_date, max_date = loc_df["date"].min(), loc_df["date"].max()

ref_date = st.sidebar.date_input("Use history up to date", max_date)
ref_date = pd.to_datetime(ref_date)

default_forecast_end = min(max_date + pd.Timedelta(days=7), ref_date + pd.Timedelta(days=14))
forecast_end_date = st.sidebar.date_input("Forecast until date", default_forecast_end)
forecast_end_date = pd.to_datetime(forecast_end_date)

loc_df_ref = loc_df[loc_df["date"] <= ref_date].copy()

DEFAULT_SEQ_LEN = int(loc_df_ref["river_discharge_m3s"].notna().sum())


# ============================================================
# EDA
# ============================================================
with st.expander("🌊 EDA: River Discharge Trend", expanded=True):
    fig = px.line(loc_df_ref, x="date", y="river_discharge_m3s", title=f"River Discharge — {location}")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# DYNAMIC SEQ LEN (based on available daily rows)
# ============================================================
# Count daily rows where target exists (acts like flood_daily availability)
available_days = int(loc_df_ref["river_discharge_m3s"].notna().sum())

# # User can optionally cap max sequence length
# user_max_seq_len = st.sidebar.slider(
#     "Max sequence length (days)",
#     min_value=MIN_SEQ_LEN,
#     max_value=max(DEFAULT_SEQ_LEN, MIN_SEQ_LEN),
#     value=DEFAULT_SEQ_LEN,
# )

# # Dynamic length: if data is short, reduce seq_len automatically
# seq_len = min(user_max_seq_len, available_days)

# Model input length is fixed — we must keep it at DEFAULT_SEQ_LEN.
# If seq_len < DEFAULT_SEQ_LEN, we will stop and tell user (unless you re-train model).
MODEL_SEQ_LEN = DEFAULT_SEQ_LEN


# ============================================================
# PREDICTION
# ============================================================
with st.expander("🤖 Prediction: Forecast until selected date", expanded=True):
    st.caption(
        "Forecast is generated day-by-day from the last N days up to your **history date**.\n\n"
        "⚠️ Assumption: future feature values are held constant at the last known day.\n"
        "For realistic forecasts, fetch weather forecast and rebuild features forward."
    )

    # st.write(f"📌 Available daily rows up to history date: **{available_days}**")
    # st.write(f"📌 Selected dynamic seq_len: **{seq_len}** days")
    # st.write(f"📌 Model expects fixed seq_len: **{MODEL_SEQ_LEN}** days")

    if forecast_end_date <= ref_date:
        st.warning("Forecast until date must be AFTER history date.")
        st.stop()

    if available_days < MODEL_SEQ_LEN:
        st.error(
            f"Not enough daily rows for this model. Need at least {MODEL_SEQ_LEN} days, "
            f"but only {available_days} are available up to {ref_date.date()}.\n\n"
            "✅ Fix options:\n"
            "- Choose a later 'Use history up to date'\n"
            "- Or retrain model with smaller SEQUENCE_LENGTH"
        )
        st.stop()

    # Always use MODEL_SEQ_LEN for the trained model
    seq = loc_df_ref.tail(MODEL_SEQ_LEN).copy()

    # st.subheader(f"Input Sequence (Last {MODEL_SEQ_LEN} Days)")
    st.dataframe(seq[["date"] + feature_cols + ["river_discharge_m3s"]])

    if st.button("🚀 Run Forecast"):
        start = time.time()

        window_features = seq[feature_cols].copy().reset_index(drop=True)
        last_feature_row = window_features.iloc[-1].copy()

        preds = []
        cur_date = seq["date"].max()

        while cur_date < forecast_end_date:
            X_scaled = scaler.transform(window_features.values)
            X_scaled = X_scaled.reshape(1, MODEL_SEQ_LEN, len(feature_cols))

            pred = float(model.predict(X_scaled, verbose=0)[0][0])

            next_date = cur_date + pd.Timedelta(days=1)
            preds.append({"date": next_date, "predicted_discharge": pred})

            window_features = pd.concat(
                [window_features.iloc[1:], pd.DataFrame([last_feature_row])],
                ignore_index=True
            )

            cur_date = next_date

        forecast_df = pd.DataFrame(preds)

        latency = round((time.time() - start) * 1000, 2)
        st.success(f"✅ Forecast completed in {latency} ms | Horizon: {len(forecast_df)} day(s)")

        hist_plot = loc_df_ref[["date", "river_discharge_m3s"]].copy()
        hist_plot["type"] = "Actual"
        hist_plot = hist_plot.rename(columns={"river_discharge_m3s": "value"})

        fc_plot = forecast_df.rename(columns={"predicted_discharge": "value"}).copy()
        fc_plot["type"] = "Forecast"

        combined = pd.concat([hist_plot, fc_plot], ignore_index=True)

        figp = px.line(
            combined,
            x="date",
            y="value",
            color="type",
            title=f"Actual (≤ {ref_date.date()}) vs Forecast (→ {forecast_end_date.date()}) — {location}",
        )
        figp.update_layout(hovermode="x unified")
        st.plotly_chart(figp, use_container_width=True)

        st.subheader("📅 Forecast Table")
        st.dataframe(forecast_df)