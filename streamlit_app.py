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

MIN_SEQ_LEN = 7  # safety lower bound


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
# SIDEBAR (include BOTH)
# ============================================================
st.sidebar.header("🔎 Filters")

location_option = st.sidebar.selectbox(
    "Location",
    ["Both"] + sorted(df["location"].unique())
)

if location_option == "Both":
    loc_df = df.copy()
else:
    loc_df = df[df["location"] == location_option].copy()

loc_df = loc_df.sort_values(["location", "date"]).reset_index(drop=True)

min_date, max_date = loc_df["date"].min(), loc_df["date"].max()

ref_date = st.sidebar.date_input("Use history up to date", max_date)
ref_date = pd.to_datetime(ref_date)

default_forecast_end = min(max_date + pd.Timedelta(days=7), ref_date + pd.Timedelta(days=14))
forecast_end_date = st.sidebar.date_input("Forecast until date", default_forecast_end)
forecast_end_date = pd.to_datetime(forecast_end_date)

loc_df_ref = loc_df[loc_df["date"] <= ref_date].copy()


# ============================================================
# EDA
# ============================================================
with st.expander("🌊 EDA: River Discharge Trend", expanded=True):
    fig = px.line(
        loc_df_ref,
        x="date",
        y="river_discharge_m3s",
        color="location" if location_option == "Both" else None,
        title=f"River Discharge — {location_option}",
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PREDICTION
# ============================================================
with st.expander("Flood Potential Prediction", expanded=True):
    if forecast_end_date <= ref_date:
        st.warning("Forecast until date must be AFTER history date.")
        st.stop()

    # Decide model sequence length based on selection
    if location_option == "Both":
        # Use minimum available days across locations to ensure each location can forecast
        available_by_loc = loc_df_ref.groupby("location")["river_discharge_m3s"].apply(lambda s: s.notna().sum())
        if available_by_loc.empty:
            st.error("No data available for prediction.")
            st.stop()
        MODEL_SEQ_LEN = int(available_by_loc.min())
    else:
        MODEL_SEQ_LEN = int(loc_df_ref["river_discharge_m3s"].notna().sum())

    if MODEL_SEQ_LEN < MIN_SEQ_LEN:
        st.error(
            f"Not enough daily rows for prediction. Need at least {MIN_SEQ_LEN} days, "
            f"but only {MODEL_SEQ_LEN} are available up to {ref_date.date()}."
        )
        st.stop()

    def forecast_for_one_location(one_loc_df_ref: pd.DataFrame) -> pd.DataFrame:
        """Forecast discharge day-by-day until forecast_end_date for a single location."""
        one_loc_df_ref = one_loc_df_ref.sort_values("date").reset_index(drop=True)

        seq = one_loc_df_ref.tail(MODEL_SEQ_LEN).copy()
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
                ignore_index=True,
            )

            cur_date = next_date

        return pd.DataFrame(preds)

    if st.button("🚀 Run Forecast"):
        start = time.time()

        if location_option == "Both":
            all_forecasts = []
            for loc, g in loc_df_ref.groupby("location"):
                g = g[g["river_discharge_m3s"].notna()].copy()
                if len(g) < MODEL_SEQ_LEN:
                    continue
                fc = forecast_for_one_location(g)
                if not fc.empty:
                    fc["location"] = loc
                    all_forecasts.append(fc)

            if not all_forecasts:
                st.error("No forecasts generated. Check data availability per location.")
                st.stop()

            forecast_df = pd.concat(all_forecasts, ignore_index=True)

        else:
            g = loc_df_ref[loc_df_ref["river_discharge_m3s"].notna()].copy()
            if len(g) < MODEL_SEQ_LEN:
                st.error("Not enough non-null discharge rows for the selected history date.")
                st.stop()
            forecast_df = forecast_for_one_location(g)
            forecast_df["location"] = location_option

        latency = round((time.time() - start) * 1000, 2)
        st.success(f"✅ Forecast completed in {latency} ms | Horizon: {forecast_df['date'].nunique()} day(s)")

        # Prepare historical plot
        hist_plot = loc_df_ref[["date", "location", "river_discharge_m3s"]].copy()
        hist_plot = hist_plot.rename(columns={"river_discharge_m3s": "value"})
        hist_plot["type"] = "Actual"

        # Prepare forecast plot
        fc_plot = forecast_df.rename(columns={"predicted_discharge": "value"}).copy()
        fc_plot["type"] = "Forecast"

        combined = pd.concat([hist_plot, fc_plot], ignore_index=True)

        figp = px.line(
            combined,
            x="date",
            y="value",
            color="location" if location_option == "Both" else None,
            line_dash="type",
            title=f"Actual (≤ {ref_date.date()}) vs Forecast (→ {forecast_end_date.date()}) — {location_option}",
        )
        figp.update_layout(hovermode="x unified")
        st.plotly_chart(figp, use_container_width=True)

        st.subheader("📅 Forecast Table")
        st.dataframe(forecast_df.sort_values(["location", "date"]).reset_index(drop=True))