import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

# ============================================================
# PATHS
# ============================================================
DATA_PATH = Path("data/features_daily.csv")
RUN_DIR = Path("run")

MODEL_PATH = RUN_DIR / "cnn_lstm_model.keras"
SCALER_STATS_PATH = RUN_DIR / "scaler_stats.npz"
FEATURES_PATH = RUN_DIR / "feature_columns.json"
METADATA_PATH = RUN_DIR / "training_metadata.json"

MIN_SEQ_LEN = 7

st.set_page_config(page_title="Flood Dashboard", layout="wide")
st.title("🌊 Flood Prediction Dashboard")


# ============================================================
# FORECAST HELPERS
# ============================================================
def forecast_for_one_location(
    one_loc_df_ref: pd.DataFrame,
    model_obj,
    scaler_stats_obj: dict,
    feature_columns: list[str],
    seq_len: int,
    forecast_until: pd.Timestamp,
) -> pd.DataFrame:
    one_loc_df_ref = one_loc_df_ref.sort_values("date").reset_index(drop=True)

    clean = one_loc_df_ref.dropna(subset=feature_columns + [TARGET_COL]).copy()
    if len(clean) < seq_len:
        return pd.DataFrame(columns=["date", "predicted_discharge"])

    seq = clean.tail(seq_len).copy()
    window_features = seq[feature_columns].copy().reset_index(drop=True)
    last_feature_row = window_features.iloc[-1].copy()

    preds = []
    cur_date = seq["date"].max()

    while cur_date < forecast_until:
        X_scaled = apply_saved_standard_scaler(
            window_features.values,
            scaler_stats_obj
        )

        X_scaled = X_scaled.reshape(1, seq_len, len(feature_columns)).astype(np.float32)
        pred = float(model_obj.predict(X_scaled, verbose=0)[0][0])

        next_date = cur_date + pd.Timedelta(days=1)
        preds.append({
            "date": next_date,
            "predicted_discharge": pred,
        })

        # Keep same exogenous feature row for recursive forecasting
        window_features = pd.concat(
            [window_features.iloc[1:], pd.DataFrame([last_feature_row])],
            ignore_index=True
        )

        cur_date = next_date

    return pd.DataFrame(preds)

# ============================================================
# HELPERS
# ============================================================
def validate_required_files():
    required = [DATA_PATH, MODEL_PATH, SCALER_STATS_PATH, FEATURES_PATH, METADATA_PATH]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        st.error("❌ Missing required files:")
        for item in missing:
            st.write("-", item)
        st.info("Run the training pipeline first so the artifacts are generated.")
        st.stop()


def apply_saved_standard_scaler(X: np.ndarray, scaler_stats: dict) -> np.ndarray:
    """
    Rebuild StandardScaler transform behavior using plain saved arrays:
    z = (x - mean) / scale
    """
    mean_ = scaler_stats["mean_"]
    scale_ = scaler_stats["scale_"]
    with_mean = scaler_stats["with_mean"]
    with_std = scaler_stats["with_std"]

    X_out = X.astype(np.float32).copy()

    if with_mean:
        X_out = X_out - mean_

    if with_std:
        safe_scale = np.where(scale_ == 0, 1.0, scale_)
        X_out = X_out / safe_scale

    return X_out


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["location"] = df["location"].astype(str).str.strip()
    return df.sort_values(["location", "date"]).reset_index(drop=True)


@st.cache_resource
def load_artifacts():
    try:
        with open(FEATURES_PATH, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        raw = np.load(SCALER_STATS_PATH, allow_pickle=False)
        scaler_stats = {
            "mean_": raw["mean_"].astype(np.float32),
            "scale_": raw["scale_"].astype(np.float32),
            "var_": raw["var_"].astype(np.float32),
            "n_features_in_": int(raw["n_features_in_"][0]),
            "with_mean": bool(raw["with_mean"][0]),
            "with_std": bool(raw["with_std"][0]),
        }

        model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        return model, scaler_stats, feature_cols, metadata

    except Exception as e:
        st.error(f"Artifact loading failed: {type(e).__name__}: {e}")
        st.stop()


validate_required_files()
df = load_data()
model, scaler_stats, feature_cols, metadata = load_artifacts()

MODEL_SEQ_LEN = int(metadata.get("sequence_length", 14))
TARGET_COL = metadata.get("target_column", "river_discharge_m3s")

if len(feature_cols) != scaler_stats["n_features_in_"]:
    st.error(
        "Feature count mismatch between saved feature list and scaler stats. "
        "Please retrain and regenerate artifacts."
    )
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("🔎 Filters")

location_option = st.sidebar.selectbox(
    "Location",
    ["Both"] + sorted(df["location"].dropna().unique().tolist())
)

if location_option == "Both":
    loc_df = df.copy()
else:
    loc_df = df[df["location"] == location_option].copy()

loc_df = loc_df.sort_values(["location", "date"]).reset_index(drop=True)

if loc_df.empty:
    st.warning("No data available for the selected location.")
    st.stop()

min_date = loc_df["date"].min()
max_date = loc_df["date"].max()

ref_date = st.sidebar.date_input(
    "Use history up to date",
    value=max_date.date(),
    min_value=min_date.date(),
    max_value=max_date.date(),
)
ref_date = pd.to_datetime(ref_date)

default_forecast_end = min(
    max_date + pd.Timedelta(days=7),
    ref_date + pd.Timedelta(days=14)
)

forecast_end_date = st.sidebar.date_input(
    "Forecast until date",
    value=default_forecast_end.date(),
    min_value=(ref_date + pd.Timedelta(days=1)).date(),
)
forecast_end_date = pd.to_datetime(forecast_end_date)

loc_df_ref = loc_df[loc_df["date"] <= ref_date].copy()




# ============================================================
# PREDICTION
# ============================================================
with st.expander("🚀 Flood Potential Prediction", expanded=True):
    if forecast_end_date <= ref_date:
        st.warning("Forecast until date must be after the history date.")
        st.stop()

    if MODEL_SEQ_LEN < MIN_SEQ_LEN:
        st.error(
            f"Saved model sequence length is {MODEL_SEQ_LEN}, which is below the minimum "
            f"supported UI threshold of {MIN_SEQ_LEN}."
        )
        st.stop()

    if st.button("Run Forecast"):
        start = time.time()

        if location_option == "Both":
            all_forecasts = []

            for loc, g in loc_df_ref.groupby("location"):
                fc = forecast_for_one_location(
                    one_loc_df_ref=g,
                    model_obj=model,
                    scaler_stats_obj=scaler_stats,
                    feature_columns=feature_cols,
                    seq_len=MODEL_SEQ_LEN,
                    forecast_until=forecast_end_date,
                )

                if not fc.empty:
                    fc["location"] = loc
                    all_forecasts.append(fc)

            if not all_forecasts:
                st.error("No forecasts generated. Check data availability per location.")
                st.stop()

            forecast_df = pd.concat(all_forecasts, ignore_index=True)

        else:
            g = loc_df_ref[loc_df_ref["location"] == location_option].copy()

            forecast_df = forecast_for_one_location(
                one_loc_df_ref=g,
                model_obj=model,
                scaler_stats_obj=scaler_stats,
                feature_columns=feature_cols,
                seq_len=MODEL_SEQ_LEN,
                forecast_until=forecast_end_date,
            )

            if forecast_df.empty:
                st.error(
                    f"Not enough valid rows for forecasting. Need at least {MODEL_SEQ_LEN} "
                    f"rows with non-null target and features."
                )
                st.stop()

            forecast_df["location"] = location_option

        latency_ms = round((time.time() - start) * 1000, 2)
        horizon_days = int(forecast_df["date"].nunique())

        st.success(f"✅ Forecast completed in {latency_ms} ms | Horizon: {horizon_days} day(s)")

        hist_plot = loc_df_ref[["date", "location", TARGET_COL]].dropna().copy()
        hist_plot = hist_plot.rename(columns={TARGET_COL: "value"})
        hist_plot["type"] = "Actual"

        fc_plot = forecast_df.rename(columns={"predicted_discharge": "value"}).copy()
        fc_plot["type"] = "Forecast"

        combined_plot = pd.concat([hist_plot, fc_plot], ignore_index=True)

        figp = px.line(
            combined_plot,
            x="date",
            y="value",
            color="location" if location_option == "Both" else None,
            line_dash="type",
            title=f"Actual (≤ {ref_date.date()}) vs Forecast (→ {forecast_end_date.date()}) — {location_option}",
        )
        figp.update_layout(hovermode="x unified")
        st.plotly_chart(figp, use_container_width=True)

        # st.subheader("📅 Forecast Table")
        # st.dataframe(
        #     forecast_df.sort_values(["location", "date"]).reset_index(drop=True),
        #     use_container_width=True
        # )


        
# # ============================================================
# # EDA
# # ============================================================
# with st.expander("🌊 EDA: River Discharge Trend", expanded=True):
#     if loc_df_ref.empty:
#         st.warning("No historical data available up to the selected reference date.")
#     else:
#         fig = px.line(
#             loc_df_ref,
#             x="date",
#             y=TARGET_COL,
#             color="location" if location_option == "Both" else None,
#             title=f"River Discharge — {location_option}",
#         )
#         fig.update_layout(hovermode="x unified")
#         st.plotly_chart(fig, use_container_width=True)


# ============================================================
# EXTREME EVENT ANALYSIS
# ============================================================
with st.expander("⚡ Extreme Rainfall vs Extreme River Discharge", expanded=True):
    percentile = st.slider(
        "Extreme event percentile threshold",
        min_value=80,
        max_value=99,
        value=95,
        step=1
    )

    needed_cols = ["date", "location", "rain_sum_mm", TARGET_COL]
    missing_cols = [c for c in needed_cols if c not in loc_df_ref.columns]

    if missing_cols:
        st.warning(f"Missing columns for extreme event analysis: {missing_cols}")
    elif loc_df_ref.empty:
        st.warning("No data available for the selected filters.")
    else:
        combined = loc_df_ref[needed_cols].dropna().copy()

        if combined.empty:
            st.warning("No valid rows available after removing null values.")
        else:
            rain_threshold = combined["rain_sum_mm"].quantile(percentile / 100)
            discharge_threshold = combined[TARGET_COL].quantile(percentile / 100)

            combined["extreme_rain"] = np.where(combined["rain_sum_mm"] >= rain_threshold, 1, 0)
            combined["extreme_discharge"] = np.where(combined[TARGET_COL] >= discharge_threshold, 1, 0)
            combined["extreme_both"] = np.where(
                (combined["extreme_rain"] == 1) & (combined["extreme_discharge"] == 1),
                1,
                0
            )

            st.write(f"🌧️ Extreme Rainfall Threshold: ≥ {rain_threshold:.2f} mm")
            st.write(f"🌊 Extreme River Discharge Threshold: ≥ {discharge_threshold:.2f} m³/s")

            fig_extreme = go.Figure()

            if location_option == "Both":
                for loc in combined["location"].unique():
                    sub = combined[combined["location"] == loc]
                    fig_extreme.add_trace(
                        go.Scatter(
                            x=sub["date"],
                            y=sub[TARGET_COL],
                            mode="lines",
                            name=f"{loc} Discharge",
                            opacity=0.6,
                        )
                    )
            else:
                fig_extreme.add_trace(
                    go.Scatter(
                        x=combined["date"],
                        y=combined[TARGET_COL],
                        mode="lines",
                        name="River Discharge",
                        opacity=0.6,
                    )
                )

            discharge_pts = combined[combined["extreme_discharge"] == 1]
            rain_pts = combined[combined["extreme_rain"] == 1]
            both_pts = combined[combined["extreme_both"] == 1]

            fig_extreme.add_trace(
                go.Scatter(
                    x=discharge_pts["date"],
                    y=discharge_pts[TARGET_COL],
                    mode="markers",
                    name=f"Extreme Discharge (≥{percentile}th%)",
                    marker=dict(color="red", size=7),
                )
            )

            fig_extreme.add_trace(
                go.Scatter(
                    x=rain_pts["date"],
                    y=rain_pts[TARGET_COL],
                    mode="markers",
                    name=f"Extreme Rainfall (≥{percentile}th%)",
                    marker=dict(color="royalblue", size=7),
                )
            )

            fig_extreme.add_trace(
                go.Scatter(
                    x=both_pts["date"],
                    y=both_pts[TARGET_COL],
                    mode="markers",
                    name="Concurrent Extreme (Rain + Discharge)",
                    marker=dict(color="purple", size=10, line=dict(color="black", width=1)),
                )
            )

            fig_extreme.update_layout(
                title=f"Combined Extreme Events: Rainfall vs River Discharge — {location_option}",
                xaxis_title="Date",
                yaxis_title="River Discharge (m³/s)",
                hovermode="x unified",
            )
            st.plotly_chart(fig_extreme, use_container_width=True)

            # st.subheader("📅 Concurrent Extreme Events")
            # extreme_both_table = combined[combined["extreme_both"] == 1][[
            #     "date",
            #     "location",
            #     "rain_sum_mm",
            #     TARGET_COL,
            #     "extreme_rain",
            #     "extreme_discharge",
            #     "extreme_both",
            # ]].sort_values("date")

            # st.dataframe(extreme_both_table, use_container_width=True)
            # st.write(f"🔢 Total concurrent extreme events: {len(extreme_both_table)}")

            # corr_val = combined["rain_sum_mm"].corr(combined[TARGET_COL])

            # summary = pd.DataFrame({
            #     "Metric": [
            #         "Total Records",
            #         "Extreme Rainfall Days",
            #         "Extreme Discharge Days",
            #         "Concurrent Extreme Days",
            #         "Rain–Discharge Correlation",
            #     ],
            #     "Value": [
            #         len(combined),
            #         int(combined["extreme_rain"].sum()),
            #         int(combined["extreme_discharge"].sum()),
            #         int(combined["extreme_both"].sum()),
            #         round(corr_val, 3) if pd.notna(corr_val) else None,
            #     ]
            # })

            # st.subheader("📊 Summary")
            # st.dataframe(summary, use_container_width=True)

            # if location_option == "Both":
            #     location_summary = (
            #         combined.groupby("location")[["extreme_rain", "extreme_discharge", "extreme_both"]]
            #         .sum()
            #         .reset_index()
            #         .melt(id_vars="location", var_name="Event Type", value_name="Count")
            #     )

            #     location_summary["Event Type"] = location_summary["Event Type"].replace({
            #         "extreme_rain": "Extreme Rain",
            #         "extreme_discharge": "Extreme Discharge",
            #         "extreme_both": "Concurrent Both",
            #     })

            #     fig_bar = px.bar(
            #         location_summary,
            #         x="location",
            #         y="Count",
            #         color="Event Type",
            #         barmode="group",
            #         title="Extreme Event Counts by Location",
            #     )
            #     st.plotly_chart(fig_bar, use_container_width=True)




