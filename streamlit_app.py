from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# PATHS
# ============================================================
DATA_PATH = Path("data/features_daily.csv")

st.set_page_config(page_title="Flood Dashboard", layout="wide")
st.title("🌊 Flood Analysis Dashboard")

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["location"] = df["location"].astype(str).str.strip()
    return df.sort_values(["location", "date"]).reset_index(drop=True)

df = load_data()

@st.cache_data
def load_forecast():
    forecast_path = Path("data/flood_forecast_output.csv")

    if not forecast_path.exists():
        return None

    df = pd.read_csv(forecast_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["location"] = df["location"].astype(str).str.strip()

    return df.sort_values(["location", "date"]).reset_index(drop=True)

forecast_df = load_forecast()

# ============================================================
# SIDEBAR FILTER
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

if loc_df.empty:
    st.warning("No data available for the selected location.")
    st.stop()



# ============================================================
# SIDEBAR FILTER (ROBUST DATE RANGE + QUICK SELECT)
# ============================================================
st.sidebar.header("📅 Date Filter")

# Ensure datetime types
min_date = pd.to_datetime("2025-01-01")
max_date = df["date"].max()

# -------------------------
# Helper: clamp date within bounds
# -------------------------
def clamp_date(date, min_d, max_d):
    return max(min(date, max_d), min_d)

# -------------------------
# Initialize session state
# -------------------------
if "start_date" not in st.session_state:
    st.session_state.start_date = min_date

if "end_date" not in st.session_state:
    st.session_state.end_date = max_date

# -------------------------
# Quick Select Buttons
# -------------------------
st.sidebar.markdown("### ⚡ Quick Select")

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("Last 3 Months"):
        proposed = max_date - pd.DateOffset(months=3)
        st.session_state.start_date = clamp_date(proposed, min_date, max_date)
        st.session_state.end_date = max_date

    if st.button("Last 6 Months"):
        proposed = max_date - pd.DateOffset(months=6)
        st.session_state.start_date = clamp_date(proposed, min_date, max_date)
        st.session_state.end_date = max_date

with col2:
    if st.button("Last 1 Year"):
        proposed = max_date - pd.DateOffset(years=1)
        st.session_state.start_date = clamp_date(proposed, min_date, max_date)
        st.session_state.end_date = max_date

    if st.button("Last 2 Years"):
        proposed = max_date - pd.DateOffset(years=2)
        st.session_state.start_date = clamp_date(proposed, min_date, max_date)
        st.session_state.end_date = max_date

# -------------------------
# Date Picker (always visible)
# -------------------------
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(
        st.session_state.start_date.date(),
        st.session_state.end_date.date()
    ),
    min_value=min_date.date(),
    max_value=max_date.date()
)

# -------------------------
# Sync manual selection back to session
# -------------------------
if isinstance(date_range, tuple):
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
else:
    start_date = pd.to_datetime(date_range)
    end_date = pd.to_datetime(date_range)

# Clamp again (safety)
start_date = clamp_date(start_date, min_date, max_date)
end_date = clamp_date(end_date, min_date, max_date)

# Update session
st.session_state.start_date = start_date
st.session_state.end_date = end_date

# -------------------------
# Apply filter
# -------------------------
loc_df = df[
    (df["date"] >= start_date) &
    (df["date"] <= end_date)
].copy()

if loc_df.empty:
    st.warning("No data available for selected date range.")
    st.stop()

# -------------------------
# Display selected range
# -------------------------
st.sidebar.markdown(
    f"**Selected:** {start_date.date()} → {end_date.date()}"
)



# ============================================================
# ⚡ EXTREME EVENT ANALYSIS
# ============================================================
with st.expander("⚡ Extreme Rainfall vs Extreme River Discharge", expanded=True):

    percentile = st.slider(
        "Extreme event percentile threshold",
        min_value=80,
        max_value=99,
        value=95,
        step=1
    )

    combined = loc_df[["date", "location", "rain_sum_mm", "river_discharge_m3s", "data_type"]].dropna().copy()

    # Thresholds
    rain_threshold = combined["rain_sum_mm"].quantile(percentile / 100)
    discharge_threshold = combined["river_discharge_m3s"].quantile(percentile / 100)

    combined["extreme_both"] = (
        (combined["rain_sum_mm"] >= rain_threshold) &
        (combined["river_discharge_m3s"] >= discharge_threshold)
    )

    st.write(f"🌧️ Rain Threshold: ≥ {rain_threshold:.2f} mm")
    st.write(f"🌊 Discharge Threshold: ≥ {discharge_threshold:.2f} m³/s")

    # Plot
    fig_extreme = go.Figure()

    for dtype in combined["data_type"].unique():
        sub_type = combined[combined["data_type"] == dtype]

        for loc in sub_type["location"].unique():
            sub = sub_type[sub_type["location"] == loc]

            fig_extreme.add_trace(
                go.Scatter(
                    x=sub["date"],
                    y=sub["river_discharge_m3s"],
                    mode="lines",
                    name=f"{loc} ({dtype})",
                    line=dict(dash="solid" if dtype == "historical" else "dash"),
                    opacity=0.6,
                )
            )

    # Extreme points
    both_pts = combined[combined["extreme_both"]]

    fig_extreme.add_trace(
        go.Scatter(
            x=both_pts["date"],
            y=both_pts["river_discharge_m3s"],
            mode="markers",
            name="Extreme (Rain + Discharge)",
            marker=dict(color="purple", size=10, line=dict(color="black", width=1)),
        )
    )

    fig_extreme.update_layout(
        title=f"Extreme Rainfall vs River Discharge ({start_date.date()} → {end_date.date()})",
        xaxis_title="Date",
        yaxis_title="River Discharge (m³/s)",
        hovermode="x unified",
    )

    st.plotly_chart(fig_extreme, use_container_width=True)

# ============================================================
# 🌧️ HOURLY RAINFALL vs 🌊 DAILY DISCHARGE
# ============================================================
with st.expander("🌧️ Hourly Rainfall vs 🌊 Daily River Discharge", expanded=True):

    weather_path = Path("data/weather_hourly.csv")
    flood_path = Path("data/flood_daily.csv")

    if not weather_path.exists() or not flood_path.exists():
        st.warning("Missing weather_hourly.csv or flood_daily.csv")
    else:
        df_weather = pd.read_csv(weather_path)
        df_flood = pd.read_csv(flood_path)

        # Weather preprocessing
        df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])
        df_weather["location"] = df_weather["location"].astype(str).str.strip()
        df_weather = df_weather[(df_weather["datetime"].dt.date >= start_date.date()) & (df_weather["datetime"].dt.date <= end_date.date())]
        df_weather["date"] = df_weather["datetime"].dt.floor("D")

        # Flood preprocessing
        df_flood["datetime"] = pd.to_datetime(df_flood["date"])
        df_flood["location"] = df_flood["location"].astype(str).str.strip()
        df_flood = df_flood[(df_flood["datetime"].dt.date >= start_date.date()) & (df_flood["datetime"].dt.date <= end_date.date())]
        df_flood["date"] = df_flood["datetime"].dt.floor("D")

        # Merge
        merged = pd.merge(
            df_weather,
            df_flood[["date", "location", "river_discharge_m3s"]],
            on=["date", "location"],
            how="left"
        )

        merged["river_discharge_m3s"] = merged["river_discharge_m3s"].fillna(0)

        if location_option != "Both":
            merged = merged[merged["location"] == location_option]

        if merged.empty:
            st.warning("No data available for selected location.")
        else:
            fig_combo = go.Figure()

            fig_combo.add_trace(
                go.Scatter(
                    x=merged["datetime"],
                    y=merged["rain"],
                    name="Hourly Rainfall (mm)",
                    mode="lines",
                    yaxis="y1"
                )
            )

            fig_combo.add_trace(
                go.Bar(
                    x=merged["datetime"],
                    y=merged["river_discharge_m3s"],
                    name="Daily River Discharge (m³/s)",
                    opacity=0.3,
                    yaxis="y2"
                )
            )

            fig_combo.update_layout(
                title=f"Hourly Rainfall vs Daily River Discharge ({start_date.date()} → {end_date.date()})",
                xaxis=dict(title="Datetime"),
                yaxis=dict(title="Rainfall (mm)", side="left"),
                yaxis2=dict(title="Discharge (m³/s)", overlaying="y", side="right"),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.1),
                bargap=0
            )

            st.plotly_chart(fig_combo, use_container_width=True)


# ============================================================
# 🔮 FORECAST DATE FILTER (SEPARATE)
# ============================================================
if forecast_df is not None and not forecast_df.empty:

    st.sidebar.header("🔮 Forecast Date Filter")

    forecast_min_date = forecast_df["date"].min()
    forecast_max_date = forecast_df["date"].max()

    forecast_range = st.sidebar.date_input(
        "Select Forecast Date Range",
        value=(
            forecast_min_date.date(),
            forecast_max_date.date()
        ),
        min_value=forecast_min_date.date(),
        max_value=forecast_max_date.date(),
        key="forecast_date_range"
    )

    if isinstance(forecast_range, tuple):
        f_start = pd.to_datetime(forecast_range[0])
        f_end = pd.to_datetime(forecast_range[1])
    else:
        f_start = pd.to_datetime(forecast_range)
        f_end = pd.to_datetime(forecast_range)

    forecast_df = forecast_df[
        (forecast_df["date"] >= f_start) &
        (forecast_df["date"] <= f_end)
    ].copy()



# ============================================================
# 🔮 FLOOD FORECAST VISUALIZATION
# ============================================================
with st.expander("🔮 Flood Forecast Analysis", expanded=True):

    if forecast_df is None or forecast_df.empty:
        st.warning("No forecast data available. Run prediction pipeline first.")
    else:
        plot_df = forecast_df.copy()

        if location_option != "Both":
            plot_df = plot_df[plot_df["location"] == location_option]

        if plot_df.empty:
            st.warning("No forecast data for selected location.")
        else:
            fig_forecast = go.Figure()

            # Predicted discharge line
            fig_forecast.add_trace(
                go.Scatter(
                    x=plot_df["date"],
                    y=plot_df["predicted_discharge"],
                    mode="lines+markers",
                    name="Predicted Discharge (m³/s)",
                    line=dict(color="blue", width=3),
                )
            )

            # Flood risk markers
            risk_colors = {
                "LOW": "green",
                "MEDIUM": "orange",
                "HIGH": "red"
            }

            for risk, color in risk_colors.items():
                sub = plot_df[plot_df["flood_risk"] == risk]

                fig_forecast.add_trace(
                    go.Scatter(
                        x=sub["date"],
                        y=sub["predicted_discharge"],
                        mode="markers",
                        name=f"{risk} Risk",
                        marker=dict(size=10, color=color, line=dict(color="black", width=1)),
                    )
                )

            fig_forecast.update_layout(
                title="🔮 Predicted River Discharge & Flood Risk",
                xaxis_title="Date",
                yaxis_title="Discharge (m³/s)",
                hovermode="x unified",
                legend=dict(orientation="h", y=1.1)
            )

            st.plotly_chart(fig_forecast, use_container_width=True)

            # ========================================================
            # 📊 Summary Metrics
            # ========================================================
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "🔴 High Risk Days",
                (plot_df["flood_risk"] == "HIGH").sum()
            )

            col2.metric(
                "🟠 Medium Risk Days",
                (plot_df["flood_risk"] == "MEDIUM").sum()
            )

            col3.metric(
                "🟢 Low Risk Days",
                (plot_df["flood_risk"] == "LOW").sum()
            )