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
        title="Extreme Rainfall vs River Discharge",
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
        df_weather = df_weather[df_weather["datetime"].dt.year == 2025]
        df_weather["date"] = df_weather["datetime"].dt.floor("D")

        # Flood preprocessing
        df_flood["datetime"] = pd.to_datetime(df_flood["date"])
        df_flood["location"] = df_flood["location"].astype(str).str.strip()
        df_flood = df_flood[df_flood["datetime"].dt.year == 2025]
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
            st.warning("No data available for selected location in 2025.")
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
                title="Hourly Rainfall vs Daily River Discharge (2025)",
                xaxis=dict(title="Datetime"),
                yaxis=dict(title="Rainfall (mm)", side="left"),
                yaxis2=dict(title="Discharge (m³/s)", overlaying="y", side="right"),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.1),
                bargap=0
            )

            st.plotly_chart(fig_combo, use_container_width=True)