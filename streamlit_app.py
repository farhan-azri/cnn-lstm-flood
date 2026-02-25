import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

DATA_PATH = "data/features_daily.csv"

st.set_page_config(page_title="Flood EDA Dashboard", layout="wide")
st.title("🌧️🌊 Flood EDA Dashboard (Hourly→Daily Pipeline)")

if not Path(DATA_PATH).exists():
    st.error(f"❌ Missing {DATA_PATH}. Run: python main.py --step all")
    st.stop()


@st.cache_data
def load_features():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["location"] = df["location"].astype(str).str.strip()
    return df.sort_values(["location", "date"]).reset_index(drop=True)


df = load_features()

# Sidebar filters
st.sidebar.header("🔎 Filters")
loc_option = st.sidebar.selectbox("Location", ["Both", "Klang", "Petaling"])

if loc_option == "Both":
    dff = df.copy()
else:
    dff = df[df["location"].str.lower() == loc_option.lower()].copy()

min_d, max_d = dff["date"].min(), dff["date"].max()
dr = st.sidebar.date_input("Date range", [min_d, max_d])
start_d, end_d = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
dff = dff[dff["date"].between(start_d, end_d)]

# ============================================================
# EDA (based on your eda.py)
# ============================================================
with st.expander("📊 Rainfall Distribution", expanded=True):
    if "rain_sum_mm" not in dff.columns:
        st.error("Column 'rain_sum_mm' not found in features_daily.csv")
    else:
        fig1 = px.histogram(
            dff,
            x="rain_sum_mm",
            color="location" if loc_option == "Both" else None,
            nbins=50,
            title="Rainfall Distribution (Daily rain_sum_mm)"
        )
        st.plotly_chart(fig1, use_container_width=True)

with st.expander("🌊 River Discharge Trend", expanded=True):
    if "river_discharge_m3s" not in dff.columns:
        st.error("Column 'river_discharge_m3s' not found in features_daily.csv")
    else:
        fig2 = px.line(
            dff,
            x="date",
            y="river_discharge_m3s",
            color="location" if loc_option == "Both" else None,
            title="River Discharge Trend (Daily)"
        )
        fig2.update_layout(hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

with st.expander("🧾 Data Preview"):
    st.dataframe(dff.tail(100))