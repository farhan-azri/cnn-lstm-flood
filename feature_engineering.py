import pandas as pd
from pathlib import Path

def build_features(
    weather_hourly_path: str = "data/weather_hourly.csv",
    flood_daily_path: str = "data/flood_daily.csv",
    out_path: str = "data/features_daily.csv",
    verbose: bool = True,
):
    """
    1) Read hourly weather
    2) Aggregate to daily per location (added temp ranges for evaporation proxy)
    3) Create advanced daily lag/rolling/EMA/delta features
    4) Merge with daily discharge target
    """

    if not Path(weather_hourly_path).exists():
        raise FileNotFoundError(f"Missing {weather_hourly_path}")
    if not Path(flood_daily_path).exists():
        raise FileNotFoundError(f"Missing {flood_daily_path}")

    w = pd.read_csv(weather_hourly_path)
    f = pd.read_csv(flood_daily_path)

    # --- Normalize keys (date + location) ---
    w["datetime"] = pd.to_datetime(w["datetime"], errors="coerce")
    w = w.dropna(subset=["datetime"]).copy()
    w["date"] = w["datetime"].dt.floor("D")

    f["date"] = pd.to_datetime(f["date"], errors="coerce").dt.floor("D")
    f = f.dropna(subset=["date"]).copy()

    w["location"] = w["location"].astype(str).str.strip()
    f["location"] = f["location"].astype(str).str.strip()

    # --- 1. Enhanced Daily Aggregation ---
    daily_weather = (
        w.groupby(["location", "date"])
        .agg(
            rain_sum_mm=("rain", "sum"),
            precip_sum_mm=("precipitation", "sum"),
            temp_mean=("temperature_2m", "mean"),
            temp_max=("temperature_2m", "max"), # NEW: For evaporation proxy
            temp_min=("temperature_2m", "min"), # NEW: For evaporation proxy
            wind_max=("wind_speed_10m", "max"),
            gust_max=("wind_gusts_10m", "max"),
            rain_max_1h=("rain", "max"),
        )
        .reset_index()
        .sort_values(["location", "date"])
    )

    # --- 2. Hydrological Proxies ---
    # Diurnal Temperature Range: High range often means clear skies, dry air, and high evaporation (dries the soil).
    daily_weather["temp_diurnal_range"] = daily_weather["temp_max"] - daily_weather["temp_min"]

    # --- 3. Lags (Past context) ---
    for lag in [1, 2, 3, 7]:
        daily_weather[f"rain_sum_lag_{lag}"] = daily_weather.groupby("location")["rain_sum_mm"].shift(lag)
        daily_weather[f"rain_max1h_lag_{lag}"] = daily_weather.groupby("location")["rain_max_1h"].shift(lag)

    # --- 4. Cumulative Rolling Sums (Antecedent Moisture) ---
    # Rolling sums tell the model if the ground is saturated from previous days.
    for window in [3, 7, 14]:
        daily_weather[f"rain_sum_roll{window}"] = (
            daily_weather.groupby("location")["rain_sum_mm"]
            .transform(lambda x: x.rolling(window, min_periods=1).sum())
        )

    # --- 5. Exponential Moving Averages (EMA) ---
    # EMA mimics the natural "recession curve" of a river. Recent rain matters more than rain 5 days ago.
    for span in [3, 7]:
        daily_weather[f"rain_ema_{span}"] = (
            daily_weather.groupby("location")["rain_sum_mm"]
            .transform(lambda x: x.ewm(span=span, adjust=False).mean())
        )

    # --- 6. Day-over-Day Changes (Deltas / Spikes) ---
    # Sudden changes in rainfall (intensity spikes) trigger flash floods.
    daily_weather["rain_change_1d"] = (
        daily_weather["rain_sum_mm"] - daily_weather["rain_sum_lag_1"]
    )
    daily_weather["rain_change_1d"] = daily_weather["rain_change_1d"].fillna(0) # Fill first day

    # --- Merge with discharge (target) ---
    if "river_discharge_m3s" not in f.columns:
        raise ValueError("flood_daily.csv missing required column: river_discharge_m3s")

    merged = pd.merge(
        daily_weather,
        f[["location", "date", "river_discharge_m3s"]],
        on=["location", "date"],
        how="inner"
    )

    merged = merged.dropna().reset_index(drop=True)

    Path("data").mkdir(exist_ok=True)
    merged.to_csv(out_path, index=False)

    if verbose:
        print(f"✅ Saved features: {out_path} ({len(merged)} rows)")
        print(f"🌟 Generated {len(merged.columns) - 3} feature columns for the CNN-LSTM.")
        if len(merged) == 0:
            print("⚠️ Merged dataset is empty — likely date/location mismatch between files.")

    return merged

if __name__ == "__main__":
    build_features()