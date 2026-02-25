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
    2) Aggregate to daily per location
    3) Create daily lag/rolling features
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

    # --- Daily aggregation from hourly ---
    daily_weather = (
        w.groupby(["location", "date"])
        .agg(
            rain_sum_mm=("rain", "sum"),
            precip_sum_mm=("precipitation", "sum"),
            temp_mean=("temperature_2m", "mean"),
            wind_max=("wind_speed_10m", "max"),
            gust_max=("wind_gusts_10m", "max"),
            rain_max_1h=("rain", "max"),
        )
        .reset_index()
        .sort_values(["location", "date"])
    )

    # --- Lag features (daily) ---
    for lag in [1, 2, 3, 7]:
        daily_weather[f"rain_sum_lag_{lag}"] = daily_weather.groupby("location")["rain_sum_mm"].shift(lag)
        daily_weather[f"rain_max1h_lag_{lag}"] = daily_weather.groupby("location")["rain_max_1h"].shift(lag)

    # --- Rolling features ---
    daily_weather["rain_sum_roll3"] = (
        daily_weather.groupby("location")["rain_sum_mm"]
        .rolling(3, min_periods=1).mean()
        .reset_index(0, drop=True)
    )
    daily_weather["rain_sum_roll7"] = (
        daily_weather.groupby("location")["rain_sum_mm"]
        .rolling(7, min_periods=1).mean()
        .reset_index(0, drop=True)
    )

    # --- Merge with discharge (target) ---
    # Recommended: only merge required column from flood file
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
        if len(merged) == 0:
            print("⚠️ Merged dataset is empty — likely date/location mismatch between files.")

    return merged