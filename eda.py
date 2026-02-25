import pandas as pd
from pathlib import Path


def build_features(
    weather_hourly_path: str = "data/weather_hourly.csv",
    flood_daily_path: str = "data/flood_daily.csv",
    out_path: str = "data/features_daily.csv",
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

    w["datetime"] = pd.to_datetime(w["datetime"])
    w["date"] = w["datetime"].dt.date
    w["date"] = pd.to_datetime(w["date"])

    f["date"] = pd.to_datetime(f["date"])

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
        daily_weather.groupby("location")["rain_sum_mm"].rolling(3).mean().reset_index(0, drop=True)
    )
    daily_weather["rain_sum_roll7"] = (
        daily_weather.groupby("location")["rain_sum_mm"].rolling(7).mean().reset_index(0, drop=True)
    )

    # --- Merge with discharge (target) ---
    merged = pd.merge(daily_weather, f, on=["location", "date"], how="inner")

    merged = merged.dropna().reset_index(drop=True)

    Path("data").mkdir(exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"✅ Saved features: {out_path} ({len(merged)} rows)")

    return merged