import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler  # for normalization

def build_features(
    weather_hourly_path: str = "data/weather_hourly.csv",
    flood_daily_path: str = "data/flood_daily.csv",
    out_path: str = "data/features_daily.csv",
    verbose: bool = True,
):
    # 1) Read raw data
    if not Path(weather_hourly_path).exists() or not Path(flood_daily_path).exists():
        raise FileNotFoundError("Missing input files")
    w = pd.read_csv(weather_hourly_path)
    f = pd.read_csv(flood_daily_path)

    # 2) Normalize keys
    w["datetime"] = pd.to_datetime(w["datetime"], errors="coerce")
    w = w.dropna(subset=["datetime"]).copy()
    w["date"] = w["datetime"].dt.floor("D")

    f["date"] = pd.to_datetime(f["date"], errors="coerce").dt.floor("D")
    f = f.dropna(subset=["date"]).copy()

    w["location"] = w["location"].astype(str).str.strip()
    f["location"] = f["location"].astype(str).str.strip()

    # 3) Daily aggregation of hourly weather
    daily_weather = (
        w.groupby(["location", "date"])
        .agg(
            rain_sum_mm=("rain", "sum"),
            precip_sum_mm=("precipitation", "sum"),
            temp_mean=("temperature_2m", "mean"),
            temp_max=("temperature_2m", "max"),
            temp_min=("temperature_2m", "min"),
            wind_max=("wind_speed_10m", "max"),
            gust_max=("wind_gusts_10m", "max"),
            rain_max_1h=("rain", "max"),
        )
        .reset_index()
        .sort_values(["location", "date"])
    )

    # 4) Hydrological proxies
    daily_weather["temp_diurnal_range"] = (
        daily_weather["temp_max"] - daily_weather["temp_min"]
    )

    # 5) Temporal lags of rainfall (1,2,3,7 days)
    for lag in [1,2,3,7]:
        daily_weather[f"rain_sum_lag_{lag}"] = (
            daily_weather.groupby("location")["rain_sum_mm"]
            .shift(lag)
        )

    # 6) Rolling sums of rainfall (3d, 7d, 14d)
    for window in [3,7,14]:
        daily_weather[f"rain_sum_roll{window}"] = (
            daily_weather.groupby("location")["rain_sum_mm"]
            .transform(lambda x: x.rolling(window, min_periods=1).sum())
        )

    # 7) Exponential moving averages (EMA) of rainfall (span 3, 7 days)
    for span in [3,7]:
        daily_weather[f"rain_ema_{span}"] = (
            daily_weather.groupby("location")["rain_sum_mm"]
            .transform(lambda x: x.ewm(span=span, adjust=False).mean())
        )

    # 8) Day-over-day rainfall change
    daily_weather["rain_delta_1d"] = (
        daily_weather["rain_sum_mm"] - daily_weather["rain_sum_lag_1"]
    ).fillna(0)

    # 9) (Optional) Antecedent precipitation index (API) example: 7-day EMA
    daily_weather["rain_api_7d"] = (
        daily_weather.groupby("location")["rain_sum_mm"]
        .transform(lambda x: x.ewm(alpha=0.2, adjust=False).mean())
    )

    # 10) Merge with daily discharge target
    if "river_discharge_m3s" not in f.columns:
        raise ValueError("Missing river_discharge_m3s in flood_daily.csv")
    merged = pd.merge(
        daily_weather,
        f[["location", "date", "river_discharge_m3s"]],
        on=["location", "date"], how="inner"
    )
    merged = merged.dropna().reset_index(drop=True)

    # 11) Merge with daily discharge target
    merged = pd.merge(
        daily_weather,
        f[["location", "date", "river_discharge_m3s"]],
        on=["location", "date"], how="inner"
    )
    merged = merged.dropna().reset_index(drop=True)

    # 12) Keep original location + create encoded columns
    location_dummies = pd.get_dummies(merged["location"], prefix="loc")
    merged = pd.concat([merged, location_dummies], axis=1)

    # 13) Seasonal features
    merged["month"] = merged["date"].dt.month
    merged["is_monsoon"] = merged["month"].isin([10,11,12,1,2,3]).astype(int)

    # 14) Extreme-event indicators
    merged["heavy_rain"] = (merged["rain_sum_mm"] > 50).astype(int)
    merged["high_wind"]  = (merged["wind_max"] > 10).astype(int)

    # 15) Runoff ratio
    merged["runoff_ratio_7d"] = merged["river_discharge_m3s"] / (
        merged["rain_sum_roll7"] + 1e-6
    )

    # 16) Normalize (exclude location + date)
    feature_cols = [
        c for c in merged.columns 
        if c not in ["date", "location", "river_discharge_m3s"]
    ]
    scaler = StandardScaler()
    merged[feature_cols] = scaler.fit_transform(merged[feature_cols])

    # ✅ 17) Reorder columns → location, date FIRST
    cols = ["location", "date"] + [c for c in merged.columns if c not in ["location", "date"]]
    merged = merged[cols]

    # 18) Save
    Path("data").mkdir(exist_ok=True)
    merged.to_csv(out_path, index=False)

    return merged
