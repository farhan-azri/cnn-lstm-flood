import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def build_features(
    weather_hourly_path: str = "data/weather_hourly.csv",
    weather_forecast_hourly_path: str = "data/weather_forecast_hourly.csv",
    flood_daily_path: str = "data/flood_daily.csv",
    out_path: str = "data/features_daily.csv",
    verbose: bool = True,
):

    # ============================================================
    # 1) Read raw data
    # ============================================================
    if not Path(weather_hourly_path).exists() or not Path(flood_daily_path).exists():
        raise FileNotFoundError("Missing input files")

    w = pd.read_csv(weather_hourly_path)
    p = pd.read_csv(weather_forecast_hourly_path)
    f = pd.read_csv(flood_daily_path)

    # ============================================================
    # 2) Normalize keys
    # ============================================================
    # Historical
    w["datetime"] = pd.to_datetime(w["datetime"], errors="coerce")
    w = w.dropna(subset=["datetime"]).copy()
    w["date"] = w["datetime"].dt.floor("D")
    w["data_type"] = "historical"   # ✅ NEW

    # Forecast
    # assume forecast has datetime OR date column
    if "datetime" in p.columns:
        p["datetime"] = pd.to_datetime(p["datetime"], errors="coerce")
        p = p.dropna(subset=["datetime"]).copy()
        p["date"] = p["datetime"].dt.floor("D")
    else:
        p["date"] = pd.to_datetime(p["date"], errors="coerce").dt.floor("D")

    p["data_type"] = "forecast"     # ✅ NEW

    # Flood
    f["date"] = pd.to_datetime(f["date"], errors="coerce").dt.floor("D")
    f = f.dropna(subset=["date"]).copy()

    # Normalize location
    for df in [w, p, f]:
        df["location"] = df["location"].astype(str).str.strip()

    # ============================================================
    # 3) Combine historical + forecast weather
    # ============================================================
    weather_all = pd.concat([w, p], ignore_index=True)

    # ============================================================
    # 4) Daily aggregation (KEEP your logic)
    # ============================================================
    daily_weather = (
        weather_all.groupby(["location", "date", "data_type"])
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

    # ============================================================
    # 5) Hydrological features (UNCHANGED)
    # ============================================================
    daily_weather["temp_diurnal_range"] = (
        daily_weather["temp_max"] - daily_weather["temp_min"]
    )

    for lag in [1, 2, 3, 7]:
        daily_weather[f"rain_sum_lag_{lag}"] = (
            daily_weather.groupby("location")["rain_sum_mm"].shift(lag)
        )

    for window in [3, 7, 14]:
        daily_weather[f"rain_sum_roll{window}"] = (
            daily_weather.groupby("location")["rain_sum_mm"]
            .transform(lambda x: x.rolling(window, min_periods=1).sum())
        )

    for span in [3, 7]:
        daily_weather[f"rain_ema_{span}"] = (
            daily_weather.groupby("location")["rain_sum_mm"]
            .transform(lambda x: x.ewm(span=span, adjust=False).mean())
        )

    daily_weather["rain_delta_1d"] = (
        daily_weather["rain_sum_mm"] - daily_weather["rain_sum_lag_1"]
    ).fillna(0)

    daily_weather["rain_api_7d"] = (
        daily_weather.groupby("location")["rain_sum_mm"]
        .transform(lambda x: x.ewm(alpha=0.2, adjust=False).mean())
    )

    # ============================================================
    # 6) Merge with discharge (ONLY historical has target)
    # ============================================================
    merged = pd.merge(
        daily_weather,
        f[["location", "date", "river_discharge_m3s"]],
        on=["location", "date"],
        how="left"   # ✅ IMPORTANT (keep forecast rows)
    )

    # ============================================================
    # 7) Encoding + features (UNCHANGED)
    # ============================================================
    location_dummies = pd.get_dummies(merged["location"], prefix="loc")
    merged = pd.concat([merged, location_dummies], axis=1)

    merged["month"] = merged["date"].dt.month
    merged["is_monsoon"] = merged["month"].isin([10,11,12,1,2,3]).astype(int)

    merged["heavy_rain"] = (merged["rain_sum_mm"] > 50).astype(int)
    merged["high_wind"] = (merged["wind_max"] > 10).astype(int)

    merged["runoff_ratio_7d"] = merged["river_discharge_m3s"] / (
        merged["rain_sum_roll7"] + 1e-6
    )

    # ============================================================
    # 8) Normalize (skip target + identifiers)
    # ============================================================
    feature_cols = [
        c for c in merged.columns
        if c not in ["date", "location", "river_discharge_m3s", "data_type"]
    ]

    scaler = StandardScaler()
    merged[feature_cols] = scaler.fit_transform(merged[feature_cols])

    # ============================================================
    # 9) Reorder columns
    # ============================================================
    cols = ["location", "date", "data_type"] + [
        c for c in merged.columns if c not in ["location", "date", "data_type"]
    ]
    merged = merged[cols]

    # ============================================================
    # 10) Save
    # ============================================================
    Path("data").mkdir(exist_ok=True)
    merged.to_csv(out_path, index=False)

    if verbose:
        print(f"✅ Saved features: {out_path} ({len(merged)} rows)")
        print("🆕 Includes historical + forecast data with 'data_type' column")

    return merged