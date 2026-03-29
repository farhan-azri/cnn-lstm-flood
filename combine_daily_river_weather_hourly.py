import pandas as pd
from pathlib import Path


def combine_daily_river_weather_hourly(
    flood_daily_path: str,
    weather_hourly_path: str,
    out_path: str,
    verbose: bool = True
):
    flood_path = Path(flood_daily_path)
    weather_path = Path(weather_hourly_path)
    out_path = Path(out_path)

    if verbose:
        print("📂 Loading datasets...")

    df_flood = pd.read_csv(flood_path)
    df_weather = pd.read_csv(weather_path)

    # ==============================
    # Clean & format
    # ==============================
    df_flood["date"] = pd.to_datetime(df_flood["date"])
    df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])

    df_flood["location"] = df_flood["location"].astype(str).str.strip()
    df_weather["location"] = df_weather["location"].astype(str).str.strip()

    # Normalize to daily (for join key)
    df_flood["date"] = df_flood["date"].dt.floor("D")
    df_weather["date"] = df_weather["datetime"].dt.floor("D")

    # ==============================
    # Merge (NO aggregation)
    # ==============================
    if verbose:
        print("🔗 Expanding daily discharge into hourly records...")

    df_combined = pd.merge(
        df_weather,
        df_flood[["date", "location", "river_discharge_m3s"]],
        on=["date", "location"],
        how="left"   # keep all hourly rows
    )

    # Optional: drop rows with no discharge
    df_combined = df_combined.dropna(subset=["river_discharge_m3s"])

    # ==============================
    # Final formatting
    # ==============================
    df_combined = df_combined.sort_values(["location", "datetime"])

    # Keep only relevant columns
    df_combined = df_combined[
        ["datetime", "location", "rain", "river_discharge_m3s"]
    ]

    # ==============================
    # Save
    # ==============================
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(out_path, index=False)

    if verbose:
        print(f"✅ Combined dataset saved → {out_path}")
        print(df_combined.head())