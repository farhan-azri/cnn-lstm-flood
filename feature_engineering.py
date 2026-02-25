import pandas as pd


def build_features():

    weather = pd.read_csv("data/weather.csv")
    flood = pd.read_csv("data/flood.csv")

    weather["date"] = pd.to_datetime(weather["date"])
    flood["date"] = pd.to_datetime(flood["date"])

    df = pd.merge(weather, flood, on=["date", "location"])

    for lag in [1, 3, 7]:
        df[f"rain_lag_{lag}"] = df.groupby("location")["rain_sum_mm"].shift(lag)
        df[f"discharge_lag_{lag}"] = df.groupby("location")["river_discharge"].shift(lag)

    df["rain_roll3"] = df.groupby("location")["rain_sum_mm"].rolling(3).mean().reset_index(0, drop=True)

    df.dropna(inplace=True)

    df.to_csv("data/features.csv", index=False)

    return df