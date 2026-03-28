import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from pathlib import Path

from openmeteo_sdk.Variable import Variable


# ============================================================
# CLIENT SETUP (same pattern as your archive function)
# ============================================================
def _client():
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


# ============================================================
# MAIN FUNCTION
# ============================================================
def extract_seasonal_hourly(
    forecast_days: int = 30,
    save_csv: bool = True,
    out_path: str = "data/weather_ensemble_forecast_hourly.csv",
):
    """
    Extract HOURLY seasonal forecast data from Open-Meteo API.
    Saves: data/weather_ensemble_forecast_hourly.csv
    """

    openmeteo = _client()
    url = "https://customer-ensemble-api.open-meteo.com/v1/ensemble"

    locations = [
        {"name": "Petaling", "latitude": 3.107260, "longitude": 101.606710},
        {"name": "Klang", "latitude": 3.043092, "longitude": 101.441392},
    ]

    hourly_vars = [
        "rain",
        "temperature_2m",
        "precipitation",
        "wind_speed_10m",
        "wind_gusts_10m",
    ]

    frames = []

    for loc in locations:
        params = {
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "hourly": hourly_vars,
            "forecast_days": forecast_days,
        }

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()

        # ============================================================
        # TIME HANDLING (same as your archive code)
        # ============================================================
        dt = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        ).tz_convert("Asia/Kuala_Lumpur").tz_localize(None)

        # ============================================================
        # VARIABLE EXTRACTION (ensemble-aware)
        # ============================================================
        hourly_variables = [
            hourly.Variables(i) for i in range(hourly.VariablesLength())
        ]

        data = {"datetime": dt, "location": loc["name"]}

        for var in hourly_variables:
            member = var.EnsembleMember()

            if var.Variable() == Variable.rain:
                data[f"rain_member{member}"] = var.ValuesAsNumpy()

            elif var.Variable() == Variable.temperature and var.Altitude() == 2:
                data[f"temperature_2m_member{member}"] = var.ValuesAsNumpy()

            elif var.Variable() == Variable.precipitation:
                data[f"precipitation_member{member}"] = var.ValuesAsNumpy()

            elif var.Variable() == Variable.wind_speed and var.Altitude() == 10:
                data[f"wind_speed_10m_member{member}"] = var.ValuesAsNumpy()

            elif var.Variable() == Variable.wind_gusts and var.Altitude() == 10:
                data[f"wind_gusts_10m_member{member}"] = var.ValuesAsNumpy()

        df = pd.DataFrame(data)
        frames.append(df)

    # ============================================================
    # CONCAT ALL LOCATIONS
    # ============================================================
    out = pd.concat(frames, ignore_index=True)

    Path("data").mkdir(exist_ok=True)

    if save_csv:
        out.to_csv(out_path, index=False)
        print(f"✅ Saved ensemble forecast hourly weather: {out_path} ({len(out)} rows)")

    return out


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    df = extract_seasonal_hourly()
    print(df.head())