
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from pathlib import Path


def _client():
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def extract_hourly_rainfall(
    start_date: str = "2010-01-01",
    end_date: str = "2026-04-30",
    save_csv: bool = True,
    out_path: str = "data/weather_hourly.csv",
):
    """
    Extract HOURLY weather from Open-Meteo Archive API.
    Saves: data/weather_hourly.csv
    """
    openmeteo = _client()
    url = "https://archive-api.open-meteo.com/v1/archive"

    locations = [
        {"name": "Petaling", "latitude": 3.107260, "longitude": 101.606710},
        {"name": "Klang", "latitude": 3.043092, "longitude": 101.441392},
    ]

    hourly_vars = [
        "temperature_2m",
        "rain",
        "precipitation",
        "wind_speed_10m",
        "wind_gusts_10m",
    ]
    # Order must match Variables(i) below.

    frames = []

    for loc in locations:
        params = {
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "start_date": start_date,
            "end_date": end_date,
            "hourly": hourly_vars,
            "timezone": "Asia/Kuala_Lumpur",
        }

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()

        dt = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        ).tz_convert("Asia/Kuala_Lumpur").tz_localize(None)

        df = pd.DataFrame(
            {
                "datetime": dt,
                "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
                "rain": hourly.Variables(1).ValuesAsNumpy(),
                "precipitation": hourly.Variables(2).ValuesAsNumpy(),
                "wind_speed_10m": hourly.Variables(3).ValuesAsNumpy(),
                "wind_gusts_10m": hourly.Variables(4).ValuesAsNumpy(),
                "location": loc["name"],
            }
        )
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    Path("data").mkdir(exist_ok=True)

    if save_csv:
        out.to_csv(out_path, index=False)
        print(f"✅ Saved hourly weather: {out_path} ({len(out)} rows)")

    return out