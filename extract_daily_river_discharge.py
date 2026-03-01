import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from pathlib import Path


def _client():
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def extract_daily_river_discharge(
    start_date: str = "2010-01-01",
    end_date: str = "2025-12-31",
    save_csv: bool = True,
    out_path: str = "data/flood_daily.csv",
):
    """
    Extract DAILY river discharge from Open-Meteo Flood API.
    Saves: data/flood_daily.csv
    """
    openmeteo = _client()
    url = "https://flood-api.open-meteo.com/v1/flood"

    locations = [
        {"name": "Petaling", "latitude": 3.107260, "longitude": 101.606710},
        {"name": "Klang", "latitude": 3.043092, "longitude": 101.441392},
    ]

    params_common = {
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["river_discharge"],
        "timezone": "Asia/Kuala_Lumpur",
    }

    frames = []

    for loc in locations:
        params = {
            **params_common,
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
        }

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        daily = response.Daily()

        dates = pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s"),
            end=pd.to_datetime(daily.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        )

        df = pd.DataFrame(
            {
                "date": dates,
                "river_discharge_m3s": daily.Variables(0).ValuesAsNumpy(),
                "location": loc["name"],
            }
        )
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    Path("data").mkdir(exist_ok=True)

    if save_csv:
        out.to_csv(out_path, index=False)
        print(f"✅ Saved daily discharge: {out_path} ({len(out)} rows)")

    return out