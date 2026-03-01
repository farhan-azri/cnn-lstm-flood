import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from pathlib import Path


def get_openmeteo_client():
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def extract_weather_hourly_data(
    start_date: str = "2010-01-01",
    end_date: str = "2025-12-31",
    save_csv: bool = True,
):
    """
    Extract hourly weather from Open-Meteo Archive API.
    Saves to data/weather_hourly.csv by default.
    """
    openmeteo = get_openmeteo_client()

    url = "https://archive-api.open-meteo.com/v1/archive"

    locations = [
        {"name": "Petaling", "latitude": 3.107260, "longitude": 101.606710},
        {"name": "Klang", "latitude": 3.043092, "longitude": 101.441392},
    ]

    # IMPORTANT: variable order must match Variables(i) below
    hourly_vars = [
        "temperature_2m",
        "rain",
        "precipitation",
        "wind_speed_10m",
        "wind_gusts_10m",
    ]

    all_hourly = []

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

        # Order must match hourly_vars
        temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        rain = hourly.Variables(1).ValuesAsNumpy()
        precipitation = hourly.Variables(2).ValuesAsNumpy()
        wind_speed_10m = hourly.Variables(3).ValuesAsNumpy()
        wind_gusts_10m = hourly.Variables(4).ValuesAsNumpy()

        hourly_data = {
            "datetime": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            ),
            "temperature_2m": temperature_2m,
            "rain": rain,
            "precipitation": precipitation,
            "wind_speed_10m": wind_speed_10m,
            "wind_gusts_10m": wind_gusts_10m,
            "location": loc["name"],
        }

        all_hourly.append(pd.DataFrame(hourly_data))

    weather_hourly_df = pd.concat(all_hourly, ignore_index=True)
    print(f"✅ Weather hourly records extracted: {len(weather_hourly_df)}")

    if save_csv:
        Path("data").mkdir(exist_ok=True)
        weather_hourly_df.to_csv("data/weather_hourly.csv", index=False)
        print("✅ Saved to data/weather_hourly.csv")

    return weather_hourly_df


def extract_flood_hourly_data(
    start_date: str = "2010-01-01",
    end_date: str = "2025-12-31",
    save_csv: bool = True,
):
    """
    Extract hourly flood/river discharge from Open-Meteo Flood API (if supported).
    Saves to data/flood_hourly.csv by default.

    NOTE:
    - Some Open-Meteo Flood endpoints support DAILY more reliably than HOURLY.
    - If hourly is not supported, this call may error.
    """
    openmeteo = get_openmeteo_client()

    url = "https://flood-api.open-meteo.com/v1/flood"

    locations = [
        {"name": "Petaling", "latitude": 3.107260, "longitude": 101.606710},
        {"name": "Klang", "latitude": 3.043092, "longitude": 101.441392},
    ]

    # If hourly not supported, switch to daily in your pipeline
    hourly_vars = ["river_discharge"]

    all_hourly = []

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

        discharge = hourly.Variables(0).ValuesAsNumpy()

        hourly_data = {
            "datetime": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            ),
            "river_discharge_m3s": discharge,
            "location": loc["name"],
        }

        all_hourly.append(pd.DataFrame(hourly_data))

    flood_hourly_df = pd.concat(all_hourly, ignore_index=True)
    print(f"✅ Flood hourly records extracted: {len(flood_hourly_df)}")

    if save_csv:
        Path("data").mkdir(exist_ok=True)
        flood_hourly_df.to_csv("data/flood_hourly.csv", index=False)
        print("✅ Saved to data/flood_hourly.csv")

    return flood_hourly_df