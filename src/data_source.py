"""
Raw data access layer.

Uses Open-Meteo's Air Quality API + Weather API. Both are free, require no
API key, and offer historical + forecast data — a good fit for a "100%
serverless, 100% free" pipeline. Swap this module out for AQICN/OpenWeather
if you'd rather use those (both need a free API key).

Docs:
  https://open-meteo.com/en/docs/air-quality-api
  https://open-meteo.com/en/docs
"""

from __future__ import annotations

import pandas as pd
import requests

from config import LATITUDE, LONGITUDE, TIMEZONE

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

AQ_HOURLY_VARS = ["us_aqi", "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
WEATHER_HOURLY_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure", "precipitation"]


def _get(url: str, params: dict) -> dict:
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_air_quality(past_days: int = 0, forecast_days: int = 4) -> pd.DataFrame:
    """Hourly AQI + pollutants. Open-Meteo air-quality API supports up to ~92 past_days."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(AQ_HOURLY_VARS),
        "timezone": TIMEZONE,
        "past_days": past_days,
        "forecast_days": forecast_days,
    }
    js = _get(AIR_QUALITY_URL, params)
    df = pd.DataFrame(js["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_weather(past_days: int = 0, forecast_days: int = 4) -> pd.DataFrame:
    """Hourly weather (forecast API also serves recent past_days)."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(WEATHER_HOURLY_VARS),
        "timezone": TIMEZONE,
        "past_days": past_days,
        "forecast_days": forecast_days,
    }
    js = _get(WEATHER_URL, params)
    df = pd.DataFrame(js["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_weather_archive(start_date: str, end_date: str) -> pd.DataFrame:
    """Historical weather for deep backfills (archive API, no past_days limit)."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(WEATHER_HOURLY_VARS),
        "timezone": TIMEZONE,
        "start_date": start_date,
        "end_date": end_date,
    }
    js = _get(WEATHER_ARCHIVE_URL, params)
    df = pd.DataFrame(js["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_raw(past_days: int = 92, forecast_days: int = 4) -> pd.DataFrame:
    """
    Merge air quality + weather into one hourly dataframe.
    past_days is capped at 92 by Open-Meteo's air-quality API — for deeper
    backfills the feature pipeline chunks calls (see feature_pipeline.py).
    """
    aq = fetch_air_quality(past_days=past_days, forecast_days=forecast_days)
    wx = fetch_weather(past_days=past_days, forecast_days=forecast_days)
    df = aq.merge(wx, on="time", how="inner")
    return df.sort_values("time").reset_index(drop=True)
