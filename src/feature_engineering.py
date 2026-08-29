"""
Turns raw hourly (weather, pollutant) readings into a daily feature table
ready for the feature store, plus the 1/2/3-day-ahead targets used for
training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import FORECAST_HORIZONS


def hourly_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly readings to one row per calendar day."""
    df = df.copy()
    df["date"] = df["time"].dt.date
    agg = df.groupby("date").agg(
        aqi=("us_aqi", "mean"),
        aqi_max=("us_aqi", "max"),
        pm2_5=("pm2_5", "mean"),
        pm10=("pm10", "mean"),
        co=("carbon_monoxide", "mean"),
        no2=("nitrogen_dioxide", "mean"),
        so2=("sulphur_dioxide", "mean"),
        ozone=("ozone", "mean"),
        temperature=("temperature_2m", "mean"),
        humidity=("relative_humidity_2m", "mean"),
        wind_speed=("wind_speed_10m", "mean"),
        pressure=("surface_pressure", "mean"),
        precipitation=("precipitation", "sum"),
    ).reset_index()
    agg["date"] = pd.to_datetime(agg["date"])
    return agg.sort_values("date").reset_index(drop=True)


def add_time_features(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["day_of_month"] = daily["date"].dt.day
    daily["month"] = daily["date"].dt.month
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)
    return daily


def add_lag_and_rolling_features(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    for lag in [1, 2, 3, 7]:
        daily[f"aqi_lag{lag}"] = daily["aqi"].shift(lag)
    daily["aqi_roll3_mean"] = daily["aqi"].shift(1).rolling(3).mean()
    daily["aqi_roll7_mean"] = daily["aqi"].shift(1).rolling(7).mean()
    daily["aqi_roll7_std"] = daily["aqi"].shift(1).rolling(7).std()
    # AQI change rate vs. yesterday (derived feature requested in the brief)
    daily["aqi_change_rate"] = (daily["aqi"] - daily["aqi_lag1"]) / daily["aqi_lag1"].replace(0, np.nan)
    return daily


def add_targets(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    for h in FORECAST_HORIZONS:
        daily[f"aqi_next_{h}d"] = daily["aqi"].shift(-h)
    return daily


def build_feature_table(raw_hourly: pd.DataFrame, for_training: bool = True) -> pd.DataFrame:
    """
    Full pipeline: hourly raw -> daily features (+ targets if for_training).
    When for_training=False (i.e. building today's feature row for live
    inference) we keep the most recent row even though it has no targets yet.
    """
    daily = hourly_to_daily(raw_hourly)
    daily = add_time_features(daily)
    daily = add_lag_and_rolling_features(daily)
    daily = add_targets(daily)

    if for_training:
        daily = daily.dropna(subset=["aqi_lag7"] + [f"aqi_next_{h}d" for h in FORECAST_HORIZONS])
    else:
        daily = daily.dropna(subset=["aqi_lag7"])

    return daily.reset_index(drop=True)


FEATURE_COLS = [
    "aqi", "aqi_max", "pm2_5", "pm10", "co", "no2", "so2", "ozone",
    "temperature", "humidity", "wind_speed", "pressure", "precipitation",
    "day_of_week", "day_of_month", "month", "is_weekend",
    "aqi_lag1", "aqi_lag2", "aqi_lag3", "aqi_lag7",
    "aqi_roll3_mean", "aqi_roll7_mean", "aqi_roll7_std", "aqi_change_rate",
]
