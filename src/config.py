"""
Central configuration for the Pearls AQI Predictor.

Everything is driven by environment variables so the exact same code runs:
  - on your laptop (with local parquet files standing in for the feature store)
  - in GitHub Actions (scheduled feature + training pipelines)
  - against a real Hopsworks project once you add HOPSWORKS_API_KEY

Nothing here requires a paid service. Weather + pollution data comes from
Open-Meteo (https://open-meteo.com), which is free and keyless.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Location — change these (or set the env vars) to predict AQI for your city.
# Defaults to Rawalpindi/Islamabad.
# ---------------------------------------------------------------------------
CITY_NAME = os.getenv("CITY_NAME", "Rawalpindi")
LATITUDE = float(os.getenv("LATITUDE", "33.6007"))
LONGITUDE = float(os.getenv("LONGITUDE", "73.0679"))
TIMEZONE = os.getenv("TIMEZONE", "auto")

# ---------------------------------------------------------------------------
# Feature store / model registry backend.
# If HOPSWORKS_API_KEY is set, we use Hopsworks (real serverless feature
# store + model registry, free tier). Otherwise we fall back to local
# parquet/joblib files under data/ and models/ so the pipelines still run
# end-to-end with zero setup.
# ---------------------------------------------------------------------------
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "aqi_predictor")
USE_HOPSWORKS = bool(HOPSWORKS_API_KEY)

FEATURE_GROUP_NAME = "aqi_daily_features"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "aqi_daily_feature_view"
MODEL_NAME = "aqi_forecast_model"

# ---------------------------------------------------------------------------
# Local fallback storage paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)

LOCAL_FEATURES_PATH = DATA_DIR / "features.parquet"
LOCAL_MODEL_PATH = MODELS_DIR / "aqi_forecast_model.joblib"
LOCAL_MODEL_META_PATH = MODELS_DIR / "aqi_forecast_model_meta.json"

# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
FORECAST_HORIZONS = [1, 2, 3]          # predict AQI 1, 2, and 3 days ahead
TARGET_COLS = [f"aqi_next_{h}d" for h in FORECAST_HORIZONS]
BACKFILL_DAYS = int(os.getenv("BACKFILL_DAYS", "120"))  # history to pull on first run

# Hazardous AQI alert threshold (US AQI scale: 151+ = "Unhealthy")
HAZARD_AQI_THRESHOLD = 151
