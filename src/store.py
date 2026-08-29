"""
Thin abstraction over the "Feature store & Model registry" box in the
project diagram.

- If HOPSWORKS_API_KEY is set -> real Hopsworks feature store + model
  registry (free tier: https://www.hopsworks.ai/).
- Otherwise -> local parquet file for features, local joblib + json for the
  model, so `python feature_pipeline.py && python training_pipeline.py`
  works immediately with no account/setup.

Everything else in the project (feature_pipeline.py, training_pipeline.py,
app.py) only talks to this module — swapping the backend later doesn't
touch any other file.
"""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Feature store
# ---------------------------------------------------------------------------

def _hopsworks_feature_group():
    import hopsworks

    project = hopsworks.login(api_key_value=config.HOPSWORKS_API_KEY, project=config.HOPSWORKS_PROJECT)
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=config.FEATURE_GROUP_NAME,
        version=config.FEATURE_GROUP_VERSION,
        description=f"Daily AQI + weather features for {config.CITY_NAME}",
        primary_key=["date"],
        event_time="date",
    )
    return fg


def write_features(df: pd.DataFrame) -> None:
    """Upsert the daily feature rows into the feature store."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if config.USE_HOPSWORKS:
        fg = _hopsworks_feature_group()
        fg.insert(df, write_options={"wait_for_job": False})
        print(f"[store] wrote {len(df)} rows to Hopsworks feature group '{config.FEATURE_GROUP_NAME}'")
        return

    # Local fallback: merge-upsert by date into a parquet file.
    if config.LOCAL_FEATURES_PATH.exists():
        existing = pd.read_parquet(config.LOCAL_FEATURES_PATH)
        existing["date"] = pd.to_datetime(existing["date"])
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"], keep="last")
    else:
        combined = df

    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_parquet(config.LOCAL_FEATURES_PATH, index=False)
    print(f"[store] wrote {len(df)} rows -> {config.LOCAL_FEATURES_PATH} ({len(combined)} rows total)")


def read_features() -> pd.DataFrame:
    """Read the full historical feature table."""
    if config.USE_HOPSWORKS:
        fg = _hopsworks_feature_group()
        df = fg.read()
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    if not config.LOCAL_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"No features found at {config.LOCAL_FEATURES_PATH}. Run feature_pipeline.py first."
        )
    df = pd.read_parquet(config.LOCAL_FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def save_model(model, model_name: str, metrics: dict, feature_cols: list, target_cols: list) -> None:
    meta = {
        "model_name": model_name,
        "trained_at": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "city": config.CITY_NAME,
    }

    if config.USE_HOPSWORKS:
        import hopsworks

        project = hopsworks.login(api_key_value=config.HOPSWORKS_API_KEY, project=config.HOPSWORKS_PROJECT)
        mr = project.get_model_registry()

        # Hopsworks wants the model dumped to disk first.
        joblib.dump(model, config.LOCAL_MODEL_PATH)
        with open(config.LOCAL_MODEL_META_PATH, "w") as f:
            json.dump(meta, f, indent=2)

        hw_model = mr.sklearn.create_model(
            name=config.MODEL_NAME,
            metrics={f"{k}_rmse": v["rmse"] for k, v in metrics.items()},
            description=f"AQI 1/2/3-day forecaster for {config.CITY_NAME}",
        )
        hw_model.save(str(config.MODELS_DIR))
        print(f"[store] registered model in Hopsworks model registry as '{config.MODEL_NAME}'")
        return

    joblib.dump(model, config.LOCAL_MODEL_PATH)
    with open(config.LOCAL_MODEL_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[store] saved model -> {config.LOCAL_MODEL_PATH}")


def load_model():
    """Returns (model, meta_dict). Always reads from local disk — for the
    Hopsworks path, the CI/CD training job downloads the model artifact to
    the same local path before the app starts (see workflows)."""
    if not config.LOCAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {config.LOCAL_MODEL_PATH}. Run training_pipeline.py first."
        )
    model = joblib.load(config.LOCAL_MODEL_PATH)
    with open(config.LOCAL_MODEL_META_PATH) as f:
        meta = json.load(f)
    return model, meta
