"""
STEP 1-3 of the "Training pipeline" section of the brief:
  1. Fetch historical (features, targets) from the feature store
  2. Train + evaluate several ML models (Ridge, Random Forest, Gradient
     Boosting, and — if TensorFlow is installed — a small dense neural net)
  3. Store the best model (by average test RMSE across the 3 forecast
     horizons) in the model registry

Run:
  python training_pipeline.py
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import store
from config import TARGET_COLS
from feature_engineering import FEATURE_COLS


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    n = len(df)
    split = int(n * (1 - test_frac))
    split = max(split, n - 30) if n > 40 else split  # keep a reasonable test window
    return df.iloc[:split], df.iloc[split:]


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_candidates() -> dict:
    candidates = {
        "ridge": Pipeline([
            ("scale", StandardScaler()),
            ("model", MultiOutputRegressor(Ridge(alpha=1.0))),
        ]),
        "random_forest": MultiOutputRegressor(
            RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
        ),
        "gradient_boosting": MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
        ),
    }

    # Optional: small dense NN if TensorFlow is available. Skipped gracefully
    # otherwise — the brief only asks us to *experiment* with DL, not require it.
    try:
        from tensorflow import keras
        from scikeras.wrappers import KerasRegressor

        def build_nn(meta):
            n_features_in = meta["n_features_in_"]
            n_outputs = meta["n_outputs_"]
            model = keras.Sequential([
                keras.layers.Input(shape=(n_features_in,)),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(n_outputs),
            ])
            model.compile(optimizer="adam", loss="mse")
            return model

        candidates["dense_nn"] = Pipeline([
            ("scale", StandardScaler()),
            ("model", KerasRegressor(model=build_nn, epochs=100, batch_size=8, verbose=0)),
        ])
    except ImportError:
        print("[training_pipeline] TensorFlow/scikeras not installed — skipping dense_nn candidate "
              "(pip install tensorflow scikeras to enable it)")

    return candidates


def run() -> None:
    df = store.read_features()
    df = df.dropna(subset=TARGET_COLS)
    if len(df) < 20:
        raise ValueError(
            f"Only {len(df)} labeled rows available — run "
            "`python feature_pipeline.py --backfill 120` first to get enough training history."
        )

    train_df, test_df = time_based_split(df)
    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COLS].values
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COLS].values

    candidates = build_candidates()
    results = {}
    fitted_models = {}

    for name, model in candidates.items():
        print(f"[training_pipeline] training {name}...")
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        except Exception as e:  # noqa: BLE001 - keep pipeline resilient to optional DL deps
            print(f"[training_pipeline] {name} failed: {e}")
            continue

        per_horizon = {}
        for i, target in enumerate(TARGET_COLS):
            per_horizon[target] = evaluate(y_test[:, i], preds[:, i])
        overall_rmse = float(np.mean([m["rmse"] for m in per_horizon.values()]))
        results[name] = {"per_horizon": per_horizon, "overall_rmse": overall_rmse}
        fitted_models[name] = model
        print(f"[training_pipeline] {name}: avg RMSE={overall_rmse:.2f} "
              f"(" + ", ".join(f"{t}={m['rmse']:.2f}" for t, m in per_horizon.items()) + ")")

    if not results:
        raise RuntimeError("All candidate models failed to train.")

    best_name = min(results, key=lambda k: results[k]["overall_rmse"])
    best_model = fitted_models[best_name]
    best_metrics = results[best_name]["per_horizon"]

    print(f"\n[training_pipeline] BEST MODEL: {best_name} "
          f"(avg RMSE={results[best_name]['overall_rmse']:.2f})\n")

    # Refit best model on the full dataset (train+test) before shipping it,
    # so the deployed model benefits from the most recent data too.
    X_full, y_full = df[FEATURE_COLS], df[TARGET_COLS].values
    best_model.fit(X_full, y_full)


    store.save_model(
        model=best_model,
        model_name=best_name,
        metrics=best_metrics,
        feature_cols=FEATURE_COLS,
        target_cols=TARGET_COLS,
    )

    # Save a full leaderboard (every candidate, every metric, every horizon —
    # not just the winner) so the dashboard and report can show the full
    # comparison, not just which model was picked.
    leaderboard_rows = []
    for name, r in results.items():
        row = {"model": name, "selected": name == best_name, "overall_rmse": r["overall_rmse"]}
        for target, m in r["per_horizon"].items():
            row[f"{target}_rmse"] = m["rmse"]
            row[f"{target}_mae"] = m["mae"]
            row[f"{target}_r2"] = m["r2"]
        leaderboard_rows.append(row)

    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values("overall_rmse").reset_index(drop=True)
    leaderboard_df.to_csv(store.config.MODELS_DIR / "leaderboard.csv", index=False)

    with open(store.config.MODELS_DIR / "leaderboard.json", "w") as f:
        json.dump(
            {"trained_at": pd.Timestamp.utcnow().isoformat(), "selected_model": best_name, "candidates": results},
            f,
            indent=2,
        )

    print(leaderboard_df.to_string(index=False))


if __name__ == "__main__":
    run()
