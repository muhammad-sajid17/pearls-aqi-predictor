"""
Lightweight EDA script — run after backfilling features to generate the
plots you'll drop into the final report (brief asks for "Perform EDA to
identify trends").

Run:
  python eda.py
Outputs PNGs into ../reports/
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

import store
from config import ROOT_DIR

REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True, parents=True)


def main():
    df = store.read_features()
    df = df.sort_values("date")

    # 1. AQI over time
    plt.figure(figsize=(10, 4))
    plt.plot(df["date"], df["aqi"], color="#2b6cb0")
    plt.title("Daily mean AQI over time")
    plt.ylabel("US AQI")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "aqi_over_time.png", dpi=150)
    plt.close()

    # 2. AQI by day of week
    plt.figure(figsize=(6, 4))
    df.groupby("day_of_week")["aqi"].mean().plot(kind="bar", color="#2b6cb0")
    plt.title("Average AQI by day of week (0=Mon)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "aqi_by_dayofweek.png", dpi=150)
    plt.close()

    # 3. AQI vs weather correlations
    corr_cols = ["aqi", "pm2_5", "pm10", "temperature", "humidity", "wind_speed", "pressure", "precipitation"]
    corr = df[corr_cols].corr()
    plt.figure(figsize=(6, 5))
    plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
    plt.yticks(range(len(corr_cols)), corr_cols)
    plt.colorbar(label="Pearson correlation")
    plt.title("AQI vs. weather/pollutant correlations")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # 4. Pollutant contributions (mean levels)
    plt.figure(figsize=(6, 4))
    df[["pm2_5", "pm10", "co", "no2", "so2", "ozone"]].mean().plot(kind="bar", color="#c53030")
    plt.title("Average pollutant concentrations")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "pollutant_means.png", dpi=150)
    plt.close()

    print(f"[eda] saved 4 plots to {REPORTS_DIR}")
    print(df[corr_cols].describe())


if __name__ == "__main__":
    main()
