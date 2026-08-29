"""
STEP 1-3 of the project brief:
  1. Fetch raw weather + pollutant data (Open-Meteo, free & keyless)
  2. Compute features (time-based, lag, rolling, AQI change rate) + targets
  3. Store features in the feature store

Run modes:
  python feature_pipeline.py                -> incremental update (last ~10 days,
                                                 enough to refresh lag/rolling features)
  python feature_pipeline.py --backfill 120  -> historical backfill for training data

Scheduled hourly in CI/CD (see .github/workflows/feature_pipeline.yml).
"""

from __future__ import annotations

import argparse
# import time
import pandas as pd

import data_source
import feature_engineering as fe
import store
from config import BACKFILL_DAYS, CITY_NAME


# def backfill(days: int) -> pd.DataFrame:
#     """
#     Pull `days` of history. Open-Meteo's air-quality API caps past_days at
#     ~92, so for longer backfills we chunk requests and concatenate.
#     """
#     chunks = []
#     remaining = days
#     end_offset = 0
#     while remaining > 0:
#         chunk_days = min(remaining, 90)
#         print(f"[feature_pipeline] fetching past_days={end_offset + chunk_days} window...")
#         raw = data_source.fetch_raw(past_days=end_offset + chunk_days, forecast_days=1)
#         chunks.append(raw)
#         remaining -= chunk_days
#         end_offset += chunk_days
#         time.sleep(1)  # be polite to the free API

#     combined = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["time"])
#     return combined.sort_values("time").reset_index(drop=True)

def backfill(days: int) -> pd.DataFrame:
    """
    Pull `days` of history via Open-Meteo's `past_days` parameter, which is
    capped at 92 days by the API itself. Requesting more returns HTTP 400,
    so we cap it here instead of hitting the error at request time.
    """
    capped_days = min(days, 92)
    if capped_days < days:
        print(f"[feature_pipeline] Open-Meteo caps history at 92 days — "
              f"backfilling {capped_days} instead of the requested {days}.")

    print(f"[feature_pipeline] fetching past_days={capped_days}...")
    raw = data_source.fetch_raw(past_days=capped_days, forecast_days=1)
    return raw


def run(backfill_days: int | None = None) -> pd.DataFrame:
    if backfill_days:
        print(f"[feature_pipeline] backfilling {backfill_days} days for {CITY_NAME}")
        raw = backfill(backfill_days)
        daily_features = fe.build_feature_table(raw, for_training=True)
    else:
        # Incremental: last 14 days is plenty to recompute lag7/rolling7 correctly
        # and to pick up "today so far" as a live (target-less) row.
        raw = data_source.fetch_raw(past_days=14, forecast_days=1)
        daily_features = fe.build_feature_table(raw, for_training=False)

    if daily_features.empty:
        print("[feature_pipeline] no rows produced — check date range / API response")
        return daily_features

    store.write_features(daily_features)
    print(f"[feature_pipeline] done: {len(daily_features)} daily feature rows written")
    return daily_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", type=int, default=0, help="Days of history to backfill")
    args = parser.parse_args()

    if args.backfill:
        run(backfill_days=args.backfill)
    else:
        run()
