# 🌫️ Pearls AQI Predictor

**An end-to-end, 100% serverless system that predicts the Air Quality Index (AQI) for a city 1, 2, and 3 days into the future**
The system automatically fetches live weather + pollution data every hour, engineers features, retrains its ML models every day, and serves live forecasts on an interactive dashboard — with zero servers to manage and zero cost to run.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Requirements Checklist](#requirements-checklist-mapped-to-the-brief)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dashboard Features](#dashboard-features)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Running the Pipelines](#running-the-pipelines)
- [Model Training & Evaluation](#model-training--evaluation)
- [Switching to a Real Feature Store (Hopsworks)](#switching-to-a-real-feature-store-hopsworks)
- [Automation (CI/CD)](#automation-cicd)
- [Deploying the Dashboard](#deploying-the-dashboard)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Troubleshooting](#troubleshooting)
- [Limitations & Future Work](#limitations--future-work)
- [Final Submission Checklist](#final-submission-checklist-from-the-brief)

---

## Overview

Air quality varies hour to hour and is driven by a mix of weather (wind, humidity, temperature, pressure) and pollutant behavior (PM2.5, PM10, CO, NO₂, SO₂, ozone) that don't move randomly — they trend, cycle weekly, and correlate with conditions the day before. This project builds a **fully automated forecasting pipeline** that:

1. Pulls fresh weather + pollution data every hour (no API key required)
2. Turns it into ML-ready features (lags, rolling stats, calendar features, AQI change rate)
3. Retrains and evaluates multiple ML models every day, keeping the best one
4. Serves the resulting 1/2/3-day AQI forecast on a live dashboard with hazard alerts and explainability

Everything runs on free-tier infrastructure — no paid compute, no managed servers, no credit card required to get started.

---

## Architecture

The system follows the brief's 4-stage serverless design exactly:

```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────────────┐
│  Weather & Pollution │────▶│ Feature pipeline │────▶│                         │
│   API (Open-Meteo)   │     │  (Python, runs   │     │   Feature store &       │
│    [Raw data — 1]    │     │  hourly)  [2]     │────▶│   Model registry  [3]   │
└─────────────────────┘     └──────────────────┘     │                         │
                                                        │  (Hopsworks free tier   │
             ┌──────────────────┐                      │  or local files)        │
             │ Training pipeline│◀─────────────────────│                         │
             │ (Python, runs    │─────────────────────▶│                         │
             │ daily)            │                      └───────────┬─────────────┘
             └──────────────────┘                                   │
                                                                      ▼
                                                          ┌───────────────────────┐
                                                          │   Web app (Streamlit)  │
                                                          │  Forecast + alerts [4] │
                                                          └───────────────────────┘

     ⏰ GitHub Actions runs the feature pipeline every hour
     ⏰ GitHub Actions runs the training pipeline every day
```

| Stage | What it does | Script |
|---|---|---|
| **1. Raw data** | Fetch hourly weather + pollutant readings | `src/data_source.py` |
| **2. Feature generation** | Hourly → daily aggregation, time/lag/rolling features, targets | `src/feature_engineering.py`, `src/feature_pipeline.py` |
| **3. Feature store & model registry** | Persist features + trained models | `src/store.py` |
| **4. Web app** | Load model + features, forecast, visualize | `src/app.py` |

---

## Requirements Checklist (mapped to the brief)

| Brief requirement | Status | Where it's implemented |
|---|:---:|---|
| Fetch raw weather + pollutant data from an external API | ✅ | `src/data_source.py` — uses [Open-Meteo](https://open-meteo.com) (free, keyless) instead of AQICN/OpenWeather, which both require signup |
| Compute features (model inputs) + targets (model outputs) | ✅ | `src/feature_engineering.py` |
| Time-based features (hour, day, month) | ✅ | `day_of_week`, `day_of_month`, `month`, `is_weekend` |
| Derived features like AQI change rate | ✅ | `aqi_change_rate`, plus lag (1/2/3/7-day) and rolling (3/7-day) features |
| Store features in a Feature Store | ✅ | `src/store.py` — Hopsworks (free tier) or local parquet fallback |
| Backfill historical (features, targets) for training | ✅ | `python feature_pipeline.py --backfill 92` |
| Fetch historical (features, targets) from the Feature Store | ✅ | `src/training_pipeline.py` |
| Train & evaluate the best ML model | ✅ | Ridge, Random Forest, Gradient Boosting, + optional dense NN |
| Evaluate with RMSE, MAE, R² | ✅ | Per-model, per-horizon, in `models/leaderboard.json` and the dashboard |
| Store trained model in the Model Registry | ✅ | `src/store.py::save_model()` |
| CI/CD: feature script every hour | ✅ | `.github/workflows/feature_pipeline.yml` |
| CI/CD: training script every day | ✅ | `.github/workflows/training_pipeline.yml` |
| Web app loads model + features from the store | ✅ | `src/app.py` |
| Computes predictions on a descriptive dashboard | ✅ | Current AQI, 3-day forecast, trend chart |
| Built with Streamlit/Gradio + Flask/FastAPI | ✅ | Streamlit |
| Perform EDA to identify trends | ✅ | `src/eda.py` → plots in `reports/` |
| Variety of models: statistical → deep learning | ✅ | Ridge (statistical) → tree ensembles → optional Keras dense NN |
| SHAP or LIME for feature importance | ✅ | SHAP, shown in the dashboard |
| Alerts for hazardous AQI levels | ✅ | Red banner when forecast ≥ AQI 151 ("Unhealthy") |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data source | [Open-Meteo](https://open-meteo.com) Weather & Air Quality APIs (free, no key) |
| Feature engineering | Python, pandas, numpy |
| Feature store / model registry | [Hopsworks](https://www.hopsworks.ai) (free tier) or local parquet/joblib fallback |
| Modeling | scikit-learn (Ridge, Random Forest, Gradient Boosting), optional TensorFlow/Keras |
| Explainability | SHAP |
| Dashboard | Streamlit + Plotly |
| Automation | GitHub Actions (scheduled workflows) |
| Package management | [uv](https://docs.astral.sh/uv/) |

---

## Project Structure

```
aqi-predictor/
├── src/
│   ├── config.py               # City, coordinates, feature-store backend switch
│   ├── data_source.py          # Open-Meteo API client (weather + AQI)
│   ├── feature_engineering.py  # Hourly → daily, time/lag/rolling features, targets
│   ├── store.py                 # Feature store + model registry (Hopsworks or local)
│   ├── feature_pipeline.py     # STEP 1-3: fetch → engineer → store features
│   ├── training_pipeline.py    # STEP 1-3: fetch → train/evaluate → register model
│   ├── aqi_utils.py            # US AQI category bands, colors, alert copy
│   ├── eda.py                  # Generates EDA plots for the report
│   └── app.py                  # Streamlit dashboard
├── .github/workflows/
│   ├── feature_pipeline.yml    # Runs feature_pipeline.py every hour
│   └── training_pipeline.yml   # Runs training_pipeline.py every day
├── data/                       # Local feature store fallback (parquet)
├── models/                     # Local model registry fallback (joblib + leaderboard)
├── reports/                    # EDA plots land here
├── requirements.txt            # Core dependencies
├── requirements-hopsworks.txt  # Optional: real feature store backend
└── .env.example                # Environment variable template
```

---

## Dashboard Features

- **Current AQI hero card** — today's AQI, color-coded by US AQI category
- **3-day forecast cards** — predicted AQI for the next 3 days, each color-coded
- **Historical trend chart** — past AQI with the forecast overlaid, hazard threshold marked
- **Hazard alert banner** — appears automatically when any forecasted day crosses AQI 151 ("Unhealthy")
- **Model comparison leaderboard** — every candidate model tried (not just the winner), with RMSE/MAE/R² per forecast horizon, and clearly marked which one was selected — including any that failed to train, with the reason why
- **SHAP feature importance** — shows what's actually driving the 1-day-ahead prediction

---

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended) — or plain `pip`/`venv`

### Installation

```bash
git clone <your-repo-url>
cd aqi-predictor
uv venv
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\activate
uv pip install -r requirements.txt
```

**macOS/Linux:**
```bash
source .venv/bin/activate
uv pip install -r requirements.txt
```

> `requirements.txt` deliberately excludes `hopsworks` — its dependency chain needs a C compiler and can fail to install on plain Windows. The project runs completely fine without it using the local-file fallback. See [Switching to a Real Feature Store](#switching-to-a-real-feature-store-hopsworks) if you want it.

---

## Configuration

Copy `.env.example` to `.env` (or set these as real environment variables / GitHub Actions secrets):

```
CITY_NAME=Rawalpindi
LATITUDE=33.6007
LONGITUDE=73.0679
TIMEZONE=auto

# Optional — leave blank to use the local file fallback
HOPSWORKS_API_KEY=
HOPSWORKS_PROJECT=aqi_predictor
```

---

## Running the Pipelines

All commands run from inside `src/`.

```bash
cd src

# 1) Backfill historical data (needed once, to have training data)
#    92 is the max Open-Meteo allows per request via past_days
python feature_pipeline.py --backfill 92

# 2) Train and register the best model
python training_pipeline.py

# 3) Launch the dashboard
python -m streamlit run app.py
```

For ongoing operation, `feature_pipeline.py` (no `--backfill` flag) does an incremental refresh of the last 14 days — this is what the hourly GitHub Action runs.

---

## Model Training & Evaluation

`training_pipeline.py`:

1. Loads all historical (features, targets) rows from the feature store
2. Splits them time-aware (train on earlier data, test on the most recent slice — never shuffled, to avoid leaking the future into training)
3. Trains multiple candidates as `MultiOutputRegressor`s (predicting all 3 horizons at once):
   - **Ridge Regression** (statistical baseline)
   - **Random Forest**
   - **Gradient Boosting**
   - **Dense neural network** (optional — only trained if `tensorflow` + `scikeras` are installed)
4. Evaluates every candidate on **RMSE, MAE, and R²** for each of the 1/2/3-day horizons
5. Selects the model with the lowest average RMSE across all 3 horizons, refits it on the full dataset, and registers it
6. Saves a full leaderboard (`models/leaderboard.json` / `.csv`) — every model tried, every metric, and which one was selected, so nothing is hidden even if a candidate failed to train

The dashboard's **Model comparison** panel renders this leaderboard directly.

---

## Switching to a Real Feature Store (Hopsworks)

By default, this project uses local parquet/joblib files as a stand-in for the feature store and model registry, so it works with zero setup. To use the real thing:

1. Create a free project at [hopsworks.ai](https://www.hopsworks.ai) (Serverless free tier)
2. Copy your API key
3. Install the extra dependency:
   ```bash
   uv pip install -r requirements-hopsworks.txt
   ```
   *(On Windows, install "Microsoft C++ Build Tools" — Desktop development with C++ workload — first if this fails: https://visualstudio.microsoft.com/visual-cpp-build-tools/)*
4. Set `HOPSWORKS_API_KEY` in `.env` or your environment

No code changes needed — `store.py` detects the key and switches from local files to real Hopsworks feature groups + model registry automatically.

---

## Automation (CI/CD)

Two scheduled GitHub Actions workflows automate the pipelines, per the brief:

| Workflow | Schedule | Purpose |
|---|---|---|
| `.github/workflows/feature_pipeline.yml` | Every hour | Fetch latest data, recompute features, write to the store |
| `.github/workflows/training_pipeline.yml` | Daily | Retrain all candidate models, register the best one |

**Setup:** In your GitHub repo, go to **Settings → Secrets and variables → Actions** and add:
- **Variables:** `CITY_NAME`, `LATITUDE`, `LONGITUDE`, `HOPSWORKS_PROJECT`
- **Secrets:** `HOPSWORKS_API_KEY` (only if using Hopsworks)

If you're using the local-file fallback (no Hopsworks key), the workflows commit the updated `data/` and `models/` files back to the repo on every run, so both the pipeline and the deployed dashboard always see the latest state.

---

## Deploying the Dashboard

Deploy `src/app.py` for free on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this repo to GitHub
2. On Streamlit Cloud, point a new app at `src/app.py`
3. Set the same environment variables there as secrets
4. The app pulls the latest model + features from the store on every page load

---

## Exploratory Data Analysis

Run `python eda.py` (from `src/`, after backfilling) to generate:

- AQI trend over time
- Average AQI by day of week (weekly seasonality)
- Correlation heatmap (AQI vs. weather/pollutant variables)
- Average pollutant concentrations

Plots are saved to `reports/*.png` — use these directly in your written report.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `HTTP 400` from Open-Meteo on backfill | `past_days` requested above the API's 92-day cap | Already handled — `feature_pipeline.py` auto-caps at 92 |
| `ImportError: DLL load failed... Application Control policy has blocked this file` | Windows security software (WDAC/EDR) blocking downloaded DLLs, often because the project sits in `Downloads` | Move the project out of `Downloads` (e.g. to `C:\Projects\aqi-predictor`), or run `Get-ChildItem -Recurse | Unblock-File` in the project folder |
| `error: uv trampoline failed to canonicalize script path` on `streamlit run` | The venv's script shims embed an absolute path that broke after moving the folder | Run `python -m streamlit run app.py` instead, or `uv pip install --reinstall streamlit` to regenerate the shim |
| `Input X contains NaN` during training | Early rows in the feature table lack enough history for 7-day lag/rolling features | Already handled — `training_pipeline.py` drops rows missing any feature or target before training |
| `twofish`/`pyjks` build failure installing `hopsworks` | `hopsworks` needs a C compiler on Windows | Skip it — it's isolated in `requirements-hopsworks.txt` and optional; install "Microsoft C++ Build Tools" only if you need the real feature store |
| `&&` not recognized in PowerShell | Windows PowerShell 5 doesn't support `&&` (bash syntax) | Run each command on its own line |

---

## Limitations & Future Work

- **Single-city scope** — the pipeline is parameterized by `CITY_NAME`/`LATITUDE`/`LONGITUDE`, but only forecasts one city at a time. Multi-city support would mean one feature group + one GitHub Actions matrix job per city.
- **Model-estimated pollution data** — Open-Meteo's air quality figures are model-based (CAMS), not raw ground-sensor readings. Swapping in AQICN or a national monitoring API would give ground-truth calibration at the cost of needing an API key.
- **Daily granularity** — forecasts are daily-aggregated; hourly-resolution forecasting would need denser lag features and more historical data.
- **Limited training history at launch** — the model improves as the hourly/daily automation accumulates more real history over time; initial accuracy with a short backfill window will be modest.
- **Deep learning candidate is optional and lightly tuned** — given the amount of daily data realistically available, tree-based models are expected to outperform the neural net; a larger multi-city, multi-year dataset would be needed for it to have a real advantage.

---

## Final Submission Checklist (from the brief)

- [x] End-to-end AQI prediction system
- [x] A scalable, automated pipeline (hourly features, daily training via GitHub Actions)
- [x] An interactive dashboard showcasing real-time and forecasted AQI data
- [ ] A detailed report documenting everything achieved *(use `reports/*.png` from `eda.py` + `models/leaderboard.csv` as source material)*

---

## Author

Muhammad Sajid — Full Stack AI Developer & Data Analyst

## License

This project was built as a submission for the Pearls AQI Predictor assignment. Feel free to fork and adapt for your own learning.
