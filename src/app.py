"""
Web app (STEP 4 of the brief): loads the model + latest features from the
feature store / model registry, computes the 3-day AQI forecast, and shows
it on a dashboard.

Run:
  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import store
from aqi_utils import categorize
from config import CITY_NAME, HAZARD_AQI_THRESHOLD, TARGET_COLS
from feature_engineering import FEATURE_COLS

st.set_page_config(page_title=f"AQI Forecast — {CITY_NAME}", page_icon="🌫️", layout="wide")

# ---------------------------------------------------------------------------
# A quiet, air-themed palette instead of default Streamlit styling —
# clear-sky blue for good air, smog grey for the chrome, hazard red reserved
# for actual alerts so it stays meaningful.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #f4f7f9; }
    .aqi-hero {
        border-radius: 18px;
        padding: 28px 32px;
        color: white;
        margin-bottom: 6px;
    }
    .aqi-hero h1 { font-size: 3.4rem; margin: 0; line-height: 1; }
    .aqi-hero p { margin: 4px 0 0 0; font-size: 1.1rem; opacity: 0.92; }
    .forecast-card {
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        color: white;
    }
    .forecast-card h3 { margin: 0; font-size: 0.95rem; font-weight: 500; opacity: 0.9; }
    .forecast-card .val { font-size: 2.2rem; font-weight: 700; margin: 6px 0; }
    .hazard-banner {
        background: #7e0023;
        color: white;
        padding: 14px 20px;
        border-radius: 10px;
        margin-bottom: 18px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=1800)
def load_features() -> pd.DataFrame:
    return store.read_features()


@st.cache_resource
def load_model():
    return store.load_model()


@st.cache_data(ttl=1800)
def load_leaderboard() -> dict | None:
    """Full model-comparison results from the last training run, if present."""
    path = store.config.MODELS_DIR / "leaderboard.json"
    if not path.exists():
        return None
    import json
    with open(path) as f:
        return json.load(f)

def render_hero(latest_aqi: float, latest_date: pd.Timestamp) -> None:
    label, color, msg = categorize(latest_aqi)
    st.markdown(
        f"""
        <div class="aqi-hero" style="background: linear-gradient(135deg, {color}CC, {color}66);">
            <p>{CITY_NAME} · as of {latest_date.strftime('%b %d, %Y')}</p>
            <h1>{latest_aqi:.0f}</h1>
            <p><strong>{label}</strong> — {msg}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_forecast_row(dates: list[pd.Timestamp], values: list[float]) -> None:
    cols = st.columns(len(values))
    for col, date, val in zip(cols, dates, values):
        label, color, _ = categorize(val)
        with col:
            st.markdown(
                f"""
                <div class="forecast-card" style="background-color: {color};">
                    <h3>{date.strftime('%a, %b %d')}</h3>
                    <div class="val">{val:.0f}</div>
                    <div>{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_trend_chart(hist: pd.DataFrame, forecast_dates, forecast_values) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["aqi"], mode="lines", name="Historical AQI",
        line=dict(color="#2b6cb0", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast_values, mode="lines+markers", name="Forecast",
        line=dict(color="#c53030", width=2, dash="dash"),
    ))
    fig.add_hline(y=HAZARD_AQI_THRESHOLD, line_dash="dot", line_color="#7e0023",
                   annotation_text="Unhealthy threshold", annotation_position="top left")
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10), height=380,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title=None, yaxis_title="US AQI",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance(model, X_sample: pd.DataFrame) -> None:
    st.subheader("What's driving the forecast")
    try:
        import shap

        # Unwrap MultiOutputRegressor -> use the estimator for horizon 1 (index 0)
        estimator = model
        if hasattr(model, "named_steps"):
            estimator = model.named_steps["model"]
        if hasattr(estimator, "estimators_"):
            estimator = estimator.estimators_[0]

        explainer = shap.Explainer(estimator, X_sample)
        shap_values = explainer(X_sample)
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        importance = pd.DataFrame({"feature": X_sample.columns, "importance": mean_abs})
        importance = importance.sort_values("importance", ascending=True).tail(10)

        fig = go.Figure(go.Bar(
            x=importance["importance"], y=importance["feature"], orientation="h",
            marker_color="#2b6cb0",
        ))
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=350,
                           plot_bgcolor="white", paper_bgcolor="white",
                           xaxis_title="mean |SHAP value| (impact on 1-day-ahead AQI)")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:  # noqa: BLE001
        st.info(f"Feature importance unavailable ({e}). Try `pip install shap`.")

def render_leaderboard(leaderboard: dict) -> None:
    st.subheader("Model comparison")
    st.caption(f"Every candidate tried during the last training run "
               f"({leaderboard['trained_at'][:19].replace('T', ' ')} UTC) — "
               f"**{leaderboard['selected_model']}** was selected for lowest average RMSE.")

    rows = []
    for name, r in leaderboard["candidates"].items():
        row = {
            "Model": f"✅ {name}" if name == leaderboard["selected_model"] else name,
            "Avg RMSE": round(r["overall_rmse"], 2),
        }
        for target, m in r["per_horizon"].items():
            label = target.replace("aqi_next_", "").replace("d", "-day")
            row[f"{label} RMSE"] = round(m["rmse"], 2)
            row[f"{label} MAE"] = round(m["mae"], 2)
            row[f"{label} R²"] = round(m["r2"], 2)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("Avg RMSE").reset_index(drop=True)
    st.dataframe(
        df.style.highlight_min(subset=["Avg RMSE"], color="#c6f6d5"),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Lower RMSE/MAE = more accurate. Higher R² (closer to 1) = better fit. "
               "Highlighted row = lowest average RMSE across all 3 forecast horizons.")

def main():
    st.title("🌫️ AQI Forecast Dashboard")
    st.caption(f"Serverless AQI prediction for {CITY_NAME} — Pearls AQI Predictor project")

    try:
        features = load_features()
        model, meta = load_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Run `python feature_pipeline.py --backfill 120` then `python training_pipeline.py` first.")
        return

    latest = features.dropna(subset=["aqi_lag7"]).iloc[[-1]]
    latest_date = latest["date"].iloc[0]
    latest_aqi = float(latest["aqi"].iloc[0])

    preds = model.predict(latest[FEATURE_COLS])[0]
    forecast_dates = [latest_date + pd.Timedelta(days=i) for i in [1, 2, 3]]

    if float(np.max(preds)) >= HAZARD_AQI_THRESHOLD:
        st.markdown(
            f'<div class="hazard-banner">⚠️ Hazardous air quality forecast — '
            f'AQI is expected to reach {np.max(preds):.0f} in the next 3 days. '
            f'Limit outdoor exposure, especially for sensitive groups.</div>',
            unsafe_allow_html=True,
        )

    render_hero(latest_aqi, latest_date)
    st.write("")
    st.subheader("3-day forecast")
    render_forecast_row(forecast_dates, list(preds))

    st.write("")
    st.subheader("Historical trend + forecast")
    render_trend_chart(features, forecast_dates, list(preds))

    col1, col2 = st.columns([2, 1])
    with col1:
        sample = features.dropna(subset=FEATURE_COLS).tail(60)[FEATURE_COLS]
        if len(sample) >= 10:
            render_feature_importance(model, sample)
    with col2:
        st.subheader("Model info")
        st.metric("Model", meta["model_name"])
        st.metric("Trained at", meta["trained_at"][:19].replace("T", " "))
        for target, m in meta["metrics"].items():
            st.write(f"**{target}** — RMSE {m['rmse']:.1f} · MAE {m['mae']:.1f} · R² {m['r2']:.2f}")

    with st.expander("Raw recent feature rows"):
        st.dataframe(features.tail(14), use_container_width=True)

    st.write("")
    leaderboard = load_leaderboard()
    if leaderboard:
        render_leaderboard(leaderboard)
    else:
        st.info("Run `python training_pipeline.py` (already done at least once, since a model is "
                 "loaded) to generate the full model-comparison leaderboard shown here.")


if __name__ == "__main__":
    main()
