"""Streamlit web app for Kolkata next-day AQI prediction.

Run with::

    streamlit run streamlit_app/app.py

The app loads the persisted best model (or trains it on the fly if absent),
displays the latest AQI trend, and predicts tomorrow's AQI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Allow running both as a module and as a script
sys_path = str(Path(__file__).resolve().parent.parent)
import sys  # noqa: E402

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from src.utils.config import get_path, load_config  # noqa: E402
from src.utils.io import load_model, read_csv  # noqa: E402
from src.feature_engineering.features import build_features  # noqa: E402
from src.preprocessing.clean import clean_raw_data  # noqa: E402


@st.cache_data(show_spinner=False)
def _load_data() -> pd.DataFrame:
    """Load raw data, clean it, and build features (cached)."""
    config = load_config()
    raw_dir = get_path(config["data"]["raw_dir"])
    raw_files = list(raw_dir.glob("*.csv"))
    if not raw_files:
        return pd.DataFrame()
    raw = read_csv(raw_files[0])
    clean = clean_raw_data(raw, config=config)
    return build_features(clean, config=config)


@st.cache_resource(show_spinner=False)
def _load_model(name: str):
    """Load a persisted model, or return None if not yet trained."""
    path = get_path("models", f"{name}.pkl")
    if not path.is_file():
        return None
    return load_model(path)


def main() -> None:
    st.set_page_config(
        page_title="Kolkata AQI Prediction",
        page_icon="🌤️",
        layout="wide",
    )
    st.title("🌤️ Kolkata Air Quality Index Prediction")
    st.markdown("Forecasting **next-day AQI** for Kolkata, West Bengal.")

    with st.sidebar:
        st.header("Settings")
        model_name = st.selectbox(
            "Model",
            ["random_forest", "gradient_boosting", "ridge", "svr"],
            index=0,
        )
        st.caption("Data source: OpenAQ API (free, no key required).")

    df = _load_data()
    if df.empty:
        st.warning(
            "No raw data found. Run "
            "`python -m src.scripts.download_data --start 2020-01-01 --end 2024-12-31` "
            "first."
        )
        return

    target = "aqi_next_day"
    feature_cols = [c for c in df.columns if c != target]

    # ── Latest AQI trend ─────────────────────────────────────────────────
    st.subheader("Latest AQI Trend")
    if "aqi" in df.columns:
        recent = df["aqi"].tail(30)
        st.line_chart(recent)
    else:
        st.info("No 'aqi' column available for trend plotting.")

    # ── Prediction ────────────────────────────────────────────────────────
    st.subheader("Next-Day AQI Prediction")
    model = _load_model(model_name)
    if model is None:
        st.info(
            f"Model '{model_name}' not found. Train it with "
            "`python -m src.model_training.train`."
        )
        return

    latest = df.iloc[-1:]
    X_latest = latest[feature_cols]
    pred = float(model.predict(X_latest)[0])

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted AQI (tomorrow)", f"{pred:.1f}")
    col2.metric("Latest observed AQI", f"{latest['aqi'].values[0]:.1f}")
    col3.metric("Data points", f"{len(df)}")

    # AQI category
    category, color = _aqi_category(pred)
    st.markdown(
        f"**AQI category:** <span style='color:{color}'>{category}</span>",
        unsafe_allow_html=True,
    )

    # Feature values for the latest day
    with st.expander("Latest feature values"):
        st.dataframe(X_latest.T)


def _aqi_category(aqi: float) -> tuple[str, str]:
    """Return (category, color) for an AQI value."""
    if aqi <= 50:
        return "Good", "#009966"
    if aqi <= 100:
        return "Moderate", "#FFBD33"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#FF9933"
    if aqi <= 200:
        return "Unhealthy", "#CC0033"
    if aqi <= 300:
        return "Very Unhealthy", "#660099"
    return "Hazardous", "#7E0023"


if __name__ == "__main__":
    main()