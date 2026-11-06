"""AeroSense — Kolkata next-day AQI dashboard (single Streamlit entry point).

Launch from the project root::

    streamlit run streamlit_app/app.py

The app loads the persisted, validation-selected model and the existing
evaluation artifacts.  It never trains a model at start-up and never uses
future observations to build prediction features.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow ``streamlit run streamlit_app/app.py`` from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.components import (  # noqa: E402
    historical,
    insights,
    overview,
    performance,
    prediction,
    project_info,
)
from streamlit_app.components.data_service import (  # noqa: E402
    DashboardDataError,
    load_historical_data,
    load_model_bundle,
)
from streamlit_app.components.theme import (  # noqa: E402
    PALETTE,
    inject_css,
)

PAGE_TITLE = "AeroSense · Kolkata Next-Day AQI"
SECTIONS = {
    "Overview": overview.render,
    "Historical air quality": historical.render,
    "AQI prediction": prediction.render,
    "Model performance": performance.render,
    "Environmental insights": insights.render,
    "Project information": project_info.render,
}

FOOTER = f"""
<div class="aqi-footer">
    <b>AeroSense</b> · Kolkata next-day AQI research dashboard.<br>
    Historical observations come from the repository's processed dataset. Forecasts are
    model-generated estimates, not live or official AQI readings. AQI in this project is
    derived from PM2.5 using US-EPA breakpoints. The selected model has near-zero
    explanatory power on the held-out test window and should not be used for health or
    policy decisions.<br>
    <span style="color:{PALETTE['muted']};">Launch:
    <code>streamlit run streamlit_app/app.py</code></span>
</div>
"""


def _configure_page() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": (
                "AeroSense — an academic next-day AQI forecasting "
                "dashboard. Historical snapshot only; no live data."
            )
        },
    )
    st.markdown(inject_css(), unsafe_allow_html=True)


def _sidebar(historical, bundle) -> str:
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding:0.2rem 0 0.6rem 0;">
                <div style="display:flex;align-items:center;gap:0.55rem;">
                    <span style="font-size:1.55rem;line-height:1;">🌿</span>
                    <div>
                        <div style="font-weight:800;font-size:1.02rem;color:#14281D;">
                            AeroSense
                        </div>
                        <div style="font-size:0.72rem;color:{PALETTE['muted']};letter-spacing:0.08em;text-transform:uppercase;">
                            Kolkata · Machine Learning
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        section = st.radio(
            "Navigation",
            options=list(SECTIONS),
            index=0,
            label_visibility="collapsed",
        )

        st.divider()
        latest_date = historical.index.max().strftime("%d %b %Y")
        first_year = int(historical.index.min().year)
        last_year = int(historical.index.max().year)
        year_label = (
            str(first_year) if first_year == last_year else f"{first_year}–{last_year}"
        )
        st.markdown(
            f"""
            <div class="aqi-callout info" style="font-size:0.78rem;line-height:1.45;">
                <b>Data status</b><br>
                Historical snapshot<br>
                Latest observation: <b>{latest_date}</b><br>
                {len(historical):,} daily rows · {year_label}<br><br>
                <b>Model</b><br>
                {bundle.model_name.replace('_', ' ').title()} ({len(bundle.feature_columns)} features)<br>
                Loaded from disk — not retrained here.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "AQI is derived from PM2.5 (US-EPA). Forecasts are model estimates, "
            "not live readings."
        )

    return section


def _render_data_error(error: Exception) -> None:
    st.error("The dashboard cannot start because a required artifact is missing or invalid.")
    st.markdown(f"**Details:** {error}")
    st.markdown(
        """
        **Recover the artifacts from the project root:**

        ```bash
        source .venv/bin/activate
        ./scripts/run_forecasting.sh
        ./scripts/run_model_training.sh
        ./scripts/run_final_evaluation.sh
        streamlit run streamlit_app/app.py
        ```
        """
    )
    st.info(
        "The dashboard loads a saved model. It deliberately does not train one "
        "at start-up, so the training and evaluation artifacts must exist first."
    )


def main() -> None:
    _configure_page()

    try:
        historical_data = load_historical_data()
        bundle = load_model_bundle()
    except DashboardDataError as exc:
        _render_data_error(exc)
        return
    except Exception as exc:  # pragma: no cover - defensive boundary
        _render_data_error(exc)
        with st.expander("Technical details", expanded=False):
            st.exception(exc)
        return

    section = _sidebar(historical_data, bundle)

    try:
        SECTIONS[section]()
    except DashboardDataError as exc:
        st.error(f"**{section} could not be rendered.** {exc}")
        st.info(
            "Check that the files in `models/` and `results/metrics/` are present "
            "and were produced by the project's evaluation scripts."
        )
    except Exception as exc:  # pragma: no cover - defensive boundary
        st.error(f"**{section} could not be rendered.** {exc}")
        with st.expander("Technical details", expanded=False):
            st.exception(exc)

    st.markdown(FOOTER, unsafe_allow_html=True)


main()
