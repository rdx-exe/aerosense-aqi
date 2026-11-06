"""Design tokens, AQI classification and reusable HTML helpers.

The dashboard uses a green / white / dark-neutral environmental palette.
Category colours are semantic (they communicate AQI severity) and are used
sparingly against the neutral surface.
"""

from __future__ import annotations

import html
import math
import numbers
from typing import Any, Iterable

import numpy as np
import pandas as pd

# ── Palette ─────────────────────────────────────────────────────────────────
PALETTE = {
    "primary": "#1B5E20",
    "primary_light": "#2E7D32",
    "primary_soft": "#E8F5E9",
    "accent": "#4CAF50",
    "accent_soft": "#F1F8E9",
    "bg": "#FFFFFF",
    "surface": "#FFFFFF",
    "surface_alt": "#F5F6F5",
    "text": "#1F2933",
    "muted": "#5F6B7A",
    "border": "#E1E5E1",
    "grid": "#ECEFF1",
    "dark": "#102A1E",
    "warning": "#B7791F",
    "danger": "#C62828",
    "forecast": "#D97706",
}

POLLUTANT_META: dict[str, dict[str, Any]] = {
    "pm25": {
        "label": "PM2.5",
        "unit": "µg/m³",
        "min": 0.0,
        "max": 500.0,
        "step": 1.0,
        "description": "Fine particulate matter (≤ 2.5 µm), the AQI driver in this dataset.",
    },
    "pm10": {
        "label": "PM10",
        "unit": "µg/m³",
        "min": 0.0,
        "max": 1500.0,
        "step": 1.0,
        "description": "Coarse particulate matter (≤ 10 µm).",
    },
    "no2": {
        "label": "NO₂",
        "unit": "µg/m³",
        "min": 0.0,
        "max": 1000.0,
        "step": 1.0,
        "description": "Nitrogen dioxide, largely from combustion and traffic.",
    },
    "so2": {
        "label": "SO₂",
        "unit": "µg/m³",
        "min": 0.0,
        "max": 1000.0,
        "step": 1.0,
        "description": "Sulphur dioxide, linked to fuel burning and industry.",
    },
    "o3": {
        "label": "O₃",
        "unit": "µg/m³",
        "min": 0.0,
        "max": 1000.0,
        "step": 1.0,
        "description": "Ground-level ozone, formed photochemically.",
    },
    "co": {
        "label": "CO",
        "unit": "mg/m³",
        "min": 0.0,
        "max": 50.0,
        "step": 0.05,
        "description": "Carbon monoxide from incomplete combustion.",
    },
}

# US-EPA AQI categories.  The repository derives AQI from PM2.5 with the
# US-EPA 2012 breakpoints, so these are the matching health categories.
AQI_BANDS: list[dict[str, Any]] = [
    {
        "lower": 0,
        "upper": 50,
        "label": "Good",
        "color": "#2E7D32",
        "guidance": "Air quality is satisfactory; little risk for the general public.",
    },
    {
        "lower": 51,
        "upper": 100,
        "label": "Moderate",
        "color": "#A68A00",
        "guidance": "Acceptable for most people; unusually sensitive individuals may notice effects.",
    },
    {
        "lower": 101,
        "upper": 150,
        "label": "Unhealthy for Sensitive Groups",
        "color": "#D9691B",
        "guidance": "Sensitive groups (children, older adults, respiratory conditions) may be affected.",
    },
    {
        "lower": 151,
        "upper": 200,
        "label": "Unhealthy",
        "color": "#C62828",
        "guidance": "Everyone may begin to experience health effects; sensitive groups are at greater risk.",
    },
    {
        "lower": 201,
        "upper": 300,
        "label": "Very Unhealthy",
        "color": "#6A1B9A",
        "guidance": "Health alert: everyone may experience more serious effects. Avoid prolonged outdoor exertion.",
    },
    {
        "lower": 301,
        "upper": 500,
        "label": "Hazardous",
        "color": "#7E0023",
        "guidance": "Emergency conditions: the entire population is likely to be affected.",
    },
]

AQI_BAND_LABELS = [band["label"] for band in AQI_BANDS]
AQI_BAND_COLORS = {band["label"]: band["color"] for band in AQI_BANDS}


def aqi_category(value: float | int | None) -> dict[str, Any]:
    """Return the AQI category record for ``value``.

    The function is deliberately total: missing/non-finite values return an
    ``Unknown`` record instead of raising, so charts and cards can render
    gracefully when a value is unavailable.
    """
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return {
            "label": "Unknown",
            "color": PALETTE["muted"],
            "lower": None,
            "upper": None,
            "guidance": "No AQI value is available.",
        }

    if not math.isfinite(number):
        return {
            "label": "Unknown",
            "color": PALETTE["muted"],
            "lower": None,
            "upper": None,
            "guidance": "No AQI value is available.",
        }

    if number < 0:
        return {
            "label": "Invalid",
            "color": PALETTE["danger"],
            "lower": None,
            "upper": None,
            "guidance": "AQI cannot be negative.",
        }

    for band in AQI_BANDS:
        if number <= band["upper"]:
            return dict(band)

    record = dict(AQI_BANDS[-1])
    record["label"] = "Beyond Hazardous"
    record["guidance"] = "AQI exceeds the standard 500-point scale."
    return record


def aqi_color(value: float | int | None) -> str:
    """Return the semantic colour for an AQI value."""
    return str(aqi_category(value)["color"])


def format_number(value: Any, digits: int = 1, suffix: str = "") -> str:
    """Format a number for display, returning ``N/A`` for missing values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    return f"{number:,.{digits}f}{suffix}"


def inject_css() -> str:
    """Return the dashboard stylesheet.

    The whole dashboard uses exactly two typefaces: Inter for interface and
    body text, JetBrains Mono for numeric and metadata accents.  The Google
    Fonts import is transparent and falls back to the generic family when the
    dashboard is viewed offline.
    """
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {{
        --aqi-primary: {PALETTE['primary']};
        --aqi-primary-light: {PALETTE['primary_light']};
        --aqi-soft: {PALETTE['primary_soft']};
        --aqi-bg: {PALETTE['bg']};
        --aqi-text: {PALETTE['text']};
        --aqi-muted: {PALETTE['muted']};
        --aqi-border: {PALETTE['border']};
        --aqi-font-sans: 'Inter', sans-serif;
        --aqi-font-mono: 'JetBrains Mono', monospace;
        --font: 'Inter', sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
    }}

    html, body, .stApp,
    .stApp p, .stApp li, .stApp label,
    .stApp button, .stApp input, .stApp textarea, .stApp select,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stWidgetLabel"] {{
        font-family: 'Inter', sans-serif !important;
    }}

    .stApp code, .stApp pre, .stApp kbd, .stApp samp,
    .stApp code *, .stApp pre *,
    .aqi-mono, .aqi-metric .value, .aqi-forecast-card .big,
    [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    .aqi-eyebrow, .aqi-badge, .aqi-metric .label,
    .aqi-forecast-card .kicker, .aqi-footer,
    [data-testid="stSidebar"] .stRadio > label {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    .stApp {{
        background: {PALETTE['bg']};
        color: {PALETTE['text']};
    }}

    .block-container {{
        max-width: 1440px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }}

    h1, h2, h3, h4 {{
        color: #14281D;
        letter-spacing: -0.02em;
    }}

    p, li, label, span {{
        color: {PALETTE['text']};
    }}

    [data-testid="stSidebar"] {{
        background: {PALETTE['surface']};
        border-right: 1px solid {PALETTE['border']};
    }}

    [data-testid="stSidebar"] .stRadio > label {{
        font-weight: 700;
        color: #14281D;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] label {{
        padding: 0.45rem 0.6rem;
        border-radius: 10px;
        margin-bottom: 0.15rem;
        transition: background 120ms ease;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
        background: rgba(46, 125, 50, 0.10);
    }}

    .aqi-hero {{
        background: {PALETTE['surface']};
        color: {PALETTE['text']};
        padding: 2.0rem 2.1rem;
        border: 1px solid {PALETTE['border']};
        border-left: 6px solid {PALETTE['primary']};
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(16, 42, 30, 0.06);
        margin-bottom: 1.2rem;
        position: relative;
        overflow: hidden;
    }}

    .aqi-hero::after {{
        content: "";
        position: absolute;
        right: -70px;
        top: -90px;
        width: 320px;
        height: 320px;
        background: radial-gradient(circle, rgba(46, 125, 50, 0.06) 0%, rgba(46, 125, 50, 0) 68%);
        pointer-events: none;
    }}

    .aqi-hero h1 {{
        color: #14281D;
        margin: 0 0 0.35rem 0;
        font-size: 2.0rem;
        line-height: 1.15;
    }}

    .aqi-hero p {{
        color: {PALETTE['muted']};
        margin: 0.25rem 0 0 0;
        max-width: 850px;
        font-size: 1rem;
    }}

    .aqi-eyebrow {{
        display: inline-block;
        font-size: 0.72rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: {PALETTE['primary']};
        font-weight: 700;
        margin-bottom: 0.55rem;
    }}

    .aqi-badge {{
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        background: {PALETTE['surface_alt']};
        color: {PALETTE['primary']};
        font-size: 0.74rem;
        font-weight: 600;
        margin-right: 0.35rem;
        margin-top: 0.75rem;
        border: 1px solid {PALETTE['border']};
    }}

    .aqi-card {{
        background: {PALETTE['surface']};
        border: 1px solid {PALETTE['border']};
        border-radius: 16px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 1px 2px rgba(16, 42, 30, 0.05);
        height: 100%;
    }}

    .aqi-metric {{
        background: {PALETTE['surface']};
        border: 1px solid {PALETTE['border']};
        border-left: 4px solid var(--accent, {PALETTE['primary']});
        border-radius: 14px;
        padding: 0.95rem 1.05rem;
        box-shadow: 0 1px 2px rgba(16, 42, 30, 0.05);
        margin-bottom: 0.35rem;
        min-height: 116px;
    }}

    .aqi-metric .label {{
        color: {PALETTE['muted']};
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }}

    .aqi-metric .value {{
        color: #14281D;
        font-size: 1.75rem;
        font-weight: 800;
        line-height: 1.1;
    }}

    .aqi-metric .sub {{
        color: {PALETTE['muted']};
        font-size: 0.80rem;
        margin-top: 0.35rem;
        line-height: 1.35;
    }}

    .aqi-pill {{
        display: inline-block;
        padding: 0.22rem 0.62rem;
        border-radius: 999px;
        color: #FFFFFF;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }}

    .aqi-forecast-card {{
        background: {PALETTE['surface']};
        border: 1px solid {PALETTE['border']};
        border-left: 5px solid {PALETTE['primary_light']};
        border-radius: 16px;
        padding: 1.4rem 1.5rem;
        box-shadow: 0 8px 20px rgba(16, 42, 30, 0.06);
        margin-bottom: 0.8rem;
    }}

    .aqi-forecast-card .kicker {{
        color: {PALETTE['muted']};
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }}

    .aqi-forecast-card .big {{
        font-size: 3.0rem;
        font-weight: 850;
        line-height: 1.0;
        color: #14281D;
        margin: 0.2rem 0 0.3rem 0;
    }}

    .aqi-callout {{
        background: {PALETTE['surface_alt']};
        border: 1px solid {PALETTE['border']};
        border-radius: 14px;
        padding: 0.85rem 1.0rem;
        color: {PALETTE['text']};
        font-size: 0.90rem;
        line-height: 1.5;
        margin: 0.6rem 0;
    }}

    .aqi-callout.warn {{
        background: #FFF8E6;
        border-color: #F0D48A;
    }}

    .aqi-callout.info {{
        background: {PALETTE['surface_alt']};
        border-color: {PALETTE['border']};
    }}

    .aqi-footer {{
        color: {PALETTE['muted']};
        font-size: 0.78rem;
        line-height: 1.5;
        border-top: 1px solid {PALETTE['border']};
        padding-top: 0.8rem;
        margin-top: 1.4rem;
    }}

    div[data-testid="stMetric"] {{
        background: {PALETTE['surface']};
        border: 1px solid {PALETTE['border']};
        border-radius: 14px;
        padding: 0.8rem 1rem;
    }}

    div[data-testid="stMetricLabel"] p {{
        color: {PALETTE['muted']};
        font-weight: 600;
    }}

    .stButton > button {{
        background: {PALETTE['primary']};
        color: #FFFFFF;
        border: 0;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.55rem 1.0rem;
    }}

    .stButton > button:hover {{
        background: {PALETTE['primary_light']};
        color: #FFFFFF;
    }}

    .stDownloadButton > button {{
        border-radius: 10px;
        border: 1px solid {PALETTE['border']};
    }}

    div[data-testid="stExpander"] {{
        background: {PALETTE['surface']};
        border: 1px solid {PALETTE['border']};
        border-radius: 14px;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.35rem;
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 0.9rem;
        font-weight: 650;
    }}

    /* BaseWeb select widgets otherwise fall back to Streamlit's bundled font. */
    .stApp [data-baseweb="select"] *,
    [data-baseweb="popover"] *,
    [data-baseweb="menu"] * {{
        font-family: 'Inter', sans-serif !important;
    }}

    .js-plotly-plot .plotly .modebar {{
        opacity: 0.25;
    }}

    .aqi-table-wrap {{
        width: 100%;
        overflow-x: auto;
        border: 1px solid {PALETTE['border']};
        border-radius: 14px;
        background: {PALETTE['surface']};
        margin: 0.35rem 0 0.55rem 0;
    }}

    table.aqi-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.86rem;
    }}

    .aqi-table thead th {{
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-size: 0.66rem;
        font-weight: 700;
        color: {PALETTE['muted']};
        background: {PALETTE['surface_alt']};
        text-align: left;
        padding: 0.62rem 0.85rem;
        border-bottom: 1px solid {PALETTE['border']};
        white-space: nowrap;
    }}

    .aqi-table tbody td {{
        font-family: 'Inter', sans-serif !important;
        color: {PALETTE['text']};
        padding: 0.55rem 0.85rem;
        border-bottom: 1px solid {PALETTE['grid']};
        vertical-align: top;
        line-height: 1.4;
    }}

    .aqi-table tbody tr:last-child td {{
        border-bottom: 0;
    }}

    .aqi-table tbody tr:hover {{
        background: {PALETTE['surface_alt']};
    }}

    .aqi-table td.num, .aqi-table th.num {{
        font-family: 'JetBrains Mono', monospace !important;
        font-variant-numeric: tabular-nums;
        text-align: right;
        white-space: nowrap;
    }}

    .aqi-table th.num {{
        text-align: right;
    }}

    @media (max-width: 768px) {{
        .aqi-hero {{
            padding: 1.25rem 1.15rem;
            border-radius: 16px;
        }}
        .aqi-hero h1 {{
            font-size: 1.45rem;
        }}
        .aqi-metric .value {{
            font-size: 1.45rem;
        }}
        .aqi-forecast-card .big {{
            font-size: 2.35rem;
        }}
        .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}
    }}
    </style>
    """


def page_header(title: str, subtitle: str, eyebrow: str | None = None) -> str:
    """Return a section heading block."""
    eyebrow_html = (
        f'<div class="aqi-eyebrow">{html.escape(eyebrow)}</div>' if eyebrow else ""
    )
    return f"""
    <div style="margin-bottom: 1.1rem;">
        {eyebrow_html}
        <h1 style="margin: 0 0 0.25rem 0; font-size: 1.75rem;">{html.escape(title)}</h1>
        <p style="margin: 0; color: {PALETTE['muted']}; max-width: 900px;">{html.escape(subtitle)}</p>
    </div>
    """


def hero(
    title: str,
    subtitle: str,
    badges: list[str] | None = None,
    eyebrow: str = "Kolkata · Environmental Analytics",
) -> str:
    """Return the overview hero block."""
    badge_html = "".join(
        f'<span class="aqi-badge">{html.escape(badge)}</span>'
        for badge in (badges or [])
    )
    return f"""
    <div class="aqi-hero">
        <div class="aqi-eyebrow">{html.escape(eyebrow)}</div>
        <h1>{html.escape(title)}</h1>
        <p>{html.escape(subtitle)}</p>
        <div>{badge_html}</div>
    </div>
    """


def metric_card(
    label: str,
    value: str,
    subtitle: str = "",
    accent: str = PALETTE["primary"],
) -> str:
    """Return a metric card as HTML."""
    safe_subtitle = (
        f'<div class="sub">{html.escape(subtitle)}</div>' if subtitle else ""
    )
    return f"""
    <div class="aqi-metric" style="--accent:{accent};">
        <div class="label">{html.escape(label)}</div>
        <div class="value">{html.escape(value)}</div>
        {safe_subtitle}
    </div>
    """


def pill(label: str, color: str) -> str:
    """Return an AQI category pill."""
    return (
        f'<span class="aqi-pill" style="background:{color};">'
        f"{html.escape(label)}</span>"
    )


def callout(text: str, kind: str = "info") -> str:
    """Return a callout block. ``kind`` is ``info`` or ``warn``."""
    css_class = "aqi-callout warn" if kind == "warn" else "aqi-callout info"
    return f'<div class="{css_class}">{text}</div>'


def _format_table_cell(value: Any) -> tuple[str, bool]:
    """Return ``(text, is_numeric)`` for a table cell."""
    if value is None:
        return "—", False
    try:
        if pd.isna(value):
            return "—", False
    except (TypeError, ValueError):
        pass

    if isinstance(value, (bool, np.bool_)):
        return ("Yes" if value else "No"), False
    if isinstance(value, numbers.Integral):
        return f"{int(value):,}", True
    if isinstance(value, numbers.Real):
        number = float(value)
        if not math.isfinite(number):
            return "—", False
        if number.is_integer():
            return f"{int(number):,}", True
        return f"{number:,.3f}".rstrip("0").rstrip("."), True
    return str(value), False


def data_table(
    frame: pd.DataFrame,
    *,
    numeric_columns: Iterable[str] | None = None,
    max_rows: int | None = None,
) -> str:
    """Render a DataFrame as an accessible, theme-native HTML table.

    Streamlit's canvas dataframe draws cell text with its own bundled font,
    which would bypass the dashboard's two-typeface system.  This helper keeps
    text in Inter and numeric columns in JetBrains Mono.
    """
    if frame is None or frame.empty:
        return ""

    view = frame.head(max_rows) if max_rows is not None else frame
    auto_numeric = {
        str(column)
        for column in view.select_dtypes(include="number").columns
    }
    if numeric_columns is not None:
        auto_numeric.update(str(column) for column in numeric_columns)

    header_cells = "".join(
        f'<th class="{"num" if str(column) in auto_numeric else ""}">'
        f"{html.escape(str(column))}</th>"
        for column in view.columns
    )

    body_rows = []
    for record in view.itertuples(index=False):
        cells = []
        for column, value in zip(view.columns, record):
            text, is_numeric = _format_table_cell(value)
            css_class = "num" if is_numeric or str(column) in auto_numeric else ""
            cells.append(f'<td class="{css_class}">{html.escape(text)}</td>')
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return (
        '<div class="aqi-table-wrap">'
        '<table class="aqi-table">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )
