from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import load_scenarios, run_pipeline  # noqa: E402
from src.load.db import get_engine  # noqa: E402
from src.load.read_db import read_pipeline_outputs  # noqa: E402


DATA_SOURCE_OPTIONS = ("sample", "live", "database")
DATA_SOURCE_LABELS = {
    "sample": "Sample CSV",
    "live": "Live Yahoo Finance",
    "database": "PostgreSQL database",
}
DATA_SOURCE_ALIASES = {
    "db": "database",
    "postgres": "database",
    "postgresql": "database",
    "live_with_fallback": "live",
}


def normalize_dashboard_data_mode(mode: str | None = None) -> str:
    """Normalize configured dashboard source names to UI-supported modes."""
    raw_mode = (mode or os.getenv("MARKET_DATA_MODE", "sample")).strip().lower()
    normalized = DATA_SOURCE_ALIASES.get(raw_mode, raw_mode)
    if normalized not in DATA_SOURCE_OPTIONS:
        raise ValueError("Dashboard data source must be sample, live, or database")
    return normalized


def render_data_source_control() -> str:
    """Render the shared dashboard data source selector."""
    default_mode = normalize_dashboard_data_mode()
    selected_mode = st.sidebar.radio(
        "Data source",
        DATA_SOURCE_OPTIONS,
        index=DATA_SOURCE_OPTIONS.index(default_mode),
        format_func=lambda mode: DATA_SOURCE_LABELS[mode],
        key="dashboard_data_source",
    )
    st.sidebar.caption("Sample uses bundled CSV data. Live uses Yahoo Finance. Database reads PostgreSQL marts.")
    return selected_mode


def load_selected_dashboard_data() -> dict:
    """Load dashboard data for the source selected in the sidebar."""
    return load_dashboard_data(render_data_source_control())


@st.cache_data(show_spinner=False)
def load_dashboard_data(mode: str | None = None) -> dict:
    """Load analytics for Streamlit pages from the selected data source."""
    data_source = normalize_dashboard_data_mode(mode)
    if data_source == "database":
        return read_pipeline_outputs(get_engine())
    return run_pipeline(
        PROJECT_ROOT,
        prefer_live=data_source == "live",
        allow_fallback=data_source == "sample",
        write_processed=False,
    )


@st.cache_data(show_spinner=False)
def load_dashboard_scenarios() -> dict:
    """Load stress scenarios for Streamlit pages."""
    return load_scenarios(PROJECT_ROOT)


def prepare_dates(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    data = df.copy()
    for column in columns:
        if column in data.columns:
            data[column] = pd.to_datetime(data[column])
    return data


def currency(value: float) -> str:
    return f"${value:,.0f}"


def percent(value: float) -> str:
    return f"{value:.2%}"


def bar_chart(df: pd.DataFrame, x: str, y: str, color: str | None = None, title: str | None = None):
    return px.bar(df, x=x, y=y, color=color, title=title, text_auto=".2s")
