from __future__ import annotations

import sys
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import load_scenarios, run_pipeline  # noqa: E402


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> dict:
    """Load analytics for Streamlit pages from sample or live market data."""
    mode = os.getenv("MARKET_DATA_MODE", "sample").strip().lower()
    if mode not in {"sample", "live", "live_with_fallback"}:
        raise ValueError("MARKET_DATA_MODE must be sample, live, or live_with_fallback")
    return run_pipeline(
        PROJECT_ROOT,
        prefer_live=mode in {"live", "live_with_fallback"},
        allow_fallback=mode != "live",
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
