from __future__ import annotations

import os
import sys
from datetime import UTC, date, datetime, timedelta
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
DATA_SOURCE_STATE_KEY = "dashboard_data_source_selected"
DATA_SOURCE_WIDGET_KEY = "_dashboard_data_source"
PRICE_HISTORY_MODE_OPTIONS = ("full", "period", "date_range", "lookback")
PRICE_HISTORY_MODE_LABELS = {
    "full": "Full history",
    "period": "Period",
    "date_range": "Date range",
    "lookback": "Rolling lookback",
}
PRICE_HISTORY_MODE_STATE_KEY = "price_history_mode_selected"
PRICE_HISTORY_MODE_WIDGET_KEY = "_price_history_mode"
PRICE_PERIOD_STATE_KEY = "price_period_selected"
PRICE_PERIOD_WIDGET_KEY = "_price_period"
PRICE_START_STATE_KEY = "price_start_selected"
PRICE_START_WIDGET_KEY = "_price_start"
PRICE_END_STATE_KEY = "price_end_selected"
PRICE_END_WIDGET_KEY = "_price_end"
PRICE_LOOKBACK_STATE_KEY = "price_lookback_days_selected"
PRICE_LOOKBACK_WIDGET_KEY = "_price_lookback_days"
YFINANCE_PERIOD_OPTIONS = ("max", "10y", "5y", "2y", "1y", "ytd", "6mo", "3mo", "1mo", "5d", "1d")


def normalize_dashboard_data_mode(mode: str | None = None) -> str:
    """Normalize configured dashboard source names to UI-supported modes."""
    raw_mode = (mode or os.getenv("MARKET_DATA_MODE", "sample")).strip().lower()
    normalized = DATA_SOURCE_ALIASES.get(raw_mode, raw_mode)
    if normalized not in DATA_SOURCE_OPTIONS:
        raise ValueError("Dashboard data source must be sample, live, or database")
    return normalized


def render_data_source_control() -> str:
    """Render the shared dashboard data source selector."""
    default_mode = _persisted_value(DATA_SOURCE_STATE_KEY, normalize_dashboard_data_mode())
    selected_mode = st.sidebar.radio(
        "Data source",
        DATA_SOURCE_OPTIONS,
        index=DATA_SOURCE_OPTIONS.index(default_mode),
        format_func=lambda mode: DATA_SOURCE_LABELS[mode],
        key=DATA_SOURCE_WIDGET_KEY,
    )
    _set_persisted_value(DATA_SOURCE_STATE_KEY, selected_mode)
    st.sidebar.caption("Sample uses bundled CSV data. Live uses Yahoo Finance. Database reads PostgreSQL marts.")
    return selected_mode


def render_price_history_controls(data_source: str) -> dict[str, str | int | None]:
    """Render live yfinance history controls and return run_pipeline keyword arguments."""
    if data_source != "live":
        return {
            "price_start": None,
            "price_end": None,
            "price_lookback_days": None,
            "price_period": None,
        }

    default_history_mode = _persisted_value(PRICE_HISTORY_MODE_STATE_KEY, "full")
    history_mode = st.sidebar.radio(
        "Yahoo history",
        PRICE_HISTORY_MODE_OPTIONS,
        index=PRICE_HISTORY_MODE_OPTIONS.index(default_history_mode),
        format_func=lambda mode: PRICE_HISTORY_MODE_LABELS[mode],
        key=PRICE_HISTORY_MODE_WIDGET_KEY,
    )
    _set_persisted_value(PRICE_HISTORY_MODE_STATE_KEY, history_mode)
    if history_mode == "period":
        default_period = _persisted_value(PRICE_PERIOD_STATE_KEY, "max")
        period = st.sidebar.selectbox(
            "Period",
            YFINANCE_PERIOD_OPTIONS,
            index=YFINANCE_PERIOD_OPTIONS.index(default_period),
            key=PRICE_PERIOD_WIDGET_KEY,
        )
        _set_persisted_value(PRICE_PERIOD_STATE_KEY, period)
        return {
            "price_start": None,
            "price_end": None,
            "price_lookback_days": None,
            "price_period": period,
        }

    if history_mode == "date_range":
        today = datetime.now(UTC).date()
        start_date = st.sidebar.date_input(
            "Start date",
            value=_persisted_value(PRICE_START_STATE_KEY, date(2020, 1, 1)),
            key=PRICE_START_WIDGET_KEY,
        )
        end_date = st.sidebar.date_input(
            "End date",
            value=_persisted_value(PRICE_END_STATE_KEY, today + timedelta(days=1)),
            key=PRICE_END_WIDGET_KEY,
        )
        if start_date >= end_date:
            st.sidebar.error("Start date must be before end date.")
            st.stop()
        _set_persisted_value(PRICE_START_STATE_KEY, start_date)
        _set_persisted_value(PRICE_END_STATE_KEY, end_date)
        return {
            "price_start": start_date.isoformat(),
            "price_end": end_date.isoformat(),
            "price_lookback_days": None,
            "price_period": None,
        }

    if history_mode == "lookback":
        lookback_days = st.sidebar.number_input(
            "Lookback days",
            min_value=1,
            max_value=10000,
            value=_persisted_value(PRICE_LOOKBACK_STATE_KEY, 756),
            step=21,
            key=PRICE_LOOKBACK_WIDGET_KEY,
        )
        _set_persisted_value(PRICE_LOOKBACK_STATE_KEY, int(lookback_days))
        return {
            "price_start": None,
            "price_end": None,
            "price_lookback_days": int(lookback_days),
            "price_period": None,
        }

    return {
        "price_start": None,
        "price_end": None,
        "price_lookback_days": None,
        "price_period": None,
    }


def load_selected_dashboard_data() -> dict:
    """Load dashboard data for the source selected in the sidebar."""
    data_source = render_data_source_control()
    price_controls = render_price_history_controls(data_source)
    return load_dashboard_data(data_source, **price_controls)


def _persisted_value(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def _set_persisted_value(key: str, value) -> None:
    st.session_state[key] = value


@st.cache_data(show_spinner=False)
def load_dashboard_data(
    mode: str | None = None,
    price_start: str | None = None,
    price_end: str | None = None,
    price_lookback_days: int | None = None,
    price_period: str | None = None,
) -> dict:
    """Load analytics for Streamlit pages from the selected data source."""
    data_source = normalize_dashboard_data_mode(mode)
    if data_source == "database":
        return read_pipeline_outputs(get_engine())
    return run_pipeline(
        PROJECT_ROOT,
        prefer_live=data_source == "live",
        allow_fallback=data_source == "sample",
        write_processed=False,
        price_start=price_start,
        price_end=price_end,
        price_lookback_days=price_lookback_days,
        price_period=price_period,
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
