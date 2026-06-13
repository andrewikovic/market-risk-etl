from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from html import escape
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import load_scenarios, run_pipeline  # noqa: E402
from src.load.db import get_engine  # noqa: E402
from src.load.read_db import read_pipeline_outputs  # noqa: E402
from src.risk.stress_testing import run_stress_test  # noqa: E402


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
PDF_ROWS_PER_SECTION = 18
PDF_COLUMNS_PER_SECTION = 8


@dataclass(frozen=True)
class RiskPack:
    generated_at: datetime
    sections: tuple[tuple[str, pd.DataFrame], ...]


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


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a dashboard table as UTF-8 CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def render_table_download(
    df: pd.DataFrame,
    file_stem: str,
    label: str = "Download CSV",
    key: str | None = None,
) -> None:
    """Render a CSV download button for one dashboard table."""
    disabled = df.empty
    st.download_button(
        label,
        data=b"" if disabled else dataframe_to_csv_bytes(df),
        file_name=f"{_slugify(file_stem)}.csv",
        mime="text/csv",
        key=key,
        disabled=disabled,
    )


def render_risk_pack_downloads(
    dashboard_data: dict[str, Any],
    scenarios: list[dict] | dict | None = None,
    stress_results: dict[str, pd.DataFrame] | None = None,
    key_prefix: str = "risk_pack",
) -> RiskPack:
    """Render CSV, HTML, and PDF exports for the generated risk pack."""
    pack = build_risk_pack(dashboard_data, scenarios=scenarios, stress_results=stress_results)
    file_stem = f"market_risk_pack_{pack.generated_at.strftime('%Y%m%d_%H%M%S')}"

    st.subheader("Risk Pack Exports")
    columns = st.columns(3)
    columns[0].download_button(
        "Download risk-pack CSV",
        data=risk_pack_to_csv_bytes(pack),
        file_name=f"{file_stem}.csv",
        mime="text/csv",
        key=f"{key_prefix}_csv",
    )
    columns[1].download_button(
        "Download risk-pack HTML",
        data=risk_pack_to_html_bytes(pack),
        file_name=f"{file_stem}.html",
        mime="text/html",
        key=f"{key_prefix}_html",
    )
    columns[2].download_button(
        "Download risk-pack PDF",
        data=risk_pack_to_pdf_bytes(pack),
        file_name=f"{file_stem}.pdf",
        mime="application/pdf",
        key=f"{key_prefix}_pdf",
    )
    return pack


def build_risk_pack(
    dashboard_data: dict[str, Any],
    scenarios: list[dict] | dict | None = None,
    stress_results: dict[str, pd.DataFrame] | None = None,
    generated_at: datetime | None = None,
) -> RiskPack:
    """Build the reusable risk-pack sections from dashboard pipeline outputs."""
    generated = generated_at or datetime.now(UTC)
    stress = stress_results or _run_pack_stress_tests(dashboard_data, scenarios)
    sections = (
        ("Portfolio Summary", _portfolio_summary_frame(dashboard_data)),
        ("VaR/ES", _var_es_frame(dashboard_data)),
        ("Stress Tests", _stress_summary_frame(stress)),
        ("Exposures", _exposures_frame(dashboard_data)),
        ("Backtesting", _backtesting_frame(dashboard_data)),
    )
    return RiskPack(generated_at=generated, sections=sections)


def risk_pack_to_csv_bytes(pack: RiskPack) -> bytes:
    """Serialize all risk-pack sections into one sectioned CSV file."""
    buffer = StringIO()
    buffer.write("Market Risk Pack\n")
    buffer.write(f"Generated At,{pack.generated_at.isoformat()}\n\n")
    for title, frame in pack.sections:
        buffer.write(f"{title}\n")
        frame.to_csv(buffer, index=False)
        buffer.write("\n")
    return buffer.getvalue().encode("utf-8")


def risk_pack_to_html_bytes(pack: RiskPack) -> bytes:
    """Serialize the risk pack as a standalone HTML report."""
    body = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        "<title>Market Risk Pack</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:32px;color:#1f2937;}",
        "h1{margin-bottom:4px;} h2{margin-top:28px;border-bottom:1px solid #d1d5db;padding-bottom:6px;}",
        "table{border-collapse:collapse;width:100%;font-size:13px;} th,td{border:1px solid #d1d5db;padding:6px;text-align:left;}",
        "th{background:#f3f4f6;} .meta{color:#6b7280;margin-top:0;}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Market Risk Pack</h1>",
        f'<p class="meta">Generated at {escape(pack.generated_at.isoformat())}</p>',
    ]
    for title, frame in pack.sections:
        body.append(f"<h2>{escape(title)}</h2>")
        body.append(frame.to_html(index=False, border=0, escape=True))
    body.extend(["</body>", "</html>"])
    return "\n".join(body).encode("utf-8")


def risk_pack_to_pdf_bytes(pack: RiskPack) -> bytes:
    """Serialize the risk pack as a lightweight text PDF without extra dependencies."""
    lines = ["Market Risk Pack", f"Generated at {pack.generated_at.isoformat()}", ""]
    for title, frame in pack.sections:
        lines.extend(_frame_to_pdf_lines(title, frame))
        lines.append("")
    return _simple_pdf(lines)


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


def _run_pack_stress_tests(
    dashboard_data: dict[str, Any],
    scenarios: list[dict] | dict | None,
) -> dict[str, pd.DataFrame] | None:
    scenario_list = scenarios
    if scenario_list is None:
        scenario_list = load_dashboard_scenarios().get("scenarios", [])
    positions = dashboard_data.get("current_positions")
    if not isinstance(positions, pd.DataFrame) or positions.empty:
        return None
    try:
        return run_stress_test(positions, scenario_list)
    except ValueError:
        return None


def _portfolio_summary_frame(dashboard_data: dict[str, Any]) -> pd.DataFrame:
    portfolio_values = _as_frame(dashboard_data.get("portfolio_values"))
    current_positions = _as_frame(dashboard_data.get("current_positions"))
    if portfolio_values.empty:
        return _status_frame("No portfolio value rows are available.")

    latest = _latest_row(portfolio_values, "value_date")
    row = {
        "portfolio_name": latest.get("portfolio_name", "Portfolio"),
        "as_of_date": latest.get("value_date", latest.get("date", "")),
        "market_value": latest.get("market_value"),
        "daily_pnl": latest.get("daily_pnl"),
        "daily_return": latest.get("daily_return"),
        "cumulative_return": latest.get("cumulative_return"),
        "holdings": len(current_positions),
    }
    return pd.DataFrame([row])


def _var_es_frame(dashboard_data: dict[str, Any]) -> pd.DataFrame:
    risk_metrics = dashboard_data.get("risk_metrics") or {}
    metric_labels = {
        "historical_var_95": "Historical VaR",
        "parametric_var_95": "Parametric VaR",
        "expected_shortfall_95": "Expected Shortfall",
    }
    rows = [
        {
            "metric": label,
            "confidence_level": 0.95,
            "value": risk_metrics.get(metric_name),
        }
        for metric_name, label in metric_labels.items()
        if metric_name in risk_metrics
    ]
    if rows:
        return pd.DataFrame(rows)

    risk_metrics_frame = _as_frame(dashboard_data.get("risk_metrics_frame"))
    if risk_metrics_frame.empty or "metric_name" not in risk_metrics_frame.columns:
        return _status_frame("No VaR or expected shortfall metrics are available.")
    mask = risk_metrics_frame["metric_name"].astype(str).str.contains("var|shortfall|es", case=False, regex=True)
    frame = risk_metrics_frame.loc[mask].copy()
    return frame if not frame.empty else _status_frame("No VaR or expected shortfall metrics are available.")


def _stress_summary_frame(stress_results: dict[str, pd.DataFrame] | None) -> pd.DataFrame:
    if not stress_results:
        return _status_frame("No stress test results are available.")
    scenario_results = _as_frame(stress_results.get("scenario_results"))
    if scenario_results.empty:
        return _status_frame("No stress test results are available.")
    columns = [
        "portfolio_name",
        "scenario_name",
        "run_timestamp",
        "current_portfolio_value",
        "shocked_portfolio_value",
        "stress_loss",
        "stress_return",
    ]
    columns = [column for column in columns if column in scenario_results.columns]
    return scenario_results.sort_values("stress_loss", ascending=False)[columns].reset_index(drop=True)


def _exposures_frame(dashboard_data: dict[str, Any]) -> pd.DataFrame:
    exposures = _as_frame(dashboard_data.get("exposures"))
    if exposures.empty:
        return _status_frame("No exposure rows are available.")
    if {"exposure_type", "market_value"}.issubset(exposures.columns):
        return exposures.sort_values(["exposure_type", "market_value"], ascending=[True, False]).reset_index(drop=True)
    if "market_value" in exposures.columns:
        return exposures.sort_values("market_value", ascending=False).reset_index(drop=True)
    return exposures.reset_index(drop=True)


def _backtesting_frame(dashboard_data: dict[str, Any]) -> pd.DataFrame:
    var_backtest = _as_frame(dashboard_data.get("var_backtest"))
    if var_backtest.empty:
        return _status_frame("No VaR backtesting rows are available.")

    date_column = "date" if "date" in var_backtest.columns else None
    sort_columns = [column for column in ["confidence_level", date_column] if column]
    sorted_frame = var_backtest.sort_values(sort_columns) if sort_columns else var_backtest.copy()
    if "confidence_level" in sorted_frame.columns:
        summary = sorted_frame.groupby("confidence_level", as_index=False).tail(1)
    else:
        summary = sorted_frame.tail(1)
    columns = [
        "portfolio_name",
        "date",
        "confidence_level",
        "total_observations",
        "number_of_exceptions",
        "expected_exceptions",
        "exception_ratio",
        "kupiec_statistic",
        "p_value",
        "pass_fail",
    ]
    columns = [column for column in columns if column in summary.columns]
    return summary[columns].reset_index(drop=True)


def _frame_to_pdf_lines(title: str, frame: pd.DataFrame) -> list[str]:
    lines = [title]
    display = frame.head(PDF_ROWS_PER_SECTION)
    display = display.iloc[:, :PDF_COLUMNS_PER_SECTION]
    table = display.to_string(index=False, max_colwidth=22, line_width=110)
    lines.extend(table.splitlines())
    if len(frame) > len(display):
        lines.append(f"... {len(frame) - len(display)} more rows")
    if frame.shape[1] > display.shape[1]:
        lines.append(f"... {frame.shape[1] - display.shape[1]} more columns")
    return lines


def _simple_pdf(lines: list[str]) -> bytes:
    page_lines = _paginate_pdf_lines(lines)
    objects: list[bytes] = []
    page_ids = [4 + idx * 2 for idx in range(len(page_lines))]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_lines)} >>".encode("ascii"))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for idx, page in enumerate(page_lines):
        page_id = 4 + idx * 2
        content_id = page_id + 1
        stream = _pdf_page_stream(page)
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode("ascii")
        )
        objects.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")

    pdf = b"%PDF-1.4\n"
    offsets = [0]
    for object_id, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{object_id} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"

    xref_offset = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    pdf += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        pdf += f"{offset:010d} 00000 n \n".encode("ascii")
    pdf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")
    return pdf


def _paginate_pdf_lines(lines: list[str], lines_per_page: int = 48) -> list[list[str]]:
    if not lines:
        return [["No risk-pack content is available."]]
    return [lines[idx : idx + lines_per_page] for idx in range(0, len(lines), lines_per_page)]


def _pdf_page_stream(lines: list[str]) -> bytes:
    commands = ["BT", "/F1 9 Tf", "50 760 Td", "12 TL"]
    for line in lines:
        commands.append(f"({_pdf_escape(line)}) Tj")
        commands.append("T*")
    commands.append("ET")
    return "\n".join(commands).encode("latin-1", errors="replace")


def _pdf_escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _latest_row(frame: pd.DataFrame, date_column: str) -> pd.Series:
    if date_column in frame.columns:
        data = frame.copy()
        data[date_column] = pd.to_datetime(data[date_column])
        return data.sort_values(date_column).iloc[-1]
    return frame.iloc[-1]


def _as_frame(value: Any) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _status_frame(message: str) -> pd.DataFrame:
    return pd.DataFrame({"status": [message]})


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "dashboard_export"
