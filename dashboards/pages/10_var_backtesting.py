from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.name == "dashboards":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboards.common import load_selected_dashboard_data, prepare_dates, render_table_download


st.set_page_config(page_title="VaR Backtesting", layout="wide")
data = load_selected_dashboard_data()
var_backtest = prepare_dates(data.get("var_backtest", None), ("date",)) if data.get("var_backtest") is not None else None

st.title("VaR Backtesting")
if var_backtest is None or var_backtest.empty:
    st.info("No VaR backtesting rows are available.")
    st.stop()

confidence_levels = sorted(var_backtest["confidence_level"].dropna().unique())
confidence = st.selectbox(
    "Confidence",
    confidence_levels,
    index=confidence_levels.index(0.95) if 0.95 in confidence_levels else 0,
    format_func=lambda value: f"{value:.1%}",
)
selected = var_backtest[var_backtest["confidence_level"] == confidence].sort_values("date")
summary = selected.iloc[-1]

cols = st.columns(5)
cols[0].metric("Observations", f"{int(summary['total_observations'])}")
cols[1].metric("Exceptions", f"{int(summary['number_of_exceptions'])}")
cols[2].metric("Expected", f"{summary['expected_exceptions']:.1f}")
cols[3].metric("Kupiec p-value", f"{summary['p_value']:.3f}")
cols[4].metric("Status", str(summary["pass_fail"]).upper())

left, right = st.columns(2)
with left:
    st.plotly_chart(
        px.bar(
            selected,
            x="date",
            y="realized_pnl",
            color="breach",
            title="Realized P&L Exceptions",
            labels={"realized_pnl": "Realized P&L"},
        ),
        width="stretch",
    )
with right:
    st.plotly_chart(
        px.bar(
            selected,
            x="date",
            y="breach_severity",
            title="Breach Severity",
            labels={"breach_severity": "Severity"},
        ),
        width="stretch",
    )

backtest_table = selected[
    [
        "date",
        "confidence_level",
        "var_estimate",
        "realized_pnl",
        "breach",
        "breach_severity",
        "exception_ratio",
        "kupiec_statistic",
        "p_value",
        "pass_fail",
    ]
]
st.dataframe(
    backtest_table,
    width="stretch",
    hide_index=True,
    column_config={
        "confidence_level": st.column_config.NumberColumn(format="%.2f"),
        "var_estimate": st.column_config.NumberColumn(format="$%.0f"),
        "realized_pnl": st.column_config.NumberColumn(format="$%.0f"),
        "breach_severity": st.column_config.NumberColumn(format="$%.0f"),
        "exception_ratio": st.column_config.NumberColumn(format="%.2f"),
        "kupiec_statistic": st.column_config.NumberColumn(format="%.3f"),
        "p_value": st.column_config.NumberColumn(format="%.3f"),
    },
)
render_table_download(backtest_table, "var_backtesting", key="var_backtesting_csv")
