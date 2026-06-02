from __future__ import annotations

import plotly.express as px
import streamlit as st

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.name == "dashboards":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboards.common import currency, load_selected_dashboard_data, percent, prepare_dates


st.set_page_config(page_title="Market Risk Analytics Platform", page_icon=":chart_with_upwards_trend:", layout="wide")

data = load_selected_dashboard_data()
portfolio_values = prepare_dates(data["portfolio_values"], ("value_date",))
exposures = data["exposures"]
risk_metrics = data["risk_metrics"]
latest_value = float(portfolio_values["market_value"].iloc[-1])
latest_return = float(portfolio_values["cumulative_return"].iloc[-1])
latest_pnl = float(portfolio_values["daily_pnl"].iloc[-1])

st.title("Market Risk Analytics Platform")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Market Value", currency(latest_value))
col2.metric("Cumulative Return", percent(latest_return))
col3.metric("Latest Daily P&L", currency(latest_pnl))
col4.metric("Historical VaR 95%", currency(risk_metrics["historical_var_95"]))

left, right = st.columns([2, 1])
with left:
    st.plotly_chart(
        px.line(
            portfolio_values,
            x="value_date",
            y="market_value",
            title="Portfolio Market Value",
            markers=True,
        ),
        width="stretch",
    )
with right:
    asset_class = exposures[exposures["exposure_type"] == "asset_class"].sort_values("market_value", ascending=False)
    st.plotly_chart(
        px.pie(asset_class, names="exposure_name", values="market_value", title="Asset-Class Exposure"),
        width="stretch",
    )

st.subheader("Risk Snapshot")
metric_cols = st.columns(4)
metric_cols[0].metric("Sharpe Ratio", f"{risk_metrics['sharpe_ratio']:.2f}")
metric_cols[1].metric("Sortino Ratio", f"{risk_metrics['sortino_ratio']:.2f}")
metric_cols[2].metric("Expected Shortfall 95%", currency(risk_metrics["expected_shortfall_95"]))
metric_cols[3].metric("Max Drawdown", percent(risk_metrics["max_drawdown"]))
