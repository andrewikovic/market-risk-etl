from __future__ import annotations

import numpy as np
import plotly.express as px
import streamlit as st

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.name == "dashboards":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboards.common import (
    currency,
    load_selected_dashboard_data,
    percent,
    prepare_dates,
    realized_portfolio_return_series,
)
from src.transform.calculate_beta import calculate_beta
from src.transform.calculate_drawdowns import calculate_drawdowns
from src.transform.calculate_volatility import calculate_sharpe_ratio, calculate_sortino_ratio


st.set_page_config(page_title="Risk Summary", layout="wide")
data = load_selected_dashboard_data()
portfolio_values = prepare_dates(data["portfolio_values"], ("value_date",))
returns = prepare_dates(data["returns"], ("return_date",))
portfolio_returns = realized_portfolio_return_series(portfolio_values)
benchmark_returns = returns[returns["ticker"] == "SPY"][["return_date", "daily_return"]]
beta_metrics = calculate_beta(
    portfolio_returns.rename("daily_return").to_frame(),
    benchmark_returns,
    rolling_window=10,
)
drawdowns = calculate_drawdowns(portfolio_values)
drawdown_series = prepare_dates(drawdowns["drawdown_series"], ("value_date",))

annualized_volatility = float(portfolio_returns.std(ddof=1) * np.sqrt(252))
sharpe = calculate_sharpe_ratio(portfolio_returns)
sortino = calculate_sortino_ratio(portfolio_returns)

st.title("Risk Summary")
cols = st.columns(4)
cols[0].metric("Annualized Volatility", percent(annualized_volatility))
cols[1].metric("Sharpe Ratio", f"{sharpe:.2f}")
cols[2].metric("Sortino Ratio", f"{sortino:.2f}")
cols[3].metric("Max Drawdown", percent(drawdowns["max_drawdown"]))

cols = st.columns(4)
cols[0].metric("Beta to SPY", f"{beta_metrics['beta']:.2f}")
cols[1].metric("Alpha", percent(beta_metrics["alpha"]))
cols[2].metric("Tracking Error", percent(beta_metrics["tracking_error"]))
cols[3].metric("Historical VaR 95%", currency(data["risk_metrics"]["historical_var_95"]))

left, right = st.columns(2)
with left:
    st.plotly_chart(
        px.area(drawdown_series, x="value_date", y="drawdown", title="Drawdown Series"),
        width="stretch",
    )
with right:
    rolling_beta = prepare_dates(beta_metrics["rolling_beta"], ("return_date",))
    st.plotly_chart(
        px.line(rolling_beta, x="return_date", y="rolling_beta", title="Rolling Beta to SPY", markers=True),
        width="stretch",
    )
