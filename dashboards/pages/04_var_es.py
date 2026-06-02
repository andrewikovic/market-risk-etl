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

from dashboards.common import currency, load_selected_dashboard_data
from src.risk.expected_shortfall import calculate_expected_shortfall
from src.risk.historical_var import calculate_historical_var
from src.risk.monte_carlo import run_correlated_monte_carlo
from src.risk.parametric_var import calculate_parametric_var


st.set_page_config(page_title="VaR and Expected Shortfall", layout="wide")
data = load_selected_dashboard_data()
portfolio_values = data["portfolio_values"]
returns = data["returns"]
weights = data["current_positions"].set_index("ticker")["weight"].to_dict()
portfolio_returns = portfolio_values["daily_return"].iloc[1:]
portfolio_value = float(portfolio_values["market_value"].iloc[-1])
confidence = st.slider("Confidence", 0.90, 0.99, 0.95, 0.01)

historical_var = calculate_historical_var(portfolio_returns, portfolio_value, confidence)
parametric_var = calculate_parametric_var(portfolio_returns, portfolio_value, confidence)
historical_es = calculate_expected_shortfall(portfolio_returns, portfolio_value, confidence)
mc = run_correlated_monte_carlo(
    returns,
    weights,
    initial_value=portfolio_value,
    horizon_days=60,
    n_simulations=1000,
    confidence_level=confidence,
    random_seed=42,
)

st.title("VaR and Expected Shortfall")
cols = st.columns(5)
cols[0].metric("Historical VaR", currency(historical_var))
cols[1].metric("Parametric VaR", currency(parametric_var))
cols[2].metric("Monte Carlo VaR", currency(mc["monte_carlo_var"]))
cols[3].metric("Historical ES", currency(historical_es))
cols[4].metric("Monte Carlo ES", currency(mc["monte_carlo_expected_shortfall"]))

loss_threshold = -historical_var / portfolio_value
hist = portfolio_returns.to_frame(name="daily_return")
hist["var_threshold"] = loss_threshold

left, right = st.columns(2)
with left:
    st.plotly_chart(
        px.histogram(hist, x="daily_return", nbins=30, title="Portfolio Return Distribution"),
        width="stretch",
    )
with right:
    x = np.sort(portfolio_returns.to_numpy())
    y = np.arange(1, len(x) + 1) / len(x)
    st.plotly_chart(
        px.line(x=x, y=y, labels={"x": "Daily Return", "y": "Cumulative Probability"}, title="Empirical Loss Threshold"),
        width="stretch",
    )
