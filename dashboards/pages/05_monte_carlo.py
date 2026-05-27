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

from dashboards.common import currency, load_dashboard_data
from src.risk.monte_carlo import run_correlated_monte_carlo


st.set_page_config(page_title="Monte Carlo Simulation", layout="wide")
data = load_dashboard_data()
returns = data["returns"]
weights = data["current_positions"].set_index("ticker")["weight"].to_dict()
initial_value = float(data["portfolio_values"]["market_value"].iloc[-1])

st.title("Monte Carlo Simulation")
col1, col2, col3 = st.columns(3)
horizon = col1.slider("Horizon Days", 20, 252, 60, 10)
simulations = col2.slider("Simulations", 250, 5000, 1000, 250)
confidence = col3.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)

mc = run_correlated_monte_carlo(
    returns,
    weights,
    initial_value=initial_value,
    horizon_days=horizon,
    n_simulations=simulations,
    confidence_level=confidence,
    random_seed=42,
)

cols = st.columns(5)
cols[0].metric("Monte Carlo VaR", currency(mc["monte_carlo_var"]))
cols[1].metric("Expected Shortfall", currency(mc["monte_carlo_expected_shortfall"]))
cols[2].metric("Probability of Loss", f"{mc['probability_of_loss']:.2%}")
cols[3].metric("Loss > 10%", f"{mc['probability_loss_gt_10']:.2%}")
cols[4].metric("Median Terminal Value", currency(mc["median_terminal_value"]))

paths = mc["paths"].iloc[:, : min(100, simulations)].reset_index(names="day")
paths_long = paths.melt(id_vars="day", var_name="simulation", value_name="portfolio_value")
left, right = st.columns(2)
with left:
    st.plotly_chart(
        px.line(paths_long, x="day", y="portfolio_value", color="simulation", title="Simulated Portfolio Paths"),
        width="stretch",
    )
with right:
    st.plotly_chart(
        px.histogram(mc["terminal_values"], x="terminal_value", nbins=50, title="Terminal Value Distribution"),
        width="stretch",
    )

st.plotly_chart(
    px.imshow(
        mc["simulated_correlation_matrix"],
        text_auto=".2f",
        aspect="auto",
        title="Simulated Asset Correlations",
    ),
    width="stretch",
)
