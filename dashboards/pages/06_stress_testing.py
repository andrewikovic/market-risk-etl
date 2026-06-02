from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.name == "dashboards":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboards.common import currency, load_dashboard_scenarios, load_selected_dashboard_data
from src.risk.stress_testing import run_stress_test


st.set_page_config(page_title="Stress Testing", layout="wide")
data = load_selected_dashboard_data()
scenarios = load_dashboard_scenarios()["scenarios"]
positions = data["current_positions"]

st.title("Stress Testing")
scenario_names = [scenario["name"] for scenario in scenarios]
selected_name = st.selectbox("Scenario", scenario_names)
selected = next(scenario for scenario in scenarios if scenario["name"] == selected_name)

custom_enabled = st.toggle("Custom shocks", value=False)
if custom_enabled:
    asset_classes = sorted(positions["asset_class"].unique())
    shocks = {}
    cols = st.columns(len(asset_classes))
    for idx, asset_class in enumerate(asset_classes):
        shocks[asset_class] = cols[idx].number_input(asset_class, value=0.0, min_value=-1.0, max_value=1.0, step=0.01)
    selected = {"name": "Custom Scenario", "shocks": shocks}

result = run_stress_test(positions, [selected])
summary = result["scenario_results"].iloc[0]
position_results = result["position_results"].sort_values("stress_loss", ascending=False)

cols = st.columns(3)
cols[0].metric("Current Value", currency(summary["current_portfolio_value"]))
cols[1].metric("Shocked Value", currency(summary["shocked_portfolio_value"]))
cols[2].metric("Stress Loss", currency(summary["stress_loss"]))

left, right = st.columns(2)
with left:
    st.plotly_chart(
        px.bar(position_results, x="ticker", y="stress_loss", color="asset_class", title="Stress Loss by Position"),
        width="stretch",
    )
with right:
    sector_loss = position_results.groupby("sector", as_index=False)["stress_loss"].sum()
    st.plotly_chart(
        px.bar(sector_loss, x="sector", y="stress_loss", title="Stress Loss by Sector"),
        width="stretch",
    )

st.dataframe(
    position_results[
        ["scenario_name", "ticker", "asset_class", "sector", "position_value", "shock", "shocked_position_value", "stress_loss"]
    ],
    width="stretch",
    hide_index=True,
)
