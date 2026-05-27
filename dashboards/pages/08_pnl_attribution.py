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

from dashboards.common import load_dashboard_data, prepare_dates


st.set_page_config(page_title="P&L Attribution", layout="wide")
data = load_dashboard_data()
portfolio_values = prepare_dates(data["portfolio_values"], ("value_date",))
position_pnl = prepare_dates(data["position_pnl"], ("value_date",))
latest_date = position_pnl["value_date"].max()
latest = position_pnl[position_pnl["value_date"] == latest_date].sort_values("daily_pnl", ascending=False)

st.title("P&L Attribution")

left, right = st.columns(2)
with left:
    st.plotly_chart(
        px.bar(portfolio_values, x="value_date", y="daily_pnl", title="Daily Portfolio P&L"),
        width="stretch",
    )
with right:
    st.plotly_chart(
        px.bar(latest, x="ticker", y="daily_pnl", color="asset_class", title="Latest Position P&L"),
        width="stretch",
    )

sector_pnl = latest.groupby("sector", as_index=False)["daily_pnl"].sum()
asset_class_pnl = latest.groupby("asset_class", as_index=False)["daily_pnl"].sum()
cols = st.columns(2)
cols[0].plotly_chart(px.bar(sector_pnl, x="sector", y="daily_pnl", title="P&L by Sector"), width="stretch")
cols[1].plotly_chart(
    px.bar(asset_class_pnl, x="asset_class", y="daily_pnl", title="P&L by Asset Class"),
    width="stretch",
)

st.dataframe(
    latest[["value_date", "ticker", "daily_pnl", "contribution_to_return", "weight", "asset_class", "sector"]],
    width="stretch",
    hide_index=True,
)
