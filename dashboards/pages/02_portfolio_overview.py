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

from dashboards.common import currency, load_selected_dashboard_data, prepare_dates, render_table_download


st.set_page_config(page_title="Portfolio Overview", layout="wide")
data = load_selected_dashboard_data()
portfolio_values = prepare_dates(data["portfolio_values"], ("value_date",))
current_positions = data["current_positions"].sort_values("position_value", ascending=False)

st.title("Portfolio Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Latest Market Value", currency(float(portfolio_values["market_value"].iloc[-1])))
col2.metric("Latest Daily P&L", currency(float(portfolio_values["daily_pnl"].iloc[-1])))
col3.metric("Holdings", f"{len(current_positions)}")

left, right = st.columns(2)
with left:
    st.plotly_chart(
        px.line(portfolio_values, x="value_date", y="market_value", title="Portfolio Value", markers=True),
        width="stretch",
    )
with right:
    st.plotly_chart(
        px.bar(portfolio_values, x="value_date", y="daily_pnl", title="Daily P&L"),
        width="stretch",
    )

st.plotly_chart(
    px.bar(
        current_positions,
        x="ticker",
        y="position_value",
        color="asset_class",
        title="Top Holdings",
        text_auto=".2s",
    ),
    width="stretch",
)
holdings_table = current_positions[
    ["ticker", "quantity", "current_price", "position_value", "weight", "asset_class", "sector"]
]
st.dataframe(
    holdings_table,
    width="stretch",
    hide_index=True,
)
render_table_download(holdings_table, "portfolio_holdings", key="portfolio_holdings_csv")
