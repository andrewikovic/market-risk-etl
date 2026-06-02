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

from dashboards.common import load_selected_dashboard_data, prepare_dates
from src.transform.calculate_returns import calculate_cumulative_returns
from src.transform.calculate_volatility import calculate_rolling_volatility


st.set_page_config(page_title="Market Overview", layout="wide")
data = load_selected_dashboard_data()
prices = prepare_dates(data["stg_prices"], ("price_date",))
returns = prepare_dates(data["returns"], ("return_date",))

st.title("Market Overview")

tickers = sorted(prices["ticker"].unique())
selected = st.multiselect("Tickers", tickers, default=tickers)
filtered_prices = prices[prices["ticker"].isin(selected)]
filtered_returns = returns[returns["ticker"].isin(selected)]

st.plotly_chart(
    px.line(filtered_prices, x="price_date", y="adjusted_close", color="ticker", title="Adjusted Close Prices"),
    width="stretch",
)

left, right = st.columns(2)
with left:
    st.plotly_chart(
        px.line(filtered_returns, x="return_date", y="daily_return", color="ticker", title="Daily Returns"),
        width="stretch",
    )
with right:
    cumulative = prepare_dates(calculate_cumulative_returns(filtered_returns), ("return_date",))
    st.plotly_chart(
        px.line(cumulative, x="return_date", y="cumulative_return", color="ticker", title="Cumulative Returns"),
        width="stretch",
    )

vol = prepare_dates(calculate_rolling_volatility(returns, windows=(5, 10, 20)), ("return_date",))
vol = vol[vol["ticker"].isin(selected)]
st.plotly_chart(
    px.line(
        vol,
        x="return_date",
        y="rolling_volatility",
        color="ticker",
        line_dash="window",
        title="Rolling Annualized Volatility",
    ),
    width="stretch",
)
