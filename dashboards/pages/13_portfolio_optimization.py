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

from dashboards.common import currency, load_selected_dashboard_data, percent, prepare_dates


st.set_page_config(page_title="Portfolio Optimization", layout="wide")
data = load_selected_dashboard_data()
frontier = prepare_dates(data.get("efficient_frontier", None), ("run_date",)) if data.get("efficient_frontier") is not None else None
optimized = (
    prepare_dates(data.get("optimized_portfolio", None), ("run_date",))
    if data.get("optimized_portfolio") is not None
    else None
)
trades = prepare_dates(data.get("rebalancing_trades", None), ("run_date",)) if data.get("rebalancing_trades") is not None else None

st.title("Portfolio Optimization")
if frontier is None or frontier.empty or optimized is None or optimized.empty:
    st.info("No optimization rows are available.")
    st.stop()

summary = optimized.iloc[0]
cols = st.columns(4)
cols[0].metric("Expected Return", percent(float(summary["expected_return"])))
cols[1].metric("Volatility", percent(float(summary["volatility"])))
cols[2].metric("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}")
cols[3].metric("Weight Sum", f"{summary['weight_sum']:.4f}")

left, right = st.columns(2)
with left:
    st.plotly_chart(
        px.line(
            frontier.sort_values("volatility"),
            x="volatility",
            y="expected_return",
            markers=True,
            title="Efficient Frontier",
            labels={"volatility": "Volatility", "expected_return": "Expected Return"},
        ),
        width="stretch",
    )
with right:
    st.plotly_chart(
        px.bar(
            optimized.sort_values("target_weight", ascending=False),
            x="ticker",
            y="target_weight",
            title="Optimized Target Weights",
            labels={"target_weight": "Target Weight"},
            text_auto=".1%",
        ),
        width="stretch",
    )

if trades is not None and not trades.empty:
    st.subheader("Rebalancing")
    st.plotly_chart(
        px.bar(
            trades.sort_values("trade_value"),
            x="ticker",
            y="trade_value",
            title="Trade Value",
            labels={"trade_value": "Trade Value"},
        ),
        width="stretch",
    )
    st.dataframe(
        trades[
            [
                "ticker",
                "current_weight",
                "target_weight",
                "weight_change",
                "trade_value",
                "price",
                "quantity_change",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "current_weight": st.column_config.NumberColumn(format="%.2%"),
            "target_weight": st.column_config.NumberColumn(format="%.2%"),
            "weight_change": st.column_config.NumberColumn(format="%.2%"),
            "trade_value": st.column_config.NumberColumn(format="$%.0f"),
            "price": st.column_config.NumberColumn(format="$%.2f"),
            "quantity_change": st.column_config.NumberColumn(format="%.2f"),
        },
    )

st.subheader("Constraints")
st.dataframe(
    optimized[
        [
            "ticker",
            "target_weight",
            "full_investment",
            "long_only",
            "min_weight_satisfied",
            "max_weight_satisfied",
        ]
    ],
    width="stretch",
    hide_index=True,
    column_config={"target_weight": st.column_config.NumberColumn(format="%.2%")},
)
