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


st.set_page_config(page_title="Risk Contributions", layout="wide")
data = load_selected_dashboard_data()
var_contributions = prepare_dates(data.get("var_contributions", None), ("metric_date",)) if data.get("var_contributions") is not None else None
risk_contributions = (
    prepare_dates(data.get("risk_contributions", None), ("metric_date",))
    if data.get("risk_contributions") is not None
    else None
)

st.title("Risk Contributions")

if var_contributions is None or var_contributions.empty:
    st.info("No VaR contribution rows are available.")
    st.stop()

confidence_levels = sorted(var_contributions["confidence_level"].dropna().unique())
confidence = st.selectbox(
    "Confidence",
    confidence_levels,
    index=confidence_levels.index(0.95) if 0.95 in confidence_levels else 0,
    format_func=lambda value: f"{value:.1%}",
)
selected_var = var_contributions[var_contributions["confidence_level"] == confidence].sort_values(
    "component_var",
    ascending=False,
)

cols = st.columns(3)
cols[0].metric("Portfolio VaR", currency(float(selected_var["portfolio_var"].iloc[0])))
cols[1].metric("Contribution Sum", currency(float(selected_var["component_var"].sum())))
cols[2].metric("Reconciliation", f"{selected_var['contribution_reconciliation_error'].iloc[0]:.6f}")

left, right = st.columns(2)
with left:
    st.plotly_chart(
        px.bar(
            selected_var,
            x="ticker",
            y="component_var",
            title="Component VaR",
            labels={"component_var": "Component VaR"},
        ),
        width="stretch",
    )
with right:
    st.plotly_chart(
        px.pie(
            selected_var,
            names="ticker",
            values="percent_contribution",
            title="VaR Contribution %",
        ),
        width="stretch",
    )

st.dataframe(
    selected_var[
        [
            "ticker",
            "weight",
            "exposure",
            "volatility",
            "marginal_var",
            "component_var",
            "percent_contribution",
        ]
    ],
    width="stretch",
    hide_index=True,
    column_config={
        "weight": st.column_config.NumberColumn(format="%.2%"),
        "exposure": st.column_config.NumberColumn(format="$%.0f"),
        "volatility": st.column_config.NumberColumn(format="%.4f"),
        "marginal_var": st.column_config.NumberColumn(format="$%.0f"),
        "component_var": st.column_config.NumberColumn(format="$%.0f"),
        "percent_contribution": st.column_config.NumberColumn(format="%.2%"),
    },
)

if risk_contributions is not None and not risk_contributions.empty:
    st.subheader("Volatility Contribution")
    risk_selected = risk_contributions.sort_values("var_percent_contribution", ascending=False)
    cols = st.columns(3)
    cols[0].metric("Portfolio Volatility", percent(float(risk_selected["portfolio_volatility"].iloc[0])))
    cols[1].metric("Portfolio VaR", currency(float(risk_selected["portfolio_var"].iloc[0])))
    cols[2].metric("VaR Reconciliation", f"{risk_selected['var_reconciliation_error'].iloc[0]:.6f}")

    st.plotly_chart(
        px.bar(
            risk_selected,
            x="ticker",
            y=["volatility_percent_contribution", "var_percent_contribution"],
            barmode="group",
            title="Risk Contribution %",
        ),
        width="stretch",
    )
