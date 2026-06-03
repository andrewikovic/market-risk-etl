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

from dashboards.common import load_selected_dashboard_data, percent, prepare_dates


st.set_page_config(page_title="Factor Model", layout="wide")
data = load_selected_dashboard_data()
factor_exposures = (
    prepare_dates(data.get("factor_exposures", None), ("metric_date",))
    if data.get("factor_exposures") is not None
    else None
)

st.title("Factor Model")
if factor_exposures is None or factor_exposures.empty:
    st.info("No factor exposure rows are available.")
    st.stop()

asset_exposures = factor_exposures[factor_exposures["exposure_level"] == "asset"].copy()
portfolio_exposures = factor_exposures[factor_exposures["exposure_level"] == "portfolio"].copy()

beta_matrix = asset_exposures.pivot(index="ticker", columns="factor", values="beta")
left, right = st.columns([2, 1])
with left:
    st.plotly_chart(
        px.imshow(
            beta_matrix,
            text_auto=".2f",
            aspect="auto",
            title="Asset Factor Betas",
            labels={"x": "Factor", "y": "Ticker", "color": "Beta"},
        ),
        width="stretch",
    )
with right:
    if not portfolio_exposures.empty:
        st.plotly_chart(
            px.bar(
                portfolio_exposures.sort_values("beta", ascending=False),
                x="factor",
                y="beta",
                title="Portfolio Factor Exposure",
            ),
            width="stretch",
        )

summary = asset_exposures.drop_duplicates("ticker")[
    ["ticker", "alpha", "residual_volatility", "idiosyncratic_variance", "r_squared", "observations"]
].sort_values("ticker")
cols = st.columns(3)
cols[0].metric("Assets", f"{summary['ticker'].nunique()}")
cols[1].metric("Factors", f"{asset_exposures['factor'].nunique()}")
cols[2].metric("Median R-squared", percent(float(summary["r_squared"].median())))

st.dataframe(
    asset_exposures[
        [
            "ticker",
            "factor",
            "beta",
            "alpha",
            "residual_volatility",
            "idiosyncratic_variance",
            "r_squared",
            "observations",
        ]
    ].sort_values(["ticker", "factor"]),
    width="stretch",
    hide_index=True,
    column_config={
        "beta": st.column_config.NumberColumn(format="%.3f"),
        "alpha": st.column_config.NumberColumn(format="%.5f"),
        "residual_volatility": st.column_config.NumberColumn(format="%.2%"),
        "idiosyncratic_variance": st.column_config.NumberColumn(format="%.8f"),
        "r_squared": st.column_config.NumberColumn(format="%.2%"),
    },
)
