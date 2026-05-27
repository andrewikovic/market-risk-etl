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

from dashboards.common import load_dashboard_data


st.set_page_config(page_title="Exposure Analytics", layout="wide")
data = load_dashboard_data()
exposures = data["exposures"]

st.title("Exposure Analytics")

exposure_type = st.selectbox(
    "Exposure Type",
    ["ticker", "sector", "asset_class", "currency", "country"],
    index=1,
)
filtered = exposures[exposures["exposure_type"] == exposure_type].sort_values("market_value", ascending=False)

left, right = st.columns(2)
with left:
    st.plotly_chart(
        px.bar(filtered, x="exposure_name", y="market_value", title=f"{exposure_type.title()} Exposure", text_auto=".2s"),
        width="stretch",
    )
with right:
    st.plotly_chart(
        px.pie(filtered, names="exposure_name", values="market_value", title=f"{exposure_type.title()} Weights"),
        width="stretch",
    )

st.subheader("Concentration Risk")
ticker = exposures[exposures["exposure_type"] == "ticker"].sort_values("weight", ascending=False)
st.plotly_chart(
    px.bar(ticker, x="exposure_name", y="weight", title="Ticker Weights", text_auto=".1%"),
    width="stretch",
)
st.dataframe(exposures, width="stretch", hide_index=True)
