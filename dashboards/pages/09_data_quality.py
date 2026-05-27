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


st.set_page_config(page_title="Data Quality", layout="wide")
data = load_dashboard_data()
checks = prepare_dates(data["data_quality"], ("check_date", "run_timestamp"))

st.title("Data Quality")

status_counts = checks.groupby("status", as_index=False).size()
severity_counts = checks.groupby("severity", as_index=False).size()

cols = st.columns(4)
for idx, status in enumerate(["PASS", "WARN", "FAIL"]):
    count = int(status_counts.loc[status_counts["status"] == status, "size"].sum())
    cols[idx].metric(status, count)
cols[3].metric("Total Checks", len(checks))

left, right = st.columns(2)
with left:
    st.plotly_chart(px.bar(status_counts, x="status", y="size", title="Checks by Status"), width="stretch")
with right:
    st.plotly_chart(px.bar(severity_counts, x="severity", y="size", title="Checks by Severity"), width="stretch")

failed_or_warn = checks[checks["status"].isin(["WARN", "FAIL"])]
st.dataframe(failed_or_warn if not failed_or_warn.empty else checks, width="stretch", hide_index=True)
