from __future__ import annotations

from numbers import Number
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Engine, text

from src.load.db import initialize_database
from src.load.load_marts import (
    load_daily_returns,
    load_data_quality_results,
    load_exposures,
    load_portfolio_values,
    load_position_pnl,
    load_risk_metrics,
    load_staging_prices,
)
from src.load.load_raw import load_assets, load_positions, load_raw_prices


PIPELINE_TABLES = [
    "mart.monte_carlo_terminal_values",
    "mart.monte_carlo_results",
    "mart.monte_carlo_runs",
    "mart.stress_test_results",
    "mart.data_quality_results",
    "mart.exposures",
    "mart.risk_metrics",
    "mart.position_pnl",
    "mart.portfolio_values",
    "mart.daily_returns",
    "staging.stg_daily_prices",
    "raw.portfolio_positions",
    "raw.assets",
    "raw.prices",
]


def truncate_pipeline_tables(engine: Engine) -> None:
    """Remove existing rows from the project-owned PostgreSQL tables."""
    table_list = ", ".join(PIPELINE_TABLES)
    with engine.begin() as connection:
        connection.execute(text(f"TRUNCATE TABLE {table_list} RESTART IDENTITY CASCADE"))


def build_risk_metrics_frame(outputs: dict[str, Any]) -> pd.DataFrame:
    """Convert the pipeline risk metric dictionary into mart.risk_metrics rows."""
    risk_metrics = outputs.get("risk_metrics") or {}
    if not risk_metrics:
        return pd.DataFrame(
            columns=[
                "portfolio_name",
                "metric_date",
                "metric_name",
                "metric_value",
                "lookback_days",
                "confidence_level",
            ]
        )

    portfolio_values = outputs.get("portfolio_values")
    if not isinstance(portfolio_values, pd.DataFrame) or portfolio_values.empty:
        raise ValueError("portfolio_values output is required to persist risk metrics")

    portfolio_name = str(portfolio_values["portfolio_name"].iloc[-1])
    metric_date = pd.to_datetime(portfolio_values["value_date"]).max().date()
    lookback_days = max(len(portfolio_values) - 1, 0)

    rows = []
    for metric_name, metric_value in risk_metrics.items():
        if not isinstance(metric_value, Number) or pd.isna(metric_value):
            continue
        rows.append(
            {
                "portfolio_name": portfolio_name,
                "metric_date": metric_date,
                "metric_name": metric_name,
                "metric_value": float(metric_value),
                "lookback_days": lookback_days,
                "confidence_level": _confidence_level(metric_name),
            }
        )
    return pd.DataFrame(rows)


def load_pipeline_outputs(
    engine: Engine,
    outputs: dict[str, Any],
    sql_dir: str | Path = "sql",
    initialize: bool = True,
    replace_existing: bool = True,
) -> dict[str, int]:
    """
    Persist the outputs produced by src.pipeline.run_pipeline into PostgreSQL.

    When replace_existing is true, all project-owned raw, staging, and mart tables
    are truncated first so local reruns stay deterministic.
    """
    if initialize:
        initialize_database(engine, sql_dir)
    if replace_existing:
        truncate_pipeline_tables(engine)

    required_frames = [
        "raw_prices",
        "assets",
        "positions",
        "stg_prices",
        "returns",
        "portfolio_values",
        "position_pnl",
        "exposures",
        "data_quality",
    ]
    missing = [name for name in required_frames if name not in outputs or not isinstance(outputs[name], pd.DataFrame)]
    if missing:
        raise ValueError(f"Pipeline outputs are missing DataFrames required for database loading: {missing}")

    risk_metrics = build_risk_metrics_frame(outputs)

    load_raw_prices(engine, outputs["raw_prices"])
    load_assets(engine, outputs["assets"])
    load_positions(engine, outputs["positions"])
    load_staging_prices(engine, outputs["stg_prices"])
    load_daily_returns(engine, outputs["returns"])
    load_portfolio_values(engine, outputs["portfolio_values"])
    load_position_pnl(engine, outputs["position_pnl"])
    load_exposures(engine, outputs["exposures"])
    if not risk_metrics.empty:
        load_risk_metrics(engine, risk_metrics)
    load_data_quality_results(engine, outputs["data_quality"])

    counts = {name: len(outputs[name]) for name in required_frames}
    counts["risk_metrics"] = len(risk_metrics)
    return counts


def _confidence_level(metric_name: str) -> float | None:
    suffix = metric_name.rsplit("_", maxsplit=1)[-1]
    if suffix.isdigit():
        value = int(suffix)
        if 0 < value < 100:
            return value / 100
    return None
