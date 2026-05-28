from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import Engine

from src.transform.calculate_drawdowns import calculate_drawdowns
from src.transform.calculate_exposures import calculate_current_positions


def read_pipeline_outputs(engine: Engine) -> dict[str, Any]:
    """Read persisted pipeline tables back into the dashboard output shape."""
    raw_prices = _read(
        engine,
        """
        SELECT ticker, price_date, open_price, high_price, low_price, close_price,
               adjusted_close, volume, source, ingested_at
        FROM raw.prices
        ORDER BY ticker, price_date
        """,
        parse_dates=["price_date", "ingested_at"],
        numeric_columns=["open_price", "high_price", "low_price", "close_price", "adjusted_close", "volume"],
    )
    assets = _read(
        engine,
        """
        SELECT ticker, asset_name, asset_class, sector, currency, country, benchmark_ticker
        FROM raw.assets
        ORDER BY ticker
        """,
    )
    positions = _read(
        engine,
        """
        SELECT portfolio_name, ticker, quantity, as_of_date, asset_class, sector, currency
        FROM raw.portfolio_positions
        ORDER BY portfolio_name, ticker, as_of_date
        """,
        parse_dates=["as_of_date"],
        numeric_columns=["quantity"],
    )
    stg_prices = _read(
        engine,
        """
        SELECT ticker, price_date, adjusted_close, volume, is_stale, is_missing
        FROM staging.stg_daily_prices
        ORDER BY ticker, price_date
        """,
        parse_dates=["price_date"],
        numeric_columns=["adjusted_close", "volume"],
    )
    returns = _read(
        engine,
        """
        SELECT ticker, return_date, daily_return, log_return
        FROM mart.daily_returns
        ORDER BY ticker, return_date
        """,
        parse_dates=["return_date"],
        numeric_columns=["daily_return", "log_return"],
    )
    portfolio_values = _read(
        engine,
        """
        SELECT portfolio_name, value_date, market_value, daily_pnl, daily_return, cumulative_return
        FROM mart.portfolio_values
        ORDER BY portfolio_name, value_date
        """,
        parse_dates=["value_date"],
        numeric_columns=["market_value", "daily_pnl", "daily_return", "cumulative_return"],
    )
    position_pnl = _read(
        engine,
        """
        SELECT portfolio_name, value_date, ticker, position_value, daily_pnl,
               contribution_to_pnl, contribution_to_return, weight
        FROM mart.position_pnl
        ORDER BY portfolio_name, value_date, ticker
        """,
        parse_dates=["value_date"],
        numeric_columns=["position_value", "daily_pnl", "contribution_to_pnl", "contribution_to_return", "weight"],
    )
    exposures = _read(
        engine,
        """
        SELECT portfolio_name, exposure_date, exposure_type, exposure_name, market_value, weight
        FROM mart.exposures
        ORDER BY portfolio_name, exposure_date, exposure_type, exposure_name
        """,
        parse_dates=["exposure_date"],
        numeric_columns=["market_value", "weight"],
    )
    data_quality = _read(
        engine,
        """
        SELECT check_name, status, severity, ticker, check_date, message, run_timestamp
        FROM mart.data_quality_results
        ORDER BY run_timestamp, check_name, ticker
        """,
        parse_dates=["check_date", "run_timestamp"],
    )
    risk_metrics_frame = _read(
        engine,
        """
        SELECT portfolio_name, metric_date, metric_name, metric_value, lookback_days, confidence_level
        FROM mart.risk_metrics
        WHERE metric_date = (SELECT MAX(metric_date) FROM mart.risk_metrics)
        ORDER BY metric_name
        """,
        parse_dates=["metric_date"],
        numeric_columns=["metric_value", "lookback_days", "confidence_level"],
    )

    current_positions = calculate_current_positions(positions, stg_prices, assets)
    return {
        "assets": assets,
        "raw_prices": raw_prices,
        "stg_prices": stg_prices,
        "returns": returns,
        "positions": positions,
        "current_positions": current_positions,
        "portfolio_values": portfolio_values,
        "position_pnl": position_pnl,
        "exposures": exposures,
        "data_quality": data_quality,
        "risk_metrics": dict(zip(risk_metrics_frame["metric_name"], risk_metrics_frame["metric_value"], strict=False)),
        "risk_metrics_frame": risk_metrics_frame,
        "drawdowns": calculate_drawdowns(portfolio_values),
    }


def _read(
    engine: Engine,
    query: str,
    parse_dates: list[str] | None = None,
    numeric_columns: list[str] | None = None,
) -> pd.DataFrame:
    data = pd.read_sql_query(query, engine, parse_dates=parse_dates)
    for column in numeric_columns or []:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data
