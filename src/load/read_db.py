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
        SELECT portfolio_name, ticker, quantity, as_of_date, asset_class, sector, currency, base_currency
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
               contribution_to_pnl, contribution_to_return, weight,
               currency, fx_rate_to_base, base_currency
        FROM mart.position_pnl
        ORDER BY portfolio_name, value_date, ticker
        """,
        parse_dates=["value_date"],
        numeric_columns=[
            "position_value",
            "daily_pnl",
            "contribution_to_pnl",
            "contribution_to_return",
            "weight",
            "fx_rate_to_base",
        ],
    )
    exposures = _read(
        engine,
        """
        SELECT portfolio_name, exposure_date, exposure_type, exposure_name, market_value, weight, base_currency
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
    var_backtest = _read(
        engine,
        """
        SELECT portfolio_name, backtest_date AS date, confidence_level, var_estimate,
               realized_pnl, breach, breach_severity, total_observations,
               number_of_exceptions, expected_exceptions, exception_ratio,
               kupiec_statistic, p_value, pass_fail
        FROM mart.var_backtest_exceptions
        ORDER BY portfolio_name, confidence_level, backtest_date
        """,
        parse_dates=["date"],
        numeric_columns=[
            "confidence_level",
            "var_estimate",
            "realized_pnl",
            "breach_severity",
            "total_observations",
            "number_of_exceptions",
            "expected_exceptions",
            "exception_ratio",
            "kupiec_statistic",
            "p_value",
        ],
    )
    var_contributions = _read(
        engine,
        """
        SELECT portfolio_name, metric_date, confidence_level, ticker, weight, exposure,
               mean_return, volatility, marginal_var, component_var,
               percent_contribution, portfolio_var, contribution_reconciliation_error
        FROM mart.var_contributions
        ORDER BY portfolio_name, metric_date, confidence_level, ticker
        """,
        parse_dates=["metric_date"],
        numeric_columns=[
            "confidence_level",
            "weight",
            "exposure",
            "mean_return",
            "volatility",
            "marginal_var",
            "component_var",
            "percent_contribution",
            "portfolio_var",
            "contribution_reconciliation_error",
        ],
    )
    risk_contributions = _read(
        engine,
        """
        SELECT portfolio_name, metric_date, confidence_level, ticker, weight, exposure,
               asset_volatility, marginal_volatility, component_volatility,
               volatility_contribution_amount, marginal_var, component_var,
               volatility_percent_contribution, var_percent_contribution,
               portfolio_volatility, portfolio_var, volatility_reconciliation_error,
               var_reconciliation_error
        FROM mart.risk_contributions
        ORDER BY portfolio_name, metric_date, confidence_level, ticker
        """,
        parse_dates=["metric_date"],
        numeric_columns=[
            "confidence_level",
            "weight",
            "exposure",
            "asset_volatility",
            "marginal_volatility",
            "component_volatility",
            "volatility_contribution_amount",
            "marginal_var",
            "component_var",
            "volatility_percent_contribution",
            "var_percent_contribution",
            "portfolio_volatility",
            "portfolio_var",
            "volatility_reconciliation_error",
            "var_reconciliation_error",
        ],
    )
    factor_exposures = _read(
        engine,
        """
        SELECT portfolio_name, metric_date, exposure_level, ticker, factor, beta,
               alpha, residual_volatility, idiosyncratic_variance, r_squared, observations
        FROM mart.factor_exposures
        ORDER BY portfolio_name, metric_date, exposure_level, ticker, factor
        """,
        parse_dates=["metric_date"],
        numeric_columns=[
            "beta",
            "alpha",
            "residual_volatility",
            "idiosyncratic_variance",
            "r_squared",
            "observations",
        ],
    )
    efficient_frontier_rows = _read(
        engine,
        """
        SELECT portfolio_name, run_date, point_number, ticker, target_return,
               expected_return, volatility, sharpe_ratio, weight
        FROM mart.efficient_frontier
        ORDER BY portfolio_name, run_date, point_number, ticker
        """,
        parse_dates=["run_date"],
        numeric_columns=[
            "point_number",
            "target_return",
            "expected_return",
            "volatility",
            "sharpe_ratio",
            "weight",
        ],
    )
    optimized_portfolio = _read(
        engine,
        """
        SELECT portfolio_name, run_date, ticker, target_weight, expected_return,
               volatility, sharpe_ratio, weight_sum, full_investment, long_only,
               min_weight_satisfied, max_weight_satisfied, target_return_error,
               target_volatility_error
        FROM mart.optimized_portfolio
        ORDER BY portfolio_name, run_date, ticker
        """,
        parse_dates=["run_date"],
        numeric_columns=[
            "target_weight",
            "expected_return",
            "volatility",
            "sharpe_ratio",
            "weight_sum",
            "target_return_error",
            "target_volatility_error",
        ],
    )
    rebalancing_trades = _read(
        engine,
        """
        SELECT portfolio_name, run_date, ticker, current_weight, target_weight,
               weight_change, trade_value, price, quantity_change
        FROM mart.rebalancing_trades
        ORDER BY portfolio_name, run_date, ticker
        """,
        parse_dates=["run_date"],
        numeric_columns=[
            "current_weight",
            "target_weight",
            "weight_change",
            "trade_value",
            "price",
            "quantity_change",
        ],
    )

    current_positions = _current_positions_from_position_pnl(position_pnl, positions, assets)
    if current_positions.empty:
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
        "var_backtest": var_backtest,
        "var_contributions": var_contributions,
        "risk_contributions": risk_contributions,
        "factor_exposures": factor_exposures,
        "efficient_frontier": _frontier_from_rows(efficient_frontier_rows),
        "optimized_portfolio": optimized_portfolio,
        "rebalancing_trades": rebalancing_trades,
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


def _frontier_from_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    grouped_rows = []
    group_cols = ["portfolio_name", "run_date", "point_number", "target_return", "expected_return", "volatility", "sharpe_ratio"]
    for keys, group in rows.groupby(group_cols, dropna=False, sort=True):
        row = dict(zip(group_cols, keys, strict=False))
        row["weights"] = dict(zip(group["ticker"], group["weight"], strict=False))
        for ticker, weight in row["weights"].items():
            row[f"weight_{ticker}"] = weight
        grouped_rows.append(row)
    return pd.DataFrame(grouped_rows)


def _current_positions_from_position_pnl(
    position_pnl: pd.DataFrame,
    positions: pd.DataFrame,
    assets: pd.DataFrame,
) -> pd.DataFrame:
    if position_pnl.empty:
        return pd.DataFrame()
    latest_date = pd.to_datetime(position_pnl["value_date"]).max()
    latest = position_pnl[pd.to_datetime(position_pnl["value_date"]) == latest_date].copy()
    position_cols = ["ticker", "quantity", "asset_class", "sector", "currency", "base_currency"]
    position_meta = positions.sort_values("as_of_date").drop_duplicates("ticker", keep="last")
    latest = latest.merge(position_meta[position_cols], on="ticker", how="left", suffixes=("", "_position"))
    asset_cols = ["ticker", "asset_name", "asset_class", "sector", "currency", "country"]
    latest = latest.merge(assets[asset_cols], on="ticker", how="left", suffixes=("", "_asset"))
    for column in ["asset_class", "sector", "currency"]:
        position_col = f"{column}_position"
        asset_col = f"{column}_asset"
        if position_col in latest.columns:
            latest[column] = latest[column].combine_first(latest[position_col]) if column in latest.columns else latest[position_col]
        if asset_col in latest.columns:
            latest[column] = latest[column].combine_first(latest[asset_col])
    if "country_asset" in latest.columns:
        latest["country"] = latest["country_asset"]
    latest["current_price"] = latest["position_value"] / latest["quantity"].replace(0, pd.NA)
    return latest[
        [
            "portfolio_name",
            "ticker",
            "quantity",
            "current_price",
            "position_value",
            "weight",
            "asset_class",
            "sector",
            "currency",
            "base_currency",
            "country",
        ]
    ].reset_index(drop=True)
