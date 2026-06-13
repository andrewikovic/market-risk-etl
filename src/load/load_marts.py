from __future__ import annotations

import pandas as pd
from sqlalchemy import Date, DateTime, Engine

from src.load.to_sql import write_frame_to_sql


def load_staging_prices(engine: Engine, stg_prices_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load cleaned daily prices into staging.stg_daily_prices."""
    write_frame_to_sql(stg_prices_df, "stg_daily_prices", engine, schema="staging", if_exists=if_exists)


def load_daily_returns(engine: Engine, returns_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load daily returns into mart.daily_returns."""
    write_frame_to_sql(returns_df, "daily_returns", engine, schema="mart", if_exists=if_exists)


def load_portfolio_values(engine: Engine, values_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load portfolio value history into mart.portfolio_values."""
    write_frame_to_sql(values_df, "portfolio_values", engine, schema="mart", if_exists=if_exists)


def load_position_pnl(engine: Engine, pnl_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load position-level P&L into mart.position_pnl."""
    columns = [
        "portfolio_name",
        "value_date",
        "ticker",
        "asset_class",
        "sector",
        "position_value",
        "daily_pnl",
        "contribution_to_pnl",
        "contribution_to_return",
        "weight",
        "currency",
        "fx_rate_to_base",
        "base_currency",
    ]
    _load_required_frame(engine, pnl_df, "position_pnl", columns, if_exists)


def load_exposures(engine: Engine, exposures_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load exposure analytics into mart.exposures."""
    columns = [
        "portfolio_name",
        "exposure_date",
        "exposure_type",
        "exposure_name",
        "market_value",
        "weight",
        "base_currency",
    ]
    _load_required_frame(engine, exposures_df, "exposures", columns, if_exists)


def load_risk_metrics(engine: Engine, risk_metrics_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load portfolio-level risk metric rows into mart.risk_metrics."""
    columns = [
        "portfolio_name",
        "metric_date",
        "metric_name",
        "metric_value",
        "lookback_days",
        "confidence_level",
    ]
    write_frame_to_sql(risk_metrics_df[columns], "risk_metrics", engine, schema="mart", if_exists=if_exists)


def load_data_quality_results(engine: Engine, checks_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load data quality check results into mart.data_quality_results."""
    write_frame_to_sql(
        checks_df,
        "data_quality_results",
        engine,
        schema="mart",
        if_exists=if_exists,
        dtype={"check_date": Date(), "run_timestamp": DateTime(timezone=True)},
    )


def load_var_backtest(engine: Engine, var_backtest_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load VaR backtesting exception rows into mart.var_backtest_exceptions."""
    if var_backtest_df.empty:
        return
    data = var_backtest_df.rename(columns={"date": "backtest_date"}).copy()
    columns = [
        "portfolio_name",
        "backtest_date",
        "confidence_level",
        "var_estimate",
        "realized_pnl",
        "breach",
        "breach_severity",
        "total_observations",
        "number_of_exceptions",
        "expected_exceptions",
        "exception_ratio",
        "kupiec_statistic",
        "p_value",
        "pass_fail",
    ]
    write_frame_to_sql(data[columns], "var_backtest_exceptions", engine, schema="mart", if_exists=if_exists)


def load_var_contributions(engine: Engine, var_contributions_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load asset-level VaR contribution rows into mart.var_contributions."""
    columns = [
        "portfolio_name",
        "metric_date",
        "confidence_level",
        "ticker",
        "weight",
        "exposure",
        "mean_return",
        "volatility",
        "marginal_var",
        "component_var",
        "percent_contribution",
        "portfolio_var",
        "contribution_reconciliation_error",
    ]
    _load_optional_frame(engine, var_contributions_df, "var_contributions", columns, if_exists)


def load_risk_contributions(engine: Engine, risk_contributions_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load asset-level volatility and VaR contribution rows into mart.risk_contributions."""
    columns = [
        "portfolio_name",
        "metric_date",
        "confidence_level",
        "ticker",
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
    ]
    _load_optional_frame(engine, risk_contributions_df, "risk_contributions", columns, if_exists)


def load_factor_exposures(engine: Engine, factor_exposures_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load factor exposure rows into mart.factor_exposures."""
    columns = [
        "portfolio_name",
        "metric_date",
        "exposure_level",
        "ticker",
        "factor",
        "beta",
        "alpha",
        "residual_volatility",
        "idiosyncratic_variance",
        "r_squared",
        "observations",
    ]
    _load_optional_frame(engine, factor_exposures_df, "factor_exposures", columns, if_exists)


def load_efficient_frontier(engine: Engine, efficient_frontier_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load efficient-frontier point weights into mart.efficient_frontier."""
    if efficient_frontier_df.empty:
        return
    rows = []
    for _, row in efficient_frontier_df.iterrows():
        weights = row.get("weights", {}) or {}
        for ticker, weight in weights.items():
            rows.append(
                {
                    "portfolio_name": row["portfolio_name"],
                    "run_date": row["run_date"],
                    "point_number": row["point_number"],
                    "ticker": ticker,
                    "target_return": row["target_return"],
                    "expected_return": row["expected_return"],
                    "volatility": row["volatility"],
                    "sharpe_ratio": row["sharpe_ratio"],
                    "weight": weight,
                }
            )
    write_frame_to_sql(pd.DataFrame(rows), "efficient_frontier", engine, schema="mart", if_exists=if_exists)


def load_optimized_portfolio(engine: Engine, optimized_portfolio_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load optimized target weights into mart.optimized_portfolio."""
    columns = [
        "portfolio_name",
        "run_date",
        "ticker",
        "target_weight",
        "expected_return",
        "volatility",
        "sharpe_ratio",
        "weight_sum",
        "full_investment",
        "long_only",
        "min_weight_satisfied",
        "max_weight_satisfied",
        "target_return_error",
        "target_volatility_error",
    ]
    _load_optional_frame(engine, optimized_portfolio_df, "optimized_portfolio", columns, if_exists)


def load_rebalancing_trades(engine: Engine, rebalancing_trades_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load optimizer-driven rebalance trades into mart.rebalancing_trades."""
    columns = [
        "portfolio_name",
        "run_date",
        "ticker",
        "current_weight",
        "target_weight",
        "weight_change",
        "trade_value",
        "price",
        "quantity_change",
    ]
    _load_optional_frame(engine, rebalancing_trades_df, "rebalancing_trades", columns, if_exists)


def _select_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = data.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame[columns]


def _load_required_frame(
    engine: Engine,
    frame: pd.DataFrame,
    table_name: str,
    columns: list[str],
    if_exists: str,
) -> None:
    write_frame_to_sql(_select_columns(frame, columns), table_name, engine, schema="mart", if_exists=if_exists)


def _load_optional_frame(
    engine: Engine,
    frame: pd.DataFrame,
    table_name: str,
    columns: list[str],
    if_exists: str,
) -> None:
    if frame.empty:
        return
    _load_required_frame(engine, frame, table_name, columns, if_exists)
