from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine


def load_staging_prices(engine: Engine, stg_prices_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load cleaned daily prices into staging.stg_daily_prices."""
    stg_prices_df.to_sql(
        "stg_daily_prices",
        engine,
        schema="staging",
        if_exists=if_exists,
        index=False,
        method="multi",
    )


def load_daily_returns(engine: Engine, returns_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load daily returns into mart.daily_returns."""
    returns_df.to_sql("daily_returns", engine, schema="mart", if_exists=if_exists, index=False, method="multi")


def load_portfolio_values(engine: Engine, values_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load portfolio value history into mart.portfolio_values."""
    values_df.to_sql("portfolio_values", engine, schema="mart", if_exists=if_exists, index=False, method="multi")


def load_position_pnl(engine: Engine, pnl_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load position-level P&L into mart.position_pnl."""
    columns = [
        "portfolio_name",
        "value_date",
        "ticker",
        "position_value",
        "daily_pnl",
        "contribution_to_pnl",
        "contribution_to_return",
        "weight",
    ]
    pnl_df[columns].to_sql("position_pnl", engine, schema="mart", if_exists=if_exists, index=False, method="multi")


def load_exposures(engine: Engine, exposures_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load exposure analytics into mart.exposures."""
    exposures_df.to_sql("exposures", engine, schema="mart", if_exists=if_exists, index=False, method="multi")


def load_data_quality_results(engine: Engine, checks_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load data quality check results into mart.data_quality_results."""
    checks_df.to_sql("data_quality_results", engine, schema="mart", if_exists=if_exists, index=False, method="multi")
