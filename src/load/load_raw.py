from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine


def load_raw_prices(engine: Engine, prices_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load raw price rows into raw.prices."""
    prices_df.to_sql("prices", engine, schema="raw", if_exists=if_exists, index=False, method="multi")


def load_assets(engine: Engine, assets_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load asset metadata into raw.assets."""
    assets_df.to_sql("assets", engine, schema="raw", if_exists=if_exists, index=False, method="multi")


def load_positions(engine: Engine, positions_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load portfolio positions into raw.portfolio_positions."""
    columns = ["portfolio_name", "ticker", "quantity", "as_of_date", "asset_class", "sector", "currency"]
    positions_df[columns].to_sql(
        "portfolio_positions",
        engine,
        schema="raw",
        if_exists=if_exists,
        index=False,
        method="multi",
    )

