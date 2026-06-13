from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine

from src.load.to_sql import write_frame_to_sql


def load_raw_prices(engine: Engine, prices_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load raw price rows into raw.prices."""
    write_frame_to_sql(prices_df, "prices", engine, schema="raw", if_exists=if_exists)


def load_assets(engine: Engine, assets_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load asset metadata into raw.assets."""
    write_frame_to_sql(assets_df, "assets", engine, schema="raw", if_exists=if_exists)


def load_positions(engine: Engine, positions_df: pd.DataFrame, if_exists: str = "append") -> None:
    """Load portfolio positions into raw.portfolio_positions."""
    columns = ["portfolio_name", "ticker", "quantity", "as_of_date", "asset_class", "sector", "currency", "base_currency"]
    write_frame_to_sql(positions_df[columns], "portfolio_positions", engine, schema="raw", if_exists=if_exists)
