from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yaml


def load_portfolio_config(config_path: str | Path, as_of_date: date | str | None = None) -> dict:
    """Load the portfolio YAML config."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if "portfolio_name" not in payload or "positions" not in payload:
        raise ValueError(f"Portfolio config is missing required keys: {config_path}")
    payload["as_of_date"] = pd.to_datetime(as_of_date or date.today()).date()
    return payload


def load_positions_from_config(config_path: str | Path, as_of_date: date | str | None = None) -> pd.DataFrame:
    """Convert the portfolio YAML positions into a clean DataFrame."""
    payload = load_portfolio_config(config_path, as_of_date=as_of_date)
    positions = pd.DataFrame(payload["positions"])
    if positions.empty:
        raise ValueError("Portfolio config contains no positions")

    positions["portfolio_name"] = payload["portfolio_name"]
    positions["base_currency"] = payload.get("base_currency", "USD")
    positions["as_of_date"] = payload["as_of_date"]
    positions["ticker"] = positions["ticker"].str.upper().str.strip()
    positions["quantity"] = pd.to_numeric(positions["quantity"], errors="coerce")
    if positions["quantity"].isna().any():
        raise ValueError("Portfolio positions contain non-numeric quantities")
    return positions[
        [
            "portfolio_name",
            "ticker",
            "quantity",
            "as_of_date",
            "asset_class",
            "sector",
            "currency",
            "base_currency",
        ]
    ].reset_index(drop=True)

