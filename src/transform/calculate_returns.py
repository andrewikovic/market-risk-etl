from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_daily_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate simple and log daily returns from adjusted close prices.

    The first observation per ticker is removed, and invalid or infinite values are
    filtered out rather than forwarded into risk calculations.
    """
    required = {"ticker", "adjusted_close"}
    missing = required - set(prices_df.columns)
    if missing:
        raise ValueError(f"prices_df is missing columns: {sorted(missing)}")

    date_col = "price_date" if "price_date" in prices_df.columns else "return_date"
    if date_col not in prices_df.columns:
        raise ValueError("prices_df must include price_date or return_date")

    prices = prices_df[["ticker", date_col, "adjusted_close"]].copy()
    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip()
    prices[date_col] = pd.to_datetime(prices[date_col])
    prices["adjusted_close"] = pd.to_numeric(prices["adjusted_close"], errors="coerce")
    prices = prices[prices["adjusted_close"].gt(0)].sort_values(["ticker", date_col])

    grouped = prices.groupby("ticker", group_keys=False)
    prices["previous_close"] = grouped["adjusted_close"].shift(1)
    prices["daily_return"] = prices["adjusted_close"] / prices["previous_close"] - 1.0
    prices["log_return"] = np.log(prices["adjusted_close"] / prices["previous_close"])

    returns = prices.rename(columns={date_col: "return_date"})[
        ["ticker", "return_date", "daily_return", "log_return"]
    ]
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(subset=["daily_return", "log_return"])
    returns["return_date"] = pd.to_datetime(returns["return_date"]).dt.date
    return returns.reset_index(drop=True)


def calculate_cumulative_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate cumulative return series by ticker."""
    if returns_df.empty:
        return pd.DataFrame(columns=["ticker", "return_date", "cumulative_return"])

    returns = returns_df[["ticker", "return_date", "daily_return"]].copy()
    returns["return_date"] = pd.to_datetime(returns["return_date"])
    returns = returns.sort_values(["ticker", "return_date"])
    returns["cumulative_return"] = (
        returns.groupby("ticker")["daily_return"].transform(lambda values: (1.0 + values).cumprod() - 1.0)
    )
    returns["return_date"] = returns["return_date"].dt.date
    return returns[["ticker", "return_date", "cumulative_return"]].reset_index(drop=True)

