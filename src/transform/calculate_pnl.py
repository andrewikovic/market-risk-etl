from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_portfolio_values(
    positions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    portfolio_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate daily portfolio market value and position-level P&L."""
    required_positions = {"ticker", "quantity"}
    missing_positions = required_positions - set(positions_df.columns)
    if missing_positions:
        raise ValueError(f"positions_df is missing columns: {sorted(missing_positions)}")
    required_prices = {"ticker", "price_date", "adjusted_close"}
    missing_prices = required_prices - set(prices_df.columns)
    if missing_prices:
        raise ValueError(f"prices_df is missing columns: {sorted(missing_prices)}")

    positions = positions_df.copy()
    positions["ticker"] = positions["ticker"].astype(str).str.upper().str.strip()
    positions["quantity"] = pd.to_numeric(positions["quantity"], errors="coerce")
    quantity = positions.groupby("ticker")["quantity"].sum()
    portfolio_name = portfolio_name or positions.get("portfolio_name", pd.Series(["Portfolio"])).iloc[0]

    prices = prices_df[["ticker", "price_date", "adjusted_close"]].copy()
    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip()
    prices["price_date"] = pd.to_datetime(prices["price_date"])
    prices["adjusted_close"] = pd.to_numeric(prices["adjusted_close"], errors="coerce")
    prices = prices[prices["ticker"].isin(quantity.index) & prices["adjusted_close"].gt(0)]

    price_matrix = (
        prices.pivot_table(index="price_date", columns="ticker", values="adjusted_close", aggfunc="last")
        .sort_index()
        .ffill()
    )
    price_matrix = price_matrix.dropna(subset=list(quantity.index), how="any")
    price_matrix = price_matrix[quantity.index]
    if price_matrix.empty:
        raise ValueError("No complete price history available for portfolio valuation")

    position_values = price_matrix.multiply(quantity, axis=1)
    market_value = position_values.sum(axis=1)
    daily_pnl = market_value.diff().fillna(0.0)
    daily_return = market_value.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cumulative_return = market_value / market_value.iloc[0] - 1.0

    portfolio_values = pd.DataFrame(
        {
            "portfolio_name": portfolio_name,
            "value_date": market_value.index.date,
            "market_value": market_value.to_numpy(dtype=float),
            "daily_pnl": daily_pnl.to_numpy(dtype=float),
            "daily_return": daily_return.to_numpy(dtype=float),
            "cumulative_return": cumulative_return.to_numpy(dtype=float),
        }
    )

    asset_returns = price_matrix.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    position_pnl = price_matrix.diff().multiply(quantity, axis=1).fillna(0.0)
    prev_position_values = position_values.shift(1)
    prev_portfolio_value = market_value.shift(1)
    prev_weights = prev_position_values.div(prev_portfolio_value, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    contribution_to_return = (prev_weights * asset_returns).fillna(0.0)
    weights = position_values.div(market_value, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    pnl_long = position_values.stack().rename("position_value").reset_index()
    pnl_long = pnl_long.rename(columns={"price_date": "value_date", "level_1": "ticker"})
    pnl_long["daily_pnl"] = position_pnl.stack().to_numpy(dtype=float)
    pnl_long["contribution_to_pnl"] = pnl_long["daily_pnl"]
    pnl_long["contribution_to_return"] = contribution_to_return.stack().to_numpy(dtype=float)
    pnl_long["weight"] = weights.stack().to_numpy(dtype=float)
    pnl_long["portfolio_name"] = portfolio_name
    pnl_long["value_date"] = pd.to_datetime(pnl_long["value_date"]).dt.date

    metadata_cols = [col for col in ["ticker", "asset_class", "sector", "currency"] if col in positions.columns]
    if len(metadata_cols) > 1:
        pnl_long = pnl_long.merge(positions[metadata_cols].drop_duplicates("ticker"), on="ticker", how="left")

    ordered = [
        "portfolio_name",
        "value_date",
        "ticker",
        "position_value",
        "daily_pnl",
        "contribution_to_pnl",
        "contribution_to_return",
        "weight",
    ]
    extra = [col for col in ["asset_class", "sector", "currency"] if col in pnl_long.columns]
    return portfolio_values.reset_index(drop=True), pnl_long[ordered + extra].reset_index(drop=True)


def calculate_pnl_attribution(
    positions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    returns_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculate position-level P&L and contribution to portfolio return."""
    _, position_pnl = calculate_portfolio_values(positions_df, prices_df)
    return position_pnl

