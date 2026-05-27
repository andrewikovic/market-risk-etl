from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_historical_var(
    portfolio_returns: pd.Series | pd.DataFrame,
    portfolio_value: float,
    confidence_level: float = 0.95,
) -> float:
    """Calculate historical VaR from empirical portfolio returns."""
    returns = _clean_returns(portfolio_returns)
    if returns.empty:
        return float("nan")
    percentile = np.percentile(returns, (1.0 - confidence_level) * 100.0)
    return float(abs(percentile) * portfolio_value)


def _clean_returns(portfolio_returns: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(portfolio_returns, pd.Series):
        series = portfolio_returns.copy()
    else:
        value_col = next(
            (col for col in ["daily_return", "portfolio_return", "return"] if col in portfolio_returns.columns),
            None,
        )
        if value_col is None:
            raise ValueError("portfolio_returns must include daily_return, portfolio_return, or return")
        series = portfolio_returns[value_col]
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

