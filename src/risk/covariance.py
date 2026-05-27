from __future__ import annotations

import numpy as np
import pandas as pd


def align_returns_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Align long-form asset returns into a clean date-by-ticker matrix."""
    if {"ticker", "return_date", "daily_return"}.issubset(returns_df.columns):
        matrix = returns_df.pivot_table(
            index="return_date",
            columns="ticker",
            values="daily_return",
            aggfunc="last",
        )
    else:
        matrix = returns_df.copy()
    matrix.index = pd.to_datetime(matrix.index)
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    return matrix.replace([np.inf, -np.inf], np.nan).dropna(how="any").sort_index()


def calculate_covariance_matrix(returns_df: pd.DataFrame, annualize: bool = False) -> pd.DataFrame:
    """Calculate the asset return covariance matrix."""
    matrix = align_returns_matrix(returns_df)
    cov = matrix.cov()
    return cov * 252 if annualize else cov


def calculate_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the asset return correlation matrix."""
    return align_returns_matrix(returns_df).corr()


def calculate_portfolio_return_series(returns_df: pd.DataFrame, weights: dict | pd.Series) -> pd.Series:
    """Calculate daily portfolio returns from aligned asset returns and weights."""
    matrix = align_returns_matrix(returns_df)
    weight_series = pd.Series(weights, dtype=float)
    missing = set(weight_series.index) - set(matrix.columns)
    if missing:
        raise ValueError(f"Missing returns for weighted assets: {sorted(missing)}")
    weight_series = weight_series.reindex(matrix.columns).fillna(0.0)
    return matrix.dot(weight_series)

