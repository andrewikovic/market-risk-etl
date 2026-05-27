from __future__ import annotations

import math

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def calculate_beta(
    portfolio_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series | pd.DataFrame,
    rolling_window: int = 60,
) -> dict:
    """
    Calculate beta, alpha, rolling beta, tracking error, and information ratio.

    Returns are aligned on shared dates when date columns or indexes are available.
    """
    portfolio = _coerce_returns(portfolio_returns, "portfolio_return")
    benchmark = _coerce_returns(benchmark_returns, "benchmark_return")
    aligned = portfolio.join(benchmark, how="inner").replace([np.inf, -np.inf], np.nan).dropna()
    if len(aligned) < 2:
        return {
            "beta": float("nan"),
            "alpha": float("nan"),
            "tracking_error": float("nan"),
            "information_ratio": float("nan"),
            "rolling_beta": pd.DataFrame(columns=["return_date", "rolling_beta"]),
        }

    benchmark_variance = aligned["benchmark_return"].var(ddof=1)
    if benchmark_variance == 0 or np.isnan(benchmark_variance):
        beta = float("nan")
    else:
        covariance = aligned["portfolio_return"].cov(aligned["benchmark_return"])
        beta = float(covariance / benchmark_variance)

    alpha = float((aligned["portfolio_return"].mean() - beta * aligned["benchmark_return"].mean()) * TRADING_DAYS)
    active_return = aligned["portfolio_return"] - aligned["benchmark_return"]
    tracking_error = float(active_return.std(ddof=1) * math.sqrt(TRADING_DAYS))
    information_ratio = (
        float(active_return.mean() * TRADING_DAYS / tracking_error)
        if tracking_error and not np.isnan(tracking_error)
        else float("nan")
    )

    rolling_cov = aligned["portfolio_return"].rolling(rolling_window, min_periods=rolling_window).cov(
        aligned["benchmark_return"]
    )
    rolling_var = aligned["benchmark_return"].rolling(rolling_window, min_periods=rolling_window).var(ddof=1)
    rolling_beta = (rolling_cov / rolling_var).replace([np.inf, -np.inf], np.nan)
    rolling_beta_df = pd.DataFrame(
        {
            "return_date": aligned.index,
            "rolling_beta": rolling_beta.to_numpy(),
        }
    )
    rolling_beta_df["return_date"] = pd.to_datetime(rolling_beta_df["return_date"]).dt.date

    return {
        "beta": beta,
        "alpha": alpha,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "rolling_beta": rolling_beta_df,
    }


def _coerce_returns(data: pd.Series | pd.DataFrame, value_name: str) -> pd.DataFrame:
    if isinstance(data, pd.Series):
        series = pd.to_numeric(data, errors="coerce")
        index = pd.to_datetime(series.index)
        return pd.DataFrame({value_name: series.to_numpy()}, index=index)

    date_col = next((col for col in ["return_date", "value_date", "date"] if col in data.columns), None)
    value_col = next(
        (col for col in ["daily_return", "portfolio_return", "benchmark_return", "return"] if col in data.columns),
        None,
    )
    if value_col is None:
        raise ValueError("returns DataFrame must include a return value column")

    frame = data.copy()
    if date_col:
        index = pd.to_datetime(frame[date_col])
    else:
        index = pd.to_datetime(frame.index)
    return pd.DataFrame({value_name: pd.to_numeric(frame[value_col], errors="coerce").to_numpy()}, index=index)

