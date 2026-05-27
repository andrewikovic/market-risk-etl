from __future__ import annotations

import math

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def calculate_rolling_volatility(
    returns_df: pd.DataFrame,
    windows: tuple[int, ...] = (20, 60, 252),
) -> pd.DataFrame:
    """Calculate annualized rolling volatility for each ticker."""
    required = {"ticker", "return_date", "daily_return"}
    missing = required - set(returns_df.columns)
    if missing:
        raise ValueError(f"returns_df is missing columns: {sorted(missing)}")

    if returns_df.empty:
        return pd.DataFrame(columns=["ticker", "return_date", "window", "rolling_volatility"])

    data = returns_df[["ticker", "return_date", "daily_return"]].copy()
    data["return_date"] = pd.to_datetime(data["return_date"])
    data["daily_return"] = pd.to_numeric(data["daily_return"], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["daily_return"])
    data = data.sort_values(["ticker", "return_date"])

    frames: list[pd.DataFrame] = []
    for ticker, group in data.groupby("ticker", sort=True):
        for window in windows:
            tmp = pd.DataFrame(
                {
                    "ticker": ticker,
                    "return_date": group["return_date"],
                    "window": window,
                    "rolling_volatility": group["daily_return"]
                    .rolling(window=window, min_periods=window)
                    .std(ddof=1)
                    * math.sqrt(TRADING_DAYS),
                }
            )
            frames.append(tmp)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if result.empty:
        return pd.DataFrame(columns=["ticker", "return_date", "window", "rolling_volatility"])
    result["return_date"] = pd.to_datetime(result["return_date"]).dt.date
    return result.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def calculate_sharpe_ratio(returns: pd.Series | pd.DataFrame, risk_free_rate_annual: float = 0.0) -> float:
    """Calculate annualized Sharpe ratio from daily returns."""
    series = _return_series(returns)
    if series.empty:
        return float("nan")
    daily_rf = (1.0 + risk_free_rate_annual) ** (1.0 / TRADING_DAYS) - 1.0
    excess = series - daily_rf
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(excess.mean() / std * math.sqrt(TRADING_DAYS))


def calculate_sortino_ratio(
    returns: pd.Series | pd.DataFrame,
    risk_free_rate_annual: float = 0.0,
    target_return_daily: float = 0.0,
) -> float:
    """Calculate annualized Sortino ratio from daily returns."""
    series = _return_series(returns)
    if series.empty:
        return float("nan")
    daily_rf = (1.0 + risk_free_rate_annual) ** (1.0 / TRADING_DAYS) - 1.0
    excess = series - daily_rf
    downside = series[series < target_return_daily] - target_return_daily
    downside_std = downside.std(ddof=1)
    if downside.empty or downside_std == 0 or np.isnan(downside_std):
        return float("nan")
    return float(excess.mean() / downside_std * math.sqrt(TRADING_DAYS))


def _return_series(returns: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(returns, pd.Series):
        series = returns.copy()
    else:
        for column in ["daily_return", "portfolio_return", "return"]:
            if column in returns.columns:
                series = returns[column].copy()
                break
        else:
            raise ValueError("returns DataFrame must contain daily_return, portfolio_return, or return")
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

