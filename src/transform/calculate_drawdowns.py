from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_drawdowns(portfolio_values: pd.Series | pd.DataFrame) -> dict:
    """Calculate drawdown series, current drawdown, max drawdown, and worst period."""
    values = _coerce_values(portfolio_values)
    if values.empty:
        return {
            "drawdown_series": pd.DataFrame(columns=["value_date", "market_value", "rolling_peak", "drawdown"]),
            "current_drawdown": float("nan"),
            "max_drawdown": float("nan"),
            "max_drawdown_start": None,
            "max_drawdown_end": None,
            "drawdown_duration": 0,
        }

    rolling_peak = values.cummax()
    drawdown = values / rolling_peak - 1.0
    trough_idx = drawdown.idxmin()
    peak_idx = values.loc[:trough_idx].idxmax()

    underwater = drawdown.lt(0)
    max_duration = _max_true_streak(underwater)
    series = pd.DataFrame(
        {
            "value_date": values.index,
            "market_value": values.to_numpy(),
            "rolling_peak": rolling_peak.to_numpy(),
            "drawdown": drawdown.to_numpy(),
        }
    )
    series["value_date"] = pd.to_datetime(series["value_date"]).dt.date

    return {
        "drawdown_series": series,
        "current_drawdown": float(drawdown.iloc[-1]),
        "max_drawdown": float(drawdown.min()),
        "max_drawdown_start": pd.to_datetime(peak_idx).date(),
        "max_drawdown_end": pd.to_datetime(trough_idx).date(),
        "drawdown_duration": int(max_duration),
    }


def _coerce_values(portfolio_values: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(portfolio_values, pd.Series):
        series = portfolio_values.copy()
        series.index = pd.to_datetime(series.index)
    else:
        date_col = "value_date" if "value_date" in portfolio_values.columns else "date"
        if date_col not in portfolio_values.columns or "market_value" not in portfolio_values.columns:
            raise ValueError("portfolio_values must contain value_date/date and market_value")
        series = pd.Series(
            pd.to_numeric(portfolio_values["market_value"], errors="coerce").to_numpy(),
            index=pd.to_datetime(portfolio_values[date_col]),
        )
    return series.replace([np.inf, -np.inf], np.nan).dropna().sort_index()


def _max_true_streak(mask: pd.Series) -> int:
    max_streak = 0
    current = 0
    for value in mask:
        if value:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak

