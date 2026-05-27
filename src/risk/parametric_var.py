from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from src.risk.historical_var import _clean_returns


def calculate_parametric_var(
    portfolio_returns: pd.Series | pd.DataFrame,
    portfolio_value: float,
    confidence_level: float = 0.95,
) -> float:
    """Calculate parametric VaR using the normal approximation."""
    returns = _clean_returns(portfolio_returns)
    if returns.empty:
        return float("nan")
    z_score = NormalDist().inv_cdf(confidence_level)
    if confidence_level == 0.95:
        z_score = 1.645
    mean_return = returns.mean()
    volatility = returns.std(ddof=1)
    if np.isnan(volatility):
        return float("nan")
    return float(max(portfolio_value * (z_score * volatility - mean_return), 0.0))

