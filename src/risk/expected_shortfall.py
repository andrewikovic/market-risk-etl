from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk.historical_var import _clean_returns


def calculate_expected_shortfall(
    portfolio_returns: pd.Series | pd.DataFrame,
    portfolio_value: float,
    confidence_level: float = 0.95,
) -> float:
    """Calculate average loss beyond the VaR threshold."""
    returns = _clean_returns(portfolio_returns)
    if returns.empty:
        return float("nan")
    threshold = np.percentile(returns, (1.0 - confidence_level) * 100.0)
    tail = returns[returns <= threshold]
    if tail.empty:
        return float("nan")
    return float(abs(tail.mean()) * portfolio_value)

