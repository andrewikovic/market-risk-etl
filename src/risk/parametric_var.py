from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from src.risk.covariance import align_returns_matrix
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


def calculate_component_var(
    returns_df: pd.DataFrame,
    weights: dict | pd.Series,
    portfolio_value: float = 100_000,
    confidence_level: float = 0.95,
    exposures: dict | pd.Series | None = None,
) -> pd.DataFrame:
    """Calculate marginal and component parametric VaR by asset."""
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be positive")
    returns_matrix = align_returns_matrix(returns_df)
    if returns_matrix.empty:
        raise ValueError("returns_df has no complete aligned return history")

    weight_series = _aligned_weights(weights, returns_matrix.columns)
    mean_vector = returns_matrix.mean()
    covariance = returns_matrix.cov()
    portfolio_variance = float(weight_series.to_numpy() @ covariance.to_numpy() @ weight_series.to_numpy())
    if portfolio_variance <= 0 or np.isnan(portfolio_variance):
        raise ValueError("portfolio variance must be positive")

    portfolio_volatility = float(np.sqrt(portfolio_variance))
    portfolio_mean = float(weight_series.dot(mean_vector))
    z_score = _normal_z_score(confidence_level)
    covariance_weight = covariance.dot(weight_series)
    marginal_return_var = z_score * covariance_weight / portfolio_volatility - mean_vector
    marginal_var = marginal_return_var * portfolio_value
    component_var = weight_series * marginal_var
    total_var = float(portfolio_value * (z_score * portfolio_volatility - portfolio_mean))
    if total_var <= 0:
        marginal_var = marginal_var * 0.0
        component_var = component_var * 0.0
        total_var = 0.0

    if exposures is None:
        exposure_series = weight_series * portfolio_value
    else:
        exposure_series = pd.Series(exposures, dtype=float).reindex(returns_matrix.columns).fillna(0.0)

    result = pd.DataFrame(
        {
            "ticker": returns_matrix.columns,
            "weight": weight_series.to_numpy(dtype=float),
            "exposure": exposure_series.to_numpy(dtype=float),
            "mean_return": mean_vector.reindex(returns_matrix.columns).to_numpy(dtype=float),
            "volatility": returns_matrix.std(ddof=1).reindex(returns_matrix.columns).to_numpy(dtype=float),
            "marginal_var": marginal_var.reindex(returns_matrix.columns).to_numpy(dtype=float),
            "component_var": component_var.reindex(returns_matrix.columns).to_numpy(dtype=float),
        }
    )
    result["percent_contribution"] = np.where(
        total_var != 0,
        result["component_var"] / total_var,
        np.nan,
    )
    result["portfolio_var"] = total_var
    result["confidence_level"] = confidence_level
    result["contribution_reconciliation_error"] = result["component_var"].sum() - total_var
    return result


def calculate_var_contributions(
    returns_df: pd.DataFrame,
    weights: dict | pd.Series,
    portfolio_value: float = 100_000,
    confidence_level: float = 0.95,
    exposures: dict | pd.Series | None = None,
) -> pd.DataFrame:
    """Alias for asset-level marginal and component VaR contributions."""
    return calculate_component_var(
        returns_df,
        weights,
        portfolio_value=portfolio_value,
        confidence_level=confidence_level,
        exposures=exposures,
    )


def _aligned_weights(weights: dict | pd.Series, columns: pd.Index) -> pd.Series:
    weight_series = pd.Series(weights, dtype=float)
    missing = set(weight_series.index) - set(columns)
    if missing:
        raise ValueError(f"Missing returns for weighted assets: {sorted(missing)}")
    weight_series = weight_series.reindex(columns).fillna(0.0)
    total_weight = weight_series.sum()
    if total_weight == 0 or np.isnan(total_weight):
        raise ValueError("weights must sum to a non-zero value")
    if not np.isclose(total_weight, 1.0):
        weight_series = weight_series / total_weight
    return weight_series


def _normal_z_score(confidence_level: float) -> float:
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    if confidence_level == 0.95:
        return 1.645
    return NormalDist().inv_cdf(confidence_level)
