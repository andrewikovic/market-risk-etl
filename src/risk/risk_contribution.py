from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk.covariance import align_returns_matrix
from src.risk.parametric_var import calculate_component_var


def calculate_asset_risk_contributions(
    returns_df: pd.DataFrame,
    weights: dict | pd.Series,
    portfolio_value: float = 100_000,
    confidence_level: float = 0.95,
    exposures: dict | pd.Series | None = None,
) -> pd.DataFrame:
    """Calculate volatility and VaR contribution by asset."""
    returns_matrix = align_returns_matrix(returns_df)
    var_contributions = calculate_component_var(
        returns_matrix,
        weights,
        portfolio_value=portfolio_value,
        confidence_level=confidence_level,
        exposures=exposures,
    ).set_index("ticker")

    weight_series = var_contributions["weight"]
    covariance = returns_matrix.cov().reindex(index=weight_series.index, columns=weight_series.index)
    portfolio_variance = float(weight_series.to_numpy() @ covariance.to_numpy() @ weight_series.to_numpy())
    if portfolio_variance <= 0 or np.isnan(portfolio_variance):
        raise ValueError("portfolio variance must be positive")

    portfolio_volatility = float(np.sqrt(portfolio_variance))
    covariance_weight = covariance.dot(weight_series)
    marginal_volatility = covariance_weight / portfolio_volatility
    component_volatility = weight_series * marginal_volatility
    volatility_error = float(component_volatility.sum() - portfolio_volatility)
    var_error = float(var_contributions["component_var"].sum() - var_contributions["portfolio_var"].iloc[0])

    result = pd.DataFrame(
        {
            "ticker": weight_series.index,
            "weight": weight_series.to_numpy(dtype=float),
            "exposure": var_contributions["exposure"].to_numpy(dtype=float),
            "asset_volatility": returns_matrix.std(ddof=1).reindex(weight_series.index).to_numpy(dtype=float),
            "marginal_volatility": marginal_volatility.to_numpy(dtype=float),
            "component_volatility": component_volatility.to_numpy(dtype=float),
            "volatility_contribution_amount": (component_volatility * portfolio_value).to_numpy(dtype=float),
            "marginal_var": var_contributions["marginal_var"].to_numpy(dtype=float),
            "component_var": var_contributions["component_var"].to_numpy(dtype=float),
        }
    )
    result["volatility_percent_contribution"] = np.where(
        portfolio_volatility != 0,
        result["component_volatility"] / portfolio_volatility,
        np.nan,
    )
    portfolio_var = float(var_contributions["portfolio_var"].iloc[0])
    result["var_percent_contribution"] = np.where(portfolio_var != 0, result["component_var"] / portfolio_var, np.nan)
    result["portfolio_volatility"] = portfolio_volatility
    result["portfolio_var"] = portfolio_var
    result["volatility_reconciliation_error"] = volatility_error
    result["var_reconciliation_error"] = var_error
    return result
