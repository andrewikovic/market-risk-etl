import numpy as np
import pandas as pd

from src.risk.parametric_var import calculate_component_var
from src.risk.risk_contribution import calculate_asset_risk_contributions


def _returns_df():
    rng = np.random.default_rng(11)
    covariance = np.array([[0.0004, 0.00012, 0.00005], [0.00012, 0.0003, 0.00004], [0.00005, 0.00004, 0.0002]])
    samples = rng.multivariate_normal([0.0005, 0.0003, 0.0002], covariance, size=300)
    return pd.DataFrame(samples, columns=["A", "B", "C"], index=pd.date_range("2023-01-01", periods=300))


def test_component_var_contributions_reconcile_to_total_var():
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}

    result = calculate_component_var(_returns_df(), weights, portfolio_value=1_000_000, confidence_level=0.99)

    assert set(result.columns).issuperset(
        {"ticker", "weight", "exposure", "marginal_var", "component_var", "percent_contribution"}
    )
    assert np.isclose(result["component_var"].sum(), result["portfolio_var"].iloc[0])
    assert np.isclose(result["percent_contribution"].sum(), 1.0)
    assert np.allclose(result["component_var"], result["weight"] * result["marginal_var"])


def test_asset_risk_contributions_include_volatility_and_var_checks():
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}

    result = calculate_asset_risk_contributions(_returns_df(), weights, portfolio_value=1_000_000)

    assert np.isclose(result["component_volatility"].sum(), result["portfolio_volatility"].iloc[0])
    assert np.isclose(result["component_var"].sum(), result["portfolio_var"].iloc[0])
    assert abs(result["volatility_reconciliation_error"].iloc[0]) < 1e-12
    assert abs(result["var_reconciliation_error"].iloc[0]) < 1e-9
    assert np.isclose(result["volatility_percent_contribution"].sum(), 1.0)
