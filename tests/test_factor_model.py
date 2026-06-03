import numpy as np
import pandas as pd

from src.risk.factor_model import estimate_factor_exposures


def test_factor_model_recovers_known_synthetic_loadings():
    dates = pd.date_range("2024-01-01", periods=80)
    factor_returns = pd.DataFrame(
        {
            "return_date": dates,
            "market": np.linspace(-0.02, 0.02, len(dates)),
            "rates": np.sin(np.linspace(0, 6, len(dates))) / 100,
        }
    )
    market = factor_returns["market"]
    rates = factor_returns["rates"]
    asset_returns = pd.DataFrame(
        {
            "return_date": dates,
            "A": 0.001 + 1.5 * market - 0.4 * rates,
            "B": -0.0005 + 0.7 * market + 1.2 * rates,
        }
    )

    result = estimate_factor_exposures(asset_returns, factor_returns, weights={"A": 0.25, "B": 0.75})
    exposures = result["asset_exposures"].pivot(index="ticker", columns="factor", values="beta")
    portfolio = result["portfolio_factor_exposure"].set_index("factor")["portfolio_beta"]

    assert np.isclose(exposures.loc["A", "market"], 1.5)
    assert np.isclose(exposures.loc["A", "rates"], -0.4)
    assert np.isclose(exposures.loc["B", "market"], 0.7)
    assert np.isclose(exposures.loc["B", "rates"], 1.2)
    assert np.isclose(portfolio.loc["market"], 0.25 * 1.5 + 0.75 * 0.7)
    assert np.isclose(portfolio.loc["rates"], 0.25 * -0.4 + 0.75 * 1.2)
    assert result["asset_summary"]["residual_volatility"].max() < 1e-12
    assert np.allclose(result["asset_summary"]["r_squared"], 1.0)
