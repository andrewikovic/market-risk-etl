import numpy as np
import pandas as pd

from src.risk.stress_testing import run_stress_test


def test_scenario_shocks_apply_with_expected_precedence():
    positions = pd.DataFrame(
        {
            "portfolio_name": ["P", "P", "P"],
            "ticker": ["SPY", "AAPL", "TLT"],
            "asset_class": ["Equity", "Equity", "Fixed Income"],
            "sector": ["Broad Market", "Technology", "Treasury Bonds"],
            "currency": ["USD", "USD", "USD"],
            "position_value": [100.0, 100.0, 100.0],
        }
    )
    scenarios = [
        {
            "name": "Shock",
            "shocks": {
                "Equity": -0.10,
                "Technology": -0.20,
                "AAPL": -0.30,
            },
        }
    ]

    result = run_stress_test(positions, scenarios)
    rows = result["position_results"].set_index("ticker")

    assert rows.loc["SPY", "shock"] == -0.10
    assert rows.loc["AAPL", "shock"] == -0.30
    assert rows.loc["TLT", "shock"] == 0.0


def test_position_losses_sum_to_portfolio_stress_loss():
    positions = pd.DataFrame(
        {
            "portfolio_name": ["P", "P"],
            "ticker": ["A", "B"],
            "asset_class": ["Equity", "Fixed Income"],
            "sector": ["Technology", "Treasury Bonds"],
            "currency": ["USD", "USD"],
            "position_value": [100.0, 200.0],
        }
    )
    scenarios = [{"name": "Mixed", "shocks": {"Equity": -0.10, "Fixed Income": 0.05}}]

    result = run_stress_test(positions, scenarios)
    position_loss = result["position_results"]["stress_loss"].sum()
    portfolio_loss = result["scenario_results"]["stress_loss"].iloc[0]

    assert np.isclose(position_loss, portfolio_loss)

