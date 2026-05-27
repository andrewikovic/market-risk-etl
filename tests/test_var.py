import numpy as np
import pandas as pd

from src.risk.expected_shortfall import calculate_expected_shortfall
from src.risk.historical_var import calculate_historical_var
from src.risk.parametric_var import calculate_parametric_var


def test_historical_var_uses_correct_percentile():
    returns = pd.Series([-0.10, -0.05, -0.02, 0.01, 0.03])
    portfolio_value = 1_000

    var = calculate_historical_var(returns, portfolio_value, confidence_level=0.95)

    assert np.isclose(var, abs(np.percentile(returns, 5)) * portfolio_value)
    assert var >= 0


def test_expected_shortfall_averages_losses_beyond_threshold():
    returns = pd.Series([-0.10, -0.08, -0.02, 0.01, 0.03])
    portfolio_value = 1_000
    threshold = np.percentile(returns, 20)
    expected = abs(returns[returns <= threshold].mean()) * portfolio_value

    es = calculate_expected_shortfall(returns, portfolio_value, confidence_level=0.80)

    assert np.isclose(es, expected)


def test_parametric_var_is_non_negative_loss_amount():
    returns = pd.Series([0.01, 0.02, -0.01, 0.00, 0.03])

    var = calculate_parametric_var(returns, 1_000, confidence_level=0.95)

    assert var >= 0

