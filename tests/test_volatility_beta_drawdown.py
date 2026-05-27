import numpy as np
import pandas as pd

from src.transform.calculate_beta import calculate_beta
from src.transform.calculate_drawdowns import calculate_drawdowns
from src.transform.calculate_volatility import calculate_rolling_volatility


def test_rolling_volatility_window_and_annualization():
    returns = pd.DataFrame(
        {
            "ticker": ["A"] * 4,
            "return_date": pd.date_range("2024-01-01", periods=4),
            "daily_return": [0.01, 0.02, 0.03, 0.04],
        }
    )

    vol = calculate_rolling_volatility(returns, windows=(3,))
    first_valid = vol.dropna(subset=["rolling_volatility"]).iloc[0]["rolling_volatility"]
    expected = np.std([0.01, 0.02, 0.03], ddof=1) * np.sqrt(252)

    assert vol["rolling_volatility"].isna().sum() == 2
    assert np.isclose(first_valid, expected)


def test_empty_volatility_input_does_not_crash():
    returns = pd.DataFrame(columns=["ticker", "return_date", "daily_return"])
    vol = calculate_rolling_volatility(returns)
    assert vol.empty


def test_beta_uses_aligned_dates_and_covariance_formula():
    dates = pd.date_range("2024-01-01", periods=5)
    benchmark = pd.DataFrame({"return_date": dates[:-1], "daily_return": [0.01, 0.02, -0.01, 0.03]})
    portfolio = pd.DataFrame(
        {
            "return_date": dates,
            "daily_return": [0.02, 0.04, -0.02, 0.06, 0.99],
        }
    )

    result = calculate_beta(portfolio, benchmark, rolling_window=3)

    assert np.isclose(result["beta"], 2.0)
    assert len(result["rolling_beta"]) == 4


def test_drawdown_zero_at_new_high_and_max_duration():
    values = pd.DataFrame(
        {
            "value_date": pd.date_range("2024-01-01", periods=7),
            "market_value": [100, 110, 105, 120, 90, 95, 130],
        }
    )

    result = calculate_drawdowns(values)
    series = result["drawdown_series"]

    assert series.loc[series["market_value"].isin([100, 110, 120, 130]), "drawdown"].eq(0).all()
    assert np.isclose(result["max_drawdown"], -0.25)
    assert result["drawdown_duration"] == 2

