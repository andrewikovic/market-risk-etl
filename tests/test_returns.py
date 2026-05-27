import numpy as np
import pandas as pd

from src.transform.calculate_returns import calculate_daily_returns


def test_daily_and_log_returns_are_calculated_correctly():
    prices = pd.DataFrame(
        {
            "ticker": ["A", "A", "A"],
            "price_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "adjusted_close": [100.0, 110.0, 121.0],
        }
    )

    returns = calculate_daily_returns(prices)

    assert len(returns) == 2
    assert np.allclose(returns["daily_return"], [0.10, 0.10])
    assert np.allclose(returns["log_return"], [np.log(1.10), np.log(1.10)])
    assert returns["return_date"].min() == pd.Timestamp("2024-01-02").date()


def test_returns_skip_first_date_per_asset():
    prices = pd.DataFrame(
        {
            "ticker": ["A", "A", "B", "B"],
            "price_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "adjusted_close": [100.0, 101.0, 50.0, 55.0],
        }
    )

    returns = calculate_daily_returns(prices)

    assert set(returns["ticker"]) == {"A", "B"}
    assert len(returns) == 2


def test_missing_and_invalid_prices_are_handled_safely():
    prices = pd.DataFrame(
        {
            "ticker": ["A", "A", "A", "A"],
            "price_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "adjusted_close": [100.0, np.nan, 0.0, 110.0],
        }
    )

    returns = calculate_daily_returns(prices)

    assert len(returns) == 1
    assert np.isfinite(returns["daily_return"]).all()
    assert np.isfinite(returns["log_return"]).all()
    assert np.isclose(returns["daily_return"].iloc[0], 0.10)
