import numpy as np
import pandas as pd
import pytest

from src.transform.calculate_returns import calculate_cumulative_returns, calculate_daily_returns


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


def test_daily_returns_require_ticker_adjusted_close_and_date():
    with pytest.raises(ValueError, match="missing columns"):
        calculate_daily_returns(pd.DataFrame({"ticker": ["A"], "price_date": [pd.Timestamp("2024-01-01")]}))

    with pytest.raises(ValueError, match="price_date or return_date"):
        calculate_daily_returns(pd.DataFrame({"ticker": ["A"], "adjusted_close": [100.0]}))


def test_daily_returns_accept_return_date_column_and_normalize_tickers():
    prices = pd.DataFrame(
        {
            "ticker": [" a ", "A"],
            "return_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "adjusted_close": [100.0, 105.0],
        }
    )

    returns = calculate_daily_returns(prices)

    assert returns["ticker"].tolist() == ["A"]
    assert returns["return_date"].tolist() == [pd.Timestamp("2024-01-02").date()]
    assert np.isclose(returns["daily_return"].iloc[0], 0.05)


def test_cumulative_returns_handle_empty_and_sort_by_asset_date():
    empty = calculate_cumulative_returns(pd.DataFrame())

    assert empty.columns.tolist() == ["ticker", "return_date", "cumulative_return"]
    assert empty.empty

    returns = pd.DataFrame(
        {
            "ticker": ["B", "A", "A"],
            "return_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"]),
            "daily_return": [0.05, 0.10, 0.20],
        }
    )

    cumulative = calculate_cumulative_returns(returns)

    assert cumulative["ticker"].tolist() == ["A", "A", "B"]
    assert np.allclose(cumulative.loc[cumulative["ticker"] == "A", "cumulative_return"], [0.20, 0.32])
