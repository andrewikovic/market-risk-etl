import numpy as np
import pandas as pd
import pytest

from src.risk.covariance import (
    align_returns_matrix,
    calculate_correlation_matrix,
    calculate_covariance_matrix,
    calculate_portfolio_return_series,
)


def test_align_returns_matrix_pivots_sorts_and_drops_invalid_rows():
    returns = pd.DataFrame(
        {
            "ticker": ["B", "A", "B", "A", "A", "B"],
            "return_date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-01", "2024-01-01", "2024-01-03", "2024-01-03"]
            ),
            "daily_return": [0.02, 0.01, 0.03, 0.02, np.inf, 0.01],
        }
    )

    matrix = align_returns_matrix(returns)

    assert matrix.index.tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert matrix.columns.tolist() == ["A", "B"]
    assert np.allclose(matrix.loc[pd.Timestamp("2024-01-01")], [0.02, 0.03])


def test_covariance_matrix_can_be_annualized():
    returns = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [0.02, 0.03, 0.04]})

    daily = calculate_covariance_matrix(returns)
    annualized = calculate_covariance_matrix(returns, annualize=True)

    assert np.allclose(annualized, daily * 252)


def test_correlation_matrix_uses_aligned_returns():
    returns = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [0.03, 0.02, 0.01]})

    correlation = calculate_correlation_matrix(returns)

    assert np.isclose(correlation.loc["A", "B"], -1.0)


def test_portfolio_return_series_aligns_weights_and_rejects_missing_assets():
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.03, 0.04]})

    portfolio_returns = calculate_portfolio_return_series(returns, {"A": 0.25, "B": 0.75})

    assert np.allclose(portfolio_returns, [0.025, 0.035])
    with pytest.raises(ValueError, match="Missing returns for weighted assets"):
        calculate_portfolio_return_series(returns, {"A": 0.5, "C": 0.5})
