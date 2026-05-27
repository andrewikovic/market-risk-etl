import numpy as np
import pandas as pd

from src.risk.monte_carlo import run_correlated_monte_carlo


def _returns_df(seed=7):
    rng = np.random.default_rng(seed)
    covariance = np.array([[0.0004, 0.00018], [0.00018, 0.0003]])
    samples = rng.multivariate_normal([0.001, 0.0005], covariance, size=500)
    dates = pd.date_range("2022-01-01", periods=500)
    wide = pd.DataFrame(samples, columns=["A", "B"], index=dates)
    return wide.reset_index(names="return_date").melt(
        id_vars="return_date",
        var_name="ticker",
        value_name="daily_return",
    )


def test_same_seed_gives_same_terminal_values():
    returns = _returns_df()
    weights = {"A": 0.6, "B": 0.4}

    first = run_correlated_monte_carlo(returns, weights, horizon_days=20, n_simulations=250, random_seed=42)
    second = run_correlated_monte_carlo(returns, weights, horizon_days=20, n_simulations=250, random_seed=42)

    assert np.allclose(first["terminal_values"], second["terminal_values"])


def test_path_matrix_shape_and_terminal_length():
    result = run_correlated_monte_carlo(
        _returns_df(),
        {"A": 0.5, "B": 0.5},
        horizon_days=15,
        n_simulations=300,
        random_seed=1,
    )

    assert result["paths"].shape == (16, 300)
    assert len(result["terminal_values"]) == 300


def test_expected_shortfall_is_worse_than_or_equal_to_var():
    result = run_correlated_monte_carlo(
        _returns_df(),
        {"A": 0.5, "B": 0.5},
        horizon_days=30,
        n_simulations=1000,
        random_seed=2,
    )

    assert result["monte_carlo_expected_shortfall"] >= result["monte_carlo_var"]


def test_simulated_correlations_approximately_match_input_correlations():
    returns = _returns_df()
    input_corr = returns.pivot(index="return_date", columns="ticker", values="daily_return").corr().loc["A", "B"]
    result = run_correlated_monte_carlo(
        returns,
        {"A": 0.5, "B": 0.5},
        horizon_days=30,
        n_simulations=2000,
        random_seed=3,
    )
    simulated_corr = result["simulated_correlation_matrix"].loc["A", "B"]

    assert abs(simulated_corr - input_corr) < 0.08

