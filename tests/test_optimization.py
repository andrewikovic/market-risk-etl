import numpy as np
import pandas as pd

from src.risk.optimization import calculate_rebalancing_trades, generate_efficient_frontier, optimize_portfolio


def _optimization_inputs():
    expected_returns = pd.Series({"A": 0.05, "B": 0.09, "C": 0.12})
    covariance = pd.DataFrame(
        [[0.0400, 0.0060, 0.0040], [0.0060, 0.0225, 0.0050], [0.0040, 0.0050, 0.0324]],
        index=expected_returns.index,
        columns=expected_returns.index,
    )
    return expected_returns, covariance


def test_efficient_frontier_returns_fully_invested_weight_sets():
    expected_returns, covariance = _optimization_inputs()

    frontier = generate_efficient_frontier(
        expected_returns,
        covariance,
        points=9,
        min_weight=0.05,
        max_weight=0.80,
    )

    assert len(frontier) == 9
    assert frontier["volatility"].gt(0).all()
    weight_cols = ["weight_A", "weight_B", "weight_C"]
    assert np.allclose(frontier[weight_cols].sum(axis=1), 1.0)
    assert frontier[weight_cols].ge(0.05 - 1e-8).all().all()
    assert frontier[weight_cols].le(0.80 + 1e-8).all().all()


def test_optimizer_supports_target_return_and_constraint_diagnostics():
    expected_returns, covariance = _optimization_inputs()

    result = optimize_portfolio(
        expected_returns,
        covariance,
        min_weight=0.05,
        max_weight=0.80,
        target_return=0.09,
    )

    assert np.isclose(result["expected_return"], 0.09)
    assert result["volatility"] > 0
    assert result["constraint_diagnostics"]["full_investment"]
    assert result["constraint_diagnostics"]["min_weight_satisfied"]
    assert result["constraint_diagnostics"]["max_weight_satisfied"]
    assert np.isclose(sum(result["weights"].values()), 1.0)


def test_optimizer_default_maximizes_sharpe_with_long_only_weights():
    expected_returns, covariance = _optimization_inputs()

    result = optimize_portfolio(expected_returns, covariance, max_weight=0.75, risk_free_rate=0.02)

    assert result["sharpe_ratio"] > 0
    assert result["constraint_diagnostics"]["long_only"]
    assert max(result["weights"].values()) <= 0.75 + 1e-8


def test_rebalancing_outputs_weight_changes_and_trade_values():
    trades = calculate_rebalancing_trades(
        current_weights={"A": 0.40, "B": 0.40, "C": 0.20},
        target_weights={"A": 0.30, "B": 0.50, "C": 0.20},
        portfolio_value=1_000,
        prices={"A": 10.0, "B": 20.0, "C": 25.0},
    )

    assert np.isclose(trades.loc[trades["ticker"] == "A", "trade_value"].iloc[0], -100.0)
    assert np.isclose(trades.loc[trades["ticker"] == "B", "quantity_change"].iloc[0], 5.0)
    assert np.isclose(trades["trade_value"].sum(), 0.0)
