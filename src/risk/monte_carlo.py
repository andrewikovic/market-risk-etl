from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk.covariance import align_returns_matrix


def run_correlated_monte_carlo(
    returns_df: pd.DataFrame,
    weights: dict | pd.Series,
    initial_value: float = 100_000,
    horizon_days: int = 252,
    n_simulations: int = 10_000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> dict:
    """Run correlated multi-asset Monte Carlo simulation using historical covariance."""
    if horizon_days <= 0 or n_simulations <= 0:
        raise ValueError("horizon_days and n_simulations must be positive")

    returns_matrix = align_returns_matrix(returns_df)
    if returns_matrix.empty:
        raise ValueError("returns_df has no complete aligned return history")

    weight_series = pd.Series(weights, dtype=float)
    missing = set(weight_series.index) - set(returns_matrix.columns)
    if missing:
        raise ValueError(f"Missing returns for weighted assets: {sorted(missing)}")
    weight_series = weight_series.reindex(returns_matrix.columns).fillna(0.0)
    if not np.isclose(weight_series.sum(), 1.0):
        weight_series = weight_series / weight_series.sum()

    mean_vector = returns_matrix.mean().to_numpy(dtype=float)
    covariance_matrix = returns_matrix.cov().to_numpy(dtype=float)
    cholesky = _stable_cholesky(covariance_matrix)
    rng = np.random.default_rng(random_seed)
    independent_shocks = rng.standard_normal((n_simulations, horizon_days, len(mean_vector)))
    simulated_asset_returns = mean_vector + independent_shocks @ cholesky.T
    simulated_portfolio_returns = np.tensordot(simulated_asset_returns, weight_series.to_numpy(), axes=([2], [0]))

    path_values = initial_value * np.cumprod(1.0 + simulated_portfolio_returns, axis=1)
    path_values = np.concatenate([np.full((n_simulations, 1), initial_value), path_values], axis=1)
    paths_df = pd.DataFrame(path_values.T, index=range(horizon_days + 1))
    paths_df.columns = [f"simulation_{idx}" for idx in range(n_simulations)]

    terminal_values = path_values[:, -1]
    terminal_returns = terminal_values / initial_value - 1.0
    loss_threshold = np.percentile(terminal_values, (1.0 - confidence_level) * 100.0)
    losses = initial_value - terminal_values
    monte_carlo_var = max(initial_value - loss_threshold, 0.0)
    tail_losses = losses[terminal_values <= loss_threshold]
    monte_carlo_es = float(tail_losses.mean()) if len(tail_losses) else float("nan")

    running_max = np.maximum.accumulate(path_values, axis=1)
    drawdowns = path_values / running_max - 1.0
    max_drawdowns = drawdowns.min(axis=1)

    return {
        "paths": paths_df,
        "terminal_values": pd.Series(terminal_values, name="terminal_value"),
        "terminal_returns": pd.Series(terminal_returns, name="terminal_return"),
        "monte_carlo_var": float(monte_carlo_var),
        "monte_carlo_expected_shortfall": float(monte_carlo_es),
        "probability_of_loss": float(np.mean(terminal_values < initial_value)),
        "probability_loss_gt_10": float(np.mean(terminal_returns < -0.10)),
        "median_terminal_value": float(np.median(terminal_values)),
        "p5_terminal_value": float(np.percentile(terminal_values, 5)),
        "p95_terminal_value": float(np.percentile(terminal_values, 95)),
        "worst_simulated_loss": float(np.max(losses)),
        "simulated_max_drawdown": float(abs(max_drawdowns.min())),
        "max_drawdowns": pd.Series(max_drawdowns, name="max_drawdown"),
        "simulated_asset_returns": simulated_asset_returns,
        "simulated_correlation_matrix": pd.DataFrame(
            np.corrcoef(simulated_asset_returns.reshape(-1, len(mean_vector)), rowvar=False),
            index=returns_matrix.columns,
            columns=returns_matrix.columns,
        ),
    }


def _stable_cholesky(covariance_matrix: np.ndarray) -> np.ndarray:
    jitter = 1e-10
    for _ in range(8):
        try:
            return np.linalg.cholesky(covariance_matrix)
        except np.linalg.LinAlgError:
            covariance_matrix = covariance_matrix + np.eye(covariance_matrix.shape[0]) * jitter
            jitter *= 10
    return np.linalg.cholesky(covariance_matrix + np.eye(covariance_matrix.shape[0]) * jitter)

