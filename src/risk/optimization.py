from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd


def generate_efficient_frontier(
    expected_returns: dict | pd.Series,
    covariance_matrix: pd.DataFrame | np.ndarray,
    points: int = 25,
    long_only: bool = True,
    min_weight: float | dict | pd.Series = 0.0,
    max_weight: float | dict | pd.Series = 1.0,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Generate an efficient frontier using constrained minimum-variance portfolios."""
    if points <= 0:
        raise ValueError("points must be positive")
    mu, cov = _coerce_inputs(expected_returns, covariance_matrix)
    lower, upper = _bounds(mu.index, long_only, min_weight, max_weight)
    min_return = _extreme_return(mu, lower, upper, maximize=False)
    max_return = _extreme_return(mu, lower, upper, maximize=True)
    targets = np.linspace(min_return, max_return, points) if points > 1 else np.array([min_return])

    rows: list[dict] = []
    for target in targets:
        weights = _solve_min_variance(mu, cov, lower, upper, target_return=float(target))
        if weights is None:
            continue
        metrics = _portfolio_metrics(weights, mu, cov, risk_free_rate)
        row = {
            "target_return": float(target),
            "expected_return": metrics["expected_return"],
            "volatility": metrics["volatility"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "weights": dict(zip(mu.index, weights)),
        }
        for ticker, weight in zip(mu.index, weights):
            row[f"weight_{ticker}"] = float(weight)
        rows.append(row)
    return pd.DataFrame(rows)


def optimize_portfolio(
    expected_returns: dict | pd.Series,
    covariance_matrix: pd.DataFrame | np.ndarray,
    long_only: bool = True,
    full_investment: bool = True,
    min_weight: float | dict | pd.Series = 0.0,
    max_weight: float | dict | pd.Series = 1.0,
    target_return: float | None = None,
    target_volatility: float | None = None,
    risk_free_rate: float = 0.0,
    objective: str = "max_sharpe",
    frontier_points: int = 75,
) -> dict:
    """Optimize portfolio weights with common box and target constraints."""
    if not full_investment:
        raise ValueError("Only fully invested portfolios are currently supported")
    if target_return is not None and target_volatility is not None:
        raise ValueError("Use target_return or target_volatility, not both")

    mu, cov = _coerce_inputs(expected_returns, covariance_matrix)
    lower, upper = _bounds(mu.index, long_only, min_weight, max_weight)
    if target_return is not None:
        weights = _solve_min_variance(mu, cov, lower, upper, target_return=target_return)
        if weights is None:
            raise ValueError("target_return is infeasible under the supplied constraints")
    elif target_volatility is not None:
        frontier = generate_efficient_frontier(
            mu,
            cov,
            points=frontier_points,
            long_only=long_only,
            min_weight=lower,
            max_weight=upper,
            risk_free_rate=risk_free_rate,
        )
        feasible = frontier[frontier["volatility"] <= target_volatility + 1e-10]
        if feasible.empty:
            raise ValueError("target_volatility is infeasible under the supplied constraints")
        selected = feasible.sort_values(["expected_return", "sharpe_ratio"], ascending=False).iloc[0]
        weights = np.array([selected[f"weight_{ticker}"] for ticker in mu.index], dtype=float)
    elif objective == "min_volatility":
        weights = _solve_min_variance(mu, cov, lower, upper)
        if weights is None:
            raise ValueError("No feasible portfolio found under the supplied constraints")
    elif objective == "max_sharpe":
        frontier = generate_efficient_frontier(
            mu,
            cov,
            points=frontier_points,
            long_only=long_only,
            min_weight=lower,
            max_weight=upper,
            risk_free_rate=risk_free_rate,
        )
        if frontier.empty:
            raise ValueError("No feasible portfolio found under the supplied constraints")
        selected = frontier.sort_values("sharpe_ratio", ascending=False).iloc[0]
        weights = np.array([selected[f"weight_{ticker}"] for ticker in mu.index], dtype=float)
    else:
        raise ValueError("objective must be 'max_sharpe' or 'min_volatility'")

    metrics = _portfolio_metrics(weights, mu, cov, risk_free_rate)
    target_return_error = metrics["expected_return"] - target_return if target_return is not None else np.nan
    target_vol_error = metrics["volatility"] - target_volatility if target_volatility is not None else np.nan
    return {
        "expected_return": metrics["expected_return"],
        "volatility": metrics["volatility"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "weights": dict(zip(mu.index, weights)),
        "constraint_diagnostics": {
            "full_investment": bool(np.isclose(weights.sum(), 1.0, atol=1e-8)),
            "weight_sum": float(weights.sum()),
            "long_only": bool(np.all(weights >= -1e-10)) if long_only else None,
            "min_weight_satisfied": bool(np.all(weights >= lower - 1e-8)),
            "max_weight_satisfied": bool(np.all(weights <= upper + 1e-8)),
            "target_return_error": float(target_return_error) if target_return is not None else np.nan,
            "target_volatility_error": float(target_vol_error) if target_volatility is not None else np.nan,
        },
    }


def calculate_rebalancing_trades(
    current_weights: dict | pd.Series,
    target_weights: dict | pd.Series,
    portfolio_value: float | None = None,
    prices: dict | pd.Series | None = None,
) -> pd.DataFrame:
    """Compare current and target weights and calculate rebalance trades."""
    current = pd.Series(current_weights, dtype=float)
    target = pd.Series(target_weights, dtype=float)
    tickers = sorted(set(current.index) | set(target.index))
    current = current.reindex(tickers).fillna(0.0)
    target = target.reindex(tickers).fillna(0.0)
    result = pd.DataFrame(
        {
            "ticker": tickers,
            "current_weight": current.to_numpy(dtype=float),
            "target_weight": target.to_numpy(dtype=float),
        }
    )
    result["weight_change"] = result["target_weight"] - result["current_weight"]
    if portfolio_value is not None:
        result["trade_value"] = result["weight_change"] * float(portfolio_value)
    else:
        result["trade_value"] = np.nan
    if prices is not None:
        price_series = pd.Series(prices, dtype=float).reindex(tickers)
        result["price"] = price_series.to_numpy(dtype=float)
        result["quantity_change"] = np.where(result["price"].gt(0), result["trade_value"] / result["price"], np.nan)
    return result


def _coerce_inputs(
    expected_returns: dict | pd.Series,
    covariance_matrix: pd.DataFrame | np.ndarray,
) -> tuple[pd.Series, pd.DataFrame]:
    mu = pd.Series(expected_returns, dtype=float)
    if mu.empty:
        raise ValueError("expected_returns must include at least one asset")
    if isinstance(covariance_matrix, pd.DataFrame):
        cov = covariance_matrix.copy().astype(float)
        missing = set(mu.index) - set(cov.index) | (set(mu.index) - set(cov.columns))
        if missing:
            raise ValueError(f"covariance_matrix is missing assets: {sorted(missing)}")
        cov = cov.reindex(index=mu.index, columns=mu.index)
    else:
        cov = pd.DataFrame(np.asarray(covariance_matrix, dtype=float), index=mu.index, columns=mu.index)
    if cov.shape != (len(mu), len(mu)):
        raise ValueError("covariance_matrix shape must match expected_returns")
    cov = (cov + cov.T) / 2.0
    return mu, cov


def _bounds(
    index: pd.Index,
    long_only: bool,
    min_weight: float | dict | pd.Series,
    max_weight: float | dict | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    lower = _bound_series(index, min_weight, default=0.0 if long_only else -1.0).to_numpy(dtype=float)
    upper = _bound_series(index, max_weight, default=1.0).to_numpy(dtype=float)
    if long_only:
        lower = np.maximum(lower, 0.0)
    if np.any(lower > upper):
        raise ValueError("min_weight cannot exceed max_weight")
    if lower.sum() > 1.0 + 1e-10 or upper.sum() < 1.0 - 1e-10:
        raise ValueError("weight bounds cannot satisfy full investment")
    return lower, upper


def _bound_series(index: pd.Index, value: float | dict | pd.Series, default: float) -> pd.Series:
    if np.isscalar(value):
        return pd.Series(float(value), index=index)
    if isinstance(value, np.ndarray) or isinstance(value, (list, tuple)):
        values = list(value)
        if len(values) != len(index):
            raise ValueError("weight bound arrays must match expected_returns length")
        return pd.Series(values, index=index, dtype=float)
    return pd.Series(value, dtype=float).reindex(index).fillna(default)


def _solve_min_variance(
    mu: pd.Series,
    cov: pd.DataFrame,
    lower: np.ndarray,
    upper: np.ndarray,
    target_return: float | None = None,
) -> np.ndarray | None:
    n_assets = len(mu)
    constraints = [np.ones(n_assets)]
    rhs = [1.0]
    if target_return is not None:
        constraints.append(mu.to_numpy(dtype=float))
        rhs.append(float(target_return))
    a_matrix = np.vstack(constraints)
    b_vector = np.array(rhs, dtype=float)
    cov_values = cov.to_numpy(dtype=float) + np.eye(n_assets) * 1e-12

    best_weights = None
    best_variance = np.inf
    for states in product((0, -1, 1), repeat=n_assets):
        states_array = np.array(states)
        free = states_array == 0
        fixed = ~free
        if free.sum() < len(rhs):
            continue

        weights = np.zeros(n_assets)
        weights[states_array == -1] = lower[states_array == -1]
        weights[states_array == 1] = upper[states_array == 1]
        adjusted_rhs = b_vector - a_matrix[:, fixed] @ weights[fixed]
        a_free = a_matrix[:, free]
        cov_free = cov_values[np.ix_(free, free)]
        cross = cov_values[np.ix_(free, fixed)] @ weights[fixed]
        kkt = np.block(
            [
                [cov_free, a_free.T],
                [a_free, np.zeros((a_free.shape[0], a_free.shape[0]))],
            ]
        )
        rhs_kkt = np.concatenate([-cross, adjusted_rhs])
        try:
            solution = np.linalg.solve(kkt, rhs_kkt)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(kkt, rhs_kkt, rcond=None)[0]
        weights[free] = solution[: free.sum()]
        if not _is_feasible(weights, lower, upper, a_matrix, b_vector):
            continue

        variance = float(weights @ cov_values @ weights)
        if variance < best_variance:
            best_variance = variance
            best_weights = weights
    return best_weights


def _is_feasible(
    weights: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    a_matrix: np.ndarray,
    b_vector: np.ndarray,
) -> bool:
    return bool(
        np.all(weights >= lower - 1e-8)
        and np.all(weights <= upper + 1e-8)
        and np.allclose(a_matrix @ weights, b_vector, atol=1e-8)
    )


def _extreme_return(mu: pd.Series, lower: np.ndarray, upper: np.ndarray, maximize: bool) -> float:
    weights = lower.copy()
    remaining = 1.0 - weights.sum()
    order = np.argsort(mu.to_numpy(dtype=float))
    if maximize:
        order = order[::-1]
    for idx in order:
        add = min(upper[idx] - lower[idx], remaining)
        weights[idx] += add
        remaining -= add
        if remaining <= 1e-12:
            break
    if remaining > 1e-8:
        raise ValueError("weight bounds cannot satisfy full investment")
    return float(weights @ mu.to_numpy(dtype=float))


def _portfolio_metrics(weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame, risk_free_rate: float) -> dict[str, float]:
    expected_return = float(weights @ mu.to_numpy(dtype=float))
    variance = float(weights @ cov.to_numpy(dtype=float) @ weights)
    volatility = float(np.sqrt(max(variance, 0.0)))
    sharpe = (expected_return - risk_free_rate) / volatility if volatility > 0 else float("nan")
    return {"expected_return": expected_return, "volatility": volatility, "sharpe_ratio": float(sharpe)}
