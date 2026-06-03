from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.transform.calculate_volatility import TRADING_DAYS


def estimate_factor_exposures(
    asset_returns_df: pd.DataFrame,
    factor_returns_df: pd.DataFrame,
    weights: dict | pd.Series | None = None,
    include_intercept: bool = True,
    annualize_residual: bool = True,
) -> dict[str, pd.DataFrame]:
    """Estimate asset and optional portfolio factor exposures using OLS regression."""
    asset_returns = _asset_return_matrix(asset_returns_df)
    factor_returns = _factor_return_matrix(factor_returns_df)
    if asset_returns.empty:
        raise ValueError("asset_returns_df has no aligned asset returns")
    if factor_returns.empty:
        raise ValueError("factor_returns_df has no factor return columns")

    aligned = asset_returns.join(factor_returns, how="inner").replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if aligned.empty:
        raise ValueError("asset and factor returns have no overlapping complete observations")

    factor_names = list(factor_returns.columns)
    factor_values = aligned[factor_names].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(aligned)), factor_values]) if include_intercept else factor_values

    exposure_rows: list[dict] = []
    summary_rows: list[dict] = []
    for ticker in asset_returns.columns:
        y = aligned[ticker].to_numpy(dtype=float)
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        offset = 1 if include_intercept else 0
        alpha = float(coefficients[0]) if include_intercept else 0.0
        betas = coefficients[offset:]
        fitted = design @ coefficients
        residuals = y - fitted
        residual_std = float(pd.Series(residuals).std(ddof=1))
        residual_volatility = residual_std * math.sqrt(TRADING_DAYS) if annualize_residual else residual_std
        total_sse = float(np.sum((y - y.mean()) ** 2))
        residual_sse = float(np.sum(residuals**2))
        r_squared = 1.0 - residual_sse / total_sse if total_sse > 0 else float("nan")

        for factor, beta in zip(factor_names, betas):
            exposure_rows.append({"ticker": ticker, "factor": factor, "beta": float(beta)})
        summary_rows.append(
            {
                "ticker": ticker,
                "alpha": alpha,
                "residual_volatility": residual_volatility,
                "idiosyncratic_variance": residual_std**2,
                "r_squared": float(r_squared),
                "observations": len(aligned),
            }
        )

    asset_exposures = pd.DataFrame(exposure_rows)
    asset_summary = pd.DataFrame(summary_rows)
    result = {
        "asset_exposures": asset_exposures,
        "asset_summary": asset_summary,
        "asset_factor_matrix": asset_exposures.pivot(index="ticker", columns="factor", values="beta").reset_index(),
    }

    if weights is not None:
        result["portfolio_factor_exposure"] = _portfolio_factor_exposure(asset_exposures, weights)
    return result


def _portfolio_factor_exposure(asset_exposures: pd.DataFrame, weights: dict | pd.Series) -> pd.DataFrame:
    matrix = asset_exposures.pivot(index="ticker", columns="factor", values="beta")
    weight_series = pd.Series(weights, dtype=float)
    missing = set(weight_series.index) - set(matrix.index)
    if missing:
        raise ValueError(f"Missing factor exposures for weighted assets: {sorted(missing)}")
    weight_series = weight_series.reindex(matrix.index).fillna(0.0)
    total = weight_series.sum()
    if total == 0 or np.isnan(total):
        raise ValueError("weights must sum to a non-zero value")
    if not np.isclose(total, 1.0):
        weight_series = weight_series / total
    portfolio_betas = matrix.multiply(weight_series, axis=0).sum(axis=0)
    return portfolio_betas.rename("portfolio_beta").reset_index().rename(columns={"index": "factor"})


def _asset_return_matrix(asset_returns_df: pd.DataFrame) -> pd.DataFrame:
    if {"ticker", "return_date", "daily_return"}.issubset(asset_returns_df.columns):
        matrix = asset_returns_df.pivot_table(
            index="return_date",
            columns="ticker",
            values="daily_return",
            aggfunc="last",
        )
    else:
        matrix = asset_returns_df.copy()
        date_col = next((col for col in ["return_date", "date", "value_date"] if col in matrix.columns), None)
        if date_col is not None:
            matrix = matrix.set_index(date_col)
    matrix.index = pd.to_datetime(matrix.index)
    matrix = matrix.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return matrix.dropna(axis=1, how="all").sort_index()


def _factor_return_matrix(factor_returns_df: pd.DataFrame) -> pd.DataFrame:
    if {"return_date", "factor", "factor_return"}.issubset(factor_returns_df.columns):
        matrix = factor_returns_df.pivot_table(
            index="return_date",
            columns="factor",
            values="factor_return",
            aggfunc="last",
        )
    else:
        matrix = factor_returns_df.copy()
        date_col = next((col for col in ["return_date", "date", "value_date"] if col in matrix.columns), None)
        if date_col is not None:
            matrix = matrix.set_index(date_col)
    matrix.index = pd.to_datetime(matrix.index)
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    return matrix.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all").sort_index()
