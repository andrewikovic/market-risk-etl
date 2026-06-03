from __future__ import annotations

import math

import numpy as np
import pandas as pd


def calculate_var_breaches(
    realized_pnl: pd.Series | pd.DataFrame,
    var_estimates: float | pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    """Track VaR exceptions where realized P&L is worse than the VaR loss estimate."""
    aligned = _align_pnl_and_var(realized_pnl, var_estimates)
    aligned["breach"] = aligned["realized_pnl"] < -aligned["var_estimate"]
    losses = -aligned["realized_pnl"]
    aligned["breach_severity"] = np.where(aligned["breach"], losses - aligned["var_estimate"], 0.0)
    return aligned[["date", "var_estimate", "realized_pnl", "breach", "breach_severity"]].reset_index(drop=True)


def kupiec_pof_test(
    exceptions: int,
    observations: int,
    confidence_level: float = 0.95,
    test_alpha: float = 0.05,
) -> dict[str, float | str | int]:
    """Run Kupiec's unconditional proportion-of-failures test."""
    if observations < 0 or exceptions < 0 or exceptions > observations:
        raise ValueError("exceptions must be between zero and observations")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    if not 0 < test_alpha < 1:
        raise ValueError("test_alpha must be between 0 and 1")
    if observations == 0:
        return {
            "total_observations": 0,
            "number_of_exceptions": 0,
            "expected_exceptions": 0.0,
            "exception_ratio": float("nan"),
            "kupiec_statistic": float("nan"),
            "p_value": float("nan"),
            "pass_fail": "fail",
        }

    expected_probability = 1.0 - confidence_level
    observed_probability = exceptions / observations
    log_null = _binomial_log_likelihood(exceptions, observations, expected_probability)
    log_alt = _binomial_log_likelihood(exceptions, observations, observed_probability)
    statistic = max(-2.0 * (log_null - log_alt), 0.0)
    p_value = math.erfc(math.sqrt(statistic / 2.0))
    expected_exceptions = observations * expected_probability

    return {
        "total_observations": observations,
        "number_of_exceptions": exceptions,
        "expected_exceptions": float(expected_exceptions),
        "exception_ratio": float(exceptions / expected_exceptions) if expected_exceptions else float("nan"),
        "kupiec_statistic": float(statistic),
        "p_value": float(p_value),
        "pass_fail": "pass" if p_value >= test_alpha else "fail",
    }


def generate_exception_report(
    realized_pnl: pd.Series | pd.DataFrame,
    var_estimates: float | pd.Series | pd.DataFrame,
    confidence_level: float = 0.95,
    test_alpha: float = 0.05,
    portfolio_name: str | None = None,
) -> pd.DataFrame:
    """Generate row-level VaR exceptions with Kupiec backtest summary fields."""
    breaches = calculate_var_breaches(realized_pnl, var_estimates)
    summary = kupiec_pof_test(
        exceptions=int(breaches["breach"].sum()),
        observations=len(breaches),
        confidence_level=confidence_level,
        test_alpha=test_alpha,
    )
    report = breaches.copy()
    report["confidence_level"] = confidence_level
    if portfolio_name is not None:
        report["portfolio_name"] = portfolio_name
    for key, value in summary.items():
        report[key] = value
    return report


def _align_pnl_and_var(
    realized_pnl: pd.Series | pd.DataFrame,
    var_estimates: float | pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    pnl = _coerce_measure(realized_pnl, ["realized_pnl", "daily_pnl", "pnl", "P&L"], "realized_pnl")
    if np.isscalar(var_estimates):
        result = pnl.copy()
        result["var_estimate"] = float(var_estimates)
    else:
        var = _coerce_measure(var_estimates, ["var_estimate", "historical_var", "parametric_var", "VaR"], "var_estimate")
        result = pnl.merge(var, on="date", how="inner")
    result["var_estimate"] = pd.to_numeric(result["var_estimate"], errors="coerce").abs()
    result["realized_pnl"] = pd.to_numeric(result["realized_pnl"], errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=["var_estimate", "realized_pnl"])
    return result.sort_values("date").reset_index(drop=True)


def _coerce_measure(data: pd.Series | pd.DataFrame, value_columns: list[str], output_col: str) -> pd.DataFrame:
    if isinstance(data, pd.Series):
        frame = data.rename(output_col).reset_index()
        frame = frame.rename(columns={frame.columns[0]: "date"})
    else:
        date_col = next((col for col in ["date", "value_date", "return_date", "metric_date"] if col in data.columns), None)
        value_col = next((col for col in value_columns if col in data.columns), None)
        if date_col is None:
            raise ValueError("DataFrame inputs must include date, value_date, return_date, or metric_date")
        if value_col is None:
            raise ValueError(f"DataFrame input must include one of: {value_columns}")
        frame = data[[date_col, value_col]].copy().rename(columns={date_col: "date", value_col: output_col})
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame


def _binomial_log_likelihood(exceptions: int, observations: int, probability: float) -> float:
    non_exceptions = observations - exceptions
    if probability <= 0:
        return 0.0 if exceptions == 0 else float("-inf")
    if probability >= 1:
        return 0.0 if non_exceptions == 0 else float("-inf")
    return exceptions * math.log(probability) + non_exceptions * math.log(1.0 - probability)
