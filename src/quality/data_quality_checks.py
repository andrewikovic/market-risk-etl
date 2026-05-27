from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd


def run_data_quality_checks(
    prices_df: pd.DataFrame,
    positions_df: pd.DataFrame | None = None,
    asset_metadata_df: pd.DataFrame | None = None,
    returns_df: pd.DataFrame | None = None,
    min_history: int = 60,
    outlier_threshold: float = 0.20,
) -> pd.DataFrame:
    """Run market-risk ETL data quality checks and return result rows."""
    run_timestamp = datetime.now(UTC)
    results: list[dict] = []

    if prices_df is None or prices_df.empty:
        results.append(_result("failed_api_pulls", "FAIL", "high", None, None, "No price data available", run_timestamp))
        return pd.DataFrame(results)

    prices = prices_df.copy()
    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip()
    prices["price_date"] = pd.to_datetime(prices["price_date"]).dt.date
    prices["adjusted_close"] = pd.to_numeric(prices["adjusted_close"], errors="coerce")

    _check_missing_prices(prices, results, run_timestamp)
    _check_duplicate_rows(prices, results, run_timestamp)
    _check_stale_prices(prices, results, run_timestamp)
    _check_negative_prices(prices, results, run_timestamp)
    _check_zero_prices(prices, results, run_timestamp)
    _check_insufficient_history(prices, results, min_history, run_timestamp)

    if positions_df is not None and asset_metadata_df is not None:
        _check_missing_metadata(positions_df, asset_metadata_df, results, run_timestamp)

    if returns_df is not None:
        _check_outlier_returns(returns_df, results, outlier_threshold, run_timestamp)

    if positions_df is not None:
        _check_portfolio_weights(positions_df, results, run_timestamp)

    return pd.DataFrame(results)[
        ["check_name", "status", "severity", "ticker", "check_date", "message", "run_timestamp"]
    ]


def _result(check_name, status, severity, ticker, check_date, message, run_timestamp) -> dict:
    return {
        "check_name": check_name,
        "status": status,
        "severity": severity,
        "ticker": ticker,
        "check_date": check_date,
        "message": message,
        "run_timestamp": run_timestamp,
    }


def _pass(results: list[dict], check_name: str, message: str, run_timestamp) -> None:
    results.append(_result(check_name, "PASS", "info", None, None, message, run_timestamp))


def _check_missing_prices(prices: pd.DataFrame, results: list[dict], run_timestamp) -> None:
    missing = prices[prices["adjusted_close"].isna()]
    if missing.empty:
        _pass(results, "missing_prices", "No missing adjusted close prices detected", run_timestamp)
        return
    for _, row in missing.iterrows():
        results.append(
            _result("missing_prices", "FAIL", "high", row["ticker"], row["price_date"], "Missing adjusted close", run_timestamp)
        )


def _check_duplicate_rows(prices: pd.DataFrame, results: list[dict], run_timestamp) -> None:
    duplicates = prices[prices.duplicated(["ticker", "price_date"], keep=False)]
    if duplicates.empty:
        _pass(results, "duplicate_rows", "No duplicate ticker/date rows detected", run_timestamp)
        return
    for _, row in duplicates.drop_duplicates(["ticker", "price_date"]).iterrows():
        results.append(
            _result("duplicate_rows", "FAIL", "high", row["ticker"], row["price_date"], "Duplicate ticker/date rows", run_timestamp)
        )


def _check_stale_prices(prices: pd.DataFrame, results: list[dict], run_timestamp) -> None:
    ordered = prices.sort_values(["ticker", "price_date"]).copy()
    ordered["is_stale"] = ordered.groupby("ticker")["adjusted_close"].transform(
        lambda values: values.eq(values.shift(1)) & values.notna()
    )
    stale = ordered[ordered["is_stale"]]
    if stale.empty:
        _pass(results, "stale_prices", "No repeated stale prices detected", run_timestamp)
        return
    for _, row in stale.iterrows():
        results.append(
            _result("stale_prices", "WARN", "medium", row["ticker"], row["price_date"], "Price equals prior observation", run_timestamp)
        )


def _check_negative_prices(prices: pd.DataFrame, results: list[dict], run_timestamp) -> None:
    negative = prices[prices["adjusted_close"].lt(0)]
    if negative.empty:
        _pass(results, "negative_prices", "No negative prices detected", run_timestamp)
        return
    for _, row in negative.iterrows():
        results.append(
            _result("negative_prices", "FAIL", "high", row["ticker"], row["price_date"], "Negative adjusted close", run_timestamp)
        )


def _check_zero_prices(prices: pd.DataFrame, results: list[dict], run_timestamp) -> None:
    zero = prices[prices["adjusted_close"].eq(0)]
    if zero.empty:
        _pass(results, "zero_prices", "No zero prices detected", run_timestamp)
        return
    for _, row in zero.iterrows():
        results.append(_result("zero_prices", "FAIL", "high", row["ticker"], row["price_date"], "Zero adjusted close", run_timestamp))


def _check_missing_metadata(
    positions: pd.DataFrame,
    metadata: pd.DataFrame,
    results: list[dict],
    run_timestamp,
) -> None:
    position_tickers = set(positions["ticker"].astype(str).str.upper())
    metadata_tickers = set(metadata["ticker"].astype(str).str.upper())
    missing = sorted(position_tickers - metadata_tickers)
    if not missing:
        _pass(results, "missing_metadata", "All position tickers have metadata", run_timestamp)
        return
    for ticker in missing:
        results.append(_result("missing_metadata", "FAIL", "high", ticker, None, "Ticker missing asset metadata", run_timestamp))


def _check_insufficient_history(
    prices: pd.DataFrame,
    results: list[dict],
    min_history: int,
    run_timestamp,
) -> None:
    counts = prices.dropna(subset=["adjusted_close"]).groupby("ticker").size()
    insufficient = counts[counts < min_history]
    if insufficient.empty:
        _pass(results, "insufficient_history", f"All assets have at least {min_history} observations", run_timestamp)
        return
    for ticker, count in insufficient.items():
        results.append(
            _result(
                "insufficient_history",
                "WARN",
                "medium",
                ticker,
                None,
                f"Only {count} price observations; expected at least {min_history}",
                run_timestamp,
            )
        )


def _check_outlier_returns(
    returns: pd.DataFrame,
    results: list[dict],
    threshold: float,
    run_timestamp,
) -> None:
    data = returns.copy()
    value_col = "daily_return" if "daily_return" in data.columns else "return"
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    if "return_date" in data.columns:
        data["return_date"] = pd.to_datetime(data["return_date"]).dt.date
    outliers = data[data[value_col].abs().gt(threshold)]
    if outliers.empty:
        _pass(results, "extreme_return_outliers", "No extreme return outliers detected", run_timestamp)
        return
    for _, row in outliers.iterrows():
        ticker = row.get("ticker")
        check_date = row.get("return_date")
        message = f"Return {row[value_col]:.2%} exceeds threshold {threshold:.2%}"
        results.append(_result("extreme_return_outliers", "WARN", "medium", ticker, check_date, message, run_timestamp))


def _check_portfolio_weights(positions: pd.DataFrame, results: list[dict], run_timestamp) -> None:
    data = positions.copy()
    if "weight" in data.columns:
        weight_sum = pd.to_numeric(data["weight"], errors="coerce").sum()
    elif "position_value" in data.columns:
        total = pd.to_numeric(data["position_value"], errors="coerce").sum()
        weight_sum = 1.0 if total != 0 else np.nan
    else:
        _pass(results, "portfolio_weights_sum", "Portfolio weights unavailable; check skipped", run_timestamp)
        return

    if np.isclose(weight_sum, 1.0, atol=1e-6):
        _pass(results, "portfolio_weights_sum", "Portfolio weights sum to 1", run_timestamp)
    else:
        results.append(
            _result(
                "portfolio_weights_sum",
                "FAIL",
                "high",
                None,
                None,
                f"Portfolio weights sum to {weight_sum:.6f}, expected 1",
                run_timestamp,
            )
        )

