from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.extract.extract_prices import ingest_prices, load_assets_config


def benchmark_tickers_from_assets(assets_df: pd.DataFrame) -> list[str]:
    """Return unique benchmark tickers configured for the asset universe."""
    if "benchmark_ticker" not in assets_df:
        return []
    return sorted(
        ticker
        for ticker in assets_df["benchmark_ticker"].dropna().astype(str).str.upper().unique()
        if ticker
    )


def ingest_benchmark_prices(
    assets_config_path: str | Path,
    start: str | None = None,
    end: str | None = None,
    csv_fallback_path: str | Path | None = None,
    prefer_live: bool = True,
) -> pd.DataFrame:
    """Ingest benchmark prices listed in the asset metadata config."""
    assets = load_assets_config(assets_config_path)
    tickers = benchmark_tickers_from_assets(assets)
    if not tickers:
        raise ValueError("No benchmark_ticker values found in assets config")
    return ingest_prices(
        tickers=tickers,
        start=start,
        end=end,
        csv_fallback_path=csv_fallback_path,
        prefer_live=prefer_live,
    )

