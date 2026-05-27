from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


PRICE_COLUMNS = [
    "ticker",
    "price_date",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "adjusted_close",
    "volume",
    "source",
]


def load_assets_config(config_path: str | Path) -> pd.DataFrame:
    """Load asset metadata from a YAML config file."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    assets = payload.get("assets", [])
    if not assets:
        raise ValueError(f"No assets found in {config_path}")
    return pd.DataFrame(assets)


def load_prices_from_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load raw OHLCV prices from a CSV fallback file."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Price CSV not found: {path}")

    prices = pd.read_csv(path, parse_dates=["price_date"])
    missing = set(PRICE_COLUMNS) - set(prices.columns)
    if missing:
        raise ValueError(f"Price CSV is missing columns: {sorted(missing)}")

    prices = prices[PRICE_COLUMNS].copy()
    prices["ticker"] = prices["ticker"].str.upper().str.strip()
    prices["price_date"] = pd.to_datetime(prices["price_date"]).dt.date
    prices["ingested_at"] = datetime.now(UTC)
    return prices.sort_values(["ticker", "price_date"]).reset_index(drop=True)


def fetch_prices_yfinance(
    tickers: Iterable[str],
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch daily adjusted close prices from Yahoo Finance via yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed") from exc

    ticker_list = sorted({ticker.upper().strip() for ticker in tickers})
    if not ticker_list:
        raise ValueError("At least one ticker is required")

    raw = yf.download(
        tickers=ticker_list,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw.empty:
        raise ValueError("yfinance returned no price rows")

    frames: list[pd.DataFrame] = []
    for ticker in ticker_list:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker not in raw.columns.get_level_values(0):
                continue
            data = raw[ticker].copy()
        else:
            data = raw.copy()

        column_map = {
            "Open": "open_price",
            "High": "high_price",
            "Low": "low_price",
            "Close": "close_price",
            "Adj Close": "adjusted_close",
            "Volume": "volume",
        }
        data = data.rename(columns=column_map)
        required = list(column_map.values())
        if not set(required).issubset(data.columns):
            continue
        data = data[required].reset_index()
        date_column = data.columns[0]
        data = data.rename(columns={date_column: "price_date"})
        data["ticker"] = ticker
        data["source"] = "yfinance"
        frames.append(data[PRICE_COLUMNS])

    if not frames:
        raise ValueError("No yfinance ticker payloads could be normalized")

    prices = pd.concat(frames, ignore_index=True)
    prices["price_date"] = pd.to_datetime(prices["price_date"]).dt.date
    prices["ingested_at"] = datetime.now(UTC)
    return prices.sort_values(["ticker", "price_date"]).reset_index(drop=True)


def ingest_prices(
    tickers: Iterable[str],
    start: str | None = None,
    end: str | None = None,
    csv_fallback_path: str | Path | None = None,
    prefer_live: bool = True,
    allow_fallback: bool = True,
) -> pd.DataFrame:
    """Ingest prices from yfinance with an optional CSV fallback for offline development."""
    ticker_set = {ticker.upper().strip() for ticker in tickers}
    if prefer_live:
        try:
            return fetch_prices_yfinance(ticker_set, start=start, end=end)
        except Exception as exc:
            if not allow_fallback or csv_fallback_path is None:
                raise
            print(f"Live yfinance ingestion failed; falling back to CSV: {exc}")

    if csv_fallback_path is None:
        raise ValueError("csv_fallback_path is required when live ingestion is disabled")

    prices = load_prices_from_csv(csv_fallback_path)
    prices = prices[prices["ticker"].isin(ticker_set)]
    if start:
        prices = prices[prices["price_date"] >= pd.to_datetime(start).date()]
    if end:
        prices = prices[prices["price_date"] < pd.to_datetime(end).date()]
    if prices.empty:
        raise ValueError("CSV fallback produced no matching price rows")
    return prices.reset_index(drop=True)
