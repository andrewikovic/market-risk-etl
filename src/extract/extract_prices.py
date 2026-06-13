from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
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

VALID_YFINANCE_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
_PERIOD_PATTERN = re.compile(r"^(?P<count>\d+)(?P<unit>d|mo|y)$")


@dataclass(frozen=True)
class PriceWindow:
    """Resolved historical price window using yfinance's exclusive end-date semantics."""

    start: date | None = None
    end: date | None = None
    period: str | None = None

    @property
    def start_arg(self) -> str | None:
        return self.start.isoformat() if self.start else None

    @property
    def end_arg(self) -> str | None:
        return self.end.isoformat() if self.end else None


def resolve_price_window(
    start: str | date | datetime | None = None,
    end: str | date | datetime | None = None,
    lookback_days: int | None = None,
    period: str | None = None,
    today: date | None = None,
) -> PriceWindow:
    """Resolve user-facing price history controls into a concrete fetch window."""
    start_date = _parse_price_date(start, "start") if start else None
    end_date = _parse_price_date(end, "end") if end else None
    normalized_period = _normalize_period(period)

    if lookback_days is not None:
        lookback_days = _validate_lookback_days(lookback_days)

    if normalized_period:
        if start_date or end_date or lookback_days is not None:
            raise ValueError("price_period cannot be combined with price_start, price_end, or price_lookback_days")
        return PriceWindow(period=normalized_period)

    if lookback_days is not None:
        anchor_date = end_date or today or datetime.now(UTC).date()
        start_date = anchor_date - timedelta(days=lookback_days)

    if start_date and end_date and start_date >= end_date:
        raise ValueError("price_start must be before price_end")

    return PriceWindow(start=start_date, end=end_date)


def filter_prices_by_window(
    prices: pd.DataFrame,
    start: str | date | datetime | None = None,
    end: str | date | datetime | None = None,
    lookback_days: int | None = None,
    period: str | None = None,
) -> pd.DataFrame:
    """Filter local fallback prices using the same controls as yfinance ingestion."""
    if prices.empty:
        return prices

    normalized_period = _normalize_period(period)
    if normalized_period:
        resolve_price_window(start=start, end=end, lookback_days=lookback_days, period=normalized_period)
        window = _period_to_csv_window(normalized_period, prices)
    else:
        window = resolve_price_window(
            start=start,
            end=end,
            lookback_days=lookback_days,
            today=_latest_exclusive_price_date(prices),
        )

    filtered = prices
    if window.start:
        filtered = filtered[filtered["price_date"] >= window.start]
    if window.end:
        filtered = filtered[filtered["price_date"] < window.end]
    return filtered


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
    lookback_days: int | None = None,
    period: str | None = None,
) -> pd.DataFrame:
    """Fetch daily adjusted close prices from Yahoo Finance via yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed") from exc

    ticker_list = sorted({clean for ticker in tickers if (clean := ticker.upper().strip())})
    if not ticker_list:
        raise ValueError("At least one ticker is required")

    window = resolve_price_window(start=start, end=end, lookback_days=lookback_days, period=period)
    download_kwargs = {
        "tickers": ticker_list,
        "auto_adjust": False,
        "progress": False,
        "group_by": "ticker",
        "threads": True,
    }
    if window.period:
        download_kwargs["period"] = window.period
    elif window.start or window.end:
        if window.start:
            download_kwargs["start"] = window.start_arg
        if window.end:
            download_kwargs["end"] = window.end_arg
    else:
        download_kwargs["period"] = "max"

    raw = yf.download(**download_kwargs)
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
    lookback_days: int | None = None,
    period: str | None = None,
    csv_fallback_path: str | Path | None = None,
    prefer_live: bool = True,
    allow_fallback: bool = True,
) -> pd.DataFrame:
    """Ingest prices from yfinance with an optional CSV fallback for offline development."""
    ticker_set = {clean for ticker in tickers if (clean := ticker.upper().strip())}
    if prefer_live:
        try:
            return fetch_prices_yfinance(
                ticker_set,
                start=start,
                end=end,
                lookback_days=lookback_days,
                period=period,
            )
        except Exception as exc:
            if not allow_fallback or csv_fallback_path is None:
                raise
            print(f"Live yfinance ingestion failed; falling back to CSV: {exc}")

    if csv_fallback_path is None:
        raise ValueError("csv_fallback_path is required when live ingestion is disabled")

    prices = load_prices_from_csv(csv_fallback_path)
    prices = prices[prices["ticker"].isin(ticker_set)]
    prices = filter_prices_by_window(
        prices,
        start=start,
        end=end,
        lookback_days=lookback_days,
        period=period,
    )
    if prices.empty:
        raise ValueError("CSV fallback produced no matching price rows")
    return prices.reset_index(drop=True)


def _parse_price_date(value: str | date | datetime, label: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        parsed = pd.to_datetime(str(value), errors="raise")
    except Exception as exc:
        raise ValueError(f"{label} must be a valid date") from exc
    return parsed.date()


def _validate_lookback_days(value: int) -> int:
    try:
        lookback_days = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("price_lookback_days must be an integer") from exc
    if lookback_days <= 0:
        raise ValueError("price_lookback_days must be greater than zero")
    return lookback_days


def _normalize_period(period: str | None) -> str | None:
    if period is None:
        return None
    normalized = period.strip().lower()
    if not normalized:
        return None
    if normalized not in VALID_YFINANCE_PERIODS:
        valid = ", ".join(sorted(VALID_YFINANCE_PERIODS))
        raise ValueError(f"price_period must be one of: {valid}")
    return normalized


def _period_to_csv_window(period: str | None, prices: pd.DataFrame) -> PriceWindow:
    normalized = _normalize_period(period)
    if not normalized or normalized == "max":
        return PriceWindow()

    anchor_date = _latest_exclusive_price_date(prices)
    if normalized == "ytd":
        return PriceWindow(start=date(anchor_date.year, 1, 1))

    match = _PERIOD_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError(f"Unsupported price_period for CSV fallback: {normalized}")

    count = int(match.group("count"))
    unit = match.group("unit")
    if unit == "d":
        start_date = anchor_date - timedelta(days=count)
    elif unit == "mo":
        start_date = (pd.Timestamp(anchor_date) - pd.DateOffset(months=count)).date()
    else:
        start_date = (pd.Timestamp(anchor_date) - pd.DateOffset(years=count)).date()
    return PriceWindow(start=start_date)


def _latest_exclusive_price_date(prices: pd.DataFrame) -> date:
    latest = pd.to_datetime(prices["price_date"], errors="coerce").max()
    if pd.isna(latest):
        return datetime.now(UTC).date()
    return latest.date() + timedelta(days=1)
