import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from src.extract.extract_prices import PRICE_COLUMNS, fetch_prices_yfinance, ingest_prices, resolve_price_window


def test_yfinance_fetch_defaults_to_full_history(monkeypatch):
    captured = {}

    def fake_download(**kwargs):
        captured.update(kwargs)
        return _yfinance_frame("AAPL")

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=fake_download))

    prices = fetch_prices_yfinance(["aapl"])

    assert captured["period"] == "max"
    assert "start" not in captured
    assert "end" not in captured
    assert prices["ticker"].tolist() == ["AAPL", "AAPL"]


def test_yfinance_fetch_uses_explicit_date_window(monkeypatch):
    captured = {}

    def fake_download(**kwargs):
        captured.update(kwargs)
        return _yfinance_frame("AAPL")

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=fake_download))

    fetch_prices_yfinance(["AAPL"], start="2024-01-01", end="2024-02-01")

    assert captured["start"] == "2024-01-01"
    assert captured["end"] == "2024-02-01"
    assert "period" not in captured


def test_price_period_cannot_be_combined_with_date_controls():
    with pytest.raises(ValueError, match="price_period cannot be combined"):
        resolve_price_window(start="2024-01-01", period="max")


def test_csv_fallback_filters_latest_lookback_days(tmp_path):
    csv_path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "ticker": ["AAPL"] * 5,
            "price_date": pd.date_range("2024-01-01", periods=5),
            "open_price": [100, 101, 102, 103, 104],
            "high_price": [101, 102, 103, 104, 105],
            "low_price": [99, 100, 101, 102, 103],
            "close_price": [100, 101, 102, 103, 104],
            "adjusted_close": [100, 101, 102, 103, 104],
            "volume": [1000, 1001, 1002, 1003, 1004],
            "source": ["sample"] * 5,
        }
    )[PRICE_COLUMNS].to_csv(csv_path, index=False)

    prices = ingest_prices(["AAPL"], lookback_days=2, csv_fallback_path=csv_path, prefer_live=False)

    assert prices["price_date"].tolist() == [pd.Timestamp("2024-01-04").date(), pd.Timestamp("2024-01-05").date()]


def _yfinance_frame(ticker: str) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=2)
    columns = pd.MultiIndex.from_product(
        [[ticker], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    )
    return pd.DataFrame(
        [
            [100.0, 101.0, 99.0, 100.5, 100.5, 1000],
            [101.0, 102.0, 100.0, 101.5, 101.5, 1100],
        ],
        index=dates,
        columns=columns,
    )
