from types import SimpleNamespace

import pandas as pd
import pytest

from src.extract import extract_benchmarks, extract_factors
from src.extract.extract_prices import (
    PRICE_COLUMNS,
    fetch_prices_yfinance,
    filter_prices_by_window,
    ingest_prices,
    load_assets_config,
    load_prices_from_csv,
    resolve_price_window,
)


def test_benchmark_tickers_from_assets_normalizes_unique_values():
    assets = pd.DataFrame({"benchmark_ticker": ["spy", "SPY", " tlt ", None, ""]})

    assert extract_benchmarks.benchmark_tickers_from_assets(assets) == ["SPY", "TLT"]
    assert extract_benchmarks.benchmark_tickers_from_assets(pd.DataFrame({"ticker": ["A"]})) == []


def test_ingest_benchmark_prices_uses_configured_benchmarks(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        extract_benchmarks,
        "load_assets_config",
        lambda path: pd.DataFrame({"benchmark_ticker": ["spy", "tlt"]}),
    )
    monkeypatch.setattr(
        extract_benchmarks,
        "ingest_prices",
        lambda **kwargs: captured.update(kwargs) or pd.DataFrame({"ticker": kwargs["tickers"]}),
    )

    result = extract_benchmarks.ingest_benchmark_prices(
        "assets.yml",
        start="2024-01-01",
        end="2024-02-01",
        csv_fallback_path="prices.csv",
        prefer_live=False,
    )

    assert captured == {
        "tickers": ["SPY", "TLT"],
        "start": "2024-01-01",
        "end": "2024-02-01",
        "csv_fallback_path": "prices.csv",
        "prefer_live": False,
    }
    assert result["ticker"].tolist() == ["SPY", "TLT"]


def test_ingest_benchmark_prices_requires_benchmarks(monkeypatch):
    monkeypatch.setattr(extract_benchmarks, "load_assets_config", lambda path: pd.DataFrame({"ticker": ["A"]}))

    with pytest.raises(ValueError, match="No benchmark_ticker"):
        extract_benchmarks.ingest_benchmark_prices("assets.yml")


def test_factor_csv_loader_sorts_dates_and_estimator_uses_loaded_factors(tmp_path, monkeypatch):
    csv_path = tmp_path / "factors.csv"
    pd.DataFrame(
        {
            "return_date": ["2024-01-03", "2024-01-01"],
            "market": [0.02, 0.01],
        }
    ).to_csv(csv_path, index=False)

    loaded = extract_factors.load_factor_returns_from_csv(csv_path)

    assert loaded["return_date"].tolist() == [pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-03").date()]

    captured = {}
    monkeypatch.setattr(
        extract_factors,
        "estimate_factor_exposures",
        lambda asset_returns, factor_returns, weights=None: captured.update(
            asset_returns=asset_returns, factor_returns=factor_returns, weights=weights
        )
        or {"asset_exposures": pd.DataFrame({"ticker": ["A"]})},
    )
    asset_returns = pd.DataFrame({"return_date": loaded["return_date"], "ticker": ["A", "A"], "daily_return": [0.01, 0.02]})

    result = extract_factors.estimate_factor_exposures_from_csv(asset_returns, csv_path, weights={"A": 1.0})

    assert result["asset_exposures"]["ticker"].tolist() == ["A"]
    assert captured["weights"] == {"A": 1.0}
    pd.testing.assert_frame_equal(captured["factor_returns"], loaded)


def test_factor_csv_loader_requires_existing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_factors.load_factor_returns_from_csv(tmp_path / "missing.csv")


def test_price_config_and_csv_loaders_validate_inputs(tmp_path):
    empty_assets = tmp_path / "assets.yml"
    empty_assets.write_text("assets: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No assets"):
        load_assets_config(empty_assets)

    missing_csv = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_prices_from_csv(missing_csv)

    bad_csv = tmp_path / "bad_prices.csv"
    pd.DataFrame({"ticker": ["A"], "price_date": ["2024-01-01"]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_prices_from_csv(bad_csv)


def test_filter_prices_by_period_controls():
    prices = pd.DataFrame(
        {
            "ticker": ["A"] * 4,
            "price_date": pd.to_datetime(["2023-12-29", "2024-01-01", "2024-01-15", "2024-02-01"]).date,
            "open_price": [1, 2, 3, 4],
            "high_price": [1, 2, 3, 4],
            "low_price": [1, 2, 3, 4],
            "close_price": [1, 2, 3, 4],
            "adjusted_close": [1, 2, 3, 4],
            "volume": [10, 20, 30, 40],
            "source": ["sample"] * 4,
        }
    )

    assert filter_prices_by_window(prices, period="max").equals(prices)
    assert filter_prices_by_window(prices, period="ytd")["price_date"].min() == pd.Timestamp("2024-01-01").date()
    assert filter_prices_by_window(prices, period="1mo")["price_date"].tolist() == [
        pd.Timestamp("2024-01-15").date(),
        pd.Timestamp("2024-02-01").date(),
    ]
    assert filter_prices_by_window(pd.DataFrame(), period="1y").empty


def test_resolve_price_window_validates_dates_and_lookbacks():
    assert resolve_price_window(end="2024-02-01", lookback_days="5").start == pd.Timestamp("2024-01-27").date()

    with pytest.raises(ValueError, match="start"):
        resolve_price_window(start="2024-02-01", end="2024-01-01")
    with pytest.raises(ValueError, match="valid date"):
        resolve_price_window(start="not-a-date")
    with pytest.raises(ValueError, match="integer"):
        resolve_price_window(lookback_days="bad")
    with pytest.raises(ValueError, match="greater than zero"):
        resolve_price_window(lookback_days=0)
    with pytest.raises(ValueError, match="price_period"):
        resolve_price_window(period="forever")


def test_yfinance_fetch_handles_import_and_payload_errors(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "yfinance", None)
    with pytest.raises(RuntimeError, match="not installed"):
        fetch_prices_yfinance(["A"])

    monkeypatch.setitem(
        __import__("sys").modules,
        "yfinance",
        SimpleNamespace(download=lambda **kwargs: pd.DataFrame()),
    )
    with pytest.raises(ValueError, match="no price rows"):
        fetch_prices_yfinance(["A"])

    with pytest.raises(ValueError, match="At least one ticker"):
        fetch_prices_yfinance([" "])


def test_yfinance_fetch_normalizes_single_ticker_frame(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=2)
    raw = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Close": [100.5, 101.5],
            "Adj Close": [100.5, 101.5],
            "Volume": [1000, 1100],
        },
        index=dates,
    )
    monkeypatch.setitem(__import__("sys").modules, "yfinance", SimpleNamespace(download=lambda **kwargs: raw))

    prices = fetch_prices_yfinance(["aapl"], period="1y")

    assert prices[PRICE_COLUMNS]["ticker"].tolist() == ["AAPL", "AAPL"]


def test_yfinance_fetch_rejects_unusable_ticker_payload(monkeypatch):
    raw = pd.DataFrame({"Open": [1.0]}, index=pd.date_range("2024-01-01", periods=1))
    monkeypatch.setitem(__import__("sys").modules, "yfinance", SimpleNamespace(download=lambda **kwargs: raw))

    with pytest.raises(ValueError, match="No yfinance ticker payloads"):
        fetch_prices_yfinance(["AAPL"])


def test_ingest_prices_live_failure_requires_fallback(tmp_path, monkeypatch, capsys):
    csv_path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "ticker": ["A"],
            "price_date": pd.to_datetime(["2024-01-01"]),
            "open_price": [1.0],
            "high_price": [1.0],
            "low_price": [1.0],
            "close_price": [1.0],
            "adjusted_close": [1.0],
            "volume": [10],
            "source": ["sample"],
        }
    )[PRICE_COLUMNS].to_csv(csv_path, index=False)
    monkeypatch.setattr("src.extract.extract_prices.fetch_prices_yfinance", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    prices = ingest_prices(["A"], csv_fallback_path=csv_path, prefer_live=True, allow_fallback=True)

    assert prices["source"].tolist() == ["sample"]
    assert "falling back to CSV" in capsys.readouterr().out

    with pytest.raises(RuntimeError, match="boom"):
        ingest_prices(["A"], csv_fallback_path=csv_path, prefer_live=True, allow_fallback=False)
    with pytest.raises(ValueError, match="csv_fallback_path"):
        ingest_prices(["A"], prefer_live=False)
    with pytest.raises(ValueError, match="no matching"):
        ingest_prices(["B"], csv_fallback_path=csv_path, prefer_live=False)
