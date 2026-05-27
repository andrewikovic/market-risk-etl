import numpy as np
import pandas as pd

from src.transform.calculate_exposures import calculate_exposures


def _sample_inputs():
    positions = pd.DataFrame(
        {
            "portfolio_name": ["P", "P", "P"],
            "ticker": ["A", "B", "C"],
            "quantity": [10, 20, 5],
        }
    )
    prices = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "price_date": pd.to_datetime(["2024-01-02"] * 3),
            "adjusted_close": [10.0, 5.0, 20.0],
        }
    )
    metadata = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "asset_class": ["Equity", "Equity", "Commodity"],
            "sector": ["Technology", "Financials", "Gold"],
            "currency": ["USD", "USD", "USD"],
            "country": ["United States", "United States", "United States"],
        }
    )
    return positions, prices, metadata


def test_ticker_weights_sum_to_one():
    exposures = calculate_exposures(*_sample_inputs())
    ticker = exposures[exposures["exposure_type"] == "ticker"]
    assert np.isclose(ticker["weight"].sum(), 1.0)


def test_sector_exposures_sum_to_total_market_value():
    exposures = calculate_exposures(*_sample_inputs())
    total = exposures.loc[exposures["exposure_type"] == "net", "market_value"].iloc[0]
    sector_total = exposures.loc[exposures["exposure_type"] == "sector", "market_value"].sum()
    assert np.isclose(sector_total, total)


def test_asset_class_exposures_sum_to_total_market_value():
    exposures = calculate_exposures(*_sample_inputs())
    total = exposures.loc[exposures["exposure_type"] == "net", "market_value"].iloc[0]
    asset_class_total = exposures.loc[exposures["exposure_type"] == "asset_class", "market_value"].sum()
    assert np.isclose(asset_class_total, total)

