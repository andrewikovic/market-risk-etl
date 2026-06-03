import numpy as np
import pandas as pd
import pytest

from src.transform.calculate_exposures import calculate_exposures
from src.transform.calculate_pnl import calculate_portfolio_values
from src.transform.currency_conversion import MissingFXRateError


def _multi_currency_inputs():
    positions = pd.DataFrame(
        {
            "portfolio_name": ["P", "P"],
            "ticker": ["US", "EU"],
            "quantity": [10, 10],
            "currency": ["USD", "EUR"],
            "base_currency": ["USD", "USD"],
            "asset_class": ["Equity", "Equity"],
            "sector": ["Domestic", "International"],
        }
    )
    prices = pd.DataFrame(
        {
            "ticker": ["US", "EU", "US", "EU"],
            "price_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "adjusted_close": [10.0, 20.0, 11.0, 21.0],
        }
    )
    metadata = pd.DataFrame(
        {
            "ticker": ["US", "EU"],
            "asset_class": ["Equity", "Equity"],
            "sector": ["Domestic", "International"],
            "currency": ["USD", "EUR"],
            "country": ["United States", "Germany"],
        }
    )
    fx_rates = pd.DataFrame(
        {
            "rate_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "from_currency": ["EUR", "EUR"],
            "to_currency": ["USD", "USD"],
            "rate": [1.1, 1.2],
        }
    )
    return positions, prices, metadata, fx_rates


def test_portfolio_values_convert_prices_and_pnl_to_base_currency():
    positions, prices, _, fx_rates = _multi_currency_inputs()

    portfolio_values, position_pnl = calculate_portfolio_values(positions, prices, fx_rates=fx_rates)

    assert np.isclose(portfolio_values["market_value"].iloc[0], 320.0)
    assert np.isclose(portfolio_values["market_value"].iloc[1], 362.0)
    assert np.isclose(portfolio_values["daily_pnl"].iloc[1], 42.0)
    latest_eur = position_pnl[(position_pnl["ticker"] == "EU") & (position_pnl["value_date"] == pd.Timestamp("2024-01-02").date())]
    assert np.isclose(latest_eur["position_value"].iloc[0], 252.0)
    assert np.isclose(latest_eur["fx_rate_to_base"].iloc[0], 1.2)


def test_exposures_sum_multi_currency_positions_in_base_currency():
    positions, prices, metadata, fx_rates = _multi_currency_inputs()

    exposures = calculate_exposures(positions, prices, metadata, fx_rates=fx_rates)

    net = exposures.loc[exposures["exposure_type"] == "net", "market_value"].iloc[0]
    ticker_total = exposures.loc[exposures["exposure_type"] == "ticker", "market_value"].sum()
    eur_exposure = exposures.loc[
        (exposures["exposure_type"] == "currency") & (exposures["exposure_name"] == "EUR"),
        "market_value",
    ].iloc[0]
    assert np.isclose(net, 362.0)
    assert np.isclose(ticker_total, net)
    assert np.isclose(eur_exposure, 252.0)


def test_missing_fx_rates_raise_explicit_error():
    positions, prices, _, _ = _multi_currency_inputs()

    with pytest.raises(MissingFXRateError, match="EUR->USD"):
        calculate_portfolio_values(positions, prices)
