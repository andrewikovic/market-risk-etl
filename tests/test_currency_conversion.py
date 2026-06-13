import numpy as np
import pandas as pd
import pytest

from src.transform.calculate_exposures import calculate_exposures
from src.transform.calculate_pnl import calculate_portfolio_values
from src.transform.currency_conversion import (
    MissingFXRateError,
    StaticFXRateProvider,
    convert_amounts_to_base,
    convert_prices_to_base,
    resolve_base_currency,
)


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


def test_static_fx_provider_supports_mapping_keys_and_inverse_rates():
    provider = StaticFXRateProvider(
        {
            ("2024-01-02", "EUR", "USD"): 1.2,
            ("GBP", "USD"): 1.3,
            "USDJPY": 150.0,
            "CAD-USD": 0.75,
        }
    )

    assert provider.get_rate(" eur ", "usd", "2024-01-02") == 1.2
    assert np.isclose(provider.get_rate("USD", "EUR", "2024-01-02"), 1 / 1.2)
    assert provider.get_rate("GBP", "USD", "2024-03-01") == 1.3
    assert provider.get_rate("USD", "JPY") == 150.0
    assert provider.get_rate("CAD", "USD") == 0.75
    assert provider.get_rate("usd", "USD") == 1.0


def test_static_fx_provider_uses_latest_frame_rate_on_or_before_date():
    rates = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05"]),
            "from_currency": ["EUR", "EUR", "USD"],
            "to_currency": ["USD", "USD", "CAD"],
            "rate": [1.1, 1.3, 1.4],
        }
    )
    provider = StaticFXRateProvider(rates)

    assert provider.get_rate("EUR", "USD", "2024-01-02") == 1.1
    assert provider.get_rate("EUR", "USD") == 1.3
    assert np.isclose(provider.get_rate("CAD", "USD", "2024-01-06"), 1 / 1.4)

    with pytest.raises(MissingFXRateError, match="EUR->CAD"):
        provider.get_rate("EUR", "CAD", "2024-01-02")


def test_static_fx_provider_validates_rate_inputs():
    with pytest.raises(ValueError, match="missing columns"):
        StaticFXRateProvider(pd.DataFrame({"from_currency": ["EUR"], "rate": [1.2]}))
    with pytest.raises(ValueError, match="Unsupported FX rate key"):
        StaticFXRateProvider({"bad": 1.0})
    with pytest.raises(ValueError, match="Unsupported FX rate key"):
        StaticFXRateProvider({("EUR", "USD", "extra", "bad"): 1.0})

    provider = StaticFXRateProvider({("EUR", "USD"): 0.0})
    with pytest.raises(ZeroDivisionError):
        provider.get_rate("USD", "EUR")

    frame_provider = StaticFXRateProvider(
        pd.DataFrame({"from_currency": ["EUR"], "to_currency": ["USD"], "rate": [0.0]})
    )
    with pytest.raises(MissingFXRateError, match="positive"):
        frame_provider.get_rate("EUR", "USD")


def test_resolve_base_currency_prefers_explicit_then_positions():
    positions = pd.DataFrame({"base_currency": [None, " eur "]})

    assert resolve_base_currency(" usd ", positions) == "USD"
    assert resolve_base_currency(None, positions) == "EUR"
    assert resolve_base_currency(None, pd.DataFrame({"ticker": ["A"]})) is None


def test_convert_prices_to_base_handles_price_columns_and_existing_rates():
    prices = pd.DataFrame(
        {
            "ticker": [" eu ", "US"],
            "value_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "current_price": [10.0, 20.0],
        }
    )

    converted = convert_prices_to_base(
        prices,
        ticker_currencies={"EU": "EUR"},
        base_currency="usd",
        fx_rates={"EUR/USD": 1.2},
        price_col="current_price",
    )

    assert converted["ticker"].tolist() == ["EU", "US"]
    assert converted["current_price"].tolist() == [12.0, 20.0]
    assert converted["current_price_local"].tolist() == [10.0, 20.0]
    assert converted["fx_rate_to_base"].tolist() == [1.2, 1.0]
    assert converted["base_currency"].tolist() == ["USD", "USD"]

    passthrough = convert_prices_to_base(prices, base_currency=None)
    assert passthrough.equals(prices)
    assert passthrough is not prices


def test_convert_prices_to_base_validates_missing_and_invalid_rates():
    with pytest.raises(ValueError, match="missing columns"):
        convert_prices_to_base(pd.DataFrame({"ticker": ["A"], "date": ["2024-01-01"]}), base_currency="USD")
    with pytest.raises(ValueError, match="date column"):
        convert_prices_to_base(pd.DataFrame({"ticker": ["A"], "adjusted_close": [1.0]}), base_currency="USD")

    bad_rate = pd.DataFrame(
        {
            "ticker": ["EU"],
            "price_date": pd.to_datetime(["2024-01-01"]),
            "adjusted_close": [10.0],
            "currency": ["EUR"],
            "fx_rate_to_base": [0.0],
        }
    )
    with pytest.raises(MissingFXRateError, match="Missing valid FX rate"):
        convert_prices_to_base(bad_rate, base_currency="USD")


def test_convert_amounts_to_base_uses_dates_and_validates_rates():
    amounts = pd.Series([10.0, 20.0], index=["a", "b"])
    currencies = pd.Series(["USD", "EUR"], index=["a", "b"])

    converted = convert_amounts_to_base(amounts, currencies, None, "USD", fx_rates={"EUR/USD": 1.25})

    assert converted["amount_base"].tolist() == [10.0, 25.0]
    assert converted["fx_rate_to_base"].tolist() == [1.0, 1.25]

    with pytest.raises(MissingFXRateError, match="latest available date"):
        convert_amounts_to_base(pd.Series([10.0]), pd.Series(["EUR"]), None, "USD")

    class ZeroProvider:
        def get_rate(self, from_currency, to_currency, rate_date=None):
            return 0.0

    with pytest.raises(MissingFXRateError, match="Missing valid FX rate"):
        convert_amounts_to_base(
            pd.Series([10.0]),
            pd.Series(["EUR"]),
            pd.Series(pd.to_datetime(["2024-01-01"])),
            "USD",
            fx_rate_provider=ZeroProvider(),
        )
