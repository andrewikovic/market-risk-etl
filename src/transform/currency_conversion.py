from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np
import pandas as pd


class MissingFXRateError(ValueError):
    """Raised when a required FX conversion rate is unavailable."""


class FXRateProvider(Protocol):
    """Interface for dependency-injected FX rate lookup."""

    def get_rate(self, from_currency: str, to_currency: str, rate_date=None) -> float:
        """Return the rate that converts from_currency amounts into to_currency."""


@dataclass
class StaticFXRateProvider:
    """FX rate provider backed by a mapping or a rate DataFrame."""

    rates: Mapping | pd.DataFrame

    def __post_init__(self) -> None:
        if isinstance(self.rates, pd.DataFrame):
            self._rates_df = _normalize_rate_frame(self.rates)
            self._rates_map = None
        else:
            self._rates_df = None
            self._rates_map = _normalize_rate_mapping(self.rates)

    def get_rate(self, from_currency: str, to_currency: str, rate_date=None) -> float:
        from_currency = _normalize_currency(from_currency)
        to_currency = _normalize_currency(to_currency)
        if from_currency == to_currency:
            return 1.0
        if self._rates_df is not None:
            return _lookup_frame_rate(self._rates_df, from_currency, to_currency, rate_date)
        return _lookup_mapping_rate(self._rates_map or {}, from_currency, to_currency, rate_date)


def resolve_base_currency(base_currency: str | None, positions: pd.DataFrame | None = None) -> str | None:
    """Resolve an explicit or position-configured base currency."""
    if base_currency:
        return _normalize_currency(base_currency)
    if positions is not None and "base_currency" in positions.columns:
        values = positions["base_currency"].dropna().astype(str).str.strip()
        if not values.empty and values.iloc[0]:
            return _normalize_currency(values.iloc[0])
    return None


def convert_prices_to_base(
    prices_df: pd.DataFrame,
    ticker_currencies: Mapping[str, str] | pd.Series | None = None,
    base_currency: str | None = None,
    fx_rates: Mapping | pd.DataFrame | None = None,
    fx_rate_provider: FXRateProvider | None = None,
    price_col: str | None = None,
    date_col: str | None = None,
) -> pd.DataFrame:
    """Convert a price frame into the requested base currency."""
    if base_currency is None:
        return prices_df.copy()

    base_currency = _normalize_currency(base_currency)
    data = prices_df.copy()
    price_col = price_col or ("adjusted_close" if "adjusted_close" in data.columns else "current_price")
    date_col = date_col or _detect_date_column(data)
    required = {"ticker", price_col, date_col}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"prices_df is missing columns for FX conversion: {sorted(missing)}")

    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data[date_col] = pd.to_datetime(data[date_col])
    data[price_col] = pd.to_numeric(data[price_col], errors="coerce")
    data["price_currency"] = _price_currencies(data, ticker_currencies, base_currency)
    provider = _resolve_provider(fx_rates, fx_rate_provider)

    rate_values = _existing_rates(data)
    missing_rate = rate_values.isna()
    for idx, row in data.loc[missing_rate].iterrows():
        currency = row["price_currency"]
        if currency == base_currency:
            rate_values.loc[idx] = 1.0
        elif provider is None:
            raise MissingFXRateError(
                f"Missing FX rate provider for {currency}->{base_currency} on {row[date_col].date()}"
            )
        else:
            rate_values.loc[idx] = provider.get_rate(currency, base_currency, row[date_col])

    invalid = rate_values.isna() | rate_values.le(0)
    if invalid.any():
        row = data.loc[invalid].iloc[0]
        raise MissingFXRateError(
            f"Missing valid FX rate for {row['price_currency']}->{base_currency} on {row[date_col].date()}"
        )

    local_col = f"{price_col}_local"
    if local_col not in data.columns:
        data[local_col] = data[price_col]
    data["fx_rate_to_base"] = rate_values.astype(float)
    data[price_col] = data[local_col] * data["fx_rate_to_base"]
    data["base_currency"] = base_currency
    return data


def convert_amounts_to_base(
    amounts: pd.Series,
    currencies: pd.Series,
    dates: pd.Series | None,
    base_currency: str,
    fx_rates: Mapping | pd.DataFrame | None = None,
    fx_rate_provider: FXRateProvider | None = None,
) -> pd.DataFrame:
    """Convert money amounts into base currency and return converted amounts plus rates."""
    base_currency = _normalize_currency(base_currency)
    provider = _resolve_provider(fx_rates, fx_rate_provider)
    amounts = pd.to_numeric(amounts, errors="coerce")
    currencies = currencies.astype(str).map(_normalize_currency)
    if dates is None:
        dates = pd.Series([pd.NaT] * len(amounts), index=amounts.index)
    else:
        dates = pd.to_datetime(dates)

    rates = pd.Series(np.nan, index=amounts.index, dtype=float)
    for idx in amounts.index:
        currency = currencies.loc[idx]
        if currency == base_currency:
            rates.loc[idx] = 1.0
        elif provider is None:
            date_text = "latest available date" if pd.isna(dates.loc[idx]) else pd.Timestamp(dates.loc[idx]).date()
            raise MissingFXRateError(f"Missing FX rate provider for {currency}->{base_currency} on {date_text}")
        else:
            rates.loc[idx] = provider.get_rate(currency, base_currency, dates.loc[idx])

    invalid = rates.isna() | rates.le(0)
    if invalid.any():
        idx = invalid[invalid].index[0]
        date_text = "latest available date" if pd.isna(dates.loc[idx]) else pd.Timestamp(dates.loc[idx]).date()
        raise MissingFXRateError(f"Missing valid FX rate for {currencies.loc[idx]}->{base_currency} on {date_text}")

    return pd.DataFrame(
        {
            "amount_base": amounts * rates,
            "fx_rate_to_base": rates.astype(float),
            "base_currency": base_currency,
        },
        index=amounts.index,
    )


def _resolve_provider(
    fx_rates: Mapping | pd.DataFrame | None,
    fx_rate_provider: FXRateProvider | None,
) -> FXRateProvider | None:
    if fx_rate_provider is not None:
        return fx_rate_provider
    if fx_rates is not None:
        return StaticFXRateProvider(fx_rates)
    return None


def _detect_date_column(data: pd.DataFrame) -> str:
    for column in ["price_date", "value_date", "return_date", "rate_date", "date"]:
        if column in data.columns:
            return column
    raise ValueError("DataFrame must include a date column for FX conversion")


def _price_currencies(
    prices: pd.DataFrame,
    ticker_currencies: Mapping[str, str] | pd.Series | None,
    base_currency: str,
) -> pd.Series:
    if "currency" in prices.columns:
        currencies = prices["currency"].copy()
    else:
        currencies = pd.Series(index=prices.index, dtype=object)
    if ticker_currencies is not None:
        currency_map = pd.Series(ticker_currencies, dtype=object)
        currency_map.index = currency_map.index.astype(str).str.upper().str.strip()
        mapped = prices["ticker"].map(currency_map)
        currencies = currencies.combine_first(mapped)
    return currencies.fillna(base_currency).astype(str).map(_normalize_currency)


def _existing_rates(data: pd.DataFrame) -> pd.Series:
    if "fx_rate_to_base" not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype=float)
    return pd.to_numeric(data["fx_rate_to_base"], errors="coerce")


def _normalize_rate_frame(rates: pd.DataFrame) -> pd.DataFrame:
    required = {"from_currency", "to_currency", "rate"}
    missing = required - set(rates.columns)
    if missing:
        raise ValueError(f"fx_rates is missing columns: {sorted(missing)}")
    data = rates.copy()
    data["from_currency"] = data["from_currency"].astype(str).map(_normalize_currency)
    data["to_currency"] = data["to_currency"].astype(str).map(_normalize_currency)
    data["rate"] = pd.to_numeric(data["rate"], errors="coerce")
    date_col = next((col for col in ["rate_date", "date", "fx_date", "price_date", "value_date"] if col in data), None)
    if date_col:
        data = data.rename(columns={date_col: "rate_date"})
        data["rate_date"] = pd.to_datetime(data["rate_date"]).dt.normalize()
    else:
        data["rate_date"] = pd.NaT
    return data.dropna(subset=["from_currency", "to_currency", "rate"]).sort_values("rate_date")


def _normalize_rate_mapping(rates: Mapping) -> dict:
    normalized = {}
    for key, value in rates.items():
        if isinstance(key, tuple) and len(key) == 2:
            from_currency, to_currency = key
            rate_date = None
        elif isinstance(key, tuple) and len(key) == 3:
            rate_date, from_currency, to_currency = key
        elif isinstance(key, str):
            clean_key = key.upper().replace("_TO_", "/").replace("-", "/")
            if "/" in clean_key:
                from_currency, to_currency = clean_key.split("/", 1)
            elif len(clean_key) == 6:
                from_currency, to_currency = clean_key[:3], clean_key[3:]
            else:
                raise ValueError(f"Unsupported FX rate key: {key}")
            rate_date = None
        else:
            raise ValueError(f"Unsupported FX rate key: {key}")
        normalized[(
            _normalize_currency(from_currency),
            _normalize_currency(to_currency),
            _normalize_date_key(rate_date),
        )] = float(value)
    return normalized


def _lookup_frame_rate(data: pd.DataFrame, from_currency: str, to_currency: str, rate_date) -> float:
    direct = _frame_candidates(data, from_currency, to_currency, rate_date)
    if not direct.empty:
        return _latest_rate(direct)
    inverse = _frame_candidates(data, to_currency, from_currency, rate_date)
    if not inverse.empty:
        return 1.0 / _latest_rate(inverse)
    date_text = "latest available date" if pd.isna(rate_date) else pd.Timestamp(rate_date).date()
    raise MissingFXRateError(f"Missing FX rate for {from_currency}->{to_currency} on {date_text}")


def _frame_candidates(data: pd.DataFrame, from_currency: str, to_currency: str, rate_date) -> pd.DataFrame:
    candidates = data[(data["from_currency"] == from_currency) & (data["to_currency"] == to_currency)]
    if candidates.empty or candidates["rate_date"].isna().all() or pd.isna(rate_date):
        return candidates
    return candidates[candidates["rate_date"] <= pd.Timestamp(rate_date).normalize()]


def _latest_rate(candidates: pd.DataFrame) -> float:
    value = float(candidates.sort_values("rate_date")["rate"].iloc[-1])
    if value <= 0:
        raise MissingFXRateError("FX rates must be positive")
    return value


def _lookup_mapping_rate(
    rates: dict[tuple[str, str, object], float],
    from_currency: str,
    to_currency: str,
    rate_date,
) -> float:
    date_key = _normalize_date_key(rate_date)
    for key in [(from_currency, to_currency, date_key), (from_currency, to_currency, None)]:
        if key in rates:
            return rates[key]
    for key in [(to_currency, from_currency, date_key), (to_currency, from_currency, None)]:
        if key in rates:
            return 1.0 / rates[key]
    date_text = "latest available date" if date_key is None else date_key.date()
    raise MissingFXRateError(f"Missing FX rate for {from_currency}->{to_currency} on {date_text}")


def _normalize_currency(currency: str) -> str:
    return str(currency).upper().strip()


def _normalize_date_key(rate_date):
    if rate_date is None or pd.isna(rate_date):
        return None
    return pd.Timestamp(rate_date).normalize()
