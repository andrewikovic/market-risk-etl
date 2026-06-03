from __future__ import annotations

import numpy as np
import pandas as pd

from src.transform.currency_conversion import (
    FXRateProvider,
    convert_amounts_to_base,
    convert_prices_to_base,
    resolve_base_currency,
)


def calculate_current_positions(
    positions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    asset_metadata_df: pd.DataFrame | None = None,
    base_currency: str | None = None,
    fx_rates: dict | pd.DataFrame | None = None,
    fx_rate_provider: FXRateProvider | None = None,
) -> pd.DataFrame:
    """Attach latest prices and metadata to current portfolio positions."""
    required = {"ticker", "quantity"}
    missing = required - set(positions_df.columns)
    if missing:
        raise ValueError(f"positions_df is missing columns: {sorted(missing)}")

    positions = positions_df.copy()
    positions["ticker"] = positions["ticker"].astype(str).str.upper().str.strip()
    positions["quantity"] = pd.to_numeric(positions["quantity"], errors="coerce")

    if asset_metadata_df is not None and not asset_metadata_df.empty:
        metadata = asset_metadata_df.copy()
        metadata["ticker"] = metadata["ticker"].astype(str).str.upper().str.strip()
        positions = positions.merge(metadata, on="ticker", how="left", suffixes=("", "_meta"))
        for column in ["asset_class", "sector", "currency", "country"]:
            meta_col = f"{column}_meta"
            if meta_col in positions.columns:
                if column in positions.columns:
                    positions[column] = positions[column].combine_first(positions[meta_col])
                else:
                    positions[column] = positions[meta_col]

    base_currency = resolve_base_currency(base_currency, positions)
    if base_currency is not None and "currency" not in positions.columns:
        positions["currency"] = base_currency
    elif base_currency is not None:
        positions["currency"] = positions["currency"].fillna(base_currency)

    if "position_value" in positions.columns:
        positions["position_value"] = pd.to_numeric(positions["position_value"], errors="coerce")
        if base_currency is not None:
            dates = positions["as_of_date"] if "as_of_date" in positions.columns else None
            positions["local_position_value"] = positions["position_value"]
            converted = convert_amounts_to_base(
                positions["position_value"],
                positions["currency"],
                dates,
                base_currency,
                fx_rates=fx_rates,
                fx_rate_provider=fx_rate_provider,
            )
            positions["position_value"] = converted["amount_base"]
            positions["fx_rate_to_base"] = converted["fx_rate_to_base"]
            positions["base_currency"] = base_currency
    else:
        ticker_currencies = (
            positions.drop_duplicates("ticker").set_index("ticker")["currency"] if "currency" in positions else None
        )
        latest_prices = _latest_prices(
            prices_df,
            ticker_currencies=ticker_currencies,
            base_currency=base_currency,
            fx_rates=fx_rates,
            fx_rate_provider=fx_rate_provider,
        )
        positions = positions.merge(latest_prices, on="ticker", how="left")
        positions["position_value"] = positions["quantity"] * positions["current_price"]

    for column in ["asset_class", "sector", "currency", "country"]:
        if column not in positions.columns:
            positions[column] = "Unknown"
        positions[column] = positions[column].fillna("Unknown")

    total_value = positions["position_value"].sum()
    positions["weight"] = np.where(total_value != 0, positions["position_value"] / total_value, np.nan)
    return positions.reset_index(drop=True)


def calculate_exposures(
    positions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    asset_metadata_df: pd.DataFrame,
    base_currency: str | None = None,
    fx_rates: dict | pd.DataFrame | None = None,
    fx_rate_provider: FXRateProvider | None = None,
) -> pd.DataFrame:
    """Calculate exposure by ticker, sector, asset class, currency, and geography."""
    positions = calculate_current_positions(
        positions_df,
        prices_df,
        asset_metadata_df,
        base_currency=base_currency,
        fx_rates=fx_rates,
        fx_rate_provider=fx_rate_provider,
    )
    total_value = positions["position_value"].sum()
    gross_value = positions["position_value"].abs().sum()
    portfolio_name = positions.get("portfolio_name", pd.Series(["Portfolio"])).iloc[0]
    exposure_date = _exposure_date(prices_df)
    base_currency_value = positions["base_currency"].dropna().iloc[0] if "base_currency" in positions.columns else None

    rows: list[dict] = []
    rows.extend(_group_exposure(positions, "ticker", "ticker", portfolio_name, exposure_date, total_value, base_currency_value))
    rows.extend(
        _group_exposure(positions, "asset_class", "asset_class", portfolio_name, exposure_date, total_value, base_currency_value)
    )
    rows.extend(_group_exposure(positions, "sector", "sector", portfolio_name, exposure_date, total_value, base_currency_value))
    rows.extend(_group_exposure(positions, "currency", "currency", portfolio_name, exposure_date, total_value, base_currency_value))
    rows.extend(_group_exposure(positions, "country", "country", portfolio_name, exposure_date, total_value, base_currency_value))
    rows.append(
        {
            "portfolio_name": portfolio_name,
            "exposure_date": exposure_date,
            "exposure_type": "gross",
            "exposure_name": "Gross Exposure",
            "market_value": gross_value,
            "weight": gross_value / abs(total_value) if total_value != 0 else np.nan,
            "base_currency": base_currency_value,
        }
    )
    rows.append(
        {
            "portfolio_name": portfolio_name,
            "exposure_date": exposure_date,
            "exposure_type": "net",
            "exposure_name": "Net Exposure",
            "market_value": total_value,
            "weight": 1.0 if total_value != 0 else np.nan,
            "base_currency": base_currency_value,
        }
    )
    return pd.DataFrame(rows)


def _latest_prices(
    prices_df: pd.DataFrame,
    ticker_currencies: pd.Series | None = None,
    base_currency: str | None = None,
    fx_rates: dict | pd.DataFrame | None = None,
    fx_rate_provider: FXRateProvider | None = None,
) -> pd.DataFrame:
    price_col = "adjusted_close" if "adjusted_close" in prices_df.columns else "current_price"
    date_col = "price_date" if "price_date" in prices_df.columns else "value_date"
    price_columns = ["ticker", date_col, price_col]
    price_columns.extend(col for col in ["currency", "fx_rate_to_base"] if col in prices_df.columns)
    prices = prices_df[price_columns].copy()
    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip()
    prices[date_col] = pd.to_datetime(prices[date_col])
    prices[price_col] = pd.to_numeric(prices[price_col], errors="coerce")
    prices = prices.dropna(subset=[price_col]).sort_values(["ticker", date_col])
    if base_currency is not None:
        prices = convert_prices_to_base(
            prices,
            ticker_currencies=ticker_currencies,
            base_currency=base_currency,
            fx_rates=fx_rates,
            fx_rate_provider=fx_rate_provider,
            price_col=price_col,
            date_col=date_col,
        )
    latest = prices.groupby("ticker", as_index=False).tail(1).rename(
        columns={price_col: "current_price", date_col: "price_date"}
    )
    columns = ["ticker", "current_price", "price_date"]
    columns.extend(col for col in ["fx_rate_to_base", "base_currency", "price_currency"] if col in latest.columns)
    return latest[columns]


def _exposure_date(prices_df: pd.DataFrame):
    for column in ["price_date", "value_date", "return_date"]:
        if column in prices_df.columns and not prices_df.empty:
            return pd.to_datetime(prices_df[column]).max().date()
    return pd.Timestamp.today().date()


def _group_exposure(
    positions: pd.DataFrame,
    group_col: str,
    exposure_type: str,
    portfolio_name: str,
    exposure_date,
    total_value: float,
    base_currency: str | None = None,
) -> list[dict]:
    grouped = positions.groupby(group_col, dropna=False)["position_value"].sum().reset_index()
    rows = []
    for _, row in grouped.iterrows():
        market_value = float(row["position_value"])
        rows.append(
            {
                "portfolio_name": portfolio_name,
                "exposure_date": exposure_date,
                "exposure_type": exposure_type,
                "exposure_name": str(row[group_col]),
                "market_value": market_value,
                "weight": market_value / total_value if total_value != 0 else np.nan,
                "base_currency": base_currency,
            }
        )
    return rows
