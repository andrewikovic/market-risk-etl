from __future__ import annotations

import numpy as np
import pandas as pd


def clean_prices(raw_prices_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw prices into one clean daily adjusted-close row per ticker/date."""
    required = {"ticker", "price_date", "adjusted_close"}
    missing = required - set(raw_prices_df.columns)
    if missing:
        raise ValueError(f"raw_prices_df is missing columns: {sorted(missing)}")

    data = raw_prices_df.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data["price_date"] = pd.to_datetime(data["price_date"]).dt.date
    data["adjusted_close"] = pd.to_numeric(data["adjusted_close"], errors="coerce")
    data["volume"] = pd.to_numeric(data.get("volume", np.nan), errors="coerce")

    data = data.sort_values(["ticker", "price_date"])
    data = data.drop_duplicates(subset=["ticker", "price_date"], keep="last")

    invalid_price = data["adjusted_close"].le(0)
    data["is_missing"] = data["adjusted_close"].isna() | invalid_price
    data.loc[invalid_price, "adjusted_close"] = np.nan
    data["is_stale"] = (
        data.groupby("ticker")["adjusted_close"]
        .transform(lambda prices: prices.eq(prices.shift(1)) & prices.notna())
        .fillna(False)
    )

    return data[
        ["ticker", "price_date", "adjusted_close", "volume", "is_stale", "is_missing"]
    ].reset_index(drop=True)

