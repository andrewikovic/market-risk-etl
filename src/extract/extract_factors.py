from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.risk.factor_model import estimate_factor_exposures


def load_factor_returns_from_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load factor return data from CSV for factor-model analytics."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Factor CSV not found: {path}")
    factors = pd.read_csv(path, parse_dates=["return_date"])
    if "return_date" not in factors.columns:
        raise ValueError("Factor CSV must include return_date")
    factors["return_date"] = pd.to_datetime(factors["return_date"]).dt.date
    return factors.sort_values("return_date").reset_index(drop=True)


def estimate_factor_exposures_from_csv(
    asset_returns_df: pd.DataFrame,
    factor_csv_path: str | Path,
    weights: dict | pd.Series | None = None,
) -> dict[str, pd.DataFrame]:
    """Load factor returns from CSV and estimate asset and portfolio factor exposures."""
    factor_returns = load_factor_returns_from_csv(factor_csv_path)
    return estimate_factor_exposures(asset_returns_df, factor_returns, weights=weights)
