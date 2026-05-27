from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_factor_returns_from_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load factor return data from CSV for future factor-model extensions."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Factor CSV not found: {path}")
    factors = pd.read_csv(path, parse_dates=["return_date"])
    if "return_date" not in factors.columns:
        raise ValueError("Factor CSV must include return_date")
    factors["return_date"] = pd.to_datetime(factors["return_date"]).dt.date
    return factors.sort_values("return_date").reset_index(drop=True)

