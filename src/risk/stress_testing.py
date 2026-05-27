from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd


def run_stress_test(positions_df: pd.DataFrame, scenarios: dict | list[dict]) -> dict[str, pd.DataFrame]:
    """
    Apply scenario shocks to positions and calculate stress losses.

    Precedence is ticker-specific shocks, then sector shocks, then asset-class
    shocks, then currency shocks.
    """
    positions = positions_df.copy()
    if "position_value" not in positions.columns:
        if {"quantity", "current_price"}.issubset(positions.columns):
            positions["position_value"] = positions["quantity"] * positions["current_price"]
        else:
            raise ValueError("positions_df must include position_value or quantity/current_price")

    for column in ["ticker", "sector", "asset_class", "currency"]:
        if column not in positions.columns:
            positions[column] = "Unknown"
        positions[column] = positions[column].fillna("Unknown").astype(str)

    scenario_list = scenarios.get("scenarios", []) if isinstance(scenarios, dict) else scenarios
    if not scenario_list:
        raise ValueError("At least one stress scenario is required")

    run_timestamp = datetime.now(UTC)
    portfolio_name = positions.get("portfolio_name", pd.Series(["Portfolio"])).iloc[0]
    current_portfolio_value = float(positions["position_value"].sum())
    position_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []

    for scenario in scenario_list:
        name = scenario["name"]
        shocks = scenario.get("shocks", {})
        scenario_positions = positions.copy()
        scenario_positions["shock"] = scenario_positions.apply(lambda row: _applicable_shock(row, shocks), axis=1)
        scenario_positions["shocked_position_value"] = scenario_positions["position_value"] * (
            1.0 + scenario_positions["shock"]
        )
        scenario_positions["stress_loss"] = (
            scenario_positions["position_value"] - scenario_positions["shocked_position_value"]
        )
        scenario_positions["stress_return"] = np.where(
            scenario_positions["position_value"] != 0,
            -scenario_positions["stress_loss"] / scenario_positions["position_value"],
            np.nan,
        )
        shocked_portfolio_value = float(scenario_positions["shocked_position_value"].sum())
        stress_loss = current_portfolio_value - shocked_portfolio_value
        stress_return = stress_loss / current_portfolio_value if current_portfolio_value else np.nan

        scenario_rows.append(
            {
                "portfolio_name": portfolio_name,
                "scenario_name": name,
                "run_timestamp": run_timestamp,
                "current_portfolio_value": current_portfolio_value,
                "shocked_portfolio_value": shocked_portfolio_value,
                "stress_loss": stress_loss,
                "stress_return": stress_return,
            }
        )
        for _, row in scenario_positions.iterrows():
            position_rows.append(
                {
                    "portfolio_name": portfolio_name,
                    "scenario_name": name,
                    "run_timestamp": run_timestamp,
                    "ticker": row["ticker"],
                    "asset_class": row["asset_class"],
                    "sector": row["sector"],
                    "currency": row["currency"],
                    "position_value": row["position_value"],
                    "shock": row["shock"],
                    "shocked_position_value": row["shocked_position_value"],
                    "stress_loss": row["stress_loss"],
                    "stress_return": row["stress_return"],
                }
            )

    return {
        "scenario_results": pd.DataFrame(scenario_rows),
        "position_results": pd.DataFrame(position_rows),
    }


def _applicable_shock(row: pd.Series, shocks: dict[str, float]) -> float:
    ticker = str(row["ticker"])
    sector = str(row["sector"])
    asset_class = str(row["asset_class"])
    currency = str(row["currency"])

    candidates = [
        f"ticker:{ticker}",
        ticker,
        f"sector:{sector}",
        sector,
        f"asset_class:{asset_class}",
        asset_class,
        f"currency:{currency}",
        currency,
    ]
    for key in candidates:
        if key in shocks:
            return float(shocks[key])
    return 0.0

