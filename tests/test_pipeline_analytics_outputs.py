import numpy as np
import pandas as pd

from src.load.read_db import _frontier_from_rows
from src.pipeline import run_pipeline


def test_pipeline_outputs_include_new_analytics_frames():
    outputs = run_pipeline(write_processed=False)

    for key in [
        "var_backtest",
        "var_contributions",
        "risk_contributions",
        "factor_exposures",
        "efficient_frontier",
        "optimized_portfolio",
        "rebalancing_trades",
    ]:
        assert key in outputs
        assert not outputs[key].empty

    assert set(outputs["var_backtest"]["confidence_level"]) == {0.95, 0.975, 0.99}
    assert np.isclose(
        outputs["var_contributions"].query("confidence_level == 0.95")["component_var"].sum(),
        outputs["var_contributions"].query("confidence_level == 0.95")["portfolio_var"].iloc[0],
    )
    assert np.isclose(outputs["optimized_portfolio"]["target_weight"].sum(), 1.0)
    assert np.isclose(outputs["rebalancing_trades"]["trade_value"].sum(), 0.0)


def test_database_frontier_rows_reconstruct_dashboard_shape():
    rows = pd.DataFrame(
        {
            "portfolio_name": ["P", "P", "P", "P"],
            "run_date": pd.to_datetime(["2024-01-02"] * 4),
            "point_number": [1, 1, 2, 2],
            "ticker": ["A", "B", "A", "B"],
            "target_return": [0.05, 0.05, 0.08, 0.08],
            "expected_return": [0.05, 0.05, 0.08, 0.08],
            "volatility": [0.10, 0.10, 0.15, 0.15],
            "sharpe_ratio": [0.50, 0.50, 0.53, 0.53],
            "weight": [0.6, 0.4, 0.3, 0.7],
        }
    )

    frontier = _frontier_from_rows(rows)

    assert len(frontier) == 2
    assert frontier.loc[frontier["point_number"] == 1, "weights"].iloc[0] == {"A": 0.6, "B": 0.4}
    assert np.isclose(frontier.loc[frontier["point_number"] == 2, "weight_B"].iloc[0], 0.7)
