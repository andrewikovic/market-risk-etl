import numpy as np
import pandas as pd

from src.load.pipeline_db import build_risk_metrics_frame


def test_build_risk_metrics_frame_shapes_pipeline_metrics_for_mart_table():
    outputs = {
        "portfolio_values": pd.DataFrame(
            {
                "portfolio_name": ["Sample Portfolio", "Sample Portfolio"],
                "value_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "market_value": [100.0, 105.0],
            }
        ),
        "risk_metrics": {
            "historical_var_95": 123.45,
            "sharpe_ratio": 1.2,
            "max_drawdown": np.nan,
        },
    }

    result = build_risk_metrics_frame(outputs)

    assert set(result["metric_name"]) == {"historical_var_95", "sharpe_ratio"}
    assert result.loc[result["metric_name"] == "historical_var_95", "confidence_level"].iloc[0] == 0.95
    assert pd.isna(result.loc[result["metric_name"] == "sharpe_ratio", "confidence_level"].iloc[0])
    assert result["portfolio_name"].eq("Sample Portfolio").all()
    assert result["metric_date"].iloc[0] == pd.Timestamp("2024-01-03").date()
    assert result["lookback_days"].eq(1).all()


def test_build_risk_metrics_frame_returns_empty_frame_without_metrics():
    result = build_risk_metrics_frame({})

    assert result.empty
    assert list(result.columns) == [
        "portfolio_name",
        "metric_date",
        "metric_name",
        "metric_value",
        "lookback_days",
        "confidence_level",
    ]
