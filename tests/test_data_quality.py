import pandas as pd

from src.quality.data_quality_checks import run_data_quality_checks


def test_data_quality_checks_emit_required_columns_and_find_issues():
    prices = pd.DataFrame(
        {
            "ticker": ["A", "A", "A", "A", "B", "B"],
            "price_date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01", "2024-01-02"]
            ),
            "adjusted_close": [100.0, 100.0, 100.0, -1.0, 0.0, None],
        }
    )
    positions = pd.DataFrame({"ticker": ["A", "C"], "weight": [0.7, 0.2]})
    metadata = pd.DataFrame({"ticker": ["A"]})
    returns = pd.DataFrame(
        {
            "ticker": ["A"],
            "return_date": pd.to_datetime(["2024-01-02"]),
            "daily_return": [0.50],
        }
    )

    results = run_data_quality_checks(
        prices,
        positions_df=positions,
        asset_metadata_df=metadata,
        returns_df=returns,
        min_history=3,
    )

    assert {
        "check_name",
        "status",
        "severity",
        "ticker",
        "check_date",
        "message",
        "run_timestamp",
    }.issubset(results.columns)
    assert "duplicate_rows" in set(results["check_name"])
    assert "negative_prices" in set(results["check_name"])
    assert "zero_prices" in set(results["check_name"])
    assert "missing_prices" in set(results["check_name"])
    assert "missing_metadata" in set(results["check_name"])
    assert "extreme_return_outliers" in set(results["check_name"])
    assert "portfolio_weights_sum" in set(results["check_name"])
    assert (results["status"] == "FAIL").any()

