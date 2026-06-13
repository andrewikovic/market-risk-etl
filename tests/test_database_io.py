from pathlib import Path

import pandas as pd
import pytest

from src.load import db as db_module
from src.load import load_marts, load_raw, pipeline_db, read_db


def test_read_coerces_configured_numeric_columns(monkeypatch):
    def fake_read_sql_query(query, engine, parse_dates=None):
        assert "SELECT" in query
        assert engine == "engine"
        assert parse_dates == ["value_date"]
        return pd.DataFrame({"amount": ["1.5", "bad"], "label": ["a", "b"]})

    monkeypatch.setattr(read_db.pd, "read_sql_query", fake_read_sql_query)

    result = read_db._read("engine", "SELECT * FROM mart.table", parse_dates=["value_date"], numeric_columns=["amount", "missing"])

    assert result["amount"].tolist() == [1.5, pytest.approx(float("nan"), nan_ok=True)]
    assert result["label"].tolist() == ["a", "b"]


def test_read_pipeline_outputs_reconstructs_dashboard_shape(monkeypatch):
    frames = [
        pd.DataFrame({"ticker": ["A"], "price_date": pd.to_datetime(["2024-01-01"]), "adjusted_close": [10.0]}),
        pd.DataFrame(
            {
                "ticker": ["A"],
                "asset_name": ["Asset A"],
                "asset_class": ["Equity"],
                "sector": ["Tech"],
                "currency": ["USD"],
                "country": ["US"],
                "benchmark_ticker": ["SPY"],
            }
        ),
        pd.DataFrame(
            {
                "portfolio_name": ["P"],
                "ticker": ["A"],
                "quantity": [2.0],
                "as_of_date": pd.to_datetime(["2024-01-01"]),
                "asset_class": ["Equity"],
                "sector": ["Tech"],
                "currency": ["USD"],
                "base_currency": ["USD"],
            }
        ),
        pd.DataFrame({"ticker": ["A"], "price_date": pd.to_datetime(["2024-01-01"]), "adjusted_close": [10.0]}),
        pd.DataFrame({"ticker": ["A"], "return_date": pd.to_datetime(["2024-01-02"]), "daily_return": [0.01]}),
        pd.DataFrame(
            {
                "portfolio_name": ["P", "P"],
                "value_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "market_value": [20.0, 22.0],
            }
        ),
        pd.DataFrame(
            {
                "portfolio_name": ["P"],
                "value_date": pd.to_datetime(["2024-01-02"]),
                "ticker": ["A"],
                "position_value": [22.0],
                "weight": [1.0],
                "currency": ["USD"],
                "fx_rate_to_base": [1.0],
                "base_currency": ["USD"],
            }
        ),
        pd.DataFrame({"exposure_type": ["net"], "market_value": [22.0]}),
        pd.DataFrame({"check_name": ["freshness"], "status": ["pass"]}),
        pd.DataFrame({"metric_name": ["historical_var_95"], "metric_value": [1.23]}),
        pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "breach": [False]}),
        pd.DataFrame({"ticker": ["A"], "component_var": [1.0]}),
        pd.DataFrame({"ticker": ["A"], "component_volatility": [0.1]}),
        pd.DataFrame({"ticker": ["A"], "factor": ["market"], "beta": [1.0]}),
        pd.DataFrame(
            {
                "portfolio_name": ["P"],
                "run_date": pd.to_datetime(["2024-01-02"]),
                "point_number": [1],
                "ticker": ["A"],
                "target_return": [0.05],
                "expected_return": [0.05],
                "volatility": [0.10],
                "sharpe_ratio": [0.5],
                "weight": [1.0],
            }
        ),
        pd.DataFrame({"ticker": ["A"], "target_weight": [1.0]}),
        pd.DataFrame({"ticker": ["A"], "trade_value": [0.0]}),
    ]

    def fake_read(*_args, **_kwargs):
        return frames.pop(0)

    monkeypatch.setattr(read_db, "_read", fake_read)

    outputs = read_db.read_pipeline_outputs("engine")

    assert outputs["risk_metrics"] == {"historical_var_95": 1.23}
    assert outputs["current_positions"]["current_price"].iloc[0] == 11.0
    assert outputs["efficient_frontier"]["weights"].iloc[0] == {"A": 1.0}
    assert outputs["drawdowns"]["current_drawdown"] == 0.0
    assert frames == []


def test_load_mart_frames_select_expected_columns(monkeypatch):
    calls = []

    def fake_to_sql(self, name, engine, schema=None, if_exists=None, index=None, method=None, dtype=None):
        calls.append(
            {
                "name": name,
                "schema": schema,
                "if_exists": if_exists,
                "columns": list(self.columns),
                "dtype": dtype,
            }
        )

    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql)
    frame = pd.DataFrame(
        {
            "portfolio_name": ["P"],
            "value_date": pd.to_datetime(["2024-01-02"]),
            "ticker": ["A"],
            "position_value": [100.0],
            "daily_pnl": [1.0],
            "contribution_to_pnl": [1.0],
            "contribution_to_return": [0.01],
            "weight": [1.0],
            "currency": ["USD"],
            "fx_rate_to_base": [1.0],
            "base_currency": ["USD"],
            "exposure_date": pd.to_datetime(["2024-01-02"]),
            "exposure_type": ["ticker"],
            "exposure_name": ["A"],
            "market_value": [100.0],
            "metric_date": pd.to_datetime(["2024-01-02"]),
            "metric_name": ["var"],
            "metric_value": [1.0],
            "lookback_days": [10],
            "confidence_level": [0.95],
            "date": pd.to_datetime(["2024-01-02"]),
            "var_estimate": [1.0],
            "realized_pnl": [-2.0],
            "breach": [True],
            "breach_severity": [1.0],
            "total_observations": [10],
            "number_of_exceptions": [1],
            "expected_exceptions": [0.5],
            "exception_ratio": [2.0],
            "kupiec_statistic": [0.1],
            "p_value": [0.9],
            "pass_fail": ["pass"],
            "exposure": [100.0],
            "mean_return": [0.01],
            "volatility": [0.2],
            "marginal_var": [1.0],
            "component_var": [1.0],
            "percent_contribution": [1.0],
            "portfolio_var": [1.0],
            "contribution_reconciliation_error": [0.0],
            "asset_volatility": [0.2],
            "marginal_volatility": [0.2],
            "component_volatility": [0.2],
            "volatility_contribution_amount": [0.2],
            "volatility_percent_contribution": [1.0],
            "var_percent_contribution": [1.0],
            "portfolio_volatility": [0.2],
            "volatility_reconciliation_error": [0.0],
            "var_reconciliation_error": [0.0],
            "exposure_level": ["asset"],
            "factor": ["market"],
            "beta": [1.0],
            "alpha": [0.0],
            "residual_volatility": [0.1],
            "idiosyncratic_variance": [0.01],
            "r_squared": [0.9],
            "observations": [20],
            "run_date": pd.to_datetime(["2024-01-02"]),
            "weights": [{"A": 1.0}],
            "point_number": [1],
            "target_return": [0.05],
            "expected_return": [0.05],
            "sharpe_ratio": [0.5],
            "target_weight": [1.0],
            "weight_sum": [1.0],
            "full_investment": [True],
            "long_only": [True],
            "min_weight_satisfied": [True],
            "max_weight_satisfied": [True],
            "target_return_error": [0.0],
            "target_volatility_error": [0.0],
            "current_weight": [1.0],
            "weight_change": [0.0],
            "trade_value": [0.0],
            "price": [100.0],
            "quantity_change": [0.0],
            "check_name": ["freshness"],
            "status": ["pass"],
            "severity": ["low"],
            "check_date": pd.to_datetime(["2024-01-02"]),
            "message": ["ok"],
            "run_timestamp": pd.to_datetime(["2024-01-02 12:00"]),
        }
    )

    load_marts.load_staging_prices("engine", frame, if_exists="replace")
    load_marts.load_daily_returns("engine", frame)
    load_marts.load_portfolio_values("engine", frame)
    load_marts.load_position_pnl("engine", frame[["portfolio_name", "value_date", "ticker", "position_value"]])
    load_marts.load_exposures("engine", frame)
    load_marts.load_risk_metrics("engine", frame)
    load_marts.load_data_quality_results("engine", frame)
    load_marts.load_var_backtest("engine", frame)
    load_marts.load_var_contributions("engine", frame)
    load_marts.load_risk_contributions("engine", frame)
    load_marts.load_factor_exposures("engine", frame)
    load_marts.load_efficient_frontier("engine", frame)
    load_marts.load_optimized_portfolio("engine", frame)
    load_marts.load_rebalancing_trades("engine", frame)
    load_marts.load_var_backtest("engine", pd.DataFrame())
    load_marts.load_var_contributions("engine", pd.DataFrame())

    table_names = [call["name"] for call in calls]
    assert "position_pnl" in table_names
    assert "efficient_frontier" in table_names
    position_call = next(call for call in calls if call["name"] == "position_pnl")
    assert position_call["columns"] == [
        "portfolio_name",
        "value_date",
        "ticker",
        "position_value",
        "daily_pnl",
        "contribution_to_pnl",
        "contribution_to_return",
        "weight",
        "currency",
        "fx_rate_to_base",
        "base_currency",
    ]
    quality_call = next(call for call in calls if call["name"] == "data_quality_results")
    assert set(quality_call["dtype"]) == {"check_date", "run_timestamp"}


def test_load_raw_frames_write_expected_tables(monkeypatch):
    calls = []

    def fake_to_sql(self, name, engine, schema=None, if_exists=None, index=None, method=None):
        calls.append((name, schema, if_exists, list(self.columns)))

    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql)
    frame = pd.DataFrame(
        {
            "portfolio_name": ["P"],
            "ticker": ["A"],
            "quantity": [1.0],
            "as_of_date": pd.to_datetime(["2024-01-02"]),
            "asset_class": ["Equity"],
            "sector": ["Tech"],
            "currency": ["USD"],
            "base_currency": ["USD"],
        }
    )

    load_raw.load_raw_prices("engine", frame, if_exists="replace")
    load_raw.load_assets("engine", frame)
    load_raw.load_positions("engine", frame)

    assert calls[0][:3] == ("prices", "raw", "replace")
    assert calls[1][:2] == ("assets", "raw")
    assert calls[2][0] == "portfolio_positions"


def test_database_helpers_use_env_and_execute_sql_files(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("DATABASE_URL=postgresql+psycopg://example/db\n", encoding="utf-8")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    assert db_module.get_database_url(env_path) == "postgresql+psycopg://example/db"

    created = {}
    monkeypatch.setattr(db_module, "create_engine", lambda url, **kwargs: created.update(url=url, kwargs=kwargs) or "engine")
    assert db_module.get_engine("postgresql://custom") == "engine"
    assert created == {"url": "postgresql://custom", "kwargs": {"pool_pre_ping": True, "future": True}}

    sql_path = tmp_path / "schema.sql"
    sql_path.write_text("CREATE SCHEMA raw; CREATE TABLE raw.prices(id int);", encoding="utf-8")
    executed = []

    class FakeConnection:
        def execute(self, statement):
            executed.append(str(statement))

    class FakeBegin:
        def __enter__(self):
            return FakeConnection()

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeEngine:
        def begin(self):
            return FakeBegin()

    db_module.execute_sql_file(FakeEngine(), sql_path)

    assert executed == ["CREATE SCHEMA raw", "CREATE TABLE raw.prices(id int)"]

    initialized = []
    monkeypatch.setattr(db_module, "execute_sql_file", lambda engine, path: initialized.append(Path(path).name))
    db_module.initialize_database("engine", tmp_path)

    assert initialized == ["01_raw_schema.sql", "02_staging_schema.sql", "03_mart_schema.sql", "04_indexes.sql"]


def test_load_pipeline_outputs_orchestrates_all_loaders(monkeypatch):
    calls = []
    outputs = {
        "raw_prices": pd.DataFrame({"x": [1]}),
        "assets": pd.DataFrame({"x": [1]}),
        "positions": pd.DataFrame({"x": [1]}),
        "stg_prices": pd.DataFrame({"x": [1]}),
        "returns": pd.DataFrame({"x": [1]}),
        "portfolio_values": pd.DataFrame(
            {
                "portfolio_name": ["P", "P"],
                "value_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "market_value": [100.0, 101.0],
            }
        ),
        "position_pnl": pd.DataFrame({"x": [1]}),
        "exposures": pd.DataFrame({"x": [1]}),
        "data_quality": pd.DataFrame({"x": [1]}),
        "risk_metrics": {"historical_var_95": 1.0},
        "var_backtest": pd.DataFrame({"x": [1, 2]}),
        "var_contributions": pd.DataFrame({"x": [1]}),
        "risk_contributions": pd.DataFrame({"x": [1]}),
        "factor_exposures": pd.DataFrame({"x": [1]}),
        "efficient_frontier": pd.DataFrame({"x": [1]}),
        "optimized_portfolio": pd.DataFrame({"x": [1]}),
        "rebalancing_trades": pd.DataFrame({"x": [1]}),
    }

    monkeypatch.setattr(pipeline_db, "initialize_database", lambda engine, sql_dir: calls.append(("initialize", sql_dir)))
    monkeypatch.setattr(pipeline_db, "truncate_pipeline_tables", lambda engine: calls.append(("truncate", engine)))

    for name in [
        "load_raw_prices",
        "load_assets",
        "load_positions",
        "load_staging_prices",
        "load_daily_returns",
        "load_portfolio_values",
        "load_position_pnl",
        "load_exposures",
        "load_risk_metrics",
        "load_var_backtest",
        "load_var_contributions",
        "load_risk_contributions",
        "load_factor_exposures",
        "load_efficient_frontier",
        "load_optimized_portfolio",
        "load_rebalancing_trades",
        "load_data_quality_results",
    ]:
        monkeypatch.setattr(pipeline_db, name, lambda engine, frame, _name=name: calls.append((_name, len(frame))))

    counts = pipeline_db.load_pipeline_outputs("engine", outputs, sql_dir="sql-dir", initialize=True, replace_existing=True)

    assert calls[0] == ("initialize", "sql-dir")
    assert calls[1] == ("truncate", "engine")
    assert ("load_risk_metrics", 1) in calls
    assert counts["risk_metrics"] == 1
    assert counts["var_backtest"] == 2


def test_load_pipeline_outputs_requires_core_frames():
    with pytest.raises(ValueError, match="missing DataFrames"):
        pipeline_db.load_pipeline_outputs("engine", {}, initialize=False, replace_existing=False)
