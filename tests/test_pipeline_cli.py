import sys

import pandas as pd
import pytest

import src.pipeline as pipeline


def test_load_scenarios_reads_configured_scenarios(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "scenarios.yml").write_text(
        "scenarios:\n  - name: selloff\n    shocks:\n      SPY: -0.05\n",
        encoding="utf-8",
    )

    assert pipeline.load_scenarios(tmp_path)["scenarios"][0]["name"] == "selloff"

    (config_dir / "scenarios.yml").write_text("", encoding="utf-8")

    assert pipeline.load_scenarios(tmp_path) == {"scenarios": []}


def test_pipeline_helper_branches():
    short_values = pd.DataFrame(
        {
            "portfolio_name": ["P"],
            "value_date": pd.to_datetime(["2024-01-01"]),
            "market_value": [100.0],
            "daily_return": [0.0],
            "daily_pnl": [0.0],
        }
    )

    assert pipeline._calculate_var_backtests(short_values, (0.95,), min_observations=5).empty

    returns = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "return_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "daily_return": [0.01, 0.02, 0.03, 0.04],
        }
    )

    factors = pipeline._default_factor_returns(returns)

    assert list(factors.columns) == ["return_date", "factor_1_AAA", "factor_2_BBB"]


def test_positive_int_parser():
    assert pipeline._positive_int("3") == 3
    with pytest.raises(pipeline.argparse.ArgumentTypeError, match="integer"):
        pipeline._positive_int("bad")
    with pytest.raises(pipeline.argparse.ArgumentTypeError, match="greater than zero"):
        pipeline._positive_int("0")


def test_main_runs_pipeline_and_loads_database(monkeypatch, capsys):
    captured = {}
    outputs = {
        "raw_prices": pd.DataFrame(
            {
                "source": ["sample", "sample"],
                "price_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            }
        ),
        "returns": pd.DataFrame({"daily_return": [0.01]}),
        "portfolio_values": pd.DataFrame({"market_value": [100.0, 101.0]}),
    }

    def fake_run_pipeline(**kwargs):
        captured["pipeline_kwargs"] = kwargs
        return outputs

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(pipeline, "get_engine", lambda database_url: captured.update(database_url=database_url) or "engine")
    monkeypatch.setattr(
        pipeline,
        "load_pipeline_outputs",
        lambda engine, loaded_outputs, **kwargs: captured.update(
            load_engine=engine,
            loaded_outputs=loaded_outputs,
            load_kwargs=kwargs,
        )
        or {"raw_prices": 2, "risk_metrics": 1},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pipeline",
            "--require-live",
            "--no-write",
            "--price-end",
            "2024-02-01",
            "--price-lookback-days",
            "10",
            "--load-db",
            "--database-url",
            "postgresql://example",
            "--skip-db-init",
        ],
    )

    pipeline.main()

    assert captured["pipeline_kwargs"] == {
        "prefer_live": True,
        "allow_fallback": False,
        "write_processed": False,
        "price_start": None,
        "price_end": "2024-02-01",
        "price_lookback_days": 10,
        "price_period": None,
    }
    assert captured["database_url"] == "postgresql://example"
    assert captured["load_engine"] == "engine"
    assert captured["loaded_outputs"] is outputs
    assert captured["load_kwargs"]["initialize"] is False
    out = capsys.readouterr().out
    assert "Pipeline completed" in out
    assert "Price rows: 2" in out
    assert "PostgreSQL load completed: raw_prices=2, risk_metrics=1" in out


def test_main_rejects_invalid_price_window(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pipeline", "--price-period", "max", "--price-start", "2024-01-01"])

    with pytest.raises(SystemExit):
        pipeline.main()
