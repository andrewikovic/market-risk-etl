from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import src.scheduler as scheduler
from src.scheduler import (
    SchedulerConfig,
    env_flag,
    next_sleep_seconds,
    parse_args,
    parse_hhmm,
    parse_positive_minutes,
    seconds_until_next_daily_run,
)


def _config(**overrides):
    values = {
        "daily_at": None,
        "interval_minutes": 1440,
        "timezone": ZoneInfo("UTC"),
        "run_on_start": False,
        "live": False,
        "require_live": False,
        "price_start": None,
        "price_end": None,
        "price_lookback_days": None,
        "price_period": None,
        "no_write": False,
        "load_db": False,
        "database_url": None,
        "skip_db_init": False,
        "log_level": "INFO",
    }
    values.update(overrides)
    return SchedulerConfig(**values)


def test_parse_hhmm_accepts_valid_daily_time():
    assert parse_hhmm("06:30") == time(hour=6, minute=30)


def test_env_flag_parses_truthy_falsey_and_default(monkeypatch):
    monkeypatch.setenv("FLAG", "yes")
    assert env_flag("FLAG") is True
    monkeypatch.setenv("FLAG", "off")
    assert env_flag("FLAG") is False
    monkeypatch.delenv("FLAG")
    assert env_flag("FLAG", default=True) is True


@pytest.mark.parametrize("value", ["24:00", "12:60", "bad"])
def test_parse_hhmm_rejects_invalid_daily_time(value):
    with pytest.raises(ValueError):
        parse_hhmm(value)


def test_parse_positive_minutes_rejects_non_positive_values():
    with pytest.raises(ValueError):
        parse_positive_minutes("0")

    with pytest.raises(ValueError):
        parse_positive_minutes("bad")


def test_seconds_until_next_daily_run_accepts_naive_datetime():
    assert seconds_until_next_daily_run(datetime(2026, 5, 28, 5, 0), time(hour=6), ZoneInfo("UTC")) == 60 * 60


def test_next_sleep_seconds_uses_daily_time_before_target():
    timezone = ZoneInfo("UTC")
    config = _config(daily_at=time(hour=6, minute=0), timezone=timezone)

    now = datetime(2026, 5, 28, 5, 30, tzinfo=timezone)

    assert next_sleep_seconds(config, now=now) == 30 * 60


def test_next_sleep_seconds_rolls_daily_time_to_tomorrow():
    timezone = ZoneInfo("UTC")
    config = _config(daily_at=time(hour=6, minute=0), timezone=timezone)

    now = datetime(2026, 5, 28, 6, 1, tzinfo=timezone)

    assert next_sleep_seconds(config, now=now) == (23 * 60 + 59) * 60


def test_parse_args_includes_price_lookback_controls(monkeypatch):
    monkeypatch.setenv("ETL_PRICE_LOOKBACK_DAYS", "365")
    monkeypatch.setenv("ETL_PRICE_END", "2026-01-01")
    monkeypatch.delenv("ETL_DAILY_AT", raising=False)
    monkeypatch.delenv("ETL_PRICE_START", raising=False)
    monkeypatch.delenv("ETL_PRICE_PERIOD", raising=False)

    config = parse_args(["--timezone", "UTC"])

    assert config.price_lookback_days == 365
    assert config.price_end == "2026-01-01"
    assert config.price_start is None
    assert config.price_period is None


def test_parse_args_rejects_conflicting_or_invalid_options(monkeypatch):
    monkeypatch.delenv("ETL_DAILY_AT", raising=False)
    monkeypatch.delenv("ETL_PRICE_LOOKBACK_DAYS", raising=False)
    monkeypatch.delenv("ETL_INTERVAL_MINUTES", raising=False)

    with pytest.raises(SystemExit):
        parse_args(["--live", "--require-live"])
    with pytest.raises(SystemExit):
        parse_args(["--timezone", "Not/AZone"])
    with pytest.raises(SystemExit):
        parse_args(["--price-period", "max", "--price-start", "2024-01-01"])
    with pytest.raises(SystemExit):
        parse_args(["--price-lookback-days", "0"])


def test_run_scheduled_pipeline_passes_config_and_loads_db(monkeypatch):
    captured = {}

    def fake_run_pipeline(**kwargs):
        captured["pipeline_kwargs"] = kwargs
        return {
            "raw_prices": pd.DataFrame({"x": [1, 2]}),
            "returns": pd.DataFrame({"x": [1]}),
            "portfolio_values": pd.DataFrame({"x": [1, 2, 3]}),
        }

    monkeypatch.setattr(scheduler, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(scheduler, "get_engine", lambda database_url: captured.update(database_url=database_url) or "engine")
    monkeypatch.setattr(
        scheduler,
        "load_pipeline_outputs",
        lambda engine, outputs, **kwargs: captured.update(load_engine=engine, load_kwargs=kwargs) or {"raw_prices": 2},
    )

    scheduler.run_scheduled_pipeline(
        _config(
            require_live=True,
            no_write=True,
            load_db=True,
            database_url="postgresql://example",
            skip_db_init=True,
            price_end="2024-02-01",
            price_lookback_days=10,
        )
    )

    assert captured["pipeline_kwargs"]["prefer_live"] is True
    assert captured["pipeline_kwargs"]["allow_fallback"] is False
    assert captured["pipeline_kwargs"]["write_processed"] is False
    assert captured["pipeline_kwargs"]["price_lookback_days"] == 10
    assert captured["database_url"] == "postgresql://example"
    assert captured["load_engine"] == "engine"
    assert captured["load_kwargs"]["initialize"] is False


def test_sleep_until_next_run_completes_or_stops(monkeypatch):
    monkeypatch.setattr(scheduler, "STOP_REQUESTED", False)
    monotonic_values = iter([0.0, 0.2, 1.1])
    monkeypatch.setattr(scheduler.time_module, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(scheduler.time_module, "sleep", lambda seconds: None)

    assert scheduler.sleep_until_next_run(1) is True

    monkeypatch.setattr(scheduler, "STOP_REQUESTED", True)
    monkeypatch.setattr(scheduler.time_module, "monotonic", lambda: 0.0)
    assert scheduler.sleep_until_next_run(1) is False
    monkeypatch.setattr(scheduler, "STOP_REQUESTED", False)


def test_run_forever_runs_on_start_and_scheduled_cycle(monkeypatch):
    calls = []
    monkeypatch.setattr(scheduler, "STOP_REQUESTED", False)

    def fake_run(config):
        calls.append(config)
        if len(calls) == 2:
            monkeypatch.setattr(scheduler, "STOP_REQUESTED", True)

    monkeypatch.setattr(scheduler, "run_scheduled_pipeline", fake_run)
    monkeypatch.setattr(scheduler, "sleep_until_next_run", lambda seconds: True)

    config = _config(run_on_start=True, interval_minutes=1)
    scheduler.run_forever(config)

    assert calls == [config, config]
    monkeypatch.setattr(scheduler, "STOP_REQUESTED", False)


def test_main_configures_signals_and_runs_forever(monkeypatch):
    config = _config(log_level="DEBUG")
    captured = {"signals": []}
    monkeypatch.setattr(scheduler, "parse_args", lambda argv: config)
    monkeypatch.setattr(scheduler, "run_forever", lambda parsed: captured.update(config=parsed))
    monkeypatch.setattr(scheduler.signal, "signal", lambda signum, handler: captured["signals"].append((signum, handler)))

    scheduler.main(["--timezone", "UTC"])

    assert captured["config"] == config
    assert [signum for signum, _ in captured["signals"]] == [scheduler.signal.SIGINT, scheduler.signal.SIGTERM]


def test_argparse_wrappers_and_stop_request():
    assert scheduler._argparse_hhmm("07:15") == time(hour=7, minute=15)
    assert scheduler._argparse_positive_minutes("5") == 5
    assert scheduler._argparse_positive_days("3") == 3
    with pytest.raises(scheduler.argparse.ArgumentTypeError):
        scheduler._argparse_hhmm("bad")
    with pytest.raises(scheduler.argparse.ArgumentTypeError):
        scheduler._argparse_positive_minutes("bad")
    with pytest.raises(scheduler.argparse.ArgumentTypeError):
        scheduler._argparse_positive_days("bad")
    with pytest.raises(scheduler.argparse.ArgumentTypeError):
        scheduler._argparse_positive_days("0")

    scheduler.STOP_REQUESTED = False
    scheduler._request_stop(15, None)
    assert scheduler.STOP_REQUESTED is True
    scheduler.STOP_REQUESTED = False
