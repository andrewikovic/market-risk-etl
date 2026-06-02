from datetime import datetime, time
from zoneinfo import ZoneInfo

import pytest

from src.scheduler import SchedulerConfig, next_sleep_seconds, parse_args, parse_hhmm, parse_positive_minutes


def test_parse_hhmm_accepts_valid_daily_time():
    assert parse_hhmm("06:30") == time(hour=6, minute=30)


@pytest.mark.parametrize("value", ["24:00", "12:60", "bad"])
def test_parse_hhmm_rejects_invalid_daily_time(value):
    with pytest.raises(ValueError):
        parse_hhmm(value)


def test_parse_positive_minutes_rejects_non_positive_values():
    with pytest.raises(ValueError):
        parse_positive_minutes("0")


def test_next_sleep_seconds_uses_daily_time_before_target():
    timezone = ZoneInfo("UTC")
    config = SchedulerConfig(
        daily_at=time(hour=6, minute=0),
        interval_minutes=1440,
        timezone=timezone,
        run_on_start=False,
        live=False,
        require_live=False,
        price_start=None,
        price_end=None,
        price_lookback_days=None,
        price_period=None,
        no_write=False,
        load_db=False,
        database_url=None,
        skip_db_init=False,
        log_level="INFO",
    )

    now = datetime(2026, 5, 28, 5, 30, tzinfo=timezone)

    assert next_sleep_seconds(config, now=now) == 30 * 60


def test_next_sleep_seconds_rolls_daily_time_to_tomorrow():
    timezone = ZoneInfo("UTC")
    config = SchedulerConfig(
        daily_at=time(hour=6, minute=0),
        interval_minutes=1440,
        timezone=timezone,
        run_on_start=False,
        live=False,
        require_live=False,
        price_start=None,
        price_end=None,
        price_lookback_days=None,
        price_period=None,
        no_write=False,
        load_db=False,
        database_url=None,
        skip_db_init=False,
        log_level="INFO",
    )

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
