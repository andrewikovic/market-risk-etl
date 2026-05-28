from datetime import datetime, time
from zoneinfo import ZoneInfo

import pytest

from src.scheduler import SchedulerConfig, next_sleep_seconds, parse_hhmm, parse_positive_minutes


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
        no_write=False,
        load_db=False,
        database_url=None,
        skip_db_init=False,
        log_level="INFO",
    )

    now = datetime(2026, 5, 28, 6, 1, tzinfo=timezone)

    assert next_sleep_seconds(config, now=now) == (23 * 60 + 59) * 60
