from __future__ import annotations

import argparse
import logging
import os
import signal
import time as time_module
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.load.db import get_engine
from src.load.pipeline_db import load_pipeline_outputs
from src.pipeline import PROJECT_ROOT, run_pipeline


LOGGER = logging.getLogger(__name__)
STOP_REQUESTED = False


@dataclass(frozen=True)
class SchedulerConfig:
    daily_at: time | None
    interval_minutes: int
    timezone: ZoneInfo
    run_on_start: bool
    live: bool
    require_live: bool
    no_write: bool
    load_db: bool
    database_url: str | None
    skip_db_init: bool
    log_level: str


def env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_hhmm(value: str) -> time:
    """Parse an HH:MM value into a time object."""
    try:
        hour_text, minute_text = value.strip().split(":", maxsplit=1)
        hour = int(hour_text)
        minute = int(minute_text)
    except ValueError as exc:
        raise ValueError("daily time must be formatted as HH:MM") from exc

    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("daily time must be between 00:00 and 23:59")
    return time(hour=hour, minute=minute)


def parse_positive_minutes(value: str) -> int:
    """Parse a positive minute interval."""
    try:
        minutes = int(value)
    except ValueError as exc:
        raise ValueError("interval minutes must be an integer") from exc
    if minutes <= 0:
        raise ValueError("interval minutes must be greater than zero")
    return minutes


def seconds_until_next_daily_run(now: datetime, daily_at: time, timezone: ZoneInfo) -> int:
    """Return seconds until the next run at daily_at in the requested timezone."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone)
    local_now = now.astimezone(timezone)
    next_run = datetime.combine(local_now.date(), daily_at, tzinfo=timezone)
    if next_run <= local_now:
        next_run += timedelta(days=1)
    return max(1, int((next_run - local_now).total_seconds()))


def next_sleep_seconds(config: SchedulerConfig, now: datetime | None = None) -> int:
    """Compute the delay before the next scheduled run."""
    if config.daily_at is not None:
        return seconds_until_next_daily_run(now or datetime.now(config.timezone), config.daily_at, config.timezone)
    return config.interval_minutes * 60


def run_scheduled_pipeline(config: SchedulerConfig) -> None:
    """Run the ETL once using scheduler configuration."""
    outputs = run_pipeline(
        prefer_live=config.live or config.require_live,
        allow_fallback=not config.require_live,
        write_processed=not config.no_write,
    )
    LOGGER.info(
        "Pipeline completed: price_rows=%s return_rows=%s portfolio_dates=%s",
        len(outputs["raw_prices"]),
        len(outputs["returns"]),
        len(outputs["portfolio_values"]),
    )

    if config.load_db:
        engine = get_engine(config.database_url)
        counts = load_pipeline_outputs(
            engine,
            outputs,
            sql_dir=PROJECT_ROOT / "sql",
            initialize=not config.skip_db_init,
            replace_existing=True,
        )
        loaded = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        LOGGER.info("PostgreSQL load completed: %s", loaded)


def sleep_until_next_run(seconds: int) -> bool:
    """Sleep in short chunks so shutdown signals can stop the scheduler promptly."""
    deadline = time_module.monotonic() + seconds
    while not STOP_REQUESTED:
        remaining = deadline - time_module.monotonic()
        if remaining <= 0:
            return True
        time_module.sleep(min(remaining, 30))
    return False


def build_parser() -> argparse.ArgumentParser:
    default_daily_at = os.getenv("ETL_DAILY_AT")
    default_interval = parse_positive_minutes(os.getenv("ETL_INTERVAL_MINUTES", "1440"))

    parser = argparse.ArgumentParser(description="Run the market-risk ETL on a schedule")
    parser.add_argument(
        "--daily-at",
        type=_argparse_hhmm,
        default=parse_hhmm(default_daily_at) if default_daily_at else None,
        help="Run once per day at HH:MM in the selected timezone",
    )
    parser.add_argument(
        "--interval-minutes",
        type=_argparse_positive_minutes,
        default=default_interval,
        help="Run every N minutes when --daily-at is not set",
    )
    parser.add_argument(
        "--timezone",
        default=os.getenv("ETL_TIMEZONE") or os.getenv("TZ") or "UTC",
        help="IANA timezone for --daily-at, for example America/Edmonton",
    )
    parser.add_argument(
        "--run-on-start",
        action="store_true",
        default=env_flag("ETL_RUN_ON_START", False),
        help="Run immediately before waiting for the first scheduled time",
    )
    parser.add_argument("--live", action="store_true", default=env_flag("ETL_LIVE", False))
    parser.add_argument("--require-live", action="store_true", default=env_flag("ETL_REQUIRE_LIVE", False))
    parser.add_argument("--no-write", action="store_true", default=env_flag("ETL_NO_WRITE", False))
    parser.add_argument("--load-db", action="store_true", default=env_flag("ETL_LOAD_DB", False))
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--skip-db-init", action="store_true", default=env_flag("ETL_SKIP_DB_INIT", False))
    parser.add_argument("--log-level", default=os.getenv("ETL_LOG_LEVEL", "INFO"))
    return parser


def parse_args(argv: list[str] | None = None) -> SchedulerConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.live and args.require_live:
        parser.error("--live and --require-live cannot both be enabled")

    try:
        timezone = ZoneInfo(args.timezone)
    except ZoneInfoNotFoundError as exc:
        parser.error(f"unknown timezone: {args.timezone}")
        raise exc

    return SchedulerConfig(
        daily_at=args.daily_at,
        interval_minutes=args.interval_minutes,
        timezone=timezone,
        run_on_start=args.run_on_start,
        live=args.live,
        require_live=args.require_live,
        no_write=args.no_write,
        load_db=args.load_db,
        database_url=args.database_url,
        skip_db_init=args.skip_db_init,
        log_level=args.log_level,
    )


def run_forever(config: SchedulerConfig) -> None:
    if config.run_on_start:
        LOGGER.info("Running scheduled ETL immediately")
        run_scheduled_pipeline(config)

    while not STOP_REQUESTED:
        seconds = next_sleep_seconds(config)
        LOGGER.info("Next ETL run in %.1f minutes", seconds / 60)
        if not sleep_until_next_run(seconds):
            break
        LOGGER.info("Starting scheduled ETL run")
        run_scheduled_pipeline(config)


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    run_forever(config)


def _argparse_hhmm(value: str) -> time:
    try:
        return parse_hhmm(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _argparse_positive_minutes(value: str) -> int:
    try:
        return parse_positive_minutes(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _request_stop(signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    LOGGER.info("Received signal %s; stopping after current wait or run", signum)
    STOP_REQUESTED = True


if __name__ == "__main__":
    main()
