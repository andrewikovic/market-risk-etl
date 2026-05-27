CREATE SCHEMA IF NOT EXISTS staging;

CREATE TABLE IF NOT EXISTS staging.stg_daily_prices (
    ticker TEXT NOT NULL,
    price_date DATE NOT NULL,
    adjusted_close NUMERIC(18, 6),
    volume BIGINT,
    is_stale BOOLEAN NOT NULL DEFAULT FALSE,
    is_missing BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (ticker, price_date)
);

