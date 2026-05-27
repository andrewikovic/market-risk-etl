CREATE INDEX IF NOT EXISTS idx_raw_prices_ticker_date ON raw.prices (ticker, price_date);
CREATE INDEX IF NOT EXISTS idx_stg_prices_ticker_date ON staging.stg_daily_prices (ticker, price_date);
CREATE INDEX IF NOT EXISTS idx_mart_returns_ticker_date ON mart.daily_returns (ticker, return_date);
CREATE INDEX IF NOT EXISTS idx_mart_portfolio_values_date ON mart.portfolio_values (portfolio_name, value_date);
CREATE INDEX IF NOT EXISTS idx_mart_position_pnl_date ON mart.position_pnl (portfolio_name, value_date);
CREATE INDEX IF NOT EXISTS idx_mart_risk_metrics_date ON mart.risk_metrics (portfolio_name, metric_date);
CREATE INDEX IF NOT EXISTS idx_mart_exposures_date ON mart.exposures (portfolio_name, exposure_date);
CREATE INDEX IF NOT EXISTS idx_mart_quality_run ON mart.data_quality_results (run_timestamp, status, severity);

