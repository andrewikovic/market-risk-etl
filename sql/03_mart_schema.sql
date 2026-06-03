CREATE SCHEMA IF NOT EXISTS mart;

CREATE TABLE IF NOT EXISTS mart.daily_returns (
    ticker TEXT NOT NULL,
    return_date DATE NOT NULL,
    daily_return NUMERIC(18, 10) NOT NULL,
    log_return NUMERIC(18, 10) NOT NULL,
    PRIMARY KEY (ticker, return_date)
);

CREATE TABLE IF NOT EXISTS mart.portfolio_values (
    portfolio_name TEXT NOT NULL,
    value_date DATE NOT NULL,
    market_value NUMERIC(18, 6) NOT NULL,
    daily_pnl NUMERIC(18, 6) NOT NULL,
    daily_return NUMERIC(18, 10) NOT NULL,
    cumulative_return NUMERIC(18, 10) NOT NULL,
    PRIMARY KEY (portfolio_name, value_date)
);

CREATE TABLE IF NOT EXISTS mart.position_pnl (
    portfolio_name TEXT NOT NULL,
    value_date DATE NOT NULL,
    ticker TEXT NOT NULL,
    position_value NUMERIC(18, 6) NOT NULL,
    daily_pnl NUMERIC(18, 6) NOT NULL,
    contribution_to_pnl NUMERIC(18, 6) NOT NULL,
    contribution_to_return NUMERIC(18, 10) NOT NULL,
    weight NUMERIC(18, 10) NOT NULL,
    currency TEXT,
    fx_rate_to_base NUMERIC(18, 10),
    base_currency TEXT,
    PRIMARY KEY (portfolio_name, value_date, ticker)
);

ALTER TABLE mart.position_pnl
    ADD COLUMN IF NOT EXISTS currency TEXT,
    ADD COLUMN IF NOT EXISTS fx_rate_to_base NUMERIC(18, 10),
    ADD COLUMN IF NOT EXISTS base_currency TEXT;

CREATE TABLE IF NOT EXISTS mart.risk_metrics (
    portfolio_name TEXT NOT NULL,
    metric_date DATE NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value NUMERIC(18, 10) NOT NULL,
    lookback_days INTEGER,
    confidence_level NUMERIC(6, 4),
    PRIMARY KEY (portfolio_name, metric_date, metric_name)
);

CREATE TABLE IF NOT EXISTS mart.exposures (
    portfolio_name TEXT NOT NULL,
    exposure_date DATE NOT NULL,
    exposure_type TEXT NOT NULL,
    exposure_name TEXT NOT NULL,
    market_value NUMERIC(18, 6) NOT NULL,
    weight NUMERIC(18, 10) NOT NULL,
    base_currency TEXT,
    PRIMARY KEY (portfolio_name, exposure_date, exposure_type, exposure_name)
);

ALTER TABLE mart.exposures
    ADD COLUMN IF NOT EXISTS base_currency TEXT;

CREATE TABLE IF NOT EXISTS mart.var_backtest_exceptions (
    portfolio_name TEXT NOT NULL,
    backtest_date DATE NOT NULL,
    confidence_level NUMERIC(6, 4) NOT NULL,
    var_estimate NUMERIC(18, 6) NOT NULL,
    realized_pnl NUMERIC(18, 6) NOT NULL,
    breach BOOLEAN NOT NULL,
    breach_severity NUMERIC(18, 6) NOT NULL,
    total_observations INTEGER NOT NULL,
    number_of_exceptions INTEGER NOT NULL,
    expected_exceptions NUMERIC(18, 6) NOT NULL,
    exception_ratio NUMERIC(18, 10),
    kupiec_statistic NUMERIC(18, 10),
    p_value NUMERIC(18, 10),
    pass_fail TEXT NOT NULL,
    PRIMARY KEY (portfolio_name, backtest_date, confidence_level)
);

CREATE TABLE IF NOT EXISTS mart.var_contributions (
    portfolio_name TEXT NOT NULL,
    metric_date DATE NOT NULL,
    confidence_level NUMERIC(6, 4) NOT NULL,
    ticker TEXT NOT NULL,
    weight NUMERIC(18, 10) NOT NULL,
    exposure NUMERIC(18, 6) NOT NULL,
    mean_return NUMERIC(18, 10) NOT NULL,
    volatility NUMERIC(18, 10) NOT NULL,
    marginal_var NUMERIC(18, 6) NOT NULL,
    component_var NUMERIC(18, 6) NOT NULL,
    percent_contribution NUMERIC(18, 10),
    portfolio_var NUMERIC(18, 6) NOT NULL,
    contribution_reconciliation_error NUMERIC(18, 10) NOT NULL,
    PRIMARY KEY (portfolio_name, metric_date, confidence_level, ticker)
);

CREATE TABLE IF NOT EXISTS mart.risk_contributions (
    portfolio_name TEXT NOT NULL,
    metric_date DATE NOT NULL,
    confidence_level NUMERIC(6, 4) NOT NULL,
    ticker TEXT NOT NULL,
    weight NUMERIC(18, 10) NOT NULL,
    exposure NUMERIC(18, 6) NOT NULL,
    asset_volatility NUMERIC(18, 10) NOT NULL,
    marginal_volatility NUMERIC(18, 10) NOT NULL,
    component_volatility NUMERIC(18, 10) NOT NULL,
    volatility_contribution_amount NUMERIC(18, 6) NOT NULL,
    marginal_var NUMERIC(18, 6) NOT NULL,
    component_var NUMERIC(18, 6) NOT NULL,
    volatility_percent_contribution NUMERIC(18, 10),
    var_percent_contribution NUMERIC(18, 10),
    portfolio_volatility NUMERIC(18, 10) NOT NULL,
    portfolio_var NUMERIC(18, 6) NOT NULL,
    volatility_reconciliation_error NUMERIC(18, 10) NOT NULL,
    var_reconciliation_error NUMERIC(18, 10) NOT NULL,
    PRIMARY KEY (portfolio_name, metric_date, confidence_level, ticker)
);

CREATE TABLE IF NOT EXISTS mart.factor_exposures (
    portfolio_name TEXT NOT NULL,
    metric_date DATE NOT NULL,
    exposure_level TEXT NOT NULL,
    ticker TEXT NOT NULL,
    factor TEXT NOT NULL,
    beta NUMERIC(18, 10) NOT NULL,
    alpha NUMERIC(18, 10),
    residual_volatility NUMERIC(18, 10),
    idiosyncratic_variance NUMERIC(18, 10),
    r_squared NUMERIC(18, 10),
    observations INTEGER,
    PRIMARY KEY (portfolio_name, metric_date, exposure_level, ticker, factor)
);

CREATE TABLE IF NOT EXISTS mart.efficient_frontier (
    portfolio_name TEXT NOT NULL,
    run_date DATE NOT NULL,
    point_number INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    target_return NUMERIC(18, 10) NOT NULL,
    expected_return NUMERIC(18, 10) NOT NULL,
    volatility NUMERIC(18, 10) NOT NULL,
    sharpe_ratio NUMERIC(18, 10),
    weight NUMERIC(18, 10) NOT NULL,
    PRIMARY KEY (portfolio_name, run_date, point_number, ticker)
);

CREATE TABLE IF NOT EXISTS mart.optimized_portfolio (
    portfolio_name TEXT NOT NULL,
    run_date DATE NOT NULL,
    ticker TEXT NOT NULL,
    target_weight NUMERIC(18, 10) NOT NULL,
    expected_return NUMERIC(18, 10) NOT NULL,
    volatility NUMERIC(18, 10) NOT NULL,
    sharpe_ratio NUMERIC(18, 10),
    weight_sum NUMERIC(18, 10) NOT NULL,
    full_investment BOOLEAN NOT NULL,
    long_only BOOLEAN,
    min_weight_satisfied BOOLEAN NOT NULL,
    max_weight_satisfied BOOLEAN NOT NULL,
    target_return_error NUMERIC(18, 10),
    target_volatility_error NUMERIC(18, 10),
    PRIMARY KEY (portfolio_name, run_date, ticker)
);

CREATE TABLE IF NOT EXISTS mart.rebalancing_trades (
    portfolio_name TEXT NOT NULL,
    run_date DATE NOT NULL,
    ticker TEXT NOT NULL,
    current_weight NUMERIC(18, 10) NOT NULL,
    target_weight NUMERIC(18, 10) NOT NULL,
    weight_change NUMERIC(18, 10) NOT NULL,
    trade_value NUMERIC(18, 6),
    price NUMERIC(18, 6),
    quantity_change NUMERIC(18, 10),
    PRIMARY KEY (portfolio_name, run_date, ticker)
);

CREATE TABLE IF NOT EXISTS mart.stress_test_results (
    portfolio_name TEXT NOT NULL,
    scenario_name TEXT NOT NULL,
    run_timestamp TIMESTAMPTZ NOT NULL,
    shocked_portfolio_value NUMERIC(18, 6) NOT NULL,
    stress_loss NUMERIC(18, 6) NOT NULL,
    stress_return NUMERIC(18, 10) NOT NULL,
    PRIMARY KEY (portfolio_name, scenario_name, run_timestamp)
);

CREATE TABLE IF NOT EXISTS mart.monte_carlo_runs (
    run_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    portfolio_name TEXT NOT NULL,
    run_timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    initial_value NUMERIC(18, 6) NOT NULL,
    horizon_days INTEGER NOT NULL,
    n_simulations INTEGER NOT NULL,
    confidence_level NUMERIC(6, 4) NOT NULL,
    lookback_days INTEGER,
    random_seed INTEGER
);

CREATE TABLE IF NOT EXISTS mart.monte_carlo_results (
    run_id BIGINT NOT NULL REFERENCES mart.monte_carlo_runs(run_id),
    metric_name TEXT NOT NULL,
    metric_value NUMERIC(18, 10) NOT NULL,
    PRIMARY KEY (run_id, metric_name)
);

CREATE TABLE IF NOT EXISTS mart.monte_carlo_terminal_values (
    run_id BIGINT NOT NULL REFERENCES mart.monte_carlo_runs(run_id),
    simulation_id INTEGER NOT NULL,
    terminal_value NUMERIC(18, 6) NOT NULL,
    terminal_return NUMERIC(18, 10) NOT NULL,
    max_drawdown NUMERIC(18, 10) NOT NULL,
    PRIMARY KEY (run_id, simulation_id)
);

CREATE TABLE IF NOT EXISTS mart.data_quality_results (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    check_name TEXT NOT NULL,
    status TEXT NOT NULL,
    severity TEXT NOT NULL,
    ticker TEXT,
    check_date DATE,
    message TEXT NOT NULL,
    run_timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
