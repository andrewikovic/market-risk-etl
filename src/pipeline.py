from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.extract.extract_positions import load_positions_from_config
from src.extract.extract_prices import ingest_prices, load_assets_config, resolve_price_window
from src.load.db import get_engine
from src.load.pipeline_db import load_pipeline_outputs
from src.quality.data_quality_checks import run_data_quality_checks
from src.risk.backtesting import generate_exception_report
from src.risk.expected_shortfall import calculate_expected_shortfall
from src.risk.factor_model import estimate_factor_exposures
from src.risk.historical_var import calculate_historical_var
from src.risk.optimization import calculate_rebalancing_trades, generate_efficient_frontier, optimize_portfolio
from src.risk.parametric_var import calculate_parametric_var
from src.risk.parametric_var import calculate_component_var
from src.risk.risk_contribution import calculate_asset_risk_contributions
from src.transform.calculate_drawdowns import calculate_drawdowns
from src.transform.calculate_exposures import calculate_current_positions, calculate_exposures
from src.transform.calculate_pnl import calculate_portfolio_values
from src.transform.calculate_returns import calculate_daily_returns
from src.transform.calculate_volatility import calculate_sharpe_ratio, calculate_sortino_ratio
from src.transform.clean_prices import clean_prices


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VAR_CONFIDENCE_LEVELS = (0.95, 0.975, 0.99)
OPTIMIZER_MAX_WEIGHT = 0.40


def run_pipeline(
    project_root: str | Path = PROJECT_ROOT,
    prefer_live: bool = False,
    allow_fallback: bool = True,
    write_processed: bool = True,
    price_start: str | None = None,
    price_end: str | None = None,
    price_lookback_days: int | None = None,
    price_period: str | None = None,
) -> dict[str, pd.DataFrame | dict]:
    """Run the local market-risk ETL pipeline using live data or sample CSV fallback."""
    root = Path(project_root)
    assets = load_assets_config(root / "config" / "assets.yml")
    positions = load_positions_from_config(root / "config" / "portfolio.yml", as_of_date="2024-02-06")
    tickers = assets["ticker"].tolist()
    raw_prices = ingest_prices(
        tickers=tickers,
        start=price_start,
        end=price_end,
        lookback_days=price_lookback_days,
        period=price_period,
        csv_fallback_path=root / "data" / "sample" / "sample_prices.csv",
        prefer_live=prefer_live,
        allow_fallback=allow_fallback,
    )
    stg_prices = clean_prices(raw_prices)
    returns = calculate_daily_returns(stg_prices)
    base_currency = str(positions.get("base_currency", pd.Series(["USD"])).dropna().iloc[0])
    portfolio_values, position_pnl = calculate_portfolio_values(positions, stg_prices, base_currency=base_currency)
    current_positions = calculate_current_positions(positions, stg_prices, assets, base_currency=base_currency)
    exposures = calculate_exposures(positions, stg_prices, assets, base_currency=base_currency)
    quality = run_data_quality_checks(
        raw_prices,
        positions_df=current_positions,
        asset_metadata_df=assets,
        returns_df=returns,
        min_history=20,
    )
    portfolio_return_series = portfolio_values["daily_return"].iloc[1:]
    latest_value = float(portfolio_values["market_value"].iloc[-1])
    drawdowns = calculate_drawdowns(portfolio_values)
    risk_metrics = {
        "historical_var_95": calculate_historical_var(portfolio_return_series, latest_value, 0.95),
        "parametric_var_95": calculate_parametric_var(portfolio_return_series, latest_value, 0.95),
        "expected_shortfall_95": calculate_expected_shortfall(portfolio_return_series, latest_value, 0.95),
        "sharpe_ratio": calculate_sharpe_ratio(portfolio_return_series),
        "sortino_ratio": calculate_sortino_ratio(portfolio_return_series),
        "max_drawdown": drawdowns["max_drawdown"],
    }
    weights = current_positions.set_index("ticker")["weight"].to_dict()
    exposures_by_ticker = current_positions.set_index("ticker")["position_value"].to_dict()
    var_backtest = _calculate_var_backtests(portfolio_values, VAR_CONFIDENCE_LEVELS)
    var_contributions = _calculate_var_contributions(
        returns,
        weights,
        exposures_by_ticker,
        latest_value,
        VAR_CONFIDENCE_LEVELS,
        portfolio_name=str(portfolio_values["portfolio_name"].iloc[-1]),
        metric_date=pd.to_datetime(portfolio_values["value_date"]).max().date(),
    )
    risk_contributions = _calculate_risk_contributions(
        returns,
        weights,
        exposures_by_ticker,
        latest_value,
        portfolio_name=str(portfolio_values["portfolio_name"].iloc[-1]),
        metric_date=pd.to_datetime(portfolio_values["value_date"]).max().date(),
    )
    factor_model = _calculate_factor_model(
        returns,
        weights,
        portfolio_name=str(portfolio_values["portfolio_name"].iloc[-1]),
        metric_date=pd.to_datetime(portfolio_values["value_date"]).max().date(),
    )
    efficient_frontier, optimized_portfolio, rebalancing_trades = _calculate_optimization(
        returns,
        current_positions,
        latest_value,
        portfolio_name=str(portfolio_values["portfolio_name"].iloc[-1]),
        run_date=pd.to_datetime(portfolio_values["value_date"]).max().date(),
    )

    outputs = {
        "assets": assets,
        "raw_prices": raw_prices,
        "stg_prices": stg_prices,
        "returns": returns,
        "positions": positions,
        "current_positions": current_positions,
        "portfolio_values": portfolio_values,
        "position_pnl": position_pnl,
        "exposures": exposures,
        "data_quality": quality,
        "risk_metrics": risk_metrics,
        "drawdowns": drawdowns,
        "var_backtest": var_backtest,
        "var_contributions": var_contributions,
        "risk_contributions": risk_contributions,
        "factor_exposures": factor_model["factor_exposures"],
        "efficient_frontier": efficient_frontier,
        "optimized_portfolio": optimized_portfolio,
        "rebalancing_trades": rebalancing_trades,
    }
    if write_processed:
        processed_dir = root / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        for name, value in outputs.items():
            if isinstance(value, pd.DataFrame):
                value.to_csv(processed_dir / f"{name}.csv", index=False)
    return outputs


def load_scenarios(project_root: str | Path = PROJECT_ROOT) -> dict:
    """Load configured stress scenarios."""
    path = Path(project_root) / "config" / "scenarios.yml"
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {"scenarios": []}


def _calculate_var_backtests(
    portfolio_values: pd.DataFrame,
    confidence_levels: tuple[float, ...],
    min_observations: int = 5,
) -> pd.DataFrame:
    values = portfolio_values.copy()
    values["value_date"] = pd.to_datetime(values["value_date"])
    values = values.sort_values("value_date").reset_index(drop=True)
    if len(values) <= min_observations:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    portfolio_name = str(values["portfolio_name"].iloc[-1])
    for confidence_level in confidence_levels:
        rows = []
        for idx in range(min_observations, len(values)):
            history = values.loc[1 : idx - 1, "daily_return"]
            var_estimate = calculate_historical_var(
                history,
                float(values.loc[idx - 1, "market_value"]),
                confidence_level=confidence_level,
            )
            rows.append(
                {
                    "date": values.loc[idx, "value_date"].date(),
                    "var_estimate": var_estimate,
                    "realized_pnl": float(values.loc[idx, "daily_pnl"]),
                }
            )
        report = generate_exception_report(
            pd.DataFrame(rows),
            pd.DataFrame(rows),
            confidence_level=confidence_level,
            portfolio_name=portfolio_name,
        )
        frames.append(report)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _calculate_var_contributions(
    returns: pd.DataFrame,
    weights: dict,
    exposures: dict,
    portfolio_value: float,
    confidence_levels: tuple[float, ...],
    portfolio_name: str,
    metric_date,
) -> pd.DataFrame:
    frames = []
    for confidence_level in confidence_levels:
        frame = calculate_component_var(
            returns,
            weights,
            portfolio_value=portfolio_value,
            confidence_level=confidence_level,
            exposures=exposures,
        )
        frame["portfolio_name"] = portfolio_name
        frame["metric_date"] = metric_date
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _calculate_risk_contributions(
    returns: pd.DataFrame,
    weights: dict,
    exposures: dict,
    portfolio_value: float,
    portfolio_name: str,
    metric_date,
) -> pd.DataFrame:
    result = calculate_asset_risk_contributions(
        returns,
        weights,
        portfolio_value=portfolio_value,
        confidence_level=0.95,
        exposures=exposures,
    )
    result["portfolio_name"] = portfolio_name
    result["metric_date"] = metric_date
    result["confidence_level"] = 0.95
    return result


def _calculate_factor_model(
    returns: pd.DataFrame,
    weights: dict,
    portfolio_name: str,
    metric_date,
) -> dict[str, pd.DataFrame]:
    factor_returns = _default_factor_returns(returns)
    result = estimate_factor_exposures(returns, factor_returns, weights=weights)
    exposures = result["asset_exposures"].merge(result["asset_summary"], on="ticker", how="left")
    exposures["portfolio_name"] = portfolio_name
    exposures["metric_date"] = metric_date
    exposures["exposure_level"] = "asset"
    if "portfolio_factor_exposure" in result:
        portfolio = result["portfolio_factor_exposure"].rename(columns={"portfolio_beta": "beta"})
        portfolio["ticker"] = "__PORTFOLIO__"
        portfolio["alpha"] = np.nan
        portfolio["residual_volatility"] = np.nan
        portfolio["idiosyncratic_variance"] = np.nan
        portfolio["r_squared"] = np.nan
        portfolio["observations"] = len(factor_returns)
        portfolio["portfolio_name"] = portfolio_name
        portfolio["metric_date"] = metric_date
        portfolio["exposure_level"] = "portfolio"
        exposures = pd.concat([exposures, portfolio], ignore_index=True)
    return {"factor_exposures": exposures}


def _default_factor_returns(returns: pd.DataFrame) -> pd.DataFrame:
    matrix = returns.pivot_table(index="return_date", columns="ticker", values="daily_return", aggfunc="last")
    matrix = matrix.sort_index()
    candidates = [
        ("market", "SPY"),
        ("rates", "TLT"),
        ("commodity", "GLD"),
    ]
    factor_data = pd.DataFrame(index=matrix.index)
    for factor_name, ticker in candidates:
        if ticker in matrix.columns:
            factor_data[factor_name] = matrix[ticker]
    if factor_data.empty:
        for idx, ticker in enumerate(matrix.columns[: min(3, len(matrix.columns))], start=1):
            factor_data[f"factor_{idx}_{ticker}"] = matrix[ticker]
    factor_data = factor_data.reset_index().rename(columns={"index": "return_date"})
    factor_data["return_date"] = pd.to_datetime(factor_data["return_date"]).dt.date
    return factor_data


def _calculate_optimization(
    returns: pd.DataFrame,
    current_positions: pd.DataFrame,
    portfolio_value: float,
    portfolio_name: str,
    run_date,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix = returns.pivot_table(index="return_date", columns="ticker", values="daily_return", aggfunc="last")
    matrix = matrix.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna(how="any")
    expected_returns = matrix.mean() * 252
    covariance_matrix = matrix.cov() * 252
    max_weight = max(OPTIMIZER_MAX_WEIGHT, 1.0 / len(expected_returns))
    frontier = generate_efficient_frontier(
        expected_returns,
        covariance_matrix,
        points=25,
        long_only=True,
        min_weight=0.0,
        max_weight=max_weight,
        risk_free_rate=0.0,
    )
    optimizer = optimize_portfolio(
        expected_returns,
        covariance_matrix,
        long_only=True,
        full_investment=True,
        min_weight=0.0,
        max_weight=max_weight,
        objective="max_sharpe",
        risk_free_rate=0.0,
    )
    current_weights = current_positions.set_index("ticker")["weight"].to_dict()
    latest_prices = current_positions.set_index("ticker")["current_price"].to_dict()
    trades = calculate_rebalancing_trades(
        current_weights,
        optimizer["weights"],
        portfolio_value=portfolio_value,
        prices=latest_prices,
    )

    frontier = frontier.copy()
    frontier["portfolio_name"] = portfolio_name
    frontier["run_date"] = run_date
    frontier["point_number"] = np.arange(1, len(frontier) + 1)

    optimized_rows = []
    diagnostics = optimizer["constraint_diagnostics"]
    for ticker, weight in optimizer["weights"].items():
        optimized_rows.append(
            {
                "portfolio_name": portfolio_name,
                "run_date": run_date,
                "ticker": ticker,
                "target_weight": weight,
                "expected_return": optimizer["expected_return"],
                "volatility": optimizer["volatility"],
                "sharpe_ratio": optimizer["sharpe_ratio"],
                **diagnostics,
            }
        )
    optimized = pd.DataFrame(optimized_rows)

    trades["portfolio_name"] = portfolio_name
    trades["run_date"] = run_date
    return frontier, optimized, trades


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the market-risk ETL pipeline")
    parser.add_argument("--live", action="store_true", help="Use yfinance before falling back to sample CSV")
    parser.add_argument(
        "--require-live",
        action="store_true",
        help="Require yfinance data and fail instead of falling back to sample CSV",
    )
    parser.add_argument("--no-write", action="store_true", help="Do not write processed CSV outputs")
    parser.add_argument("--load-db", action="store_true", help="Load pipeline outputs into PostgreSQL")
    parser.add_argument("--database-url", help="Override DATABASE_URL for this run")
    parser.add_argument("--price-start", help="Inclusive price history start date, formatted as YYYY-MM-DD")
    parser.add_argument("--price-end", help="Exclusive price history end date, formatted as YYYY-MM-DD")
    parser.add_argument(
        "--price-lookback-days",
        type=_positive_int,
        help="Fetch N calendar days of price history ending at --price-end or today",
    )
    parser.add_argument(
        "--price-period",
        help="yfinance period to fetch, such as 1y, 5y, ytd, or max. Cannot be combined with date controls.",
    )
    parser.add_argument(
        "--skip-db-init",
        action="store_true",
        help="Skip PostgreSQL schema initialization before loading outputs",
    )
    args = parser.parse_args()
    try:
        resolve_price_window(
            start=args.price_start,
            end=args.price_end,
            lookback_days=args.price_lookback_days,
            period=args.price_period,
        )
    except ValueError as exc:
        parser.error(str(exc))

    outputs = run_pipeline(
        prefer_live=args.live or args.require_live,
        allow_fallback=not args.require_live,
        write_processed=not args.no_write,
        price_start=args.price_start,
        price_end=args.price_end,
        price_lookback_days=args.price_lookback_days,
        price_period=args.price_period,
    )
    print("Pipeline completed")
    print(f"Price source: {', '.join(sorted(outputs['raw_prices']['source'].dropna().unique()))}")
    print(f"Price rows: {len(outputs['raw_prices'])}")
    price_dates = pd.to_datetime(outputs["raw_prices"]["price_date"])
    print(f"Price date range: {price_dates.min().date()} to {price_dates.max().date()}")
    print(f"Return rows: {len(outputs['returns'])}")
    print(f"Portfolio dates: {len(outputs['portfolio_values'])}")
    if args.load_db:
        engine = get_engine(args.database_url)
        counts = load_pipeline_outputs(
            engine,
            outputs,
            sql_dir=PROJECT_ROOT / "sql",
            initialize=not args.skip_db_init,
            replace_existing=True,
        )
        loaded = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        print(f"PostgreSQL load completed: {loaded}")


if __name__ == "__main__":
    main()
