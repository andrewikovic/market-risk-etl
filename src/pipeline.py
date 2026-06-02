from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.extract.extract_positions import load_positions_from_config
from src.extract.extract_prices import ingest_prices, load_assets_config, resolve_price_window
from src.load.db import get_engine
from src.load.pipeline_db import load_pipeline_outputs
from src.quality.data_quality_checks import run_data_quality_checks
from src.risk.expected_shortfall import calculate_expected_shortfall
from src.risk.historical_var import calculate_historical_var
from src.risk.parametric_var import calculate_parametric_var
from src.transform.calculate_drawdowns import calculate_drawdowns
from src.transform.calculate_exposures import calculate_current_positions, calculate_exposures
from src.transform.calculate_pnl import calculate_portfolio_values
from src.transform.calculate_returns import calculate_daily_returns
from src.transform.calculate_volatility import calculate_sharpe_ratio, calculate_sortino_ratio
from src.transform.clean_prices import clean_prices


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    portfolio_values, position_pnl = calculate_portfolio_values(positions, stg_prices)
    current_positions = calculate_current_positions(positions, stg_prices, assets)
    exposures = calculate_exposures(positions, stg_prices, assets)
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
