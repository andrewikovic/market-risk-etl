from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, text


DEFAULT_DATABASE_URL = "postgresql+psycopg://risk_user:risk_password@localhost:5432/market_risk"


def get_database_url(env_path: str | Path | None = None) -> str:
    """Read the PostgreSQL database URL from the environment."""
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def get_engine(database_url: str | None = None) -> Engine:
    """Create a SQLAlchemy engine for PostgreSQL."""
    return create_engine(database_url or get_database_url(), pool_pre_ping=True, future=True)


def execute_sql_file(engine: Engine, sql_path: str | Path) -> None:
    """Execute a SQL file containing one or more DDL statements."""
    path = Path(sql_path)
    sql = path.read_text(encoding="utf-8")
    statements = [statement.strip() for statement in sql.split(";") if statement.strip()]
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def initialize_database(engine: Engine, sql_dir: str | Path) -> None:
    """Create all database schemas, tables, and indexes."""
    path = Path(sql_dir)
    for filename in [
        "01_raw_schema.sql",
        "02_staging_schema.sql",
        "03_mart_schema.sql",
        "04_indexes.sql",
    ]:
        execute_sql_file(engine, path / filename)

