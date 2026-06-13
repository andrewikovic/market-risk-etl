from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import Engine


MAX_INSERT_PARAMETERS = 60_000


def write_frame_to_sql(
    frame: pd.DataFrame,
    table_name: str,
    engine: Engine,
    *,
    schema: str,
    if_exists: str = "append",
    dtype: dict[str, Any] | None = None,
) -> None:
    """Write a DataFrame with bounded multi-row INSERT batches."""
    column_count = max(len(frame.columns), 1)
    chunksize = max(1, MAX_INSERT_PARAMETERS // column_count)
    frame.to_sql(
        table_name,
        engine,
        schema=schema,
        if_exists=if_exists,
        index=False,
        method="multi",
        chunksize=chunksize,
        dtype=dtype,
    )
