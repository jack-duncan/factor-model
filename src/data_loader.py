"""WRDS data pulls with local caching."""

import pandas as pd

from src.config import RAW_DATA_DIR, WRDS_USERNAME


def connect_wrds():
    """Establish WRDS connection."""
    import wrds

    return wrds.Connection(wrds_username=WRDS_USERNAME)


def fetch_crsp_monthly(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """Pull CRSP monthly stock data from WRDS."""
    raise NotImplementedError


def fetch_ff_factors(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """Pull Fama-French factor returns from WRDS."""
    raise NotImplementedError


def load_cached(name: str) -> pd.DataFrame | None:
    """Load a cached parquet file if it exists."""
    import os

    path = f"{RAW_DATA_DIR}/{name}.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def save_cache(df: pd.DataFrame, name: str) -> None:
    """Cache a DataFrame as parquet."""
    df.to_parquet(f"{RAW_DATA_DIR}/{name}.parquet", index=False)