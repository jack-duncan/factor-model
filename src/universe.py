"""Top-N universe construction by market cap."""

import pandas as pd

from src.config import UNIVERSE_SIZE


def build_universe(crsp: pd.DataFrame, date_col: str = "date", mcap_col: str = "mcap") -> pd.DataFrame:
    """Select the top UNIVERSE_SIZE stocks by market cap each period.

    Returns a DataFrame with (date, permno) pairs defining the investable universe.
    """
    raise NotImplementedError