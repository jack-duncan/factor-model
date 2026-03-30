"""Top-N universe construction by market cap."""

import pandas as pd

from src.config import UNIVERSE_SIZE


def build_universe(crsp, n=UNIVERSE_SIZE):
    """Select the top n stocks by market cap each month.

    Returns
    -------
    universe : DataFrame
        (date, permno, mcap, ticker, comnam) for each stock-month in universe.
    ticker_map : DataFrame
        Latest ticker and company name per permno (for UI search).
    """
    monthly = crsp.copy()
    monthly["date"] = pd.to_datetime(monthly["date"])
    monthly = monthly.dropna(subset=["mcap"])

    # Rank by market cap within each month, keep top n
    universe = (
        monthly
        .groupby("date", group_keys=False)
        .apply(lambda g: g.nlargest(n, "mcap"))
        [["date", "permno", "mcap", "ticker", "comnam"]]
        .reset_index(drop=True)
    )

    # Build ticker map: most recent ticker/name for each permno
    latest = (
        monthly
        .sort_values("date")
        .drop_duplicates(subset=["permno"], keep="last")
        [["permno", "ticker", "comnam"]]
        .reset_index(drop=True)
    )

    return universe, latest
