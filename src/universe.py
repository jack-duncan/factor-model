"""Top-N universe construction by market cap."""

import pandas as pd

from src.config import UNIVERSE_SIZE, sic_to_sector


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
    monthly["_rank"] = monthly.groupby("date")["mcap"].rank(ascending=False, method="first")
    universe = (
        monthly[monthly["_rank"] <= n]
        [["date", "permno", "mcap", "ticker", "comnam"]]
        .reset_index(drop=True)
    )
    monthly.drop(columns=["_rank"], inplace=True)

    # Build ticker map: most recent ticker/name/sector for each permno
    cols = ["permno", "ticker", "comnam"] + (["siccd"] if "siccd" in monthly.columns else [])
    latest = (
        monthly
        .sort_values("date")
        .drop_duplicates(subset=["permno"], keep="last")
        [cols]
        .reset_index(drop=True)
    )
    if "siccd" in latest.columns:
        latest["sector"] = latest["siccd"].apply(sic_to_sector)
        latest = latest.drop(columns=["siccd"])

    return universe, latest
