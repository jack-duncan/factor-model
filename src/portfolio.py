"""Portfolio construction and return computation."""

import pandas as pd

from src.factor_model import full_sample_regression


def portfolio_returns(permnos, crsp, weight_scheme="equal"):
    """Compute portfolio return series for a set of stocks.

    Parameters
    ----------
    permnos : list of permno identifiers.
    crsp : DataFrame with columns [permno, date, ret].
    weight_scheme : 'equal' or 'mcap'.

    Returns
    -------
    DataFrame with columns [date, ret].
    """
    df = crsp[crsp["permno"].isin(permnos)].copy()
    df["date"] = pd.to_datetime(df["date"])

    if weight_scheme == "equal":
        port = df.groupby("date")["ret"].mean().reset_index()
    elif weight_scheme == "mcap":
        def _wcalc(g):
            w = g["mcap"] / g["mcap"].sum()
            return (w * g["ret"]).sum()
        port = df.groupby("date").apply(_wcalc, include_groups=False).reset_index()
        port.columns = ["date", "ret"]
    else:
        raise ValueError(f"Unknown weight_scheme: {weight_scheme}")

    return port.sort_values("date").reset_index(drop=True)


def portfolio_factor_exposures(port_returns, factor_returns, selected_factors):
    """Run full-sample regression on portfolio returns."""
    return full_sample_regression(port_returns, factor_returns, selected_factors)


def stock_factor_exposures(permnos, crsp, factor_returns, selected_factors):
    """Run full-sample regression for each individual stock.

    Returns dict: {permno: statsmodels result or None}.
    """
    results = {}
    for p in permnos:
        stock = crsp[crsp["permno"] == p][["date", "ret"]].copy()
        stock["date"] = pd.to_datetime(stock["date"])
        results[p] = full_sample_regression(stock, factor_returns, selected_factors)
    return results
