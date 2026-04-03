"""Portfolio construction and return computation."""

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src.factor_model import full_sample_regression


def portfolio_returns(permnos, crsp, weight_scheme="equal", shares=None, dollars=None):
    """Compute portfolio return series for a set of stocks.

    Parameters
    ----------
    permnos : list of permno identifiers.
    crsp : DataFrame with columns [permno, date, ret, prc, mcap].
    weight_scheme : 'equal', 'mcap', or 'shares'.
    shares : dict {permno: n_shares}, required when weight_scheme='shares'.

    Returns
    -------
    DataFrame with columns [date, ret].
    """
    df = crsp[crsp["permno"].isin(permnos)].copy()
    df["date"] = pd.to_datetime(df["date"])

    if weight_scheme == "equal":
        port = df.groupby("date")["ret"].mean().reset_index()

    elif weight_scheme == "mcap":
        def _mcap_wcalc(g):
            w = g["mcap"] / g["mcap"].sum()
            return (w * g["ret"]).sum()
        port = df.groupby("date").apply(_mcap_wcalc, include_groups=False).reset_index()
        port.columns = ["date", "ret"]

    elif weight_scheme == "shares":
        if not shares:
            # Fall back to equal weight if no shares provided
            port = df.groupby("date")["ret"].mean().reset_index()
        else:
            df["n_shares"] = df["permno"].map(shares).fillna(0)
            # Signed position: negative for shorts.
            # Use GROSS exposure as denominator so long+short doesn't cancel to zero.
            # Negative weight on a short means: short rises → negative contribution → correct P&L.
            df["position"] = df["n_shares"] * df["prc"].abs()
            def _shares_wcalc(g):
                gross = g["position"].abs().sum()
                if gross == 0:
                    return 0.0
                w = g["position"] / gross  # signed weights; shorts are negative
                return (w * g["ret"]).sum()
            port = df.groupby("date").apply(_shares_wcalc, include_groups=False).reset_index()
            port.columns = ["date", "ret"]
    elif weight_scheme == "dollar":
        if not dollars:
            port = df.groupby("date")["ret"].mean().reset_index()
        else:
            # Dollar amount is the position directly (signed: negative = short)
            df["dollar_pos"] = df["permno"].map(dollars).fillna(0)
            def _dollar_wcalc(g):
                gross = g["dollar_pos"].abs().sum()
                if gross == 0:
                    return 0.0
                w = g["dollar_pos"] / gross
                return (w * g["ret"]).sum()
            port = df.groupby("date").apply(_dollar_wcalc, include_groups=False).reset_index()
            port.columns = ["date", "ret"]

    else:
        raise ValueError(f"Unknown weight_scheme: {weight_scheme}")

    return port.sort_values("date").reset_index(drop=True)


def optimize_beta_neutral(stock_results, selected_permnos, current_weights=None):
    """Find beta-neutral, alpha-maximizing weights (Treynor-Black LP).

    Solves the true Treynor-Black problem as a Linear Program:

        maximize  scores^T w
        subject to  beta_mkt^T w = 0   (market neutral)
                    w_i in [-1, 1]      (position limits)

    where scores = alpha_i / sigma2_eps_i (appraisal ratio).

    LP optimal solutions are at vertices of the feasible polytope, so many
    weights will be 0 or ±1. Zero weights are meaningful — those stocks
    do not contribute alpha relative to their beta cost. Non-zero weights
    are the globally optimal allocation.

    After solving, normalizes to gross exposure = 1.

    Parameters
    ----------
    stock_results    : dict {permno: statsmodels result or None}
    selected_permnos : list of permnos
    current_weights  : unused, kept for API compatibility

    Returns
    -------
    dict {permno: weight} with gross exposure = 1, or None if infeasible.
    """
    permnos = [p for p in selected_permnos if stock_results.get(p) is not None]
    if len(permnos) < 2:
        return None

    alphas, betas_mkt, idio_vars = [], [], []
    for p in permnos:
        res = stock_results[p]
        alphas.append(res.params.get("const", 0.0))
        betas_mkt.append(res.params.get("mktrf", 0.0))
        idio_vars.append(max(res.mse_resid, 1e-10))

    alphas = np.array(alphas)
    betas_mkt = np.array(betas_mkt)
    scores = alphas / np.array(idio_vars)
    n = len(permnos)

    res = linprog(
        c=-scores,                          # maximize scores^T w
        A_eq=betas_mkt.reshape(1, -1),      # beta^T w = 0
        b_eq=np.array([0.0]),
        bounds=[(-1.0, 1.0)] * n,
        method="highs",
    )

    gross = np.abs(res.x).sum() if res.x is not None else 0
    if not res.success or gross < 1e-8:
        return None

    w_norm = res.x / gross
    return dict(zip(permnos, w_norm))


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
