"""Portfolio construction and return computation."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

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


def optimize_beta_neutral(stock_results, selected_permnos):
    """Find beta-neutral, alpha-maximizing weights (Treynor-Black).

    Objective : maximize  sum_i  alpha_i / sigma2_eps_i * w_i
    Constraints:
        beta_mkt^T w = 0   (market neutral)
        sum(w)       = 0   (dollar neutral: long $ = short $)
    Bounds: w_i in [-1, 1] per stock; normalized to gross exposure = 1 after solve.

    Parameters
    ----------
    stock_results : dict {permno: statsmodels result or None}
    selected_permnos : list of permnos

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
    scores = alphas / np.array(idio_vars)   # Treynor-Black appraisal scores

    n = len(permnos)

    def objective(w):
        return -float(np.dot(w, scores))

    def jac(w):
        return -scores

    constraints = [
        {"type": "eq", "fun": lambda w: np.dot(betas_mkt, w)},  # beta neutral only
    ]
    bounds = [(-1.0, 1.0)] * n
    w0 = np.zeros(n)

    res = minimize(objective, w0, jac=jac, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-12, "maxiter": 1000})

    gross = np.abs(res.x).sum()
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
