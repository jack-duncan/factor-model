"""OLS factor decomposition: full-sample, rolling, and attribution."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from src.config import MIN_OBS, ROLLING_WINDOW


def _align_returns_factors(returns, factor_returns, selected_factors):
    """Merge return series with factor returns on year-month, compute excess return.

    Parameters
    ----------
    returns : DataFrame with columns [date, ret] (or a Series with date index).
    factor_returns : DataFrame with date + factor columns + rf.
    selected_factors : list of factor column names to include.

    Returns
    -------
    merged : DataFrame with excess_ret and selected factor columns, indexed by date.
    """
    if isinstance(returns, pd.Series):
        ret_df = returns.reset_index()
        ret_df.columns = ["date", "ret"]
    else:
        ret_df = returns[["date", "ret"]].copy()

    ret_df["date"] = pd.to_datetime(ret_df["date"])
    ret_df["ym"] = ret_df["date"].dt.to_period("M")

    ff = factor_returns.copy()
    ff["date"] = pd.to_datetime(ff["date"])
    ff["ym"] = ff["date"].dt.to_period("M")

    merged = ret_df.merge(ff, on="ym", suffixes=("", "_ff"))
    merged["excess_ret"] = merged["ret"] - merged["rf"]

    keep = ["date", "excess_ret"] + [f for f in selected_factors if f in merged.columns]
    merged = merged[keep].dropna().sort_values("date").reset_index(drop=True)
    return merged


def full_sample_regression(returns, factor_returns, selected_factors):
    """Full-sample OLS: excess_ret ~ alpha + sum(beta_i * factor_i).

    Returns statsmodels RegressionResultsWrapper.
    """
    data = _align_returns_factors(returns, factor_returns, selected_factors)
    if len(data) < MIN_OBS:
        return None

    y = data["excess_ret"]
    X = sm.add_constant(data[selected_factors])
    result = sm.OLS(y, X).fit()
    return result


def estimate_factor_exposures(returns, factor_returns, selected_factors,
                              window=ROLLING_WINDOW, min_obs=MIN_OBS):
    """Rolling OLS regression of excess returns on factors.

    Returns DataFrame of time-varying betas with date index.
    """
    data = _align_returns_factors(returns, factor_returns, selected_factors)
    if len(data) < min_obs:
        return pd.DataFrame()

    y = data["excess_ret"]
    X = sm.add_constant(data[selected_factors])

    model = RollingOLS(y, X, window=window, min_nobs=min_obs)
    result = model.fit()

    betas = result.params.copy()
    betas["date"] = data["date"].values

    # Rolling idiosyncratic return: actual excess return minus fitted
    fitted = (result.params * X.values).sum(axis=1)
    betas["idiosyncratic"] = y.values - fitted

    betas = betas.dropna(subset=["const"])
    return betas


def factor_attribution(result, factor_returns, selected_factors):
    """Decompose returns into factor contributions over time.

    Parameters
    ----------
    result : statsmodels RegressionResultsWrapper from full_sample_regression.
    factor_returns : DataFrame with date + factor columns.

    Returns
    -------
    DataFrame with date and columns: alpha, each factor contribution, residual.
    """
    ff = factor_returns.copy()
    ff["date"] = pd.to_datetime(ff["date"])

    betas = {f: result.params.get(f, 0) for f in selected_factors}
    alpha = result.params.get("const", 0)

    contrib = ff[["date"]].copy()
    contrib["alpha"] = alpha

    total_factor = 0
    for f in selected_factors:
        if f in ff.columns:
            contrib[f] = betas[f] * ff[f]
            total_factor = total_factor + contrib[f]

    # Residual is implied (excess return - alpha - factor contributions)
    # We store factor contributions; residual computed in visualization
    return contrib.dropna().sort_values("date").reset_index(drop=True)


def variance_decomposition(result, factor_returns, selected_factors):
    """Compute fraction of return variance explained by each factor.

    Returns dict: {factor_name: fraction, 'idiosyncratic': fraction}.
    """
    ff = factor_returns[selected_factors].dropna()
    cov = ff.cov()

    betas = np.array([result.params.get(f, 0) for f in selected_factors])
    total_var = result.mse_total * result.nobs / (result.nobs - 1)

    decomp = {}
    for i, f in enumerate(selected_factors):
        var_contrib = betas[i] ** 2 * cov.iloc[i, i]
        decomp[f] = max(var_contrib / total_var, 0)

    decomp["idiosyncratic"] = max(1 - sum(decomp.values()), 0)
    return decomp
