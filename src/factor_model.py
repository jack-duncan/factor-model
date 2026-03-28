"""OLS factor decomposition."""

import pandas as pd
import statsmodels.api as sm

from src.config import FACTORS, MIN_OBS, ROLLING_WINDOW


def estimate_factor_exposures(
    returns: pd.Series,
    factors: pd.DataFrame,
    window: int = ROLLING_WINDOW,
    min_obs: int = MIN_OBS,
) -> pd.DataFrame:
    """Rolling OLS regression of asset returns on factor returns.

    Returns DataFrame of time-varying betas.
    """
    raise NotImplementedError


def full_sample_regression(
    returns: pd.Series,
    factors: pd.DataFrame,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Full-sample OLS regression. Returns statsmodels results object."""
    raise NotImplementedError


def factor_attribution(
    betas: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Decompose returns into factor contributions."""
    raise NotImplementedError