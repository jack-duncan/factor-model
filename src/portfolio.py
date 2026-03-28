"""Portfolio construction and weighted-average exposures."""

import pandas as pd


def equal_weight_portfolio(universe: pd.DataFrame) -> pd.DataFrame:
    """Assign equal weights to all stocks in the universe each period."""
    raise NotImplementedError


def mcap_weight_portfolio(universe: pd.DataFrame, mcap_col: str = "mcap") -> pd.DataFrame:
    """Assign market-cap weights to stocks in the universe each period."""
    raise NotImplementedError


def portfolio_factor_exposures(
    weights: pd.DataFrame,
    stock_betas: pd.DataFrame,
) -> pd.DataFrame:
    """Compute weighted-average factor exposures for the portfolio."""
    raise NotImplementedError