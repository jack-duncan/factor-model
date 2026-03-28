"""Plotly chart builders for the Streamlit app."""

import pandas as pd
import plotly.graph_objects as go


def plot_rolling_betas(betas: pd.DataFrame) -> go.Figure:
    """Line chart of rolling factor exposures over time."""
    raise NotImplementedError


def plot_factor_attribution(attribution: pd.DataFrame) -> go.Figure:
    """Stacked bar chart of return attribution by factor."""
    raise NotImplementedError


def plot_factor_correlation(factors: pd.DataFrame) -> go.Figure:
    """Heatmap of factor return correlations."""
    raise NotImplementedError