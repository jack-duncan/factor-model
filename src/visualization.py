"""Plotly chart builders for the Streamlit app."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import FACTOR_DISPLAY_NAMES


def _display(name):
    return FACTOR_DISPLAY_NAMES.get(name, name)


def plot_exposure_heatmap(results_dict, selected_factors):
    """Heatmap of factor betas across stocks + portfolio.

    Parameters
    ----------
    results_dict : dict {label: statsmodels result or None}
    selected_factors : list of factor names
    """
    rows = []
    for label, res in results_dict.items():
        if res is None:
            continue
        row = {"label": label}
        for f in selected_factors:
            row[f] = res.params.get(f, np.nan)
        rows.append(row)

    if not rows:
        return go.Figure().add_annotation(text="No results", showarrow=False)

    df = pd.DataFrame(rows).set_index("label")
    labels = df.index.tolist()
    factors = [f for f in selected_factors if f in df.columns]
    z = df[factors].values

    # Build significance markers
    text_matrix = []
    for label, res in results_dict.items():
        if res is None:
            continue
        row_text = []
        for f in factors:
            beta = res.params.get(f, np.nan)
            pval = res.pvalues.get(f, 1.0)
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            row_text.append(f"{beta:.3f}{stars}")
        text_matrix.append(row_text)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[_display(f) for f in factors],
        y=labels,
        text=text_matrix,
        texttemplate="%{text}",
        colorscale="RdBu_r",
        zmid=0,
        colorbar=dict(title="Beta"),
    ))
    fig.update_layout(
        title="Factor Exposures",
        height=max(300, 60 * len(labels)),
        margin=dict(l=120),
    )
    return fig


def plot_rolling_betas(betas, selected_factors):
    """Line chart of rolling factor exposures over time."""
    if betas.empty:
        return go.Figure().add_annotation(text="Insufficient data for rolling betas", showarrow=False)

    fig = go.Figure()
    for f in selected_factors:
        if f in betas.columns:
            fig.add_trace(go.Scatter(
                x=betas["date"], y=betas[f],
                name=_display(f), mode="lines",
            ))

    fig.update_layout(
        title="Rolling Factor Exposures",
        xaxis_title="Date",
        yaxis_title="Beta",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def plot_factor_attribution(contrib, returns_df, selected_factors):
    """Stacked area chart of factor contributions to returns.

    Parameters
    ----------
    contrib : DataFrame from factor_attribution (date, alpha, factor columns).
    returns_df : DataFrame with [date, ret] for the actual return series.
    selected_factors : list of factor names.
    """
    if contrib.empty:
        return go.Figure().add_annotation(text="No attribution data", showarrow=False)

    fig = go.Figure()

    factors_present = [f for f in selected_factors if f in contrib.columns]
    for f in factors_present:
        fig.add_trace(go.Scatter(
            x=contrib["date"], y=contrib[f],
            name=_display(f), stackgroup="one",
            mode="lines", line=dict(width=0),
        ))

    fig.add_trace(go.Scatter(
        x=contrib["date"], y=contrib["alpha"],
        name="Alpha", stackgroup="one",
        mode="lines", line=dict(width=0),
    ))

    # Overlay actual excess return
    if returns_df is not None and not returns_df.empty:
        ret = returns_df.copy()
        ret["date"] = pd.to_datetime(ret["date"])
        merged = contrib[["date"]].merge(ret, on="date", how="left")
        fig.add_trace(go.Scatter(
            x=merged["date"], y=merged["ret"],
            name="Actual Return", mode="lines",
            line=dict(color="black", width=1.5, dash="dot"),
        ))

    fig.update_layout(
        title="Return Attribution",
        xaxis_title="Date",
        yaxis_title="Return",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def plot_risk_decomposition(decomp):
    """Donut chart of variance decomposition by factor.

    Parameters
    ----------
    decomp : dict {factor_name: fraction}
    """
    labels = [_display(k) if k != "idiosyncratic" else "Idiosyncratic" for k in decomp]
    values = list(decomp.values())

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        hole=0.4, textinfo="label+percent",
        textposition="outside",
    )])
    fig.update_layout(
        title="Variance Decomposition",
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
        height=450,
    )
    return fig


def plot_factor_correlation(factor_returns, selected_factors):
    """Heatmap of factor return correlations."""
    factors_present = [f for f in selected_factors if f in factor_returns.columns]
    corr = factor_returns[factors_present].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=[_display(f) for f in factors_present],
        y=[_display(f) for f in factors_present],
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        colorbar=dict(title="Corr"),
    ))
    fig.update_layout(
        title="Factor Correlation Matrix",
        height=500,
    )
    return fig
