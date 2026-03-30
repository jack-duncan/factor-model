"""Streamlit app – Factor Lens Dashboard."""

import logging
import os

import pandas as pd
import streamlit as st

from src.config import (
    ALL_FACTORS,
    FACTOR_DISPLAY_NAMES,
    FACTORS_CONSTRUCTED,
    FACTORS_PREBUILT,
    MIN_OBS,
    PROCESSED_DATA_DIR,
    ROLLING_WINDOW,
    START_DATE,
)
from src.factor_model import (
    estimate_factor_exposures,
    factor_attribution,
    full_sample_regression,
    variance_decomposition,
)
from src.portfolio import portfolio_returns, stock_factor_exposures
from src.visualization import (
    plot_exposure_heatmap,
    plot_factor_attribution,
    plot_factor_correlation,
    plot_risk_decomposition,
    plot_rolling_betas,
)

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Factor Lens", layout="wide")


# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    """Load all processed data from parquet cache."""
    files = {
        "crsp": "crsp_monthly.parquet",
        "universe": "universe.parquet",
        "ticker_map": "ticker_map.parquet",
        "factor_returns": "factor_returns.parquet",
    }
    data = {}
    raw_dir = "data/raw"
    for key, fname in files.items():
        directory = raw_dir if key == "crsp" else PROCESSED_DATA_DIR
        path = os.path.join(directory, fname)
        if os.path.exists(path):
            data[key] = pd.read_parquet(path)
        else:
            data[key] = None
    return data


def check_data_available(data):
    """Check if required datasets are loaded."""
    missing = [k for k in ["crsp", "universe", "ticker_map", "factor_returns"] if data.get(k) is None]
    return len(missing) == 0, missing


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Factor Lens")

    # Refresh data button
    if st.button("Refresh Data from WRDS", type="secondary"):
        with st.spinner("Pulling data from WRDS..."):
            try:
                from src.data_loader import refresh_data
                refresh_data()
                load_data.clear()
                st.success("Data refreshed!")
                st.rerun()
            except Exception as e:
                st.error(f"Refresh failed: {e}")

    st.divider()

    data = load_data()
    data_ok, missing = check_data_available(data)

    if not data_ok:
        st.error(f"Missing data: {', '.join(missing)}. Click 'Refresh Data' to pull from WRDS.")
        st.stop()

    crsp = data["crsp"]
    universe = data["universe"]
    ticker_map = data["ticker_map"]
    factor_returns = data["factor_returns"]

    # Date range
    crsp["date"] = pd.to_datetime(crsp["date"])
    factor_returns["date"] = pd.to_datetime(factor_returns["date"])

    min_date = crsp["date"].min().date()
    max_date = crsp["date"].max().date()

    st.subheader("Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=pd.Timestamp(START_DATE).date(),
                                   min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("End", value=max_date,
                                 min_value=min_date, max_value=max_date)

    st.divider()

    # Ticker selection
    st.subheader("Portfolio")
    ticker_options = (
        ticker_map.dropna(subset=["ticker"])
        .assign(label=lambda d: d["ticker"] + " — " + d["comnam"].str.title())
        .sort_values("ticker")
    )
    selected_labels = st.multiselect(
        "Search tickers",
        options=ticker_options["label"].tolist(),
        default=[],
        placeholder="Type to search...",
    )
    # Map labels back to permnos
    selected_permnos = ticker_options[ticker_options["label"].isin(selected_labels)]["permno"].tolist()

    weight_scheme = st.radio("Weighting", ["equal", "mcap"], horizontal=True)

    st.divider()

    # Factor selection
    st.subheader("Factors")
    available_factors = [f for f in ALL_FACTORS if f in factor_returns.columns and factor_returns[f].notna().any()]
    selected_factors = st.multiselect(
        "Select factors",
        options=available_factors,
        default=[f for f in FACTORS_PREBUILT if f in available_factors],
        format_func=lambda f: FACTOR_DISPLAY_NAMES.get(f, f),
    )

    st.divider()

    # Rolling window
    st.subheader("Rolling Window")
    window = st.slider("Months", min_value=12, max_value=120, value=ROLLING_WINDOW, step=6)


# ── Filter data to date range ───────────────────────────────────────────────

date_mask_crsp = (crsp["date"] >= pd.Timestamp(start_date)) & (crsp["date"] <= pd.Timestamp(end_date))
crsp_filtered = crsp[date_mask_crsp]

date_mask_ff = (factor_returns["date"] >= pd.Timestamp(start_date)) & (factor_returns["date"] <= pd.Timestamp(end_date))
ff_filtered = factor_returns[date_mask_ff]


# ── Main content ─────────────────────────────────────────────────────────────

if not selected_permnos:
    st.info("Select one or more tickers in the sidebar to begin analysis.")
    st.stop()

if not selected_factors:
    st.warning("Select at least one factor in the sidebar.")
    st.stop()

# Compute portfolio returns
port_ret = portfolio_returns(selected_permnos, crsp_filtered, weight_scheme)

# Per-stock regressions
stock_results = stock_factor_exposures(selected_permnos, crsp_filtered, ff_filtered, selected_factors)

# Portfolio regression
port_result = full_sample_regression(port_ret, ff_filtered, selected_factors)

# Build labels
permno_to_ticker = dict(zip(ticker_map["permno"], ticker_map["ticker"]))

tab_holdings, tab_exposures, tab_rolling, tab_attribution = st.tabs(
    ["Portfolio Holdings", "Factor Exposures", "Rolling Exposures", "Attribution"]
)


# ── Tab 1: Holdings ──────────────────────────────────────────────────────────

with tab_holdings:
    st.subheader("Portfolio Holdings")
    holdings = ticker_map[ticker_map["permno"].isin(selected_permnos)].copy()
    holdings["ticker"] = holdings["ticker"].fillna("N/A")
    holdings["comnam"] = holdings["comnam"].str.title()

    n = len(holdings)
    if weight_scheme == "equal":
        holdings["weight"] = f"{100/n:.1f}%" if n > 0 else "0%"
    else:
        # Show latest mcap weights
        latest_month = crsp_filtered.groupby("permno")["date"].max().reset_index()
        latest = crsp_filtered.merge(latest_month, on=["permno", "date"])
        latest = latest[latest["permno"].isin(selected_permnos)]
        total_mcap = latest["mcap"].sum()
        latest["weight"] = (latest["mcap"] / total_mcap * 100).round(1).astype(str) + "%"
        holdings = holdings.merge(latest[["permno", "weight"]], on="permno", how="left")

    st.dataframe(
        holdings[["ticker", "comnam", "weight"]].rename(
            columns={"ticker": "Ticker", "comnam": "Company", "weight": "Weight"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    if port_ret is not None and not port_ret.empty:
        cum_ret = (1 + port_ret["ret"]).cumprod() - 1
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cumulative Return", f"{cum_ret.iloc[-1]:.1%}")
        with col2:
            ann_ret = (1 + cum_ret.iloc[-1]) ** (12 / len(port_ret)) - 1
            st.metric("Annualized Return", f"{ann_ret:.1%}")
        with col3:
            ann_vol = port_ret["ret"].std() * (12 ** 0.5)
            st.metric("Annualized Volatility", f"{ann_vol:.1%}")


# ── Tab 2: Factor Exposures ─────────────────────────────────────────────────

with tab_exposures:
    st.subheader("Factor Exposures")

    results_dict = {}
    for p, res in stock_results.items():
        label = permno_to_ticker.get(p, str(p))
        results_dict[label] = res
    if port_result is not None:
        results_dict["Portfolio"] = port_result

    fig = plot_exposure_heatmap(results_dict, selected_factors)
    st.plotly_chart(fig, use_container_width=True)

    # R-squared summary
    if port_result is not None:
        st.markdown(f"**Portfolio R²:** {port_result.rsquared:.3f} | "
                    f"**Adj R²:** {port_result.rsquared_adj:.3f} | "
                    f"**Obs:** {int(port_result.nobs)}")

    # Factor correlation
    st.subheader("Factor Correlations")
    fig_corr = plot_factor_correlation(ff_filtered, selected_factors)
    st.plotly_chart(fig_corr, use_container_width=True)


# ── Tab 3: Rolling Exposures ────────────────────────────────────────────────

with tab_rolling:
    st.subheader("Rolling Factor Exposures")

    roll_target = st.radio(
        "Show rolling betas for:",
        ["Portfolio"] + [permno_to_ticker.get(p, str(p)) for p in selected_permnos],
        horizontal=True,
    )

    if roll_target == "Portfolio":
        roll_ret = port_ret
    else:
        # Find permno for selected ticker
        target_permno = [p for p in selected_permnos if permno_to_ticker.get(p, str(p)) == roll_target]
        if target_permno:
            roll_ret = crsp_filtered[crsp_filtered["permno"] == target_permno[0]][["date", "ret"]]
        else:
            roll_ret = pd.DataFrame(columns=["date", "ret"])

    betas = estimate_factor_exposures(roll_ret, ff_filtered, selected_factors, window=window, min_obs=MIN_OBS)
    fig_roll = plot_rolling_betas(betas, selected_factors)
    st.plotly_chart(fig_roll, use_container_width=True)


# ── Tab 4: Attribution ───────────────────────────────────────────────────────

with tab_attribution:
    st.subheader("Return Attribution")

    if port_result is not None:
        col1, col2 = st.columns([2, 1])

        with col1:
            contrib = factor_attribution(port_result, ff_filtered, selected_factors)
            fig_attr = plot_factor_attribution(contrib, port_ret, selected_factors)
            st.plotly_chart(fig_attr, use_container_width=True)

        with col2:
            decomp = variance_decomposition(port_result, ff_filtered, selected_factors)
            fig_risk = plot_risk_decomposition(decomp)
            st.plotly_chart(fig_risk, use_container_width=True)
    else:
        st.warning("Insufficient data for attribution analysis.")
