"""Streamlit app – Factor Lens Dashboard."""

import logging
import os

import numpy as np
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

    min_date = pd.Timestamp(START_DATE).date()
    max_date = crsp["date"].max().date()

    st.subheader("Date Range")
    default_start = min_date
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=default_start,
                                   min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("End", value=max_date,
                                 min_value=min_date, max_value=max_date)

    st.divider()

    # Ticker selection
    st.subheader("Portfolio")
    _tm = ticker_map.dropna(subset=["ticker"]).copy()
    _tm["comnam"] = _tm["comnam"].str.title()
    _dup_tickers = _tm["ticker"].duplicated(keep=False)
    _tm["label"] = _tm["ticker"] + " — " + _tm["comnam"]
    _tm.loc[_dup_tickers, "label"] = (
        _tm.loc[_dup_tickers, "ticker"] + " — " + _tm.loc[_dup_tickers, "comnam"]
        + " (" + _tm.loc[_dup_tickers, "permno"].astype(str) + ")"
    )
    ticker_options = _tm.sort_values("ticker")

    # Random portfolio generator
    rand_col1, rand_col2 = st.columns(2)
    with rand_col1:
        rand_n = st.number_input("# of stocks", min_value=1, max_value=20, value=5, step=1)
    with rand_col2:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        if st.button("Randomize", use_container_width=True):
            sample_size = min(int(rand_n), len(ticker_options))
            sampled = ticker_options.sample(n=sample_size, random_state=None)
            st.session_state["ticker_multiselect"] = sampled["label"].tolist()
            if st.session_state.get("weight_scheme_radio") == "shares":
                sampled_permnos = sampled["permno"].tolist()
                # Get latest price and mcap for each sampled stock
                latest = (
                    crsp[crsp["permno"].isin(sampled_permnos)]
                    .sort_values("date")
                    .drop_duplicates("permno", keep="last")
                    .set_index("permno")[["prc", "mcap"]]
                )
                latest["prc"] = latest["prc"].abs()
                # Largest mcap stock anchors at mean=10 shares
                anchor = latest["mcap"].idxmax()
                ref_position = 10 * latest.loc[anchor, "prc"]
                for p in sampled_permnos:
                    if p in latest.index and latest.loc[p, "prc"] > 0:
                        mean_shares = ref_position / latest.loc[p, "prc"]
                    else:
                        mean_shares = 10.0
                    std = max(1.0, mean_shares * 0.2)
                    shares = max(1, int(np.round(np.random.normal(mean_shares, std))))
                    st.session_state[f"shares_{p}"] = float(shares)

    selected_labels = st.multiselect(
        "Search tickers",
        options=ticker_options["label"].tolist(),
        default=[],
        placeholder="Type to search...",
        max_selections=20,
        key="ticker_multiselect",
    )
    # Map labels back to permnos
    selected_permnos = ticker_options[ticker_options["label"].isin(selected_labels)]["permno"].tolist()

    weight_scheme = st.radio("Weighting", ["equal", "mcap", "shares"], horizontal=True,
                             key="weight_scheme_radio")

    # Shares input — one number field per selected ticker
    shares_input = {}
    if weight_scheme == "shares" and selected_permnos:
        st.caption("Enter number of shares held:")
        permno_to_label = dict(zip(ticker_options["permno"], ticker_options["ticker"]))
        for p in selected_permnos:
            label = permno_to_label.get(p, str(p))
            shares_input[p] = st.number_input(
                label, value=100.0, step=1.0, format="%.3f", key=f"shares_{p}"
            )

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

    # Rolling window — bounds driven entirely by date range and factor count
    st.subheader("Rolling Window")
    date_range_months = max(7, (pd.Timestamp(end_date).to_period("M") - pd.Timestamp(start_date).to_period("M")).n + 1)
    n_regressors = len(selected_factors) + 1 if selected_factors else 2
    win_min = n_regressors + 1
    win_max = date_range_months - 1
    win_min = min(win_min, win_max)
    default_window = max(win_min, min(ROLLING_WINDOW, win_max))

    # Reset to default whenever date range or factor count changes
    range_key = (start_date, end_date, len(selected_factors))
    if st.session_state.get("_window_range_key") != range_key:
        st.session_state["window_slider"] = default_window
        st.session_state["_window_range_key"] = range_key
    else:
        # Clamp in case bounds tightened without a range change
        st.session_state["window_slider"] = max(win_min, min(st.session_state.get("window_slider", default_window), win_max))

    window = st.slider("Estimation window (months)", min_value=win_min, max_value=win_max,
                       step=1, key="window_slider")

    st.divider()

    # Refresh data button — bottom of sidebar
    if st.button("Refresh Data from WRDS", type="secondary", use_container_width=True):
        with st.spinner("Pulling data from WRDS..."):
            try:
                from src.data_loader import refresh_data
                refresh_data()
                load_data.clear()
                st.success("Data refreshed!")
                st.rerun()
            except Exception as e:
                st.error(f"Refresh failed: {e}")


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
port_ret = portfolio_returns(selected_permnos, crsp_filtered, weight_scheme,
                             shares=shares_input if weight_scheme == "shares" else None)

# Per-stock regressions
stock_results = stock_factor_exposures(selected_permnos, crsp_filtered, ff_filtered, selected_factors)

# Portfolio regression
port_result = full_sample_regression(port_ret, ff_filtered, selected_factors)

# Build labels
permno_to_ticker = dict(zip(ticker_map["permno"], ticker_map["ticker"]))

# ── Factor equation banner ────────────────────────────────────────────────────
if port_result is not None:
    alpha = port_result.params["const"]
    factor_terms = []
    for f in selected_factors:
        coef = port_result.params.get(f, 0)
        sign = "+" if coef >= 0 else "-"
        name = FACTOR_DISPLAY_NAMES.get(f, f).replace("-", r"\text{-}")
        factor_terms.append(rf"{sign} {abs(coef):.3f} \cdot \text{{{name}}}")

    alpha_sign = "+" if alpha >= 0 else "-"
    latex_eq = (
        rf"r_p - r_f = {alpha_sign} {abs(alpha):.3f} "
        + " ".join(factor_terms)
        + r" + \varepsilon"
    )
    st.latex(latex_eq)

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
    elif weight_scheme == "mcap":
        latest_month = crsp_filtered.groupby("permno")["date"].max().reset_index()
        latest = crsp_filtered.merge(latest_month, on=["permno", "date"])
        latest = latest[latest["permno"].isin(selected_permnos)]
        total_mcap = latest["mcap"].sum()
        latest["weight"] = (latest["mcap"] / total_mcap * 100).round(1).astype(str) + "%"
        holdings = holdings.merge(latest[["permno", "weight"]], on="permno", how="left")
    else:  # shares
        latest_month = crsp_filtered.groupby("permno")["date"].max().reset_index()
        latest = crsp_filtered.merge(latest_month, on=["permno", "date"])
        latest = latest[latest["permno"].isin(selected_permnos)]
        latest["n_shares"] = latest["permno"].map(shares_input).fillna(0)
        latest["position"] = latest["n_shares"] * latest["prc"].abs()
        total_pos = latest["position"].sum()
        latest["weight"] = (latest["position"] / total_pos * 100).round(1).astype(str) + "%" if total_pos > 0 else "0%"
        holdings = holdings.merge(latest[["permno", "weight"]], on="permno", how="left")

    display_cols = {"ticker": "Ticker", "comnam": "Company", "sector": "Sector", "weight": "Weight"}
    if "sector" in holdings.columns:
        show_cols = ["ticker", "comnam", "sector", "weight"]
    else:
        show_cols = ["ticker", "comnam", "weight"]
    st.dataframe(
        holdings[show_cols].rename(columns=display_cols),
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

    ctrl_col1, ctrl_col2 = st.columns([2, 3])
    with ctrl_col1:
        roll_target = st.radio(
            "Show rolling betas for:",
            ["Portfolio"] + [permno_to_ticker.get(p, str(p)) for p in selected_permnos],
            horizontal=True,
        )

    if roll_target == "Portfolio":
        roll_ret = port_ret
    else:
        target_permno = [p for p in selected_permnos if permno_to_ticker.get(p, str(p)) == roll_target]
        if target_permno:
            roll_ret = crsp_filtered[crsp_filtered["permno"] == target_permno[0]][["date", "ret"]]
        else:
            roll_ret = pd.DataFrame(columns=["date", "ret"])

    safe_min_obs = max(n_regressors + 1, min(MIN_OBS, window - 1))
    betas = estimate_factor_exposures(roll_ret, ff_filtered, selected_factors, window=window, min_obs=safe_min_obs)

    # Display range: two-ended date slider anchored to the selected date range
    roll_min_date = start_date
    roll_max_date = end_date

    range_key = (roll_min_date, roll_max_date)
    if st.session_state.get("_roll_disp_key") != range_key:
        st.session_state["roll_disp_range"] = (roll_min_date, roll_max_date)
        st.session_state["_roll_disp_key"] = range_key
    else:
        lo, hi = st.session_state.get("roll_disp_range", (roll_min_date, roll_max_date))
        lo = max(lo, roll_min_date)
        hi = min(hi, roll_max_date)
        if lo >= hi:
            lo, hi = roll_min_date, roll_max_date
        st.session_state["roll_disp_range"] = (lo, hi)

    with ctrl_col2:
        disp_start, disp_end = st.slider(
            "Display range",
            min_value=roll_min_date,
            max_value=roll_max_date,
            key="roll_disp_range",
            format="MMM YYYY",
        )

    if not betas.empty:
        betas["date"] = pd.to_datetime(betas["date"])
        betas_display = betas[
            (betas["date"] >= pd.Timestamp(disp_start)) &
            (betas["date"] <= pd.Timestamp(disp_end))
        ]
    else:
        betas_display = betas

    fig_roll = plot_rolling_betas(betas_display, selected_factors)
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
