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
from src.portfolio import optimize_beta_neutral, portfolio_returns, stock_factor_exposures
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
            scheme = st.session_state.get("weight_scheme_radio")
            if scheme == "shares":
                sampled_permnos = sampled["permno"].tolist()
                latest = (
                    crsp[crsp["permno"].isin(sampled_permnos)]
                    .sort_values("date")
                    .drop_duplicates("permno", keep="last")
                    .set_index("permno")[["prc", "mcap"]]
                )
                latest["prc"] = latest["prc"].abs()
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
            elif scheme == "dollar":
                for p in sampled["permno"].tolist():
                    amt = max(1.0, round(np.random.normal(100.0, 30.0), 2))
                    st.session_state[f"dollar_{p}"] = amt

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

    weight_scheme = st.radio("Weighting", ["equal", "mcap", "shares", "dollar"], horizontal=True,
                             key="weight_scheme_radio")

    # Shares input — one number field per selected ticker
    shares_input = {}
    dollar_input = {}
    permno_to_label = dict(zip(ticker_options["permno"], ticker_options["ticker"]))
    if weight_scheme == "shares" and selected_permnos:
        st.caption("Enter number of shares (negative = short):")
        for p in selected_permnos:
            label = permno_to_label.get(p, str(p))
            shares_input[p] = st.number_input(
                label, value=100.0, step=1.0, format="%.3f", key=f"shares_{p}"
            )
    elif weight_scheme == "dollar" and selected_permnos:
        st.caption("Enter dollar amount (negative = short):")
        for p in selected_permnos:
            label = permno_to_label.get(p, str(p))
            dollar_input[p] = st.number_input(
                label, value=100.0, step=1.0, format="%.2f", key=f"dollar_{p}"
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
port_ret = portfolio_returns(
    selected_permnos, crsp_filtered, weight_scheme,
    shares=shares_input if weight_scheme == "shares" else None,
    dollars=dollar_input if weight_scheme == "dollar" else None,
)

# Per-stock regressions
stock_results = stock_factor_exposures(selected_permnos, crsp_filtered, ff_filtered, selected_factors)

# Portfolio regression
port_result = full_sample_regression(port_ret, ff_filtered, selected_factors)

# Beta-neutral optimization — always run for current portfolio
opt_weights = optimize_beta_neutral(stock_results, selected_permnos)

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

    # Get latest price for each selected stock
    latest_month = crsp_filtered.groupby("permno")["date"].max().reset_index()
    latest_prices = crsp_filtered.merge(latest_month, on=["permno", "date"])
    latest_prices = latest_prices[latest_prices["permno"].isin(selected_permnos)][["permno", "prc", "mcap"]].copy()
    latest_prices["price"] = latest_prices["prc"].abs()

    n = len(holdings)
    if weight_scheme == "equal":
        holdings["weight"] = f"{100/n:.1f}%" if n > 0 else "0%"
        holdings = holdings.merge(latest_prices[["permno", "price"]], on="permno", how="left")
        holdings["n_shares"] = "-"
        holdings["balance"] = "-"
        total_value = None
    elif weight_scheme == "mcap":
        total_mcap = latest_prices["mcap"].sum()
        latest_prices["weight"] = (latest_prices["mcap"] / total_mcap * 100).round(1).astype(str) + "%"
        holdings = holdings.merge(latest_prices[["permno", "weight", "price"]], on="permno", how="left")
        holdings["n_shares"] = "-"
        holdings["balance"] = "-"
        total_value = None
    elif weight_scheme == "shares":
        latest_prices["n_shares"] = latest_prices["permno"].map(shares_input).fillna(0)
        latest_prices["position"] = latest_prices["n_shares"] * latest_prices["price"]  # signed
        gross_exposure = latest_prices["position"].abs().sum()
        net_exposure = latest_prices["position"].sum()
        if gross_exposure > 0:
            latest_prices["weight"] = (latest_prices["position"] / gross_exposure * 100).round(1).astype(str) + "%"
        else:
            latest_prices["weight"] = "0%"
        latest_prices["balance_val"] = latest_prices["position"]
        holdings = holdings.merge(latest_prices[["permno", "weight", "price", "n_shares", "balance_val"]], on="permno", how="left")
        holdings["n_shares"] = holdings["n_shares"].apply(lambda x: f"{int(x):,}" if x != "-" and not (isinstance(x, float) and x != x) else "-")
        holdings["balance"] = holdings["balance_val"].apply(lambda x: f"${x:,.2f}" if x != "-" and not (isinstance(x, float) and x != x) else "-")
        holdings = holdings.drop(columns=["balance_val"])
        total_value = (gross_exposure, net_exposure)
    else:  # dollar
        latest_prices["dollar_pos"] = latest_prices["permno"].map(dollar_input).fillna(0)
        gross_exposure = latest_prices["dollar_pos"].abs().sum()
        net_exposure = latest_prices["dollar_pos"].sum()
        if gross_exposure > 0:
            latest_prices["weight"] = (latest_prices["dollar_pos"] / gross_exposure * 100).round(1).astype(str) + "%"
        else:
            latest_prices["weight"] = "0%"
        # Derive implied shares from dollar amount / price
        latest_prices["n_shares_implied"] = latest_prices.apply(
            lambda r: r["dollar_pos"] / r["price"] if r["price"] > 0 else 0, axis=1
        )
        holdings = holdings.merge(latest_prices[["permno", "weight", "price", "dollar_pos", "n_shares_implied"]], on="permno", how="left")
        holdings["n_shares"] = holdings["n_shares_implied"].apply(lambda x: f"{x:,.2f}" if not (isinstance(x, float) and x != x) else "-")
        holdings["balance"] = holdings["dollar_pos"].apply(lambda x: f"${x:,.2f}" if not (isinstance(x, float) and x != x) else "-")
        holdings = holdings.drop(columns=["dollar_pos", "n_shares_implied"])
        total_value = (gross_exposure, net_exposure)

    holdings["price"] = holdings["price"].apply(lambda x: f"${x:,.2f}" if not (isinstance(x, float) and x != x) else "-")

    display_cols = {
        "ticker": "Ticker", "comnam": "Company", "sector": "Sector",
        "weight": "Weight", "price": "Share Price", "n_shares": "# Shares", "balance": "Balance",
    }
    base_cols = ["ticker", "comnam"]
    if "sector" in holdings.columns:
        base_cols.append("sector")
    base_cols += ["weight", "price", "n_shares", "balance"]
    show_cols = [c for c in base_cols if c in holdings.columns]

    st.dataframe(
        holdings[show_cols].rename(columns=display_cols),
        use_container_width=True,
        hide_index=True,
    )

    # ── Current portfolio metrics ─────────────────────────────────────────────
    if port_ret is not None and not port_ret.empty:
        cum_ret = (1 + port_ret["ret"]).cumprod() - 1
        final_cum = cum_ret.iloc[-1]
        col1, col2, col3 = st.columns(3)

        with col1:
            if total_value is not None:
                gross, _ = total_value
                st.metric("Gross Exposure", f"${gross:,.2f}")
            st.metric("Cumulative Return", f"{final_cum:.1%}")

        with col2:
            if total_value is not None:
                _, net = total_value
                st.metric("Net Exposure", f"${net:,.2f}")
            if final_cum > -1:
                ann_ret = (1 + final_cum) ** (12 / len(port_ret)) - 1
                st.metric("Annualized Return", f"{ann_ret:.1%}")
            else:
                st.metric("Annualized Return", "N/A")

        with col3:
            ann_vol = port_ret["ret"].std() * (12 ** 0.5)
            st.metric("Annualized Volatility", f"{ann_vol:.1%}")

    # ── Beta-neutral optimized weights ────────────────────────────────────────
    if opt_weights is not None:
        st.subheader("Beta-Neutral Optimized Weights")
        st.caption("Treynor-Black: maximizes α/σ²ₑ subject to β_mkt = 0. Net exposure unconstrained.")

        opt_rows = []
        for p in selected_permnos:
            res = stock_results.get(p)
            ticker = permno_to_ticker.get(p, str(p))
            w_opt = opt_weights.get(p, 0.0)
            alpha = res.params.get("const", float("nan")) if res else float("nan")
            beta_mkt = res.params.get("mktrf", float("nan")) if res else float("nan")
            opt_rows.append({
                "Ticker": ticker,
                "Opt Weight": f"{w_opt:+.1%}",
                "Direction": "Long" if w_opt > 0 else "Short",
                "Alpha (monthly)": f"{alpha:.4f}" if res else "—",
                "MKT-RF β": f"{beta_mkt:.3f}" if res else "—",
            })
        opt_df = pd.DataFrame(opt_rows).sort_values("Opt Weight", ascending=False)
        st.dataframe(opt_df, use_container_width=True, hide_index=True)

        # ── Optimized portfolio return metrics ────────────────────────────────
        df_opt = crsp_filtered[crsp_filtered["permno"].isin(selected_permnos)].copy()
        df_opt["w"] = df_opt["permno"].map(opt_weights).fillna(0)
        opt_port_ret = (
            df_opt.groupby("date")
            .apply(lambda g: (g["w"] * g["ret"]).sum(), include_groups=False)
            .reset_index()
        )
        opt_port_ret.columns = ["date", "ret"]
        opt_port_ret = opt_port_ret.sort_values("date")

        if not opt_port_ret.empty:
            opt_cum = (1 + opt_port_ret["ret"]).cumprod() - 1
            opt_final = opt_cum.iloc[-1]
            opt_beta = sum(
                opt_weights.get(p, 0) * (stock_results[p].params.get("mktrf", 0) if stock_results.get(p) else 0)
                for p in selected_permnos
            )

            oc1, oc2, oc3 = st.columns(3)
            with oc1:
                st.metric("Portfolio β", f"{opt_beta:.4f}")
                st.metric("Cumulative Return", f"{opt_final:.1%}")
            with oc2:
                opt_net = sum(opt_weights.get(p, 0) for p in selected_permnos)
                st.metric("Net Exposure (normalized)", f"{opt_net:+.1%}")
                if opt_final > -1:
                    opt_ann = (1 + opt_final) ** (12 / len(opt_port_ret)) - 1
                    st.metric("Annualized Return", f"{opt_ann:.1%}")
                else:
                    st.metric("Annualized Return", "N/A")
            with oc3:
                opt_vol = opt_port_ret["ret"].std() * (12 ** 0.5)
                st.metric("Annualized Volatility", f"{opt_vol:.1%}")


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
