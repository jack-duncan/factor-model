"""Streamlit app – Factor Model Dashboard."""

import streamlit as st

st.set_page_config(page_title="Factor Lens", layout="wide")
st.title("Factor Lens")
st.markdown("Interactive factor model analysis for US equities.")

# ── Sidebar controls ──
with st.sidebar:
    st.header("Settings")
    # TODO: date range picker
    # TODO: universe size slider
    # TODO: factor selection
    # TODO: rolling window slider

# ── Main content ──
tab_exposures, tab_attribution, tab_portfolio = st.tabs(
    ["Factor Exposures", "Attribution", "Portfolio"]
)

with tab_exposures:
    st.subheader("Rolling Factor Exposures")
    st.info("Select a stock or portfolio to view rolling betas.")

with tab_attribution:
    st.subheader("Return Attribution")
    st.info("Decompose returns into factor and residual components.")

with tab_portfolio:
    st.subheader("Portfolio Analysis")
    st.info("Build a portfolio and view its aggregate factor exposures.")