"""Factor construction: quintile-spread returns from stock characteristics."""

import logging

import numpy as np
import pandas as pd

from src.config import (
    COMPUSTAT_LAG_MONTHS,
    FACTORS_CONSTRUCTED,
    LOOKBACK_MONTHS,
)

logger = logging.getLogger(__name__)


# ── Core utility ─────────────────────────────────────────────────────────────

def _quintile_spread_return(df, date_col, char_col, ret_col="ret"):
    """Compute monthly long-short return from a characteristic.

    Each month: sort stocks by *char_col*, equal-weight long top quintile,
    short bottom quintile.  Returns a Series indexed by date.
    """
    out = []
    for dt, grp in df.groupby(date_col):
        valid = grp.dropna(subset=[char_col, ret_col])
        if len(valid) < 10:
            continue
        valid = valid.copy()
        valid["q"] = pd.qcut(valid[char_col], 5, labels=False, duplicates="drop")
        if valid["q"].nunique() < 2:
            continue
        top = valid.loc[valid["q"] == valid["q"].max(), ret_col].mean()
        bot = valid.loc[valid["q"] == valid["q"].min(), ret_col].mean()
        out.append({"date": dt, "factor_ret": top - bot})

    if not out:
        return pd.DataFrame(columns=["date", "factor_ret"])
    return pd.DataFrame(out)


# ── CRSP-only factors ────────────────────────────────────────────────────────

def _trailing_volatility(crsp):
    """Trailing 12-month return standard deviation per stock-month."""
    df = crsp[["permno", "date", "ret"]].copy()
    df = df.sort_values(["permno", "date"])
    df["volatility"] = (
        df.groupby("permno")["ret"]
        .transform(lambda x: x.rolling(LOOKBACK_MONTHS, min_periods=6).std())
    )
    return df[["permno", "date", "volatility"]].dropna(subset=["volatility"])


def _trailing_liquidity(crsp):
    """Trailing 12-month average dollar volume per stock-month."""
    df = crsp[["permno", "date", "prc", "vol"]].copy()
    df = df.sort_values(["permno", "date"])
    df["dollar_vol"] = np.abs(df["prc"]) * df["vol"]
    df["liquidity"] = (
        df.groupby("permno")["dollar_vol"]
        .transform(lambda x: x.rolling(LOOKBACK_MONTHS, min_periods=6).mean())
    )
    # Log-transform to reduce skew
    df["liquidity"] = np.log1p(df["liquidity"])
    return df[["permno", "date", "liquidity"]].dropna(subset=["liquidity"])


def _log_size(crsp):
    """Log market cap per stock-month."""
    df = crsp[["permno", "date", "mcap"]].copy()
    df["log_size"] = np.log(df["mcap"])
    return df[["permno", "date", "log_size"]].dropna(subset=["log_size"])


# ── Compustat-based factors ──────────────────────────────────────────────────

def _merge_compustat_to_crsp(crsp, compustat, ccm_link):
    """Map Compustat fundamentals to CRSP stock-months with lag."""
    if compustat is None or ccm_link is None:
        return None

    comp = compustat.copy()
    link = ccm_link.copy()

    # Merge Compustat with CCM link to get permno
    comp = comp.merge(link[["gvkey", "permno", "linkdt", "linkenddt"]], on="gvkey")
    comp = comp[
        (comp["datadate"] >= comp["linkdt"])
        & (comp["datadate"] <= comp["linkenddt"])
    ]
    comp = comp.drop(columns=["linkdt", "linkenddt"])

    # Apply lag: fundamentals available COMPUSTAT_LAG_MONTHS after fiscal year end
    comp["avail_date"] = comp["datadate"] + pd.DateOffset(months=COMPUSTAT_LAG_MONTHS)
    comp["avail_date"] = comp["avail_date"] + pd.offsets.MonthEnd(0)

    # For each CRSP stock-month, find the most recent available fundamental
    crsp_slim = crsp[["permno", "date", "mcap"]].copy()
    crsp_slim = crsp_slim.sort_values(["permno", "date"])

    merged = pd.merge_asof(
        crsp_slim.sort_values("date"),
        comp[["permno", "avail_date", "at", "lt", "dltt", "dlc", "ceq", "sale", "revt"]]
        .sort_values("avail_date")
        .rename(columns={"avail_date": "date"}),
        on="date",
        by="permno",
        direction="backward",
    )
    return merged


def _leverage(merged_comp):
    """Leverage = (long-term debt + current debt) / total assets."""
    if merged_comp is None:
        return None
    df = merged_comp.copy()
    df["dltt"] = df["dltt"].fillna(0)
    df["dlc"] = df["dlc"].fillna(0)
    df["leverage"] = (df["dltt"] + df["dlc"]) / df["at"]
    df = df.replace([np.inf, -np.inf], np.nan)
    return df[["permno", "date", "leverage"]].dropna(subset=["leverage"])


def _growth(merged_comp):
    """Growth = year-over-year sales growth."""
    if merged_comp is None:
        return None
    df = merged_comp[["permno", "date", "sale"]].copy()
    df = df.sort_values(["permno", "date"])
    df["prev_sale"] = df.groupby("permno")["sale"].shift(12)
    df["growth"] = (df["sale"] - df["prev_sale"]) / df["prev_sale"].abs()
    df = df.replace([np.inf, -np.inf], np.nan)
    return df[["permno", "date", "growth"]].dropna(subset=["growth"])


def _value(merged_comp):
    """Value = book equity / market cap (book-to-market)."""
    if merged_comp is None:
        return None
    df = merged_comp.copy()
    df["value"] = df["ceq"] / df["mcap"]
    df = df.replace([np.inf, -np.inf], np.nan)
    return df[["permno", "date", "value"]].dropna(subset=["value"])


# ── Master builder ───────────────────────────────────────────────────────────

def build_all_factor_returns(crsp, universe, ff, compustat=None, ccm_link=None):
    """Construct all factor return series and merge with pre-built FF factors.

    Returns
    -------
    DataFrame with columns: date, mktrf, smb, hml, rmw, cma, umd, rf,
    volatility, liquidity, log_size, leverage, growth, value.
    """
    crsp = crsp.copy()
    crsp["date"] = pd.to_datetime(crsp["date"])

    # Restrict to universe stocks for factor construction
    univ = universe[["date", "permno"]].copy()
    univ["date"] = pd.to_datetime(univ["date"])
    crsp_univ = crsp.merge(univ, on=["date", "permno"], how="inner")

    # ── CRSP-only characteristics ──
    vol_char = _trailing_volatility(crsp_univ)
    liq_char = _trailing_liquidity(crsp_univ)
    size_char = _log_size(crsp_univ)

    # Build factor return series
    constructed = {}

    logger.info("Building volatility factor...")
    vol_data = crsp_univ.merge(vol_char, on=["permno", "date"], how="inner")
    constructed["volatility"] = _quintile_spread_return(vol_data, "date", "volatility")

    logger.info("Building liquidity factor...")
    liq_data = crsp_univ.merge(liq_char, on=["permno", "date"], how="inner")
    constructed["liquidity"] = _quintile_spread_return(liq_data, "date", "liquidity")

    logger.info("Building log_size factor...")
    size_data = crsp_univ.merge(size_char, on=["permno", "date"], how="inner")
    constructed["log_size"] = _quintile_spread_return(size_data, "date", "log_size")

    # ── Compustat-based characteristics ──
    merged_comp = _merge_compustat_to_crsp(crsp_univ, compustat, ccm_link)

    if merged_comp is not None:
        lev_char = _leverage(merged_comp)
        gro_char = _growth(merged_comp)
        val_char = _value(merged_comp)

        if lev_char is not None:
            logger.info("Building leverage factor...")
            lev_data = crsp_univ.merge(lev_char, on=["permno", "date"], how="inner")
            constructed["leverage"] = _quintile_spread_return(lev_data, "date", "leverage")

        if gro_char is not None:
            logger.info("Building growth factor...")
            gro_data = crsp_univ.merge(gro_char, on=["permno", "date"], how="inner")
            constructed["growth"] = _quintile_spread_return(gro_data, "date", "growth")

        if val_char is not None:
            logger.info("Building value factor...")
            val_data = crsp_univ.merge(val_char, on=["permno", "date"], how="inner")
            constructed["value"] = _quintile_spread_return(val_data, "date", "value")
    else:
        logger.warning("Compustat data unavailable — skipping leverage, growth, value factors.")

    # ── Combine everything ──
    ff = ff.copy()
    ff["date"] = pd.to_datetime(ff["date"])
    result = ff.copy()

    for name, factor_df in constructed.items():
        if factor_df.empty:
            logger.warning("Factor '%s' produced no returns — skipping.", name)
            continue
        factor_df = factor_df.rename(columns={"factor_ret": name})
        factor_df["date"] = pd.to_datetime(factor_df["date"])
        result = result.merge(factor_df, on="date", how="left")

    # Fill any constructed factors that are missing entirely
    for f in FACTORS_CONSTRUCTED:
        if f not in result.columns:
            result[f] = np.nan

    result = result.sort_values("date").reset_index(drop=True)
    logger.info(
        "Factor returns: %d months, %d factors available.",
        len(result),
        result.drop(columns=["date", "rf"]).notna().any().sum(),
    )
    return result
