"""WRDS data pulls with local caching and incremental refresh."""

import json
import logging
import os
from datetime import date

import numpy as np
import pandas as pd

from src.config import (
    METADATA_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    START_DATE,
    VALID_EXCHCD,
    VALID_SHRCD,
    WRDS_USERNAME,
)

logger = logging.getLogger(__name__)


# ── WRDS connection ──────────────────────────────────────────────────────────

def connect_wrds():
    """Establish WRDS connection."""
    import wrds
    return wrds.Connection(wrds_username=WRDS_USERNAME)


def query(conn, sql):
    """Run SQL against WRDS via raw psycopg2.

    Bypasses the pandas 2.3 / SQLAlchemy 1.4 incompatibility in the wrds
    library by dropping directly to the underlying DBAPI connection.
    """
    pgconn = conn.connection.connection
    with pgconn.cursor() as cur:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    for c in df.columns:
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.notna().sum() > 0:
            df[c] = converted
    return df


# ── Data fetchers ────────────────────────────────────────────────────────────

def fetch_crsp_monthly(conn, start_date, end_date):
    """Pull CRSP monthly stock data joined with names table."""
    shrcd_list = ",".join(str(s) for s in VALID_SHRCD)
    exchcd_list = ",".join(str(e) for e in VALID_EXCHCD)

    df = query(conn, f"""
        SELECT m.permno, m.date, m.ret, m.retx, m.prc, m.shrout, m.vol,
               n.exchcd, n.shrcd, n.ticker, n.comnam
        FROM crsp.msf m
        LEFT JOIN crsp.msenames n
            ON m.permno = n.permno
            AND m.date BETWEEN n.namedt AND n.nameendt
        WHERE m.date BETWEEN '{start_date}' AND '{end_date}'
          AND n.shrcd IN ({shrcd_list})
          AND n.exchcd IN ({exchcd_list})
        ORDER BY m.date, m.permno
    """)

    df["date"] = pd.to_datetime(df["date"])
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    df["prc"] = pd.to_numeric(df["prc"], errors="coerce")
    df["shrout"] = pd.to_numeric(df["shrout"], errors="coerce")
    df["mcap"] = np.abs(df["prc"]) * df["shrout"]
    df = df.dropna(subset=["ret"])
    return df


def fetch_ff_factors(conn, start_date, end_date):
    """Pull Fama-French 5 factors + momentum + risk-free rate.

    FF3 + momentum lives in ff.factors_monthly.
    RMW and CMA live in ff.fivefactors_monthly.
    We join them on date so all 7 factors end up in one table.
    """
    df = query(conn, f"""
        SELECT f3.date, f3.mktrf, f3.smb, f3.hml, f3.umd, f3.rf,
               f5.rmw, f5.cma
        FROM ff.factors_monthly f3
        LEFT JOIN ff.fivefactors_monthly f5
            ON f3.date = f5.date
        WHERE f3.date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY f3.date
    """)
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_compustat_annual(conn, start_date, end_date):
    """Pull Compustat annual fundamentals. Returns None if unavailable."""
    try:
        df = query(conn, f"""
            SELECT gvkey, datadate, at, lt, dltt, dlc, ceq, sale, revt, fyear
            FROM comp.funda
            WHERE datadate BETWEEN '{start_date}' AND '{end_date}'
              AND indfmt = 'INDL'
              AND datafmt = 'STD'
              AND popsrc = 'D'
              AND consol = 'C'
            ORDER BY gvkey, datadate
        """)
        df["datadate"] = pd.to_datetime(df["datadate"])
        return df
    except Exception as e:
        logger.warning("Compustat query failed (no access?): %s", e)
        return None


def fetch_ccm_link(conn):
    """Pull CRSP-Compustat link table."""
    try:
        df = query(conn, """
            SELECT gvkey, lpermno AS permno, linkdt, linkenddt, linktype, linkprim
            FROM crsp.ccmxpf_lnkhist
            WHERE linktype IN ('LU', 'LC')
              AND linkprim IN ('P', 'C')
        """)
        df["linkdt"] = pd.to_datetime(df["linkdt"])
        df["linkenddt"] = pd.to_datetime(df["linkenddt"])
        # Fill missing end dates with far future
        df["linkenddt"] = df["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
        return df
    except Exception as e:
        logger.warning("CCM link query failed: %s", e)
        return None


# ── Caching ──────────────────────────────────────────────────────────────────

def load_cached(name, directory=RAW_DATA_DIR):
    """Load a cached parquet file if it exists."""
    path = os.path.join(directory, f"{name}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def save_cache(df, name, directory=RAW_DATA_DIR):
    """Cache a DataFrame as parquet."""
    os.makedirs(directory, exist_ok=True)
    df.to_parquet(os.path.join(directory, f"{name}.parquet"), index=False)


# ── Metadata for incremental pulls ──────────────────────────────────────────

def load_metadata():
    """Read pull metadata (last pull dates per dataset)."""
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            return json.load(f)
    return {}


def save_metadata(name, last_date):
    """Update metadata with the last pull date for a dataset."""
    meta = load_metadata()
    meta[name] = str(last_date)
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def incremental_fetch(conn, fetch_fn, name, start_date, end_date, dedup_cols=None):
    """Fetch new data since last pull, concatenate with cache, and save.

    Parameters
    ----------
    fetch_fn : callable(conn, start_date, end_date) -> DataFrame or None
    dedup_cols : list of columns to use for deduplication (default: all)
    """
    meta = load_metadata()
    cached = load_cached(name)

    if cached is not None and name in meta:
        effective_start = meta[name]
        logger.info("%s: incremental pull from %s", name, effective_start)
    else:
        effective_start = start_date
        logger.info("%s: full pull from %s", name, effective_start)

    new_data = fetch_fn(conn, effective_start, end_date)
    if new_data is None:
        return cached  # graceful fallback (e.g. Compustat unavailable)

    if cached is not None and not new_data.empty:
        combined = pd.concat([cached, new_data], ignore_index=True)
        if dedup_cols:
            combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
        else:
            combined = combined.drop_duplicates(keep="last")
    elif cached is not None:
        combined = cached
    else:
        combined = new_data

    save_cache(combined, name)
    save_metadata(name, str(end_date))
    return combined


# ── Pipeline orchestrator ────────────────────────────────────────────────────

def refresh_data(conn=None):
    """Run the full data pipeline: fetch, build universe, construct factors.

    Returns a dict of all loaded DataFrames.
    """
    from src.factors import build_all_factor_returns
    from src.universe import build_universe

    close_conn = False
    if conn is None:
        conn = connect_wrds()
        close_conn = True

    end_date = str(date.today())

    # 1. Fetch raw data
    crsp = incremental_fetch(
        conn, fetch_crsp_monthly, "crsp_monthly",
        START_DATE, end_date, dedup_cols=["permno", "date"],
    )
    ff = incremental_fetch(
        conn, fetch_ff_factors, "ff_factors",
        START_DATE, end_date, dedup_cols=["date"],
    )
    compustat = incremental_fetch(
        conn, lambda c, s, e: fetch_compustat_annual(c, s, e),
        "compustat_annual", START_DATE, end_date,
        dedup_cols=["gvkey", "datadate"],
    )
    ccm_link = load_cached("ccm_link")
    if ccm_link is None:
        ccm_link = fetch_ccm_link(conn)
        if ccm_link is not None:
            save_cache(ccm_link, "ccm_link")

    if close_conn:
        conn.close()

    # 2. Build universe
    universe, ticker_map = build_universe(crsp)
    save_cache(universe, "universe", PROCESSED_DATA_DIR)
    save_cache(ticker_map, "ticker_map", PROCESSED_DATA_DIR)

    # 3. Construct factor returns
    factor_returns = build_all_factor_returns(crsp, universe, ff, compustat, ccm_link)
    save_cache(factor_returns, "factor_returns", PROCESSED_DATA_DIR)

    return {
        "crsp": crsp,
        "ff_factors": ff,
        "compustat": compustat,
        "ccm_link": ccm_link,
        "universe": universe,
        "ticker_map": ticker_map,
        "factor_returns": factor_returns,
    }
