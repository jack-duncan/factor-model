"""Project-wide constants and configuration."""

import os
from dotenv import load_dotenv

load_dotenv()

# ── WRDS credentials ──
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "")

# ── Universe ──
UNIVERSE_SIZE = 3000
REBALANCE_FREQ = "M"  # monthly

# ── Date range ──
START_DATE = "2015-01-01"
END_DATE = None  # None = present

# ── CRSP filters ──
VALID_SHRCD = [10, 11]       # ordinary common shares
VALID_EXCHCD = [1, 2, 3]     # NYSE, AMEX, NASDAQ

# ── Factor definitions ──
FACTORS_PREBUILT = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]
FACTORS_CONSTRUCTED = ["volatility", "liquidity", "leverage", "growth", "log_size", "value"]
ALL_FACTORS = FACTORS_PREBUILT

FACTOR_DISPLAY_NAMES = {
    "mktrf": "MKT-RF",
    "smb": "SMB",
    "hml": "HML",
    "rmw": "RMW",
    "cma": "CMA",
    "umd": "Momentum",
    "volatility": "Volatility",
    "liquidity": "Liquidity",
    "leverage": "Leverage",
    "growth": "Growth",
    "log_size": "Log Size",
    "value": "Value",
}

# ── Estimation ──
ROLLING_WINDOW = 60  # months
MIN_OBS = 36  # minimum observations for regression
LOOKBACK_MONTHS = 12  # for trailing characteristics (vol, liquidity)
COMPUSTAT_LAG_MONTHS = 6  # avoid look-ahead bias on fundamentals

# ── SIC code → broad sector mapping ──
# Ranges follow the standard SIC division structure.
def sic_to_sector(sic):
    if sic is None or (isinstance(sic, float) and sic != sic):
        return "Unknown"
    sic = int(sic)
    if sic < 1000:   return "Agriculture"
    if sic < 1500:   return "Mining"
    if sic < 1800:   return "Construction"
    if sic < 4000:   return "Manufacturing"
    if sic < 5000:   return "Transport & Utilities"
    if sic < 5200:   return "Wholesale Trade"
    if sic < 6000:   return "Retail Trade"
    if sic < 6800:   return "Finance & Real Estate"
    if sic < 9000:   return "Services"
    return "Government / Other"


# ── Paths ──
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
METADATA_PATH = "data/raw/_metadata.json"
