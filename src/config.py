"""Project-wide constants and configuration."""

import os
from dotenv import load_dotenv

load_dotenv()

# ── WRDS credentials ──
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "")

# ── Universe ──
UNIVERSE_SIZE = 1000
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
ALL_FACTORS = FACTORS_PREBUILT + FACTORS_CONSTRUCTED

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

# ── Paths ──
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
METADATA_PATH = "data/raw/_metadata.json"
