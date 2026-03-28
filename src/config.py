"""Project-wide constants and configuration."""

import os
from dotenv import load_dotenv

load_dotenv()

# ── WRDS credentials ──
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "")

# ── Universe ──
UNIVERSE_SIZE = 500
REBALANCE_FREQ = "M"  # monthly

# ── Date range ──
START_DATE = "2000-01-01"
END_DATE = "2025-12-31"

# ── Fama-French factors ──
FACTORS = [
    "Mkt-RF",
    "SMB",
    "HML",
    "RMW",
    "CMA",
]

# ── Estimation ──
ROLLING_WINDOW = 60  # months
MIN_OBS = 36  # minimum observations for regression

# ── Paths ──
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"