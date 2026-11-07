"""Central configuration: universe, FRED series, publication lags and model settings."""
from __future__ import annotations

ASSETS = ["SPY", "TLT", "IEF", "GLD", "DBC", "EFA", "EEM"]
ASSET_LABELS = {
    "SPY": "US equities", "TLT": "US Treasuries 20y+", "IEF": "US Treasuries 7-10y",
    "GLD": "Gold", "DBC": "Commodities", "EFA": "Developed ex-US equities", "EEM": "Emerging equities",
}

# Monthly macro series: (FRED id, publication lag in months after the reference month ends).
# A lag of 1 means the value for March is first usable at the end of April.
MONTHLY_SERIES = {"INDPRO": 1, "PAYEMS": 1, "UNRATE": 1, "CPIAUCSL": 1}

# Higher-frequency series: (FRED id, publication lag in calendar days).
# NFCI is dated on the Friday of the reference week but released the following Wednesday.
DAILY_SERIES = {"NFCI": 5, "T10Y2Y": 1, "BAA10Y": 1, "VIXCLS": 1}

FRED_SERIES = list(MONTHLY_SERIES) + list(DAILY_SERIES)

PRICE_START = "2006-01-01"   # DBC (commodities) starts trading Feb-2006
FRED_START = "1985-01-01"

MIN_TRAIN_MONTHS = 60        # ML needs 5 years of asset history before the first prediction
REGIME_MIN_MONTHS = 60       # GMM needs 5 years of macro history before the first label
N_CLUSTERS = 4

MAX_WEIGHT = 0.35
MIN_WEIGHT = 0.0
TILT_STRENGTH = 1.0          # weight shift per unit of (p_i - mean p)

COMMISSION_BPS = 1.0
SLIPPAGE_BPS = 2.0
SEED = 7
