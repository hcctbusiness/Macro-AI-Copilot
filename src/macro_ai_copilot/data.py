"""Data access through QuantCore (Yahoo Finance prices, FRED public CSV — no API key)."""
from __future__ import annotations

import pandas as pd
import quantcore as qc

from . import config


def load_macro(start: str = config.FRED_START, refresh: bool = False) -> pd.DataFrame:
    """Raw FRED series at their native frequency (no lagging applied here)."""
    return qc.data.fred(config.FRED_SERIES, start=start, refresh=refresh)


def load_prices(start: str = config.PRICE_START, refresh: bool = False) -> pd.DataFrame:
    """Daily adjusted closes for the ETF universe, common history only."""
    px = qc.data.prices(config.ASSETS, start=start, refresh=refresh)
    return px[config.ASSETS].dropna()
