import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from macro_ai_copilot import config  # noqa: E402


@pytest.fixture
def raw_macro():
    """Synthetic FRED-like raw data: monthly series dated on the 1st, daily/weekly series."""
    rng = np.random.default_rng(0)
    m = pd.date_range("1995-01-01", "2012-12-01", freq="MS")
    d = pd.bdate_range("1995-01-02", "2012-12-31")
    w = pd.date_range("1995-01-06", "2012-12-28", freq="W-FRI")
    monthly = pd.DataFrame({
        "INDPRO": 100 * np.exp(np.cumsum(rng.normal(0.002, 0.006, len(m)))),
        "PAYEMS": 120000 * np.exp(np.cumsum(rng.normal(0.001, 0.002, len(m)))),
        "UNRATE": 5 + np.cumsum(rng.normal(0, 0.1, len(m))),
        "CPIAUCSL": 150 * np.exp(np.cumsum(rng.normal(0.002, 0.002, len(m)))),
    }, index=m)
    daily = pd.DataFrame({
        "T10Y2Y": 1 + np.cumsum(rng.normal(0, 0.02, len(d))),
        "BAA10Y": 2 + np.abs(np.cumsum(rng.normal(0, 0.02, len(d)))),
        "VIXCLS": 15 * np.exp(np.cumsum(rng.normal(0, 0.03, len(d))) * 0.1),
    }, index=d)
    weekly = pd.DataFrame({"NFCI": np.cumsum(rng.normal(0, 0.05, len(w)))}, index=w)
    return pd.concat([monthly, daily, weekly], axis=1, sort=True)[config.FRED_SERIES]


@pytest.fixture
def prices():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2006-01-02", "2013-12-31")
    r = rng.normal(0.0003, 0.01, size=(len(idx), len(config.ASSETS)))
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), idx, config.ASSETS)
