"""Macro feature engineering with explicit publication-lag handling.

Every feature is indexed by month-end date *t* and uses only information that
had been published by the close of *t*:

* Monthly series (FRED dates them on the 1st of the reference month) are moved to
  the reference month-end and then shifted forward by their publication lag
  (default 1 month: March CPI becomes usable at the end of April).
* Weekly/daily series are shifted forward by a lag in days, then sampled with the
  last available value at each month-end.

Caveat: FRED serves the latest revised vintage, not real-time vintages, so
revisions to payrolls/IP are a residual source of optimism (documented in README).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def lag_monthly(s: pd.Series, months: int) -> pd.Series:
    """Re-index a monthly FRED series to the date it became public (month-end)."""
    if months < 1:
        raise ValueError("monthly macro data needs a publication lag of at least 1 month")
    s = s.dropna()
    ref_end = s.index.to_period("M").to_timestamp(how="end").normalize()
    out = pd.Series(s.values, index=ref_end, name=s.name)
    out.index = out.index + pd.offsets.MonthEnd(months)
    return out


def lag_daily(s: pd.Series, days: int) -> pd.Series:
    """Shift a daily/weekly series forward by ``days`` and sample at month-ends."""
    s = s.dropna()
    shifted = pd.Series(s.values, index=s.index + pd.Timedelta(days=days), name=s.name)
    return shifted.resample("ME").last().ffill()


def align_macro(raw: pd.DataFrame,
                monthly: dict[str, int] | None = None,
                daily: dict[str, int] | None = None) -> pd.DataFrame:
    """Point-in-time monthly panel of the raw series (levels)."""
    monthly = config.MONTHLY_SERIES if monthly is None else monthly
    daily = config.DAILY_SERIES if daily is None else daily
    cols = {}
    for sid, lag in monthly.items():
        cols[sid] = lag_monthly(raw[sid], lag)
    for sid, lag in daily.items():
        cols[sid] = lag_daily(raw[sid], lag)
    panel = pd.DataFrame(cols).sort_index()
    panel.index = panel.index.normalize()
    panel = panel.resample("ME").last()
    # Forward-fill at most 2 months for occasional publication gaps (e.g. 2025 shutdown).
    return panel.ffill(limit=2)


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Transform point-in-time levels into stationary macro features."""
    f = pd.DataFrame(index=panel.index)
    # Growth
    f["indpro_yoy"] = panel["INDPRO"].pct_change(12, fill_method=None) * 100
    f["indpro_3m_ann"] = ((panel["INDPRO"] / panel["INDPRO"].shift(3)) ** 4 - 1) * 100
    f["payems_yoy"] = panel["PAYEMS"].pct_change(12, fill_method=None) * 100
    f["unrate_chg12"] = panel["UNRATE"].diff(12)
    # Inflation
    f["cpi_yoy"] = panel["CPIAUCSL"].pct_change(12, fill_method=None) * 100
    f["cpi_impulse"] = f["cpi_yoy"] - f["cpi_yoy"].shift(6)
    # Financial conditions
    f["nfci"] = panel["NFCI"]
    f["nfci_chg3"] = panel["NFCI"].diff(3)
    f["curve_10y2y"] = panel["T10Y2Y"]
    f["credit_baa10y"] = panel["BAA10Y"]
    f["credit_chg3"] = panel["BAA10Y"].diff(3)
    f["log_vix"] = np.log(panel["VIXCLS"])
    return f.dropna()


def expanding_zscore(df: pd.DataFrame, min_periods: int = 36) -> pd.DataFrame:
    """Standardise each column using only data up to and including each date."""
    mu = df.expanding(min_periods).mean()
    sd = df.expanding(min_periods).std()
    return (df - mu) / sd.replace(0, np.nan)


def growth_inflation_scores(feat: pd.DataFrame) -> pd.DataFrame:
    """Composite growth and inflation scores (expanding z-scores, no look-ahead)."""
    z = expanding_zscore(feat[["indpro_yoy", "payems_yoy", "unrate_chg12", "cpi_yoy"]])
    growth = (z["indpro_yoy"] + z["payems_yoy"] - z["unrate_chg12"]) / 3
    return pd.DataFrame({"growth": growth, "inflation": z["cpi_yoy"]}).dropna()
