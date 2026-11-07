import numpy as np
import pandas as pd
import pytest

from macro_ai_copilot import features


def test_monthly_series_published_one_month_after_reference_month():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3, freq="MS"), name="X")
    out = features.lag_monthly(s, 1)
    # January value (dated 2020-01-01) is first usable at the end of February.
    assert out.index[0] == pd.Timestamp("2020-02-29")
    assert out.loc["2020-02-29"] == 1.0
    assert out.loc["2020-04-30"] == 3.0


def test_monthly_lag_below_one_rejected():
    s = pd.Series([1.0], index=pd.DatetimeIndex(["2020-01-01"]))
    with pytest.raises(ValueError):
        features.lag_monthly(s, 0)


def test_daily_lag_shifts_forward():
    s = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2020-01-30", "2020-01-31"]))
    out = features.lag_daily(s, 1)
    assert out.loc["2020-01-31"] == 1.0  # Jan-31 print not usable until Feb-1
    assert out.loc["2020-02-29"] == 2.0


def test_features_do_not_use_future_data(raw_macro):
    """Perturbing raw data after a cut-off must leave all features up to the cut-off unchanged."""
    cut = pd.Timestamp("2008-06-30")
    base = features.build_features(features.align_macro(raw_macro))
    shocked = raw_macro.copy()
    shocked.loc[shocked.index > cut] *= 3.0
    alt = features.build_features(features.align_macro(shocked))
    pd.testing.assert_frame_equal(base.loc[:cut], alt.loc[:cut])
    assert not np.allclose(base.loc["2009"].values, alt.loc["2009"].values)


def test_expanding_zscore_is_point_in_time(raw_macro):
    f = features.build_features(features.align_macro(raw_macro))
    z_full = features.expanding_zscore(f)
    z_part = features.expanding_zscore(f.loc[:"2005-12-31"])
    pd.testing.assert_frame_equal(z_full.loc[:"2005-12-31"], z_part)
