import numpy as np
import pandas as pd

from macro_ai_copilot import model


def _toy(n=90, assets=("A", "B", "C")):
    rng = np.random.default_rng(5)
    idx = pd.date_range("2005-01-31", periods=n, freq="ME")
    X = pd.DataFrame(rng.normal(size=(n, 4)), idx, list("wxyz"))
    mret = pd.DataFrame(rng.normal(0, 0.03, size=(n, len(assets))), idx, list(assets))
    return X, mret


def test_relative_targets_use_next_month():
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    mret = pd.DataFrame({"A": [0.0, 0.05, -0.02], "B": [0.0, -0.05, 0.02]}, index=idx)
    y = model.relative_targets(mret)
    assert y.loc[idx[0], "A"] == 1.0 and y.loc[idx[1], "A"] == 0.0
    assert y.iloc[-1].isna().all()


def test_walk_forward_has_no_leakage():
    X, mret = _toy()
    y = model.relative_targets(mret)
    p = model.walk_forward_predict(X, y, min_train=40)
    t = p.index[10]
    # Change everything realised after t: predictions up to t must not move.
    mret2 = mret.copy()
    mret2.loc[mret2.index > t] *= -1
    X2 = X.copy()
    X2.loc[X2.index > t] += 10
    p2 = model.walk_forward_predict(X2, model.relative_targets(mret2), min_train=40)
    pd.testing.assert_frame_equal(p.loc[:t], p2.loc[:t])
    assert p.index[0] > X.index[39]
    assert ((p > 0) & (p < 1)).all().all()


def test_gbm_and_diagnostics_run():
    X, mret = _toy(80)
    y = model.relative_targets(mret)
    p = model.walk_forward_predict(X, y, kind="gbm", min_train=50, retrain_every=6)
    d = model.prediction_diagnostics(p, mret)
    assert 0 <= d["Hit rate"] <= 1 and -1 <= d["Mean rank IC"] <= 1
