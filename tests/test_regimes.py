import numpy as np
import pandas as pd

from macro_ai_copilot import features, regimes


def _feat(raw_macro):
    return features.build_features(features.align_macro(raw_macro))


def test_quadrant_rule():
    idx = pd.date_range("2020-01-31", periods=8, freq="ME")
    scores = pd.DataFrame({"growth": np.arange(8.0), "inflation": 0.0}, index=idx)
    feat = pd.DataFrame({"cpi_yoy": [5, 5, 4, 4, 3, 3, 2, 9.0]}, index=idx)
    q = regimes.quadrant_regimes(scores, feat, window=6)
    assert list(q.index) == list(idx[6:])
    assert q.iloc[0] == "Goldilocks"   # growth up, inflation down
    assert q.iloc[1] == "Reflation"    # growth up, inflation up
    assert set(q.unique()) <= set(regimes.QUADRANTS)


def test_gmm_labels_are_expanding_window_only(raw_macro):
    feat = _feat(raw_macro).iloc[:110]
    full = regimes.fit_gmm_expanding(feat, n_clusters=3, min_months=60)
    part = regimes.fit_gmm_expanding(feat.iloc[:80], n_clusters=3, min_months=60)
    pd.testing.assert_frame_equal(full.iloc[: len(part)], part)


def test_gmm_output_shape_and_probabilities(raw_macro):
    feat = _feat(raw_macro).iloc[:90]
    out = regimes.fit_gmm_expanding(feat, n_clusters=3, min_months=60, refit_every=6)
    assert len(out) == 90 - 59
    assert out["regime"].between(0, 2).all()
    p = out[[f"p_{k}" for k in range(3)]]
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (out["regime"].values == p.values.argmax(axis=1)).all()


def test_gmm_regimes_ordered_by_stress():
    rng = np.random.default_rng(3)
    calm = rng.normal(-1, 0.2, size=(80, 3))
    stress = rng.normal(2, 0.2, size=(40, 3))
    X = pd.DataFrame(np.vstack([calm, stress]), columns=regimes.STRESS_COLS,
                     index=pd.date_range("2000-01-31", periods=120, freq="ME"))
    out = regimes.fit_gmm_expanding(X, n_clusters=2, min_months=100)
    assert (out["regime"] == 1).all()  # final rows are the stressed cluster → highest label
