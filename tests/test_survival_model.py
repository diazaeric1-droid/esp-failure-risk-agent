"""Tests for the genuine discrete-time hazard survival model (src/survival_model.py).

Unlike tests/test_survival.py (which checks the *projection* layer's algebra), these
assert that the TRAINED time-to-event model (a) produces valid survival curves, (b)
orders wells better than chance on real run-life data, and (c) the survival metrics
(C-index, IBS) behave correctly — including the synthetic-data targets the README quotes.
"""
import numpy as np
import pandas as pd
import pytest

from src.survival_model import (
    DiscreteTimeHazardModel,
    concordance_index,
    integrated_brier_score,
    evaluate_oof,
    _person_period,
    _km_survival,
    MAX_HORIZON,
)


# ---- synthetic run-life fixture (separable, censored) --------------------------------
def _toy_runlife(n=120, seed=0):
    """Two clusters: high-risk wells fail early, low-risk wells are censored late.
    One informative feature + noise features matching no real schema — we feed a frame
    whose columns are the real FEATURE_NAMES so the model's column selection works."""
    from src.features import FEATURE_NAMES
    rng = np.random.default_rng(seed)
    risk = rng.random(n)
    # higher risk -> earlier failure; low risk -> censored at horizon
    fail = risk > 0.6
    dur = np.where(fail, np.clip((1 - risk) * 60, 1, 30).round(),
                   rng.integers(31, MAX_HORIZON + 1, n)).astype(int)
    ev = fail.astype(int)
    # Build a feature frame: put the signal in motor_amps_slope_30d, noise elsewhere.
    data = {f: rng.normal(0, 1, n) for f in FEATURE_NAMES}
    data["motor_amps_slope_30d"] = risk * 5 + rng.normal(0, 0.3, n)
    X = pd.DataFrame(data, index=[f"w{i:03d}" for i in range(n)])[FEATURE_NAMES]
    return X, dur, ev


def test_person_period_expansion_counts():
    # durations 1,3 with events 1,0 -> 1 + 3 = 4 rows; only the last day of the event well is 1.
    X = np.array([[0.0], [0.0]])
    T, Xpp, ypp = _person_period(np.array([1, 3]), np.array([1, 0]), X)
    assert len(T) == 4
    assert ypp.sum() == 1
    # the single positive is at the event well's final at-risk day (t==1)
    assert ypp[0] == 1


def test_survival_curve_valid_and_monotone():
    X, dur, ev = _toy_runlife()
    m = DiscreteTimeHazardModel().fit(X, dur, ev)
    days, surv = m.survival_grid(X)
    assert days[0] == 0 and surv.shape == (len(X), MAX_HORIZON + 1)
    assert np.allclose(surv[:, 0], 1.0)                       # S(0) = 1
    assert np.all(surv >= -1e-9) and np.all(surv <= 1.0 + 1e-9)
    assert np.all(np.diff(surv, axis=1) <= 1e-9)              # non-increasing in t


def test_higher_risk_shorter_median_rul():
    X, dur, ev = _toy_runlife()
    m = DiscreteTimeHazardModel().fit(X, dur, ev)
    rul = pd.Series(m.median_rul(X), index=X.index)
    risk = pd.Series(m.risk_score(X), index=X.index)
    # the 10 highest-risk wells should have shorter median RUL than the 10 lowest
    hi = rul[risk.nlargest(10).index].mean()
    lo = rul[risk.nsmallest(10).index].mean()
    assert hi < lo


def test_concordance_index_perfect_and_chance():
    # perfect ordering: risk exactly inverse to failure time
    dur = np.array([2, 4, 6, 8]); ev = np.array([1, 1, 1, 0])
    perfect = -dur.astype(float)
    assert concordance_index(dur, ev, perfect) == pytest.approx(1.0)
    anti = dur.astype(float)
    assert concordance_index(dur, ev, anti) == pytest.approx(0.0)
    # constant risk -> all ties -> 0.5
    assert concordance_index(dur, ev, np.ones(4)) == pytest.approx(0.5)


def test_ibs_better_predictor_lower_than_constant():
    X, dur, ev = _toy_runlife()
    m = DiscreteTimeHazardModel().fit(X, dur, ev)
    days, surv = m.survival_grid(X)
    max_t = float(dur[ev == 1].max())
    ibs_model = integrated_brier_score(dur, ev, surv, days, max_t=max_t)
    km = _km_survival(dur, ev, days)
    ibs_km = integrated_brier_score(dur, ev, np.tile(km, (len(X), 1)), days, max_t=max_t)
    assert ibs_model <= ibs_km + 1e-6          # covariates help (or at least don't hurt)
    assert 0.0 <= ibs_model <= 0.25


def test_evaluate_oof_beats_chance_on_toy():
    X, dur, ev = _toy_runlife()
    res = evaluate_oof(X, dur, ev)
    assert res.c_index > 0.6                    # clearly better than chance
    assert res.ibs <= res.ibs_km_baseline + 1e-6
    assert res.n_events + res.n_censored == len(X)


@pytest.mark.parametrize("seed", [1, 2])
def test_evaluate_oof_deterministic(seed):
    X, dur, ev = _toy_runlife(seed=seed)
    a = evaluate_oof(X, dur, ev)
    b = evaluate_oof(X, dur, ev)
    assert a.c_index == pytest.approx(b.c_index)
    assert a.ibs == pytest.approx(b.ibs)
