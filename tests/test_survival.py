"""Tests for the survival / RUL projection layer (src/survival.py).

These verify the mathematical invariants of the constant-hazard projection — they
do NOT assert any real-data lifetime, because there is no trained time-to-event
model (see src/survival.py docstring).
"""
import numpy as np
import pandas as pd
import pytest

from src.survival import (
    WINDOW_DAYS,
    daily_hazard,
    expected_rul,
    fleet_rul,
    median_rul_days,
    survival_curve,
)


def test_survival_curve_starts_at_one():
    days, surv = survival_curve(0.3, horizon_days=180)
    assert days[0] == 0
    assert surv[0] == pytest.approx(1.0)


def test_survival_curve_monotonic_non_increasing_and_bounded():
    for p30 in (0.0, 0.05, 0.3, 0.6, 0.95, 0.999):
        days, surv = survival_curve(p30, horizon_days=180)
        assert np.all(surv >= 0.0) and np.all(surv <= 1.0), p30
        diffs = np.diff(surv)
        assert np.all(diffs <= 1e-12), f"S(t) must be non-increasing for p30={p30}"


def test_survival_at_window_matches_p30():
    # By construction S(WINDOW_DAYS) == 1 - p30.
    for p30 in (0.1, 0.4, 0.8):
        days, surv = survival_curve(p30, horizon_days=WINDOW_DAYS)
        assert surv[WINDOW_DAYS] == pytest.approx(1.0 - p30, abs=1e-9)


def test_zero_risk_flat_curve():
    days, surv = survival_curve(0.0, horizon_days=180)
    assert np.allclose(surv, 1.0)
    assert daily_hazard(0.0) == pytest.approx(0.0)


def test_higher_p30_shorter_median_rul():
    # Higher 30-day failure probability => shorter (or equal-capped) median RUL.
    rul_low = median_rul_days(0.1, horizon_days=180)
    rul_mid = median_rul_days(0.4, horizon_days=180)
    rul_high = median_rul_days(0.85, horizon_days=180)
    assert rul_high < rul_mid < rul_low


def test_expected_rul_within_horizon_or_flagged():
    horizon = 180
    for p30 in (0.001, 0.05, 0.2, 0.5, 0.9, 0.999):
        rul = expected_rul(p30, horizon_days=horizon)
        if isinstance(rul, str):
            assert rul == f">{horizon}d"          # honest cap flag
        else:
            assert 0 <= rul <= horizon


def test_low_risk_well_is_flagged_not_extrapolated():
    # A tiny p30 should not cross 50% within the horizon → flagged string.
    rul = expected_rul(0.001, horizon_days=180)
    assert isinstance(rul, str) and rul == ">180d"


def test_high_risk_well_crosses_within_horizon():
    rul = expected_rul(0.9, horizon_days=180)
    assert isinstance(rul, int) and 0 < rul <= 180


def test_fleet_rul_sorted_ascending_one_row_per_well():
    probs = pd.Series(
        {"well_01": 0.8, "well_02": 0.1, "well_03": 0.45, "well_04": 0.02},
        name="risk",
    )
    df = fleet_rul(probs, horizon_days=180)
    # one row per well, no dupes
    assert len(df) == len(probs)
    assert set(df["well_id"]) == set(probs.index)
    # sorted ascending by median_rul_days (soonest failure first)
    assert list(df["median_rul_days"]) == sorted(df["median_rul_days"])
    # soonest-failure well is the highest-p30 well
    assert df.iloc[0]["well_id"] == "well_01"
    assert set(df.columns) == {"well_id", "p30", "median_rul_days"}


def test_fleet_rul_caps_at_horizon():
    probs = pd.Series({"safe": 0.0001, "risky": 0.95})
    df = fleet_rul(probs, horizon_days=90)
    assert df["median_rul_days"].max() <= 90
    # the safe well's curve never crosses 0.5 → capped exactly at horizon
    safe_row = df[df["well_id"] == "safe"].iloc[0]
    assert safe_row["median_rul_days"] == 90


def test_daily_hazard_inverts_p30():
    # h satisfies p30 = 1 - (1 - h)**WINDOW_DAYS.
    for p30 in (0.2, 0.5, 0.75):
        h = daily_hazard(p30)
        recovered = 1.0 - (1.0 - h) ** WINDOW_DAYS
        assert recovered == pytest.approx(p30, abs=1e-9)
