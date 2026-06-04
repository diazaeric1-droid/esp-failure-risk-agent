"""Smoke tests for feature engineering and model wrapper."""
import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES, featurize_well


def make_scada(days: int = 60, **overrides) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    base = {
        "date": pd.date_range("2026-01-01", periods=days),
        "bfpd": rng.normal(2400, 100, days),
        "intake_pressure_psi": rng.normal(130, 15, days),
        "motor_temp_f": rng.normal(290, 5, days),
        "motor_amps": rng.normal(62, 3, days),
        "runtime_pct": rng.normal(99, 1, days),
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_featurize_well_returns_all_features():
    df = make_scada()
    feats = featurize_well(df)
    assert set(feats) == set(FEATURE_NAMES)
    assert all(np.isfinite(list(feats.values())))


def test_featurize_detects_amps_creep():
    days = 60
    creep_df = make_scada(motor_amps=np.linspace(60, 88, days))
    flat_df = make_scada(motor_amps=np.full(days, 62.0))
    creep = featurize_well(creep_df)
    flat = featurize_well(flat_df)
    assert creep["motor_amps_slope_30d"] > flat["motor_amps_slope_30d"]
    assert creep["high_amps_days_30d"] >= flat["high_amps_days_30d"]


def test_featurize_tolerates_missing_optional_channels():
    # Old 5-channel exports (no drive_freq_hz / current_imbalance_pct) must still
    # produce the full feature schema via healthy-default backfill.
    df = make_scada()
    feats = featurize_well(df)
    assert set(feats) == set(FEATURE_NAMES)
    assert feats["drive_freq_last7_mean"] == 58.0
    assert feats["current_imbalance_max_30d"] == 3.0


def test_classify_failure_mode_electrical_and_scale():
    from src.explainer import classify_failure_mode
    elec, _ = classify_failure_mode(
        {"current_imbalance_max_30d": 18.0, "high_imbalance_days_30d": 5})
    assert "Electrical" in elec
    scale, _ = classify_failure_mode(
        {"current_imbalance_max_30d": 3.0, "motor_amps_slope_30d": 0.4,
         "motor_temp_slope_30d": 0.3, "bfpd_cv_30d": 0.03, "downtime_days_30d": 0})
    assert "Scale" in scale
