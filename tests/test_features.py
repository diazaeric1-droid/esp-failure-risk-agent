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
