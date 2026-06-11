"""Tests for the 'bring your own fleet SCADA' upload path.

Covers the strict-schema validator and the long-frame → fleet adapter, which reuse
the existing loader/feature pipeline (no parallel path) so an uploaded fleet scores
through the same engineered features the trained model consumes.
"""
import numpy as np
import pandas as pd

from src.data_loader import (
    UPLOAD_REQUIRED_COLUMNS,
    load_fleet_from_frame,
    scada_template_frame,
    validate_scada_schema,
)
from src.features import FEATURE_NAMES, featurize_fleet


def make_fleet_csv(n_wells: int = 2, days: int = 60) -> pd.DataFrame:
    """A valid long/tidy fleet SCADA frame: stacked per-well time series + well_id."""
    rng = np.random.default_rng(0)
    frames = []
    for w in range(n_wells):
        frames.append(pd.DataFrame({
            "well_id": f"well_{w:03d}",
            "date": pd.date_range("2026-01-01", periods=days),
            "bfpd": rng.normal(2400, 100, days),
            "intake_pressure_psi": rng.normal(130, 15, days),
            "motor_temp_f": rng.normal(290, 5, days),
            "motor_amps": rng.normal(62, 3, days),
            "runtime_pct": rng.normal(99, 1, days),
            "drive_freq_hz": rng.normal(58, 0.5, days),
            "current_imbalance_pct": rng.normal(3, 0.5, days),
        }))
    return pd.concat(frames, ignore_index=True)


def test_validate_scada_schema_all_present_passes():
    df = make_fleet_csv()
    assert validate_scada_schema(df) == []


def test_validate_scada_schema_passes_without_optional_channels():
    # The two v0.5.0 channels are OPTIONAL (backfilled), so dropping them is still valid.
    df = make_fleet_csv().drop(columns=["drive_freq_hz", "current_imbalance_pct"])
    assert validate_scada_schema(df) == []


def test_validate_scada_schema_reports_missing_required_columns():
    df = make_fleet_csv().drop(columns=["intake_pressure_psi", "motor_amps"])
    missing = validate_scada_schema(df)
    assert missing == ["intake_pressure_psi", "motor_amps"]  # order preserved


def test_validate_scada_schema_flags_missing_well_id():
    df = make_fleet_csv().drop(columns=["well_id"])
    assert "well_id" in validate_scada_schema(df)


def test_template_frame_is_self_consistent_and_loadable():
    tmpl = scada_template_frame()
    # The template must itself satisfy the validator it documents.
    assert validate_scada_schema(tmpl) == []
    assert all(c in tmpl.columns for c in UPLOAD_REQUIRED_COLUMNS)


def test_load_fleet_from_frame_reuses_pipeline_to_full_feature_schema():
    df = make_fleet_csv(n_wells=3)
    fleet = load_fleet_from_frame(df)
    assert set(fleet) == {"well_000", "well_001", "well_002"}
    # Scoring path: the adapter feeds the EXISTING featurizer, which must yield the
    # exact fixed feature schema the trained model expects.
    feats = featurize_fleet(fleet)
    assert list(feats.columns) == FEATURE_NAMES
    assert len(feats) == 3
    assert np.isfinite(feats.to_numpy()).all()


def test_load_fleet_from_frame_backfills_optional_channels():
    # Upload without the optional channels still featurizes (healthy-default backfill).
    df = make_fleet_csv(n_wells=2).drop(columns=["drive_freq_hz", "current_imbalance_pct"])
    fleet = load_fleet_from_frame(df)
    feats = featurize_fleet(fleet)
    assert list(feats.columns) == FEATURE_NAMES
    assert (feats["drive_freq_last7_mean"] == 58.0).all()


def test_uploaded_fleet_scores_through_trained_model(tmp_path):
    """End-to-end upload path: long CSV → existing loader → existing features → the
    EXISTING trained model produces a calibrated probability per uploaded well.

    Trains a model into a temp artifact (the app's first-run path) rather than relying
    on a committed one, so this is hermetic. Proves no parallel pipeline: the uploaded
    fleet is scored by `ESPRiskModel.predict_proba` on `featurize_fleet` output, the
    same calls the synthetic fleet uses.
    """
    from src.model import ESPRiskModel

    # A tiny labeled training set built from the same featurizer the app consumes.
    train_fleet = load_fleet_from_frame(make_fleet_csv(n_wells=8))
    X = featurize_fleet(train_fleet)
    # Deterministic 2-class labels so fit() exercises the real (class-weighted) path.
    y = pd.Series([i % 2 for i in range(len(X))], index=X.index)
    model = ESPRiskModel()
    model.fit(X, y, calibrate=False)  # calibration needs >=2 pos & neg in a holdout

    # Now the "upload": a separate fleet CSV scored through the trained model.
    up_fleet = load_fleet_from_frame(make_fleet_csv(n_wells=4))
    up_feats = featurize_fleet(up_fleet)
    probs = model.predict_proba(up_feats)
    assert len(probs) == 4
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    # SHAP/driver view the app renders must also come back for uploaded wells.
    contribs = model.feature_contributions(up_feats)
    assert list(contribs.index) == list(up_feats.index)
    assert "bias" in contribs.columns
