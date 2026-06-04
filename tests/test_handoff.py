"""Pipeline stage-2 handoff: ESP scores a well → AFE-ready WellDiagnosis."""
from pathlib import Path

import pytest

from src.handoff import DIAGNOSIS_SCHEMA, _map_mode, diagnose

REPO = Path(__file__).resolve().parent.parent
WELL = REPO / "data" / "synthetic" / "well_013.csv"
MODEL = REPO / "artifacts" / "esp_risk_model.joblib"


def test_mode_mapping_covers_the_taxonomy():
    assert _map_mode("Scale / abrasive buildup")[0] == "scale_treatment"
    assert _map_mode("Gas interference — intake pressure collapse")[0] == "gas_lift_optimization"
    assert _map_mode("Electrical — current imbalance")[0] == "esp_swap"
    assert _map_mode("Unclear — multiple weak signals")[0] == "esp_swap"  # default


@pytest.mark.skipif(not (WELL.exists() and MODEL.exists()),
                    reason="needs generated data + trained artifact")
def test_diagnose_emits_valid_afe_diagnosis():
    diag = diagnose(WELL, deferred_bopd=0.0, baseline_bopd=220.0, model_path=MODEL)
    assert diag["schema"] == DIAGNOSIS_SCHEMA
    assert 0.0 <= diag["esp_risk_score"] <= 1.0
    assert diag["incremental_rate_bopd"] > 0
    assert 0 < diag["expected_uplift_decline_per_yr"] < 2
    # required AFE fields are all present and non-empty
    for k in ("well_id", "api_number", "field", "operator", "intervention", "primary_diagnosis"):
        assert isinstance(diag[k], str) and diag[k].strip()
