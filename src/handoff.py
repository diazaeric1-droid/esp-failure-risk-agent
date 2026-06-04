"""Pipeline handoff (stage 2 of 3): score a well handed over by the Daily
Production Digest and emit an AFE-ready ``WellDiagnosis``.

    Daily Production Digest  ──WellAlert──▶  [ESP Failure-Risk Agent]  ──WellDiagnosis──▶  AFE Copilot

The SAME well's SCADA flows through: the ESP loader tolerates the digest's
schema (it backfills the v0.5 drive-frequency / current-imbalance channels with
healthy defaults), so a digest fleet CSV can be scored directly. We:

  1. featurize + score the 30-day failure probability,
  2. run the deterministic failure-mode classifier (detection stays deterministic),
  3. map the mode → a priced AFE intervention + an uplift estimate,
  4. emit a WellDiagnosis that the AFE Copilot's drafter / renderer consumes.

WellDiagnosis schema (``pe-pipeline/well-diagnosis/v1``) is a superset of the AFE
Copilot's ``AFEDiagnosis`` (extra keys like ``esp_risk_score`` are ignored by its
``from_pe_copilot`` loader), so the contract stays decoupled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data_loader import load_well_scada
from .explainer import classify_failure_mode
from .features import FEATURE_NAMES, featurize_well
from .model import ESPRiskModel

DIAGNOSIS_SCHEMA = "pe-pipeline/well-diagnosis/v1"
DEFAULT_MODEL = "artifacts/esp_risk_model.joblib"

# Map the deterministic failure mode → (priced AFE intervention, recovery fraction).
# The recovery fraction estimates the uplift the workover protects/restores, used
# only when the upstream alert didn't already quantify a deferred rate.
MODE_TO_INTERVENTION = [
    ("Scale", "scale_treatment", 0.18),
    ("Gas interference", "gas_lift_optimization", 0.15),
    ("Gas lock", "gas_lift_optimization", 0.15),
    ("Downthrust", "esp_swap", 0.20),
    ("Electrical", "esp_swap", 0.20),
]
DEFAULT_INTERVENTION = ("esp_swap", 0.15)


def _map_mode(mode: str) -> tuple[str, float]:
    for key, interv, frac in MODE_TO_INTERVENTION:
        if key in mode:
            return interv, frac
    return DEFAULT_INTERVENTION


def diagnose(scada_csv, well_id: str | None = None, deferred_bopd: float = 0.0,
             baseline_bopd: float = 0.0, model_path=DEFAULT_MODEL,
             field: str = "Synthetic Delaware Basin",
             operator: str = "Synthetic Operator LLC") -> dict:
    df = load_well_scada(scada_csv)
    well_id = well_id or Path(scada_csv).stem
    feats = featurize_well(df)
    X = pd.DataFrame([feats])[FEATURE_NAMES]
    model = ESPRiskModel.load(model_path)
    risk = float(model.predict_proba(X)[0])
    mode, evidence = classify_failure_mode(feats)
    interv, frac = _map_mode(mode)

    # Uplift = what the workover protects/restores: the upstream-quantified deferral
    # if present, else a mode-dependent fraction of the well's recent rate (floored).
    uplift = round(max(deferred_bopd, frac * baseline_bopd, 20.0), 1)

    return {
        "schema": DIAGNOSIS_SCHEMA,
        "well_id": well_id,
        "api_number": "TBD-ASSIGN",
        "field": field,
        "operator": operator,
        "intervention": interv,
        "primary_diagnosis": f"{mode}. ESP 30-day failure risk {risk:.0%}. {evidence}",
        "incremental_rate_bopd": uplift,
        "expected_uplift_decline_per_yr": 0.6,
        "requested_by": "ESP Failure-Risk Agent (auto)",
        # ESP-specific metadata (ignored by the AFE dataclass; useful for the chain log)
        "esp_risk_score": round(risk, 4),
        "suspected_mode": mode,
    }


def main():
    parser = argparse.ArgumentParser(description="Score a well and emit an AFE-ready WellDiagnosis.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--scada", help="Path to a single well's SCADA CSV")
    src.add_argument("--alerts", help="A WellAlert JSON (list) from the digest; the top alert is used")
    parser.add_argument("--well", default=None)
    parser.add_argument("--deferred-bopd", type=float, default=0.0)
    parser.add_argument("--baseline-bopd", type=float, default=0.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default=None, help="Write the diagnosis JSON here (default: stdout)")
    args = parser.parse_args()

    if args.alerts:
        alerts = json.loads(Path(args.alerts).read_text())
        if not alerts:
            raise SystemExit("No alerts to diagnose.")
        top = alerts[0]
        diag = diagnose(top["scada_csv"], well_id=top.get("well_id"),
                        deferred_bopd=top.get("deferred_bopd", 0.0),
                        baseline_bopd=top.get("baseline_bopd", 0.0), model_path=args.model)
    else:
        diag = diagnose(args.scada, well_id=args.well, deferred_bopd=args.deferred_bopd,
                        baseline_bopd=args.baseline_bopd, model_path=args.model)

    payload = json.dumps(diag, indent=2)
    if args.out:
        Path(args.out).write_text(payload)
        print(f"Wrote WellDiagnosis for {diag['well_id']} "
              f"(risk {diag['esp_risk_score']:.0%}, → {diag['intervention']}) to {args.out}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
