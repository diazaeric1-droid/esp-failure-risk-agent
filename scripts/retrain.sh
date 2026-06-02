#!/usr/bin/env bash
#
# retrain.sh — regenerate realistic data + retrain the calibrated, class-weighted model.
#
# This refreshes the committed artifacts (artifacts/esp_risk_model.joblib,
# artifacts/training_report.json) which currently ship from the OLD separable
# generator (AUROC 1.0) so the live demo runs out-of-the-box. Running this:
#
#   1. Regenerates synthetic SCADA with overlapping failure signatures + ~5%
#      label noise, so the classes genuinely overlap (no more AUROC=1.0).
#   2. Retrains XGBoost with scale_pos_weight (class imbalance), guarded Platt
#      probability calibration, and reports stratified K-fold CV (mean ± std) —
#      the honest metric on this small, imbalanced dataset.
#
# Requires the ML extras:  pip install -e ".[ml]"
#
set -euo pipefail

# Run from the repository root regardless of where the script is invoked from.
cd "$(dirname "$0")/.."

echo "==> [1/2] Regenerating realistic synthetic SCADA (overlap + ~5% label noise)…"
python data/synthetic/generate.py

echo "==> [2/2] Retraining class-weighted, calibrated XGBoost (+ K-fold CV metrics)…"
python -m src.train

echo "==> Done. Refreshed artifacts/esp_risk_model.joblib and artifacts/training_report.json."
echo "    Inspect AUROC (K-fold CV) in the training output above — treat ~1.0 as a red flag."
