# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] — 2026-06-02

- Self-heal stale Streamlit bytecode cache at startup: purge `src/` `__pycache__`
  and evict cached `src` modules so newly-added functions reload from current source
  after a redeploy. Fixes the startup ImportError cascade seen after adding new
  symbols to existing modules (the app no longer needs a manual Reboot to pick them up).

## [0.4.0] — 2026-06-02

### Added
- **Class weighting** (`scale_pos_weight ≈ n_neg/n_pos`) + **Platt probability
  calibration** (sigmoid `CalibratedClassifierCV`, guarded so it falls back to
  raw probabilities on very small / single-class samples).
- **Stratified K-fold cross-validation** reporting AUROC mean ± std — the honest
  metric on a small, imbalanced dataset (the single held-out split is high
  variance and no longer reported alone).
- **Realistic synthetic data**: overlapping failure signatures (varying onset &
  severity), sub-threshold degradation in ~25% of healthy wells, and ~5% label
  noise — so the classes genuinely overlap and AUROC is no longer 1.0.
- **Decision economics** (`src/economics.py`): expected-value-optimal alert
  threshold that minimises expected fleet cost (failure cost vs. intervention
  cost), with the resulting expected $ savings surfaced in the dashboard.
- **Model registry + monitoring** (`src/registry.py`): versioned metric registry,
  input-range validation of incoming features, and score-drift detection via the
  Population Stability Index (PSI).
- **Experimental sequence model** (`src/sequence_model.py`): a small Temporal-CNN
  baseline-vs-sequence comparison. Opt-in only — `torch` is an optional import and
  the module is never loaded on the deployed path.
- `scripts/retrain.sh`: one command to regenerate realistic data and retrain the
  class-weighted, calibrated model with K-fold metrics.

### Changed
- Corrected metric naming throughout to **top-10%** (was the ambiguous "top-10").
- Accurate wording on calibration (Platt/sigmoid, guarded) and SHAP (XGBoost
  `pred_contribs` / Tree SHAP values, not the full `shap` library).

## [0.3.0]

### Added
- Class weighting, Platt calibration, and stratified K-fold CV groundwork in the
  model wrapper; hardened synthetic generator (overlapping + noisy classes).

## [0.2.0]

### Added
- Streamlit dashboard (`demo/app.py`): fleet ranking, per-well time series, top
  driver contributions, and on-demand Claude explanations.
