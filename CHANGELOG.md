# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.2] — 2026-06-11

### Added
- **Oracle / Bayes-optimal ceiling** (`src/oracle.py`) — because the synthetic generator's
  label process is known, we compute the best AUROC / precision@top-10% / Brier *any* model
  could attain given the ~5% irreducible label noise, and report the model **against** that
  ceiling. Result: model OOF AUROC ≈ 0.85 vs ceiling ≈ 0.85 → the model captures ~100% of the
  attainable above-chance signal, i.e. it sits at the noise floor (not a defect). Surfaced in
  the training console, `artifacts/training_report.json` (`oracle_ceiling` + `signal_capture`,
  for CI), and a new 📐 *Oracle Ceiling* panel + header chip in the app.
- **Genuine survival / time-to-event model** (`src/survival_model.py`) — a trained
  **discrete-time logistic hazard** (person-period; Singer & Willett 2003 / Cox 1972 lineage)
  fit on real **run-life ground truth**, evaluated **out-of-fold** with proper survival metrics:
  time-dependent **C-index ≈ 0.86** and **Integrated Brier Score ≈ 0.070** (beats a Kaplan–Meier
  baseline of 0.081 by ~13%). Implemented with numpy/sklearn — no new runtime dependency.
  `python -m src.survival_model` writes `artifacts/survival_report.json`; the training run also
  reports it.
- The generator now emits **run-life ground truth** — `time_to_event_days` and `event_observed`
  (right-censored healthy wells included) in `labels.csv` — drawn from an *independent* RNG so the
  SCADA channels (and thus the classifier data + oracle ceiling) stay byte-identical.

### Changed
- **Survival/RUL is now a real model, not a projection.** The app's per-well survival curve and
  fleet RUL ranking are powered by the trained discrete-time hazard model (curve *shape* learned
  from data), with C-index/IBS shown inline. This corrects the earlier framing where "survival/RUL"
  was a constant-hazard transform of the 30-day probability; that transform (`src/survival.py`) is
  retained only as a clearly-labeled fallback. README, app text, and the `survival` citation updated
  so the claim matches the code.
- Roadmap: the "survival / time-to-failure (run-life) model" item is delivered (was a v0.6 TODO).

## [0.7.1] — 2026-06-07
### Changed
- **Light theme** — suite-wide migration from dark/navy to a professional light palette (white surfaces, `plotly_white` charts, navy/blue accents retained); transparent fixed header so the title never clips. `runtime.txt` pinned to Python 3.11.

## [0.7.0] — 2026-06-06

### Added
- **Fleet explorer (multipage)** — a Fleet Overview (fleet KPIs; a **sortable per-well table**
  with lift / lateral / basin·formation from the shared registry, 30-day failure risk, suspected
  mode, median RUL, latest BFPD/intake/amps; plus the reliability curve, decision-economics
  threshold, drift/PSI, and the fleet RUL ranking) and a **drill-down page per well**
  (`st.navigation`) with its risk, SHAP contribution bar, SCADA chart, survival/RUL curve, and
  the BYOK AI explanation. No model/calibration/eval changes — UI only.

## [0.6.0] — 2026-06-06

### Added
- **Unified dark + navy suite theme** and a **cross-app sidebar suite navigator** —
  consistent look and one-click navigation across the production-engineering app suite.
- **Survival / remaining-useful-life (RUL) modeling**: a per-well **time-to-failure
  (survival) curve** plus a **fleet RUL ranking** (soonest-failure first), tied to the
  decision-economics alert threshold so the ranking reflects the economic intervention point.
- **Per-well SHAP contribution bar** — red bars raise risk, green bars lower it.
- **Real-data adapter path** (Texas RRC / NDIC / Volve schema mapping). *Honest:* the demo
  still runs on synthetic data with known ground truth; no real-data metrics are claimed.
- **Shared fleet registry**: Permian field/formation identity stays consistent across the suite.

### Changed
- Swept the deprecated `use_container_width` argument (→ `width="stretch"`); requires
  **streamlit>=1.50**.

## [0.5.0] — 2026-06-03

### Added
- **Two physically-real SCADA channels**: `drive_freq_hz` (VSD output frequency)
  and `current_imbalance_pct` (3-phase motor current imbalance) — the first signals
  an ESP analyst pulls up, and diagnostic of failure modes the 5-channel schema
  couldn't express. Optional in the loader (healthy defaults backfill old exports).
- **Two new failure modes** in the generator: **gas lock** (pump-off cycling —
  flow crashes intermittently, runtime cycles, drive frequency ramps) and
  **electrical / motor short** (current imbalance climbs). Five modes total.
- **Deterministic failure-mode classifier** (`classify_failure_mode`) that grounds
  the LLM rationale (scale · gas interference · gas lock · downthrust · electrical),
  shown in the dashboard and digest. Detection stays deterministic; the LLM narrates.
- **Alert-system metrics from out-of-fold predictions**: precision@k / recall@k now
  computed across the whole fleet (was a ~3-well test slice), plus a **reliability
  diagram** and **Brier score** in the dashboard.
- New features: `current_imbalance_last7_mean`, `current_imbalance_max_30d`,
  `high_imbalance_days_30d`, `drive_freq_last7_mean`, `drive_freq_slope_30d`.
- `failure_mode` tag in `labels.csv`; model-artifact SHA-256 recorded in the registry.

### Fixed
- **Platt calibration was silently disabled on scikit-learn ≥1.6** (`cv='prefit'`
  was removed in 1.8, raising into the guarded fallback). Now uses `FrozenEstimator`
  with a legacy `cv='prefit'` fallback — calibration actually runs again.
- **SHAP ↔ calibration mismatch**: `feature_contributions` decomposed the raw booster
  while `predict_proba` returned a *separately-trained* calibrated model. The
  calibrator now wraps the same booster Tree SHAP explains, so drivers and the shown
  probability reconcile (verified: Spearman(raw margin, calibrated p) = 1.00).
- **Shipped model ↔ reported metric decoupling**: both now use the same procedure;
  metrics are OOF, and a training-time score distribution is stored for honest
  **PSI drift** (was comparing two halves of the same live scores).
- **Per-day slopes** use actual elapsed days, not the sample index — correct on real
  historian data with gaps.
- `explain_well` raises a typed `MissingAPIKey` instead of a bare `KeyError`; the
  dashboard and ranker degrade gracefully to the deterministic diagnosis with no key.
- Committed `artifacts/` now ship the **realistic** model (AUROC ≈ 0.85 OOF, calibrated)
  out-of-the-box — no local retrain required, no more AUROC = 1.0 stand-in.
- Version strings aligned to 0.5.0 (`pyproject.toml`, `__init__.py`).

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
