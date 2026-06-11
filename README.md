---
title: ESP Failure-Risk Agent
emoji: ⚙️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: true
license: mit
---

# ESP Failure Risk Agent

> An open-source ML + LLM system that ranks ESP wells by 30-day failure risk and writes a plain-English explanation for each.

Built by a Staff Production Engineer (ex-OXY, ex-Shell) who spent 9 years troubleshooting ESPs by hand.

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://esp-failure-risk.streamlit.app)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)

**Try it now → [esp-failure-risk.streamlit.app](https://esp-failure-risk.streamlit.app)**

---

## What it does

Drop in 60 days of SCADA-style readings for a fleet of ESP wells (bfpd, intake pressure, motor temp, motor amps, runtime %, **drive frequency**, **3-phase current imbalance**). The system:

1. **Engineers features** — rolling means, per-day slopes, anomaly counts, ratios, and the electrical/VSD signals an ESP analyst checks first (current-imbalance peak, drive-frequency trend) — from the raw time series
2. **Scores each well** with a gradient-boosted classifier trained to predict failure in the next 30 days, with Platt-calibrated probabilities
3. **Classifies the suspected failure mode deterministically** (scale · gas interference · gas lock · downthrust · electrical/short) — then has Claude narrate the rationale *for that mode*, so the LLM can't invent a diagnosis the data doesn't support
4. **Explains the top drivers** for each high-risk well in plain English via Claude over the model's Tree SHAP contributions
5. **Outputs a daily digest** ranking wells by risk with suspected mode + one-paragraph rationale — ready to drop in a production engineer's morning email

This is the project that answers the question every digital-team interviewer asks: *"can you actually build and ship an ML system end-to-end?"*

## Why this matters

ESP failures cost an operator $250k–$500k each (workover + deferred production). Most operators react after the failure happens because their engineers can't watch 200+ wells daily. This system turns that into a 30-second morning scan.

## Quick start

```bash
git clone https://github.com/<your-user>/esp-failure-risk-agent
cd esp-failure-risk-agent

# Apple Silicon: install OpenMP runtime first (XGBoost dependency)
brew install libomp   # macOS only; Linux distros bundle OpenMP

pip install -e ".[ml]"
cp .env.example .env  # add ANTHROPIC_API_KEY

# Generate synthetic training data (100 wells, 60 days each, ~12% failure rate)
python data/synthetic/generate.py

# Train baseline XGBoost model
python -m src.train

# Score the fleet and produce a digest
python -m src.ranker --top 10

# Streamlit dashboard
streamlit run demo/app.py
```

## Architecture

```
SCADA CSV ──► features.py ──► model.py (XGBoost + Platt) ──► risk score
                                      │                            │
                                      ▼                            ▼
                         classify_failure_mode()          feature_contributions (Tree SHAP)
                              (deterministic)                       │
                                      └──────────┬─────────────────┘
                                                 ▼
                                       explainer.py (Claude narrates the mode)
                                                 ▼
                                       ranker.py → daily digest
```

The ML model produces a 30-day failure probability + Tree SHAP feature contributions. A small deterministic classifier maps the features to a suspected failure mode; Claude then narrates the rationale *for that mode*. Probabilities are Platt-calibrated (sigmoid) when the positive count allows, with a guarded fallback to raw XGBoost outputs on very small samples. The calibrator wraps the *same* booster that Tree SHAP explains, so the drivers and the displayed probability reconcile (the calibrated score is a monotone transform of the SHAP-decomposed margin).

## Model performance

All headline metrics come from **out-of-fold (OOF) predictions** — each well is scored by a stratified-K-fold model that never trained on it — so the number describes generalisation, not memorisation. The shipped artifact uses the *same* procedure (class-weighted XGBoost + Platt calibration), so the reported metric actually describes what's on disk.

Data and the trained artifact aren't committed (they're `.gitignore`d) — the app regenerates them deterministically (seed=7) on first run and trains automatically, so the demo shows the **realistic** model with no manual step. Because the generator now produces overlapping, noisy classes, there is **no AUROC = 1.0 stand-in** to trip over:

| Metric | Value | Oracle ceiling | What it means |
|---|---|---|---|
| AUROC (OOF CV, mean ± std) | **≈ 0.85 ± 0.17** | **0.85** | ranking quality on overlapping, noisy classes |
| Precision @ top-10% | **≈ 0.90** | 1.00 | of the 10 wells you'd work this week, ~9 really fail |
| Recall @ top-10% | **≈ 0.53** | — | fraction of all failures caught in that top-10% alert list |
| Brier score (OOF) | **≈ 0.10** | 0.05 | probability calibration (lower is better) |

The synthetic generator deliberately varies failure onset/severity, adds sub-threshold degradation to ~25% of healthy wells, and injects ~5% label noise, so the classes genuinely overlap — **treat any near-1.0 AUROC as a red flag, not a win.** Regenerate any time with `python data/synthetic/generate.py && python -m src.train`.

### Is ~0.85 AUROC "good"? — the oracle ceiling

Because the synthetic labels come from a **known** process, there's an information-theoretic ceiling on *any* model. The generator flips ~5% of labels at random (surprise failures / mislabels), and that flip is **independent of the features**, so no model can recover it. The Bayes-optimal ("oracle") predictor scores each well by its true-class probability `P(observed=1 | true class)`; grading those probabilities against the same noisy labels gives the attainable ceiling (`src/oracle.py`, surfaced in `artifacts/training_report.json` and the app's 📐 *Oracle Ceiling* panel):

> **Model OOF AUROC ≈ 0.85 vs oracle ceiling ≈ 0.85 → the model captures ~100% of the attainable above-chance signal.**

In other words, the realistic ~0.85 is the model sitting essentially **at the noise floor**, not below some ideal — the ~0.15 of "missing" AUROC is irreducible label noise, not a model defect. (This seed flips 5 healthy wells to "failed"; their features look healthy, so even a perfect ranker can't lift them above the truly-degrading wells.) The training run prints model-vs-ceiling and writes `oracle_ceiling` + `signal_capture` to the report so CI can assert the model stays near the ceiling rather than chasing an arbitrary AUROC floor.

### Survival / time-to-failure (a real trained model, not a projection)

The per-well **survival curve S(t)** and **remaining-useful-life (RUL)** come from a genuine **discrete-time logistic-hazard** time-to-event model (`src/survival_model.py`), trained on the synthetic *run-life* ground truth — each well's `time_to_event_days` + `event_observed` (right-censored healthy wells included), which the generator now emits. It is a person-period model (Singer & Willett 2003; Cox 1972 lineage) whose hazard **shape** is learned from data, evaluated **out-of-fold** with proper survival metrics:

| Survival metric (OOF) | Value | What it means |
|---|---|---|
| Time-dependent **C-index** | **≈ 0.86** | concordance: how often the model orders wells by failure time correctly (0.5 = chance) |
| **Integrated Brier Score** | **≈ 0.070** | time-integrated survival calibration (lower is better) |
| IBS — Kaplan–Meier baseline | 0.081 | covariate-free reference; the model beats it by ~13% |

Run it standalone with `python -m src.survival_model` (writes `artifacts/survival_report.json`). The earlier constant-hazard transform of `p30` (`src/survival.py`) is kept only as a clearly-labeled fallback when the trained model can't load.

### Validation methodology (read before quoting a number)

This is a **cross-sectional snapshot**: one engineered feature row per well at a fixed observation date, labelled "failed within the next 30 days." There is no within-well time ordering across rows, so stratified K-fold (with OOF metrics) is the honest protocol. The natural next step — once the pipeline ingests *rolling* observation windows per well — is **forward-chaining / grouped-by-well cross-validation** so a well's adjacent windows can't straddle train and validation and leak future information. That's the right answer to "is this time-series-safe?", and it's the v0.6 item below.

## Roadmap

- [x] v0.1 — XGBoost baseline + Claude explanations + daily digest
- [x] v0.2 — Streamlit dashboard
- [x] v0.3 — Class weighting, Platt calibration, stratified K-fold CV, realistic (overlapping + noisy) synthetic data
- [x] v0.4 — Decision economics (EV-optimal alert threshold), model registry, input-range + PSI drift monitoring
- [x] v0.5 — Drive-frequency + current-imbalance channels, gas-lock & electrical failure modes, deterministic failure-mode classifier, OOF precision@k / recall@k, reliability curve + Brier, SHAP↔calibration reconciled
- [x] v0.7 — **Oracle/Bayes ceiling** for honest metric framing (model vs attainable AUROC/precision/Brier); **genuine survival / time-to-failure model** (discrete-time logistic hazard on run-life ground truth, OOF C-index + Integrated Brier Score)
- [ ] v0.8 — Rolling windows per well + **forward-chaining / grouped CV**
- [ ] v0.9 — Real-time scoring pipeline (polling SCADA historian) + per-well nameplate-aware thresholds

## Part of a multi-agent pipeline

This is the **predict** stage of a detect → predict → authorize chain: the
[Daily Production Digest](../daily-production-digest) flags a pump-failure signature,
this agent scores the well's 30-day failure risk + classifies the mode, and the
[AFE Copilot](../afe-copilot) drafts the authorization. The well is handed over as a
JSON `WellAlert` and scored via `python -m src.handoff` (the ESP loader tolerates the
digest's SCADA schema). See [`../pe-pipeline/PIPELINE.md`](../pe-pipeline/PIPELINE.md) and run `python3 ../pe-pipeline/pe_chain.py`.

## License

MIT.

## Contact

Eric Diaz II — [LinkedIn](https://www.linkedin.com/in/eric-a-diaz2) — diaz.a.eric1@gmail.com

Available for senior AI/ML engineering roles and ESP-focused consulting engagements with E&P operators.
