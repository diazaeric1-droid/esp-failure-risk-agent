---
title: ESP Failure-Risk Agent
emoji: ⚙️
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: 1.50.0
app_file: demo/app.py
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

| Metric | Value | What it means |
|---|---|---|
| AUROC (OOF CV, mean ± std) | **≈ 0.85 ± 0.17** | ranking quality on overlapping, noisy classes |
| Precision @ top-10% | **≈ 0.90** | of the 10 wells you'd work this week, ~9 really fail |
| Recall @ top-10% | **≈ 0.53** | fraction of all failures caught in that top-10% alert list |
| Brier score (OOF) | **≈ 0.10** | probability calibration (lower is better) |

The synthetic generator deliberately varies failure onset/severity, adds sub-threshold degradation to ~25% of healthy wells, and injects ~5% label noise, so the classes genuinely overlap — **treat any near-1.0 AUROC as a red flag, not a win.** Regenerate any time with `python data/synthetic/generate.py && python -m src.train`.

### Validation methodology (read before quoting a number)

This is a **cross-sectional snapshot**: one engineered feature row per well at a fixed observation date, labelled "failed within the next 30 days." There is no within-well time ordering across rows, so stratified K-fold (with OOF metrics) is the honest protocol. The natural next step — once the pipeline ingests *rolling* observation windows per well — is **forward-chaining / grouped-by-well cross-validation** so a well's adjacent windows can't straddle train and validation and leak future information. That's the right answer to "is this time-series-safe?", and it's the v0.6 item below.

## Roadmap

- [x] v0.1 — XGBoost baseline + Claude explanations + daily digest
- [x] v0.2 — Streamlit dashboard
- [x] v0.3 — Class weighting, Platt calibration, stratified K-fold CV, realistic (overlapping + noisy) synthetic data
- [x] v0.4 — Decision economics (EV-optimal alert threshold), model registry, input-range + PSI drift monitoring
- [x] v0.5 — Drive-frequency + current-imbalance channels, gas-lock & electrical failure modes, deterministic failure-mode classifier, OOF precision@k / recall@k, reliability curve + Brier, SHAP↔calibration reconciled
- [ ] v0.6 — Rolling windows per well + **forward-chaining / grouped CV**; survival / time-to-failure (run-life) model
- [ ] v0.7 — Real-time scoring pipeline (polling SCADA historian) + per-well nameplate-aware thresholds

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
