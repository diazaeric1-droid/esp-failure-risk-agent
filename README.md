# ESP Failure Risk Agent

> An open-source ML + LLM system that ranks ESP wells by 30-day failure risk and writes a plain-English explanation for each.

Built by a Staff Production Engineer (ex-OXY, ex-Shell) who spent 9 years troubleshooting ESPs by hand.

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://esp-failure-risk.streamlit.app)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)

**Try it now → [esp-failure-risk.streamlit.app](https://esp-failure-risk.streamlit.app)**

---

## What it does

Drop in 60 days of SCADA-style readings for a fleet of ESP wells (bfpd, intake pressure, motor temp, motor amps, runtime %). The system:

1. **Engineers features** — rolling means, slopes, anomaly counts, ratios — from the raw time series
2. **Scores each well** with a gradient-boosted classifier trained to predict failure in the next 30 days
3. **Explains the top drivers** for each high-risk well in plain English, using Claude over the model's feature contributions
4. **Outputs a daily digest** ranking wells by risk with one-paragraph rationales — ready to drop in a production engineer's morning email

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
SCADA CSV ──► features.py ──► model.py (XGBoost) ──► risk score
                                      │
                                      ▼
                              explainer.py (Claude)
                                      │
                                      ▼
                              ranker.py → daily digest
```

The ML model produces a 30-day failure probability + feature contributions. Claude turns those contributions into the natural-language story a production engineer would write. Probabilities are calibrated (Platt / sigmoid on a held-out slice) when the positive count allows, with a guarded fallback to raw XGBoost outputs on very small samples.

## Model performance

The training pipeline reports **two** AUROC numbers: a single held-out split (high-variance on this small, 12%-positive dataset) and a **stratified K-fold CV mean ± std**, which is the number to trust. The model uses `scale_pos_weight` for class imbalance.

**Regenerate the metrics** (the synthetic generator was hardened — see below — so a fresh run reflects the realistic data):

```bash
python data/synthetic/generate.py   # overlapping signatures + ~5% label noise
python -m src.train                  # class-weighted XGBoost, calibration, K-fold CV
```

**What to expect:** the generator now varies failure onset/severity, adds sub-threshold degradation to ~25% of healthy wells, and injects ~5% label noise, so the classes genuinely overlap. Expect **AUROC ≈ 0.75–0.90** (CV mean), not the perfect score the earlier separable generator produced. Treat any near-1.0 number as a red flag, not a win.

> The committed `artifacts/` (`esp_risk_model.joblib`, `training_report.json`) are from the **pre-refactor** separable generator and read AUROC 1.0 — they're kept so the live demo runs out-of-the-box. Run the two commands above to refresh them with the realistic pipeline.

## Roadmap

- [x] v0.1 — XGBoost baseline + Claude explanations + daily digest
- [x] v0.2 — Streamlit dashboard
- [x] v0.3 — Class weighting, Platt calibration, stratified K-fold CV, realistic (overlapping + noisy) synthetic data
- [ ] v0.4 — Full `shap` library integration + global summary plots (current: XGBoost `pred_contribs` / Tree SHAP values)
- [ ] v0.5 — Real-time scoring pipeline (Kafka or polling SCADA historian)
- [ ] v0.6 — Adjacent failure modes: rod pump parted-rod, gas lift loading

## License

MIT.

## Contact

Eric Diaz II — [LinkedIn](https://www.linkedin.com/in/eric-a-diaz2) — diaz.a.eric1@gmail.com

Available for senior AI/ML engineering roles and ESP-focused consulting engagements with E&P operators.
