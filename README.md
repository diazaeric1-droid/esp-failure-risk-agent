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

The ML model produces a calibrated probability + feature contributions. Claude turns those contributions into the natural-language story a production engineer would write.

## Model performance

Synthetic eval (held-out 30%):
- **AUROC:** 0.89
- **Precision @ top-10:** 0.80
- **Recall @ top-10:** 0.65

On real operator data, expect lower numbers — synthetic data is cleaner than reality. Use this as a baseline to beat.

## Roadmap

- [x] v0.1 — XGBoost baseline + Claude explanations + daily digest
- [x] v0.2 — Streamlit dashboard
- [ ] v0.3 — Calibration and threshold tuning per asset
- [ ] v0.4 — SHAP-based feature contributions instead of permutation importance
- [ ] v0.5 — Real-time scoring pipeline (Kafka or polling SCADA historian)
- [ ] v0.6 — Adjacent failure modes: rod pump parted-rod, gas lift loading

## License

MIT.

## Contact

Eric Diaz II — [LinkedIn](https://www.linkedin.com/in/eric-a-diaz2) — diaz.a.eric1@gmail.com

Available for senior AI/ML engineering roles and ESP-focused consulting engagements with E&P operators.
