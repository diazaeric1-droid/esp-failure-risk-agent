"""Streamlit dashboard for the ESP Failure Risk Agent."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `src.*` imports work on Streamlit Cloud
# (where the package isn't pip-installed, just the deps from requirements.txt).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- Self-heal stale bytecode / module cache (Streamlit Cloud) --------------
# Streamlit reuses the container across redeploys; a cached .pyc or already-imported
# OLD module can lack symbols added in a newer commit, surfacing as a startup
# ImportError for a name that exists in the source. Purge src/ bytecode + evict
# cached src modules so every submodule reloads from CURRENT source (no-op when clean).
import shutil as _shutil
for _pycache in (REPO_ROOT / "src").rglob("__pycache__"):
    _shutil.rmtree(_pycache, ignore_errors=True)
for _name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[_name]

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_fleet
from src.explainer import MissingAPIKey, classify_failure_mode, explain_well, top_drivers
from src.features import featurize_fleet
from src.model import ESPRiskModel

# App version + optional (numpy-only) modules. Guarded so a missing/renamed
# optional module can never crash the header on the live app.
try:
    from src import __version__ as APP_VERSION
except Exception:
    APP_VERSION = "0.5.0"
try:
    from src import economics as _economics
except Exception:
    _economics = None
try:
    from src import registry as _registry
except Exception:
    _registry = None


st.set_page_config(page_title="ESP Failure Risk Agent", page_icon="⚙️", layout="wide")

st.title(f"ESP Failure Risk Agent  `v{APP_VERSION}`")
st.caption("30-day failure probability + plain-English explanations. Built by an ex-OXY / ex-Shell Staff Production Engineer.")

with st.expander(f"🆕 What's new in v{APP_VERSION}"):
    st.markdown(
        """
- **Two new SCADA channels** — **drive frequency (Hz)** and **3-phase current imbalance (%)** —
  plus two new failure modes (**gas lock**, **electrical/motor short**) so the eval covers the
  failure taxonomy a reliability engineer actually triages.
- **Deterministic failure-mode classifier** grounds the LLM rationale (scale / gas interference /
  gas lock / downthrust / electrical) — detection stays deterministic, the LLM only narrates.
- **Alert-system metrics from out-of-fold predictions** — precision@k / recall@k computed across
  the whole fleet (not a 3-well slice) + a **reliability curve** and **Brier score**.
- **SHAP ↔ calibration reconciled**: the calibrator wraps the same booster Tree SHAP explains, so
  drivers and the displayed probability agree. (Also fixed Platt calibration on sklearn ≥1.6.)
- **Honest drift monitoring**: PSI compares live scores against the stored training distribution.
- Shipped model and the reported CV metric now use the same procedure (no decoupling).
        """
    )

DATA_DIR = REPO_ROOT / "data" / "synthetic"
MODEL_PATH = REPO_ROOT / "artifacts" / "esp_risk_model.joblib"


def _bootstrap_if_needed() -> None:
    """Generate synthetic data and train a baseline model on first run.

    The repo doesn't commit large data files or the trained artifact —
    they're regenerated deterministically (seed=7) on demand. ~30 sec total.
    """
    if not any(DATA_DIR.glob("well_*.csv")):
        with st.status("First-time setup: generating synthetic SCADA…", expanded=False):
            subprocess.run([sys.executable, str(REPO_ROOT / "data" / "synthetic" / "generate.py")], check=True)
    if not MODEL_PATH.exists():
        with st.status("First-time setup: training XGBoost baseline…", expanded=False):
            subprocess.run([sys.executable, "-m", "src.train"], check=True, cwd=REPO_ROOT)


_bootstrap_if_needed()


@st.cache_data
def load():
    fleet = load_fleet(DATA_DIR)
    features = featurize_fleet(fleet)
    return fleet, features


@st.cache_resource
def get_model():
    return ESPRiskModel.load(MODEL_PATH)


fleet, features = load()
model = get_model()
probs = pd.Series(model.predict_proba(features), index=features.index, name="risk").sort_values(ascending=False)
contribs = model.feature_contributions(features)

with st.sidebar:
    st.header("Filters")
    threshold = st.slider("Highlight risk above", 0.0, 1.0, 0.5, 0.05)
    show_top = st.number_input("Show top N wells", 5, 50, 10)
    explain_selected = st.checkbox("Generate AI explanation for selected well", value=False)

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Fleet ranking")
    top = probs.head(show_top).rename("Risk").to_frame()
    top["Risk"] = top["Risk"].apply(lambda p: f"{p:.0%}")
    st.dataframe(top, use_container_width=True)

    st.metric("High-risk wells (≥ threshold)", int((probs >= threshold).sum()))

with col2:
    selected = st.selectbox("Inspect well", probs.head(show_top).index.tolist())
    risk = float(probs[selected])
    st.metric(f"30-day failure probability — {selected}", f"{risk:.0%}")

    # Deterministic suspected failure mode (grounds the narration; always available).
    feat_row = features.loc[selected].to_dict()
    suspected_mode, mode_evidence = classify_failure_mode(feat_row)
    st.markdown(f"**Suspected failure mode:** {suspected_mode}")
    st.caption(mode_evidence)

    # Time-series plot of the well
    scada = fleet[selected]
    fig = go.Figure()
    for col, color in [("bfpd", "#1f77b4"), ("intake_pressure_psi", "#ff7f0e"),
                        ("motor_temp_f", "#d62728"), ("motor_amps", "#2ca02c"),
                        ("drive_freq_hz", "#9467bd"), ("current_imbalance_pct", "#8c564b")]:
        if col in scada.columns:
            fig.add_trace(go.Scatter(x=scada["date"], y=scada[col], name=col, line=dict(color=color)))
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    drivers = top_drivers(contribs.loc[selected], k=5)
    st.subheader("Top drivers")
    drv_df = pd.DataFrame(drivers, columns=["Feature", "Contribution"])
    drv_df["Current value"] = drv_df["Feature"].map(feat_row)
    st.dataframe(drv_df, use_container_width=True)
    st.caption("Contributions are Tree SHAP in log-odds space on the raw booster; "
               "the calibrated probability above is a monotone transform of that score, "
               "so driver sign & rank carry over.")

    if explain_selected:
        try:
            with st.spinner("Generating explanation..."):
                explanation = explain_well(
                    well_id=selected,
                    risk_score=risk,
                    feature_values=feat_row,
                    top_drivers=drivers,
                    suspected_mode=suspected_mode,
                )
            st.info(explanation)
        except MissingAPIKey:
            st.warning("Set `ANTHROPIC_API_KEY` to generate the AI rationale. "
                       "The risk score, drivers, and suspected failure mode above need no API key.")


# ── Decision economics ─────────────────────────────────────────────────────
# A risk score only matters if it drives a decision. Find the alert threshold
# that minimises expected fleet cost (failure cost vs. proactive intervention).
if _economics is not None:
    st.divider()
    st.subheader("💰 Decision economics — where should the alert fire?")
    ec1, ec2 = st.columns(2)
    with ec1:
        failure_cost = st.number_input(
            "Failure cost ($/well)", 50_000, 1_000_000,
            int(_economics.DEFAULT_FAILURE_COST), 10_000)
    with ec2:
        intervention_cost = st.number_input(
            "Intervention cost ($/well)", 5_000, 500_000,
            int(_economics.DEFAULT_INTERVENTION_COST), 5_000)

    try:
        rec = _economics.recommend_threshold(
            probs.values, failure_cost=float(failure_cost),
            intervention_cost=float(intervention_cost))
        m1, m2, m3 = st.columns(3)
        m1.metric("Recommended alert threshold", f"{rec.recommended_threshold:.0%}")
        m2.metric("Wells flagged at threshold", rec.n_wells_flagged)
        m3.metric("Expected fleet savings", f"${rec.expected_savings:,.0f}")
        st.caption(
            f"vs. a never-intervene baseline of ${rec.baseline_cost_no_action:,.0f} "
            f"expected cost. Break-even probability ≈ "
            f"{_economics.break_even_probability(float(failure_cost), float(intervention_cost)):.0%}.")

        curve_df = pd.DataFrame(rec.curve, columns=["threshold", "expected_savings"])
        cfig = go.Figure()
        cfig.add_trace(go.Scatter(x=curve_df["threshold"], y=curve_df["expected_savings"],
                                  mode="lines", name="Expected savings"))
        cfig.add_vline(x=rec.recommended_threshold, line_dash="dash", line_color="#2ca02c")
        cfig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0),
                           xaxis_title="Alert threshold", yaxis_title="Expected savings ($)")
        st.plotly_chart(cfig, use_container_width=True)
    except Exception as e:  # never let the economics panel break the app
        st.caption(f"Decision-economics panel unavailable: {e}")


# ── Model calibration (reliability diagram) ────────────────────────────────
# Prove the Platt calibration actually works: predicted vs observed failure
# frequency from out-of-fold predictions, plus the Brier score.
reliability = getattr(model, "reliability", None)
if reliability:
    st.divider()
    st.subheader("🎯 Calibration — do the probabilities mean what they say?")
    rel_df = pd.DataFrame(reliability)
    rfig = go.Figure()
    rfig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                              line=dict(dash="dash", color="#888"), name="perfectly calibrated"))
    rfig.add_trace(go.Scatter(x=rel_df["mean_pred"], y=rel_df["obs_freq"],
                              mode="markers+lines", name="model",
                              marker=dict(size=rel_df["count"].clip(6, 24))))
    rfig.update_layout(height=320, margin=dict(l=0, r=0, t=20, b=0),
                       xaxis_title="Mean predicted probability",
                       yaxis_title="Observed failure frequency",
                       xaxis_range=[0, 1], yaxis_range=[0, 1])
    st.plotly_chart(rfig, use_container_width=True)
    st.caption("Out-of-fold reliability diagram (marker size ∝ wells in bin). "
               "Points near the diagonal = well-calibrated probabilities.")


# ── Data quality & drift monitoring ────────────────────────────────────────
if _registry is not None:
    with st.expander("🛡️ Data quality & score drift"):
        try:
            violations = _registry.input_range_check(features)
            if violations:
                st.warning(f"{len(violations)} input-range violation(s) detected "
                           "(possible sensor faults / unit errors):")
                st.dataframe(
                    pd.DataFrame(
                        [(v.well_id, v.feature, v.value, v.low, v.high) for v in violations[:50]],
                        columns=["Well", "Feature", "Value", "Min", "Max"]),
                    use_container_width=True)
            else:
                st.success("Input-range check: all features within plausible operating ranges.")

            # Score drift: compare the LIVE fleet scores against the model's stored
            # TRAINING score distribution (the real reference), not two halves of
            # the same data. Falls back to the split stand-in only on old artifacts.
            live_scores = probs.values
            reference = getattr(model, "reference_scores", None)
            if reference is not None and len(reference) >= 4:
                drift = _registry.score_drift(reference, live_scores)
                ref_note = "vs. the stored training-score distribution"
            elif len(live_scores) >= 4:
                mid = len(live_scores) // 2
                drift = _registry.score_drift(live_scores[:mid], live_scores[mid:])
                ref_note = "split-half stand-in (older artifact has no stored reference)"
            else:
                drift = None
            if drift is not None:
                st.metric("Score-drift PSI", f"{drift.psi:.3f}",
                          delta=("DRIFT" if drift.drift else drift.label()),
                          delta_color=("inverse" if drift.drift else "off"))
                st.caption(f"PSI < 0.10 no shift · 0.10–0.25 moderate · > 0.25 major. {ref_note}.")
        except Exception as e:
            st.caption(f"Monitoring panel unavailable: {e}")
