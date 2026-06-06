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

import theme
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
    from src import survival as _survival
except Exception:
    _survival = None
try:
    from src import registry as _registry
except Exception:
    _registry = None


theme.setup_page("ESP Failure-Risk Agent", icon="⚙️")
theme.suite_nav("esp")

theme.header(
    "ESP Failure-Risk Agent",
    subtitle="30-day failure probability + plain-English explanations. Built by an ex-OXY / ex-Shell Staff Production Engineer.",
    chips=[(f"v{APP_VERSION}", "ver"), ("OOF AUROC ≈0.85", "eval")],
)

with st.expander(f"🆕 What's new in v{APP_VERSION}"):
    st.markdown(
        """
- **Unified dark + navy suite theme** + a **cross-app sidebar suite navigator** — one consistent
  look and one-click navigation across the production-engineering app suite.
- **Survival / remaining-useful-life modeling** — a per-well **time-to-failure (survival) curve**
  plus a **fleet RUL ranking** (soonest-failure first), tied to the decision-economics threshold.
- **Per-well SHAP contribution bar** — red bars raise risk, green bars lower it.
- **Real-data adapter path** (Texas RRC / NDIC / Volve schema mapping). *Honest:* the demo still
  runs on synthetic data with known ground truth — no real-data metrics are claimed.
- **Shared fleet registry** — Permian field/formation identity stays consistent across the suite.
- Swept the deprecated `use_container_width` (→ `width="stretch"`); now requires streamlit>=1.50.
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
    byok_key = st.text_input(
        "🔑 Anthropic API key (optional)", type="password",
        help="Bring your own key — used only for this session, never stored. Powers the AI "
             "explanation. Get one at console.anthropic.com. Everything else works without it.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Fleet ranking")
    top = probs.head(show_top).rename("Risk").to_frame()
    top["Risk"] = top["Risk"].apply(lambda p: f"{p:.0%}")
    st.dataframe(top, width="stretch")

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

    # Time-series plot of the well (suite colorway handles the multi-series colors)
    scada = fleet[selected]
    fig = go.Figure()
    for col in ("bfpd", "intake_pressure_psi", "motor_temp_f", "motor_amps",
                "drive_freq_hz", "current_imbalance_pct"):
        if col in scada.columns:
            fig.add_trace(go.Scatter(x=scada["date"], y=scada[col], name=col))
    st.plotly_chart(theme.style_fig(fig, height=350), width="stretch")

    drivers = top_drivers(contribs.loc[selected], k=8)
    st.subheader("Top drivers")
    drv_df = pd.DataFrame(drivers, columns=["Feature", "Contribution"])
    drv_df["Current value"] = drv_df["Feature"].map(feat_row)
    st.dataframe(drv_df, width="stretch")
    st.caption("Contributions are Tree SHAP in log-odds space on the raw booster; "
               "the calibrated probability above is a monotone transform of that score, "
               "so driver sign & rank carry over.")

    # Signed per-feature SHAP contributions for the selected well (red = raises
    # risk, green = lowers it), sorted by |contribution|. Same Tree SHAP values
    # as the driver table — drivers already comes back signed and ranked by |x|.
    shap_feats = [f for f, _ in drivers][::-1]      # smallest |x| at top → largest at bottom
    shap_vals = [c for _, c in drivers][::-1]
    bar_colors = [theme.RED if v >= 0 else theme.GREEN for v in shap_vals]
    sfig = go.Figure(go.Bar(
        x=shap_vals, y=shap_feats, orientation="h",
        marker_color=bar_colors,
        hovertemplate="%{y}: %{x:+.2f} log-odds<extra></extra>",
    ))
    sfig.update_layout(title="SHAP contributions (log-odds)",
                       xaxis_title="← lowers risk   ·   raises risk →")
    st.plotly_chart(theme.style_fig(sfig, height=320, legend=False), width="stretch")

    if explain_selected:
        try:
            client = None
            if byok_key:
                from anthropic import Anthropic
                client = Anthropic(api_key=byok_key)
            with st.spinner("Generating explanation..."):
                explanation = explain_well(
                    well_id=selected,
                    risk_score=risk,
                    feature_values=feat_row,
                    top_drivers=drivers,
                    suspected_mode=suspected_mode,
                    client=client,
                )
            st.info(explanation)
        except MissingAPIKey:
            st.warning("Enter your **Anthropic API key** in the sidebar to generate the AI rationale. "
                       "The risk score, drivers, and suspected failure mode above need no key.")


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
        cfig.add_vline(x=rec.recommended_threshold, line_dash="dash", line_color=theme.GREEN)
        cfig.update_layout(xaxis_title="Alert threshold", yaxis_title="Expected savings ($)")
        st.plotly_chart(theme.style_fig(cfig, height=300), width="stretch")
    except Exception as e:  # never let the economics panel break the app
        st.caption(f"Decision-economics panel unavailable: {e}")


# ── Time-to-failure (RUL / survival projection) ────────────────────────────
# Turn the calibrated 30-day failure probability into a forward survival curve
# under a constant-hazard-within-window assumption (NOT a trained time-to-event
# model — see src/survival.py). Median RUL = day S(t) crosses 50%.
if _survival is not None:
    st.divider()
    st.subheader("⏳ Time-to-failure — projected survival & remaining-useful-life")
    st.caption(
        "RUL is **model-projected on synthetic data** under a constant-hazard "
        "assumption (per-day hazard h = 1 − (1 − p₃₀)^(1/30)); it is a projection of "
        "the existing calibrated probability, not a trained time-to-event model. A "
        "real-data adapter (`src/real_data.py`, Volve/NDIC) is wired, but the demo "
        "runs the synthetic generator.")

    HORIZON = 180
    try:
        # (a) Selected-well survival curve with 50% line + median-RUL marker.
        p30_sel = float(probs[selected])
        days, surv = _survival.survival_curve(p30_sel, horizon_days=HORIZON)
        med_rul = _survival.expected_rul(p30_sel, horizon_days=HORIZON)
        med_num = _survival.median_rul_days(p30_sel, horizon_days=HORIZON)

        tcol1, tcol2 = st.columns([2, 1])
        with tcol1:
            sv_fig = go.Figure()
            sv_fig.add_trace(go.Scatter(
                x=days, y=surv, mode="lines", name="S(t) survival",
                line=dict(color=theme.BLUE, width=3),
                hovertemplate="day %{x}: S=%{y:.0%}<extra></extra>"))
            sv_fig.add_hline(y=0.5, line_dash="dot", line_color=theme.GREY,
                             annotation_text="50%", annotation_position="right")
            if isinstance(med_rul, int):
                sv_fig.add_vline(x=med_rul, line_dash="dash", line_color=theme.RED,
                                 annotation_text=f"median RUL ≈ {med_rul}d",
                                 annotation_position="top")
            sv_fig.update_layout(
                title=f"Projected survival — {selected}",
                xaxis_title="days from today", yaxis_title="P(survives past day t)",
                yaxis_range=[0, 1.02], xaxis_range=[0, HORIZON])
            st.plotly_chart(theme.style_fig(sv_fig, height=340), width="stretch")
        with tcol2:
            rul_label = med_rul if isinstance(med_rul, str) else f"{med_rul} days"
            st.metric(f"Median RUL — {selected}", rul_label)
            st.caption(f"30-day failure probability p₃₀ = {p30_sel:.0%}. "
                       "Median RUL = day projected survival crosses 50%.")

        # (b) Fleet RUL ranking — soonest failure first, colored RED→GREEN.
        rul_df = _survival.fleet_rul(probs, horizon_days=HORIZON)
        med_fleet = float(rul_df["median_rul_days"].median())
        st.metric("Median fleet RUL", f"{med_fleet:.0f} days")

        top_rul = rul_df.head(12).iloc[::-1]   # bottom-up so soonest is on top
        rmin, rmax = top_rul["median_rul_days"].min(), top_rul["median_rul_days"].max()
        span = max(rmax - rmin, 1e-9)
        def _urgency_color(v):
            # soonest (small RUL) -> RED, later -> GREEN
            frac = (v - rmin) / span
            return theme.RED if frac < 0.34 else (theme.AMBER if frac < 0.67 else theme.GREEN)
        bar_colors = [_urgency_color(v) for v in top_rul["median_rul_days"]]
        rul_fig = go.Figure(go.Bar(
            x=top_rul["median_rul_days"], y=top_rul["well_id"], orientation="h",
            marker_color=bar_colors,
            hovertemplate="%{y}: median RUL %{x:.0f}d<extra></extra>"))
        rul_fig.update_layout(
            title="Fleet RUL ranking (soonest projected failure first)",
            xaxis_title="median remaining-useful-life (days)", yaxis_title="")
        st.plotly_chart(theme.style_fig(rul_fig, height=380, legend=False),
                        width="stretch")

        # (c) Tie to decision economics: wells projected to fail within the quarter.
        QUARTER = 90
        within_q = rul_df[rul_df["median_rul_days"] <= QUARTER]
        n_q = int(len(within_q))
        # Reuse the economics-panel failure cost if the user set one; else default.
        try:
            fc = float(failure_cost)            # set by the decision-economics panel
        except NameError:
            fc = float(_economics.DEFAULT_FAILURE_COST) if _economics is not None else 350_000.0
        addressable = n_q * fc
        st.info(
            f"**{n_q}** well(s) projected to fail within the quarter (median RUL ≤ {QUARTER}d) "
            f"— **${addressable:,.0f}** addressable failure cost at "
            f"${fc:,.0f}/well.")
    except Exception as e:  # never let the RUL panel break the app
        st.caption(f"Time-to-failure panel unavailable: {e}")


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
                              line=dict(dash="dash", color=theme.GREY), name="perfectly calibrated"))
    rfig.add_trace(go.Scatter(x=rel_df["mean_pred"], y=rel_df["obs_freq"],
                              mode="markers+lines", name="model",
                              marker=dict(size=rel_df["count"].clip(6, 24))))
    rfig.update_layout(xaxis_title="Mean predicted probability",
                       yaxis_title="Observed failure frequency",
                       xaxis_range=[0, 1], yaxis_range=[0, 1])
    st.plotly_chart(theme.style_fig(rfig, height=320), width="stretch")
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
                    width="stretch")
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
