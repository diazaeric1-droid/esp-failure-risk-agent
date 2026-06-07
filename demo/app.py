"""Streamlit dashboard for the ESP Failure Risk Agent.

Multipage (``st.navigation`` + ``st.Page``): a Fleet Overview page (fleet KPIs, a
sortable per-well table, and the fleet-level analytics — decision economics,
reliability curve, drift/PSI, and the fleet RUL ranking) plus one drill-down page
per well (its risk metric, suspected failure mode + evidence, SCADA chart,
top-drivers table, SHAP contribution bar, survival/RUL curve, and the BYOK AI
explanation).

Detection / scoring stays deterministic; the per-well AI explanation is
BYOK-optional (everything else renders with no API key). The model, calibration,
SHAP, survival, and eval logic are untouched — this file only reorganizes the UI.
Heavy loads are cached on string args.
"""
from __future__ import annotations

import subprocess
import sys
from functools import partial
from pathlib import Path

# Ensure repo root is on sys.path so `src.*` imports work on Streamlit Cloud, and
# the demo dir so the vendored `theme` / `fleet_registry` resolve regardless of cwd
# (Streamlit adds the entrypoint dir at runtime; AppTest / other contexts may not).
DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
for _p in (str(REPO_ROOT), str(DEMO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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

import fleet_registry
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


DATA_DIR = REPO_ROOT / "data" / "synthetic"
MODEL_PATH = REPO_ROOT / "artifacts" / "esp_risk_model.joblib"
HORIZON = 180  # projection horizon (days) for the survival / RUL layer


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


# ---- cached heavy loads (string args so they hash/cache cleanly) -----------

@st.cache_data
def load():
    fleet = load_fleet(DATA_DIR)
    features = featurize_fleet(fleet)
    return fleet, features


@st.cache_resource
def get_model():
    return ESPRiskModel.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def _scored():
    """Cache the fleet scoring pass: probs (sorted desc) + per-well contributions."""
    _, features = load()
    model = get_model()
    probs = pd.Series(
        model.predict_proba(features), index=features.index, name="risk"
    ).sort_values(ascending=False)
    contribs = model.feature_contributions(features)
    return probs, contribs


# ---- shared helpers --------------------------------------------------------

def _back_to_overview() -> None:
    target = globals().get("overview")
    try:
        st.page_link(target if target is not None else "app.py",
                     label="← Back to Fleet overview", icon="📊")
    except Exception:
        pass


def _last_scada(scada: pd.DataFrame) -> dict:
    """Latest BFPD / intake / amps for the fleet table (deterministic, no scoring)."""
    if scada is None or not len(scada):
        return {"bfpd": float("nan"), "intake": float("nan"), "amps": float("nan")}
    last = scada.iloc[-1]
    g = lambda k: float(last[k]) if k in scada.columns and pd.notna(last[k]) else float("nan")
    return {"bfpd": g("bfpd"), "intake": g("intake_pressure_psi"), "amps": g("motor_amps")}


# =====================================================================
# PAGE: Fleet overview
# =====================================================================

def render_overview() -> None:
    theme.header(
        "ESP Failure-Risk Agent",
        subtitle="30-day failure probability + plain-English explanations. "
                 "Built by an ex-OXY / ex-Shell Staff Production Engineer.",
        chips=[(f"v{APP_VERSION}", "ver"), ("OOF AUROC ≈0.85", "eval"),
               ("fleet explorer", "info")],
    )

    with st.expander(f"🆕 What's new in v{APP_VERSION}"):
        st.markdown(
            """
- **Fleet explorer (multipage)** — a Fleet Overview plus a **drill-down page per well**
  (`st.navigation`): each well page carries its own risk metric, suspected failure mode,
  SCADA chart, top-drivers table, SHAP contribution bar, survival/RUL curve, and the BYOK
  AI explanation.
- **Sortable fleet table** — one row per well with lift, lateral, basin·formation (shared
  registry), 30-day failure risk %, suspected failure mode, median RUL days, and the latest
  BFPD / intake / amps.
- **Fleet-level analytics on the overview** — decision-economics threshold chart, the
  out-of-fold **reliability curve**, **score-drift / PSI** monitoring, and the **fleet RUL
  ranking** all live here; the threshold / top-N controls drive the overview views.
- Per-well SHAP contribution bar (red raises risk, green lowers it) + a per-well projected
  survival curve with median RUL — moved into each well's page.
- Survival / RUL stays a **model-derived projection** under a constant-hazard assumption
  (see `src/survival.py`), not a trained time-to-event model.
            """
        )

    fleet, features = load()
    probs, _ = _scored()

    # --- controls that drive the fleet views (overview-scoped) --------------
    with st.sidebar:
        st.header("Filters")
        threshold = st.slider("Highlight risk above", 0.0, 1.0, 0.5, 0.05)
        show_top = st.number_input("Show top N wells", 5, 50, 10)

    # --- fleet KPIs ----------------------------------------------------------
    rul_df = None
    med_fleet = float("nan")
    if _survival is not None:
        try:
            rul_df = _survival.fleet_rul(probs, horizon_days=HORIZON)
            med_fleet = float(rul_df["median_rul_days"].median())
        except Exception:
            rul_df = None

    st.subheader("Fleet snapshot")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Wells", int(len(probs)))
    k2.metric("High-risk wells (≥ threshold)", int((probs >= threshold).sum()))
    k3.metric("Median fleet risk", f"{float(probs.median()):.0%}")
    k4.metric("Median fleet RUL",
              f"{med_fleet:.0f} days" if med_fleet == med_fleet else "—")

    # --- sortable fleet table -----------------------------------------------
    st.subheader("Fleet table")
    st.caption("One row per well — sort any column. Open a well from the **Wells** "
               "section in the sidebar to drill in (risk, SHAP, survival, AI rationale).")
    rul_by_well = (dict(zip(rul_df["well_id"], rul_df["median_rul_days"]))
                   if rul_df is not None else {})
    rows = []
    for well_id in probs.index:
        meta = fleet_registry.get(well_id)
        feat_row = features.loc[well_id].to_dict()
        mode, _ = classify_failure_mode(feat_row)
        last = _last_scada(fleet.get(well_id))
        rows.append({
            "Well": well_id,
            "Lift": meta.lift,
            "Lateral (ft)": meta.lateral_length_ft,
            "Basin · Formation": f"{meta.basin} · {meta.formation}",
            "30-day risk %": round(float(probs[well_id]) * 100.0, 1),
            "Suspected failure mode": mode,
            "Median RUL (days)": rul_by_well.get(well_id, float("nan")),
            "Latest BFPD": round(last["bfpd"], 0),
            "Intake psi": round(last["intake"], 0),
            "Motor amps": round(last["amps"], 1),
        })
    table = pd.DataFrame(rows)
    st.dataframe(table, width="stretch", hide_index=True,
                 column_config={
                     "30-day risk %": st.column_config.NumberColumn(format="%.1f%%"),
                 })

    # --- fleet-level analytics ----------------------------------------------
    _economics_panel(probs, threshold)
    _survival_fleet_panel(rul_df, med_fleet)
    _reliability_panel()
    _drift_panel(features, probs)


def _economics_panel(probs: pd.Series, threshold: float) -> None:
    # A risk score only matters if it drives a decision. Find the alert threshold
    # that minimises expected fleet cost (failure cost vs. proactive intervention).
    if _economics is None:
        return
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


def _survival_fleet_panel(rul_df, med_fleet: float) -> None:
    # Fleet RUL ranking — soonest projected failure first. Per-well survival curves
    # live on each well's drill-down page.
    if _survival is None or rul_df is None:
        return
    st.divider()
    st.subheader("⏳ Fleet remaining-useful-life ranking")
    st.caption(
        "RUL is **model-projected on synthetic data** under a constant-hazard "
        "assumption (per-day hazard h = 1 − (1 − p₃₀)^(1/30)); it is a projection of "
        "the existing calibrated probability, not a trained time-to-event model. A "
        "real-data adapter (`src/real_data.py`, Volve/NDIC) is wired, but the demo "
        "runs the synthetic generator.")
    try:
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

        # Tie to decision economics: wells projected to fail within the quarter.
        QUARTER = 90
        within_q = rul_df[rul_df["median_rul_days"] <= QUARTER]
        n_q = int(len(within_q))
        fc = float(_economics.DEFAULT_FAILURE_COST) if _economics is not None else 350_000.0
        addressable = n_q * fc
        st.info(
            f"**{n_q}** well(s) projected to fail within the quarter (median RUL ≤ {QUARTER}d) "
            f"— **${addressable:,.0f}** addressable failure cost at "
            f"${fc:,.0f}/well.")
    except Exception as e:  # never let the RUL panel break the app
        st.caption(f"Fleet RUL panel unavailable: {e}")


def _reliability_panel() -> None:
    # Prove the Platt calibration actually works: predicted vs observed failure
    # frequency from out-of-fold predictions, plus the Brier score.
    model = get_model()
    reliability = getattr(model, "reliability", None)
    if not reliability:
        return
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


def _drift_panel(features: pd.DataFrame, probs: pd.Series) -> None:
    if _registry is None:
        return
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
            reference = getattr(get_model(), "reference_scores", None)
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


# =====================================================================
# PAGE: per-well drill-down
# =====================================================================

def render_well(well_id: str) -> None:
    fleet, features = load()
    probs, contribs = _scored()
    meta = fleet_registry.get(well_id)

    theme.header(
        f"{well_id} · {meta.name}",
        subtitle=f"{meta.lift} · {meta.basin} · {meta.formation} · {meta.area}",
        chips=[(f"v{APP_VERSION}", "ver"), (meta.peer_group, "info")],
    )
    _back_to_overview()

    if well_id not in features.index:
        st.warning("No featurized history for this well.")
        return

    risk = float(probs[well_id])
    st.metric(f"30-day failure probability — {well_id}", f"{risk:.0%}")

    # Deterministic suspected failure mode (grounds the narration; always available).
    feat_row = features.loc[well_id].to_dict()
    suspected_mode, mode_evidence = classify_failure_mode(feat_row)
    st.markdown(f"**Suspected failure mode:** {suspected_mode}")
    st.caption(mode_evidence)

    # Time-series plot of the well (suite colorway handles the multi-series colors)
    scada = fleet[well_id]
    fig = go.Figure()
    for col in ("bfpd", "intake_pressure_psi", "motor_temp_f", "motor_amps",
                "drive_freq_hz", "current_imbalance_pct"):
        if col in scada.columns:
            fig.add_trace(go.Scatter(x=scada["date"], y=scada[col], name=col))
    st.plotly_chart(theme.style_fig(fig, height=350), width="stretch")

    drivers = top_drivers(contribs.loc[well_id], k=8)
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

    # ── Time-to-failure (RUL / survival projection) for this well ───────────
    _well_survival(well_id, risk)

    # ── BYOK AI explanation (everything above needs no key) ─────────────────
    st.divider()
    st.subheader("🤖 AI rationale (BYOK-optional)")
    byok_key = st.text_input(
        "🔑 Anthropic API key (optional)", type="password", key=f"byok_{well_id}",
        help="Bring your own key — used only for this session, never stored. Powers the AI "
             "explanation. Get one at console.anthropic.com. Everything else works without it.")
    if st.button("Generate AI explanation", key=f"explain_{well_id}"):
        try:
            client = None
            if byok_key:
                from anthropic import Anthropic
                client = Anthropic(api_key=byok_key)
            with st.spinner("Generating explanation..."):
                explanation = explain_well(
                    well_id=well_id,
                    risk_score=risk,
                    feature_values=feat_row,
                    top_drivers=drivers,
                    suspected_mode=suspected_mode,
                    client=client,
                )
            st.info(explanation)
        except MissingAPIKey:
            st.warning("Enter your **Anthropic API key** above to generate the AI rationale. "
                       "The risk score, drivers, suspected failure mode, and survival curve "
                       "need no key.")

    _back_to_overview()


def _well_survival(well_id: str, risk: float) -> None:
    # Turn the calibrated 30-day failure probability into a forward survival curve
    # under a constant-hazard-within-window assumption (NOT a trained time-to-event
    # model — see src/survival.py). Median RUL = day S(t) crosses 50%.
    if _survival is None:
        return
    st.divider()
    st.subheader("⏳ Time-to-failure — projected survival & remaining-useful-life")
    st.caption(
        "RUL is **model-projected on synthetic data** under a constant-hazard "
        "assumption (per-day hazard h = 1 − (1 − p₃₀)^(1/30)); it is a projection of "
        "the existing calibrated probability, not a trained time-to-event model.")
    try:
        days, surv = _survival.survival_curve(risk, horizon_days=HORIZON)
        med_rul = _survival.expected_rul(risk, horizon_days=HORIZON)

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
                title=f"Projected survival — {well_id}",
                xaxis_title="days from today", yaxis_title="P(survives past day t)",
                yaxis_range=[0, 1.02], xaxis_range=[0, HORIZON])
            st.plotly_chart(theme.style_fig(sv_fig, height=340), width="stretch")
        with tcol2:
            rul_label = med_rul if isinstance(med_rul, str) else f"{med_rul} days"
            st.metric(f"Median RUL — {well_id}", rul_label)
            st.caption(f"30-day failure probability p₃₀ = {risk:.0%}. "
                       "Median RUL = day projected survival crosses 50%.")
    except Exception as e:  # never let the RUL panel break the app
        st.caption(f"Time-to-failure panel unavailable: {e}")


# =====================================================================
# Shared setup (runs every rerun) + navigation
# =====================================================================

theme.setup_page("ESP Failure-Risk Agent", icon="⚙️")
theme.suite_nav("esp")
_bootstrap_if_needed()

_fleet, _ = load()

overview = st.Page(render_overview, title="Fleet overview", icon="📊", default=True)
wells = [
    st.Page(partial(render_well, wid), title=wid, url_path=wid)
    for wid in sorted(_fleet)
]
st.navigation({"Fleet": [overview], "Wells": wells}).run()
