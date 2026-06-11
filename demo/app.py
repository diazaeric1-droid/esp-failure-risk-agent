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

import io
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
# --- warm-container module self-heal (vendored top-level modules) -----------
# Streamlit Cloud reuses the container across redeploys; a cached OLD `theme` /
# `fleet_registry` in sys.modules (or a stale .pyc) lacks symbols added in a newer
# commit -> AttributeError (e.g. theme.how_to). Drop their bytecode + evict the cached
# modules so the imports below reload from the CURRENT commit's source.
import shutil as _sh_heal
_sh_heal.rmtree(Path(__file__).resolve().parent / "__pycache__", ignore_errors=True)
for _stale in ("theme", "fleet_registry"):
    sys.modules.pop(_stale, None)

import theme
from src.data_loader import (
    UPLOAD_OPTIONAL_COLUMNS,
    UPLOAD_REQUIRED_COLUMNS,
    load_fleet,
    load_fleet_from_frame,
    scada_template_frame,
    validate_scada_schema,
)
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
    from src import survival_model as _survival_model  # genuine trained time-to-event model
except Exception:
    _survival_model = None
try:
    from src import oracle as _oracle                  # Bayes-optimal ceiling for honest framing
except Exception:
    _oracle = None
try:
    from src import registry as _registry
except Exception:
    _registry = None


DATA_DIR = REPO_ROOT / "data" / "synthetic"
MODEL_PATH = REPO_ROOT / "artifacts" / "esp_risk_model.joblib"
HORIZON = 180  # projection horizon (days) for the survival / RUL layer

# Data-source toggle values (sidebar radio). Synthetic (demo) is the default; the
# upload path scores a user's own fleet SCADA through the SAME loader/features/model.
SRC_SYNTHETIC = "Synthetic (demo)"
SRC_UPLOAD = "Upload your own fleet SCADA CSV"


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
    return _score_features(features)


def _score_features(features: pd.DataFrame):
    """Score an engineered feature frame with the TRAINED model: (probs desc, contribs).

    Shared by the synthetic default and the upload path so both go through the exact
    same `ESPRiskModel.predict_proba` + Tree-SHAP `feature_contributions` — no parallel
    scoring logic. `features` must already be `featurize_fleet(...)` output.
    """
    model = get_model()
    probs = pd.Series(
        model.predict_proba(features), index=features.index, name="risk"
    ).sort_values(ascending=False)
    contribs = model.feature_contributions(features)
    return probs, contribs


@st.cache_data(show_spinner=False)
def _load_uploaded(csv_bytes: bytes):
    """Load + featurize a user's uploaded fleet SCADA CSV, reusing the existing pipeline.

    Pure data step (cached on the raw bytes): parse the long/tidy CSV, split it into the
    same `{well_id: DataFrame}` shape as the on-disk fleet via the EXISTING loader, then
    run the EXISTING `featurize_fleet`. Returns (fleet, features). Schema validation is
    done by the caller BEFORE this runs, so a bad upload never reaches here.
    """
    df = pd.read_csv(io.BytesIO(csv_bytes), parse_dates=["date"])
    fleet = load_fleet_from_frame(df)
    features = featurize_fleet(fleet)
    return fleet, features


@st.cache_resource(show_spinner=False)
def get_survival_model():
    """Fit the genuine discrete-time hazard (time-to-event) model on the synthetic
    run-life ground truth. Cached so it trains once per session. Returns None if the
    module or the run-life labels aren't available."""
    if _survival_model is None:
        return None
    from src.data_loader import load_labels
    labels = load_labels(DATA_DIR / "labels.csv")
    if not {"time_to_event_days", "event_observed"} <= set(labels.columns):
        return None
    _, features = load()
    return _survival_model.fit_on_labels(features, labels)


@st.cache_data(show_spinner=False)
def survival_metrics():
    """OOF survival metrics (C-index, IBS) for the trained time-to-event model."""
    if _survival_model is None:
        return None
    try:
        return _survival_model.evaluate_from_disk(
            str(DATA_DIR), str(DATA_DIR / "labels.csv")).as_dict()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def oracle_ceiling():
    """Oracle / Bayes-optimal ceiling + the model's share of attainable signal."""
    if _oracle is None:
        return None
    try:
        from src.data_loader import load_labels
        labels = load_labels(DATA_DIR / "labels.csv").set_index("well_id")["failed_within_30d"]
        ceiling = _oracle.compute_oracle_ceiling(labels)
        probs, _ = _scored()
        # Use the artifact's OOF AUROC if present; else fall back to the live ranking.
        rep = REPO_ROOT / "artifacts" / "training_report.json"
        model_auroc = None
        if rep.exists():
            import json
            model_auroc = json.loads(rep.read_text()).get("auroc_cv_mean")
        cap = _oracle.signal_capture(model_auroc, ceiling.auroc) if model_auroc else None
        return {"ceiling": ceiling.as_dict(), "model_auroc": model_auroc, "capture": cap}
    except Exception:
        return None


# ---- shared helpers --------------------------------------------------------

def _back_to_overview() -> None:
    target = globals().get("overview")
    try:
        st.page_link(target if target is not None else "app.py",
                     label="← Back to Fleet Overview", icon="📊")
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

def _schema_caption() -> str:
    """Human-readable required/optional column list for the upload UI + template."""
    req = ", ".join(f"`{c}`" for c in UPLOAD_REQUIRED_COLUMNS)
    opt = ", ".join(f"`{c}`" for c in UPLOAD_OPTIONAL_COLUMNS)
    return (
        f"**Strict schema** — one long/tidy CSV for the whole fleet (one row per "
        f"well-day). **Required columns:** {req}. **Optional** (backfilled with "
        f"healthy defaults if absent): {opt}. `date` is parsed as a calendar date; "
        f"each well needs ~30–60 days of history for the trend/volatility features. "
        f"Scored with the exact trained model + feature pipeline the demo uses. "
        f"**Nothing is stored server-side** — the file is parsed in memory for this "
        f"session only.")


def _resolve_overview_source():
    """Sidebar data-source control + (for the upload path) strict validation.

    Returns ``(fleet, features, probs, contribs, is_upload)``. For the default synthetic
    source this is the cached fleet/scoring. For the upload source it validates the CSV
    BEFORE scoring — on missing required columns it shows a precise ``st.error`` and
    ``st.stop()``s (never crashes), then reuses the existing loader → features → model.
    """
    with st.sidebar:
        st.header("Data source")
        data_source = st.radio(
            "Fleet to score", [SRC_SYNTHETIC, SRC_UPLOAD], index=0, key="data_source",
            help="Synthetic = the modeled demo fleet with known ground truth (powers the "
                 "eval panels). Upload = score YOUR own fleet SCADA through the same "
                 "trained model + feature pipeline. Nothing is stored server-side.")
        uploaded = None
        if data_source == SRC_UPLOAD:
            uploaded = st.file_uploader("Fleet SCADA CSV", type=["csv"], key="scada_upload")
            st.download_button(
                "⬇ Download a template CSV", data=scada_template_frame().to_csv(index=False),
                file_name="esp_scada_template.csv", mime="text/csv",
                help="Header row + one example row in the exact required schema.")

    if data_source != SRC_UPLOAD:
        fleet, features = load()
        probs, contribs = _scored()
        return fleet, features, probs, contribs, False

    # --- upload path --------------------------------------------------------
    st.caption(_schema_caption())
    if uploaded is None:
        st.info("Upload a fleet SCADA CSV in the sidebar to score your own wells, or "
                "download the template to see the exact format. Nothing is stored "
                "server-side.")
        st.stop()

    # STRICT validation BEFORE any scoring — read the header only, report exact gaps.
    try:
        head = pd.read_csv(io.BytesIO(uploaded.getvalue()), nrows=0)
    except Exception as e:
        st.error(f"Could not read the uploaded CSV: {e}")
        st.stop()
    missing = validate_scada_schema(head)
    if missing:
        st.error(
            "Uploaded SCADA CSV is missing required column(s): "
            f"**{', '.join(missing)}**.\n\n"
            f"Required columns: {', '.join(UPLOAD_REQUIRED_COLUMNS)}. "
            f"Optional (backfilled if absent): {', '.join(UPLOAD_OPTIONAL_COLUMNS)}. "
            "Download the template CSV in the sidebar for the exact format.")
        st.stop()

    try:
        fleet, features = _load_uploaded(uploaded.getvalue())
    except Exception as e:  # malformed values, unparseable dates, empty file, etc.
        st.error(f"Could not process the uploaded SCADA CSV: {e}\n\n"
                 "Check that `date` is a parseable date and the channel columns are "
                 "numeric. Download the template CSV in the sidebar for a known-good example.")
        st.stop()
    if not len(features):
        st.error("No wells found in the uploaded CSV after parsing — ensure there is at "
                 "least one `well_id` with SCADA rows.")
        st.stop()

    probs, contribs = _score_features(features)
    st.success(f"Scored **{len(features)}** uploaded well(s) with the trained model. "
               "Nothing is stored server-side.")
    return fleet, features, probs, contribs, True


def render_overview() -> None:
    is_upload_sel = st.session_state.get("data_source") == SRC_UPLOAD
    theme.header(
        "ESP Failure-Risk Agent",
        subtitle="30-day failure probability + plain-English explanations. "
                 "Built by an ex-OXY / ex-Shell Staff Production Engineer.",
        chips=[(f"v{APP_VERSION}", "ver"), ("OOF AUROC ≈0.85 (≈ oracle ceiling)", "eval"),
               ("trained survival model", "info")],
    )
    if is_upload_sel:
        theme.data_badge("real", "Your uploaded fleet SCADA — scored in-session with the "
                                 "trained model + feature pipeline; nothing stored server-side.")
    else:
        theme.data_badge("synthetic", "Modeled SCADA + labeled failures with known ground truth — no public dataset has ESP telemetry or failure labels.")

    theme.how_to(
        "- **What it predicts** — each ESP well's **30-day failure probability** (a "
        "calibrated risk %) plus a projected **remaining useful life (RUL)** in days.\n"
        "- **Inputs** — engineered features from well SCADA: pump-**intake pressure**, "
        "**motor temperature**, **motor amps** (incl. 3-phase current imbalance), and "
        "**runtime** / drive frequency.\n"
        "- **Reading the SHAP drivers** — on each well page the driver bar shows what "
        "moves that well's risk: **red bars raise** the failure risk, **green bars lower** "
        "it, sized by each feature's log-odds contribution.\n"
        "- **Fleet table → drill-down** — start on this Fleet Overview, sort the table by "
        "30-day risk %, then open any well from the **Wells** section in the sidebar for its "
        "drivers, survival/RUL curve, and AI rationale."
    )

    with st.expander(f"🆕 What's New in v{APP_VERSION}"):
        st.markdown(
            """
- **Genuine survival model** — a trained **discrete-time logistic hazard** (`src/survival_model.py`)
  fit on the synthetic run-life ground truth (right-censored healthy wells included), evaluated
  out-of-fold with proper survival metrics (**time-dependent C-index** and **Integrated Brier
  Score** vs a Kaplan–Meier baseline). The per-well survival curve and fleet RUL ranking now come
  from *this* learned hazard shape, not a transform of the 30-day probability. (The old
  constant-hazard projection in `src/survival.py` remains a labeled fallback.)
- **Oracle / Bayes ceiling** — because the generator's label process is known, we compute the
  **best AUROC / precision@top-10% / Brier any model could reach** given the irreducible label
  noise, and report the model *against* that ceiling (📐 panel below). It reframes the realistic
  ~0.85 honestly: the model sits essentially **at the noise floor**, not below some ideal.
- **Bring your own fleet SCADA** — switch the sidebar **Data source** to *Upload your own fleet
  SCADA CSV* to score YOUR wells with the exact same trained model + feature pipeline. Strict,
  documented schema (one long/tidy CSV: `well_id` + the core channels; the two v0.5.0 channels are
  optional and backfilled) with a downloadable template; the upload is validated up-front and
  parsed in memory only — **nothing is stored server-side**.
- **Fleet explorer (multipage)** — a Fleet Overview plus a **drill-down page per well**: risk,
  suspected failure mode, SCADA chart, top-drivers + SHAP bar, the trained survival curve, and the
  BYOK AI explanation.
- **Fleet-level analytics on the overview** — decision-economics threshold chart, the out-of-fold
  **reliability curve**, the **oracle-ceiling** panel, **score-drift / PSI** monitoring, and the
  **fleet RUL ranking**.
            """
        )

    fleet, features, probs, contribs, is_upload = _resolve_overview_source()

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

    st.subheader("Fleet Snapshot")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Wells", int(len(probs)))
    k2.metric("High-risk wells (≥ threshold)", int((probs >= threshold).sum()))
    k3.metric("Median fleet risk", f"{float(probs.median()):.0%}")
    k4.metric("Median fleet RUL",
              f"{med_fleet:.0f} days" if med_fleet == med_fleet else "—")

    # --- sortable fleet table -----------------------------------------------
    st.subheader("Fleet Table")
    st.caption("One row per well — sort any column. Open a well from the **Wells** "
               "section in the sidebar to drill in (risk, SHAP, survival, AI rationale).")
    rul_by_well = (dict(zip(rul_df["well_id"], rul_df["median_rul_days"]))
                   if rul_df is not None else {})
    rows = []
    for well_id in probs.index:
        feat_row = features.loc[well_id].to_dict()
        mode, _ = classify_failure_mode(feat_row)
        last = _last_scada(fleet.get(well_id))
        row = {"Well": well_id}
        # Synthetic fleet: enrich with the shared registry (and keep only ESP wells).
        # Uploaded fleet: the registry's Permian identity is meaningless for the user's
        # own wells, so score every uploaded well and skip the placeholder metadata.
        if not is_upload:
            meta = fleet_registry.get(well_id)
            if meta.lift != "ESP":
                continue
            row.update({
                "Lift": meta.lift,
                "Lateral (ft)": meta.lateral_length_ft,
                "Basin · Formation": f"{meta.basin} · {meta.formation}",
            })
        row.update({
            "30-day risk %": round(float(probs[well_id]) * 100.0, 1),
            "Suspected failure mode": mode,
            "Median RUL (days)": rul_by_well.get(well_id, float("nan")),
            "Latest BFPD": round(last["bfpd"], 0),
            "Intake psi": round(last["intake"], 0),
            "Motor amps": round(last["amps"], 1),
        })
        rows.append(row)
    table = pd.DataFrame(rows)
    st.dataframe(table, width="stretch", hide_index=True,
                 column_config={
                     "30-day risk %": st.column_config.NumberColumn(format="%.1f%%"),
                 })
    st.download_button("⬇ Download risk table (CSV)", data=table.to_csv(index=False),
                       file_name="esp_risk_fleet.csv", mime="text/csv")

    # --- fleet-level analytics ----------------------------------------------
    # Economics / survival-RUL / reliability / drift all operate on the model + the
    # active fleet's scores, so they work on an uploaded fleet too. The oracle ceiling
    # is computed from the SYNTHETIC generator's known labels, so it's synthetic-only.
    _economics_panel(probs, threshold)
    _survival_fleet_panel(rul_df, med_fleet, features)
    _reliability_panel()
    if not is_upload:
        _oracle_panel()
    _drift_panel(features, probs)


def _economics_panel(probs: pd.Series, threshold: float) -> None:
    # A risk score only matters if it drives a decision. Find the alert threshold
    # that minimises expected fleet cost (failure cost vs. proactive intervention).
    if _economics is None:
        return
    st.divider()
    st.subheader("💰 Decision Economics — Where Should the Alert Fire?")
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
        theme.source_note(
            "Expected fleet savings ($) vs. alert threshold; dashed line = cost-minimizing "
            "threshold. Savings = failure cost ($/well) avoided − intervention cost ($/well) "
            "spent, summed over wells flagged at each threshold.")
    except Exception as e:  # never let the economics panel break the app
        st.caption(f"Decision-economics panel unavailable: {e}")


def _survival_fleet_panel(rul_df, med_fleet: float, features: pd.DataFrame) -> None:
    # Fleet RUL ranking — soonest failure first. Prefer the GENUINE trained time-to-event
    # model (discrete-time hazard fit on run-life ground truth); fall back to the
    # constant-hazard projection only if the trained model isn't available. `features`
    # is the ACTIVE fleet (synthetic or uploaded) so the ranking reflects what's scored.
    surv_model = get_survival_model()
    sm_metrics = survival_metrics()

    use_trained = surv_model is not None and _survival_model is not None
    if use_trained:
        table = _survival_model.fleet_survival_table(surv_model, features)
        rul_col = table.set_index("well_id")["median_rul_days"]
        med_fleet = float(rul_col.median())
    elif _survival is not None and rul_df is not None:
        table = rul_df.rename(columns={"median_rul_days": "median_rul_days"})
        rul_col = table.set_index("well_id")["median_rul_days"]
    else:
        return

    st.divider()
    st.subheader("⏳ Fleet Remaining-Useful-Life (RUL) Ranking")
    if use_trained:
        c_idx = sm_metrics["c_index"] if sm_metrics else float("nan")
        ibs = sm_metrics["ibs"] if sm_metrics else float("nan")
        ibs_km = sm_metrics["ibs_km_baseline"] if sm_metrics else float("nan")
        st.caption(
            "RUL comes from a **genuine trained time-to-event model** — a discrete-time "
            "logistic hazard fit on the synthetic run-life ground truth (right-censored "
            "healthy wells included), NOT a transform of the 30-day probability. "
            f"Out-of-fold **C-index = {c_idx:.2f}** (0.5 = chance), **IBS = {ibs:.3f}** "
            f"(Kaplan–Meier baseline {ibs_km:.3f}). Median RUL = day projected survival "
            "S(t) crosses 50%.")
    else:
        st.caption(
            "RUL is a **constant-hazard projection** of the calibrated 30-day risk "
            "(h = 1 − (1 − p₃₀)^(1/30)) — the trained time-to-event model is "
            "unavailable in this environment.")
    theme.references(["survival"])
    try:
        st.metric("Median fleet RUL", f"{med_fleet:.0f} days")

        top_rul = table.head(12).iloc[::-1]   # bottom-up so soonest is on top
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
        title = ("Fleet RUL Ranking — Trained Hazard Model (Soonest Failure First)"
                 if use_trained else
                 "Fleet RUL Ranking — Constant-Hazard Projection (Soonest First)")
        rul_fig.update_layout(
            title=title,
            xaxis_title="median remaining-useful-life (days)", yaxis_title="")
        st.plotly_chart(theme.style_fig(rul_fig, height=380, legend=False),
                        width="stretch")
        src = ("Median RUL (days) per well from the trained discrete-time hazard model, "
               "soonest first; bar color flags urgency (red = soonest)." if use_trained else
               "Median RUL (days) per well, soonest first. Constant-hazard projection of "
               "the calibrated 30-day risk; bar color flags urgency (red = soonest).")
        theme.source_note(src)

        # Tie to decision economics: wells projected to fail within the quarter.
        QUARTER = 90
        n_q = int((rul_col <= QUARTER).sum())
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
    st.subheader("🎯 Calibration — Do the Probabilities Mean What They Say?")
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
    theme.source_note(
        "Mean predicted probability vs. observed failure frequency, binned, from "
        "out-of-fold cross-validation (Platt-calibrated). Diagonal = perfect calibration.")


def _oracle_panel() -> None:
    # Is ~0.85 AUROC "good"? Answer it honestly: the synthetic generator has a KNOWN
    # label process, so there's a Bayes-optimal CEILING. We show the model against it.
    oc = oracle_ceiling()
    if not oc or not oc.get("ceiling"):
        return
    c = oc["ceiling"]
    m_auroc = oc.get("model_auroc")
    cap = oc.get("capture")
    st.divider()
    st.subheader("📐 Oracle Ceiling — Is ~0.85 AUROC Good? (Honest Framing)")
    st.caption(
        "The synthetic generator injects ~5% **label noise** (surprise failures / "
        "mislabels) that is independent of the features, so there is a **Bayes-optimal "
        "ceiling** on any model. We compute it by scoring the generator's true-class "
        "probabilities against the same noisy labels — the headline number is only "
        "meaningful *next to* this ceiling.")
    o1, o2, o3 = st.columns(3)
    with o1:
        if m_auroc is not None:
            st.metric("Model OOF AUROC", f"{m_auroc:.3f}",
                      delta=f"ceiling {c['auroc']:.3f}", delta_color="off")
        else:
            st.metric("Oracle AUROC ceiling", f"{c['auroc']:.3f}")
    with o2:
        if cap is not None:
            st.metric("Attainable signal captured", f"{cap['above_chance']*100:.0f}%",
                      help="(model AUROC − 0.5) / (ceiling AUROC − 0.5): the share of "
                           "discriminating signal above chance the model recovers.")
        else:
            st.metric("Precision@top-10% ceiling", f"{c['precision_at_top10pct']:.2f}")
    with o3:
        st.metric("Brier ceiling (lowest)", f"{c['brier']:.3f}")
    msg = (f"Of {c['n_wells']} wells, **{c['n_true_failures']}** are truly failure-bound; "
           f"**{c['n_label_flips']}** labels are flipped by noise → {c['n_observed_positives']} "
           f"observed positives. Those flips are unpredictable from data, which is exactly "
           f"why even a perfect model tops out near **AUROC {c['auroc']:.2f}** here — "
           f"so a realistic ~0.85 is the model sitting **at the noise floor**, not a defect.")
    if cap is not None and cap["above_chance"] >= 0.95:
        st.success(msg)
    else:
        st.info(msg)
    theme.source_note(
        "Oracle / Bayes-optimal ceiling (src/oracle.py): best attainable AUROC, "
        "precision@top-10%, and Brier given the generator's irreducible label noise, "
        "scored against the realised labels. 'Attainable signal captured' = "
        "(model − 0.5)/(ceiling − 0.5).")


def _drift_panel(features: pd.DataFrame, probs: pd.Series) -> None:
    if _registry is None:
        return
    with st.expander("🛡️ Data Quality & Score Drift"):
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
    theme.data_badge("synthetic", "Modeled SCADA + labeled failures with known ground truth — no public dataset has ESP telemetry or failure labels.")
    theme.well_cross_links("esp", well_id)
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
    st.subheader("Top Drivers")
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
    sfig.update_layout(title="SHAP Contributions (Log-Odds)",
                       xaxis_title="← lowers risk   ·   raises risk →")
    st.plotly_chart(theme.style_fig(sfig, height=320, legend=False), width="stretch")
    theme.source_note(
        "Per-feature Tree SHAP contributions (log-odds) for this well — red raises the "
        "30-day failure risk, green lowers it; sorted by magnitude.")
    theme.references(["shap"])

    # ── Time-to-failure (RUL / survival projection) for this well ───────────
    _well_survival(well_id, risk)

    # ── BYOK AI explanation (everything above needs no key) ─────────────────
    st.divider()
    st.subheader("🤖 AI Rationale (BYOK-Optional)")
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
    # Per-well survival curve. Prefer the GENUINE trained time-to-event model (discrete-
    # time hazard) — a real S(t|x) estimated from run-life data — and fall back to the
    # constant-hazard projection of p30 only if the trained model is unavailable.
    _, features = load()
    surv_model = get_survival_model()
    sm_metrics = survival_metrics()
    use_trained = (surv_model is not None and _survival_model is not None
                   and well_id in features.index)

    if use_trained:
        days, surv_all = surv_model.survival_grid(features.loc[[well_id]])
        surv = surv_all[0]
        rul_arr = surv_model.median_rul(features.loc[[well_id]])
        med_rul = int(rul_arr[0])
        horizon = surv_model.max_horizon
        capped = med_rul >= horizon
    elif _survival is not None:
        days, surv = _survival.survival_curve(risk, horizon_days=HORIZON)
        med_rul = _survival.expected_rul(risk, horizon_days=HORIZON)
        horizon = HORIZON
        capped = isinstance(med_rul, str)
    else:
        return

    st.divider()
    st.subheader("⏳ Time-to-Failure — Survival Curve & Remaining-Useful-Life (RUL)")
    if use_trained:
        c_idx = sm_metrics["c_index"] if sm_metrics else float("nan")
        ibs = sm_metrics["ibs"] if sm_metrics else float("nan")
        st.caption(
            "S(t) is from a **genuine trained time-to-event model** — a discrete-time "
            "logistic hazard h(t|x) fit on the synthetic run-life ground truth, so the "
            "curve's *shape* is learned from data (not a transform of p₃₀). "
            f"Out-of-fold **C-index = {c_idx:.2f}**, **IBS = {ibs:.3f}**. "
            "Median RUL = day S(t) crosses 50%.")
    else:
        st.caption(
            "S(t) is a **constant-hazard projection** of the calibrated 30-day risk "
            "(h = 1 − (1 − p₃₀)^(1/30)) — the trained time-to-event model is "
            "unavailable in this environment.")
    theme.references(["survival"])
    try:
        med_is_num = isinstance(med_rul, (int, float)) and not capped
        tcol1, tcol2 = st.columns([2, 1])
        with tcol1:
            sv_fig = go.Figure()
            sv_fig.add_trace(go.Scatter(
                x=days, y=surv, mode="lines", name="S(t) survival",
                line=dict(color=theme.BLUE, width=3),
                hovertemplate="day %{x}: S=%{y:.0%}<extra></extra>"))
            sv_fig.add_hline(y=0.5, line_dash="dot", line_color=theme.GREY,
                             annotation_text="50%", annotation_position="right")
            if med_is_num:
                sv_fig.add_vline(x=int(med_rul), line_dash="dash", line_color=theme.RED,
                                 annotation_text=f"median RUL ≈ {int(med_rul)}d",
                                 annotation_position="top")
            sv_fig.update_layout(
                title=f"Survival — {well_id}",
                xaxis_title="days from today", yaxis_title="P(survives past day t)",
                yaxis_range=[0, 1.02], xaxis_range=[0, horizon])
            st.plotly_chart(theme.style_fig(sv_fig, height=340), width="stretch")
            note = ("Survival S(t|x) = P(no failure past day t) from the trained discrete-"
                    "time hazard model; median RUL (days) = where S(t) crosses 50%."
                    if use_trained else
                    "Projected survival S(t) under a constant-hazard fit to the calibrated "
                    "30-day risk; median RUL (days) = where S(t) crosses 50%.")
            theme.source_note(note)
        with tcol2:
            if capped:
                rul_label = f">{horizon}d"
            else:
                rul_label = f"{int(med_rul)} days"
            st.metric(f"Median RUL — {well_id}", rul_label)
            st.caption(f"30-day failure probability p₃₀ = {risk:.0%}. "
                       "Median RUL = day survival crosses 50%.")
    except Exception as e:  # never let the RUL panel break the app
        st.caption(f"Time-to-failure panel unavailable: {e}")


# =====================================================================
# Shared setup (runs every rerun) + navigation
# =====================================================================

theme.setup_page("ESP Failure-Risk Agent", icon="⚙️")
theme.suite_nav("esp")
_bootstrap_if_needed()

_fleet, _ = load()

overview = st.Page(render_overview, title="Fleet Overview", icon="📊", default=True)
wells = [
    st.Page(partial(render_well, wid), title=wid, url_path=wid)
    for wid in sorted(_fleet)
]
st.navigation({"Fleet": [overview], "Wells": wells}).run()
