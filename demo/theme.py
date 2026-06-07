"""Shared light + navy UI theme for the Upstream Copilot Suite.

Vendored identically into every app (next to the Streamlit entrypoint) so all
demos share one look:

- light background, navy ``#1F3A5F`` brand accent (professional, modern)
- standardized ``set_page_config`` + injected CSS (KPI cards, tabs, chips)
- a flex header with title / subtitle / right-aligned chips
- ``style_fig`` — one Plotly light template + suite colorway for every chart
- ``references`` / ``how_to`` / ``source_note`` — consistent, sourced annotations

Pure presentation: depends only on ``streamlit`` (and Plotly figures passed to
``style_fig``). Importing it has no side effects beyond defining helpers.

Usage
-----
    import theme
    theme.setup_page("Capital Program Optimizer", icon="🛢️")
    theme.header(
        "Capital Program Optimizer",
        subtitle="Risked economics + MILP allocation under budget & rig limits",
        chips=[("v0.1.0", "ver"), ("MILP optimal", "eval")],
    )
    ...
    st.plotly_chart(theme.style_fig(fig, height=340), width="stretch")
"""
from __future__ import annotations

from html import escape

import streamlit as st

# ---- brand tokens ----------------------------------------------------------
NAVY = "#1F3A5F"   # primary brand / totals
BLUE = "#4F81BD"   # secondary / positive series
RED = "#C0504D"    # loss / downside
GREEN = "#2ca02c"  # funded / healthy
AMBER = "#E8A33D"  # warning
PURPLE = "#9467bd"
TEAL = "#56c3c9"
GREY = "#9b9b9b"   # neutral / non-recoverable

# light surface tokens (aligned with .streamlit/config.toml — light, modern, professional)
BG = "#ffffff"
PANEL = "#ffffff"
BORDER = "#e5e7eb"
TEXT = "#1f2937"
MUTED = "#6b7280"
GRID = "#eef1f5"

# ordered colorway for multi-series charts
COLORWAY = [BLUE, AMBER, RED, GREEN, PURPLE, TEAL, GREY, "#d6c14e"]

# one font family for the whole suite (UI + charts)
FONT = "-apple-system, Segoe UI, Roboto, sans-serif"

_CHIP_STYLE = {
    "ver": "background:#e7eef7; color:#1F3A5F; border:1px solid #cfe0f5;",
    "eval": "background:#e7f6ec; color:#1b7a3d; border:1px solid #b7e0c4;",
    "info": "background:#e8f0fb; color:#1c4f8a; border:1px solid #c7dcf5;",
    "warn": "background:#fdf3e2; color:#9a6a16; border:1px solid #f0d9a8;",
}

CSS = f"""
<style>
    /* Clear Streamlit's fixed top toolbar so the header title isn't clipped. Target the
       legacy class AND the newer test-ids so it holds across Streamlit versions (older
       builds ship a taller toolbar / a renamed container). */
    .block-container,
    [data-testid="stMainBlockContainer"],
    [data-testid="stAppViewBlockContainer"] {{padding-top: 5rem; padding-bottom: 2rem; max-width: 1400px;}}

    /* The fixed top header bar: make it transparent and click-through so it never
       renders as a leftover dark/opaque bar over the light page, regardless of the
       build's header height. Title content sits safely below via the padding above. */
    [data-testid="stHeader"],
    [data-testid="stAppHeader"],
    header[data-testid] {{background: transparent !important; box-shadow: none !important;
                          border-bottom: none !important;}}

    /* KPI cards */
    [data-testid="stMetric"] {{
        background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px;
        padding: 0.6rem 0.85rem; box-shadow: 0 1px 2px rgba(16,24,40,0.05);
    }}
    [data-testid="stMetricValue"] {{font-size: 1.3rem; line-height: 1.2;}}
    [data-testid="stMetricLabel"] {{font-size: 0.75rem; font-weight: 600; opacity: 0.85;}}
    [data-testid="stMetricDelta"] {{font-size: 0.75rem;}}

    /* tabs */
    .stTabs [data-baseweb="tab-list"] {{gap: 8px;}}
    .stTabs [data-baseweb="tab"] {{padding: 0.4rem 1.1rem; font-weight: 600;}}
    hr {{margin: 0.4rem 0 !important;}}

    /* header */
    div.app-header {{
        display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;
        padding: 0.1rem 0 0.6rem 0; border-bottom: 1px solid {BORDER};
        margin-bottom: 0.8rem;
    }}
    .app-title {{font-size: 1.5rem; font-weight: 700; line-height: 1.1; color: {TEXT};}}
    .app-subtitle {{font-size: 0.85rem; color: {MUTED}; margin-top: 0.15rem;}}
    .app-chips {{margin-left: auto; display: flex; gap: 0.4rem; align-items: center;
                 flex-wrap: wrap;}}
    .suite-chip {{padding: 0.22rem 0.7rem; border-radius: 10px; font-size: 0.75rem;
                  font-weight: 600; white-space: nowrap;}}

    /* inline status flags */
    div.flag-high {{background:#fdeaea; color:#b42318; padding:0.3rem 0.7rem;
                    border-radius:6px; display:inline-block; margin:0.15rem;
                    font-size:0.8rem; font-weight:600; border:1px solid #f4c7c2;}}
    div.flag-ok {{background:#e7f6ec; color:#1b7a3d; padding:0.3rem 0.7rem;
                  border-radius:6px; display:inline-block; margin:0.15rem;
                  font-size:0.8rem; font-weight:600; border:1px solid #b7e0c4;}}
    div.flag-warn {{background:#fdf3e2; color:#9a6a16; padding:0.3rem 0.7rem;
                    border-radius:6px; display:inline-block; margin:0.15rem;
                    font-size:0.8rem; font-weight:600; border:1px solid #f0d9a8;}}

    /* cross-app well deep-links */
    .xwell {{font-size:0.8rem; color:{MUTED}; margin:0.15rem 0 0.7rem 0;}}
    .xwell a {{color:{NAVY}; text-decoration:none; font-weight:600;}}
    .xwell a:hover {{text-decoration:underline; color:{BLUE};}}

    /* data-provenance badge (real vs synthetic) */
    .data-badge {{display:inline-block; padding:0.25rem 0.7rem; border-radius:8px;
                  font-size:0.72rem; font-weight:700; letter-spacing:0.03em;
                  margin:0.1rem 0 0.7rem 0;}}

    /* sidebar suite navigation */
    .snav-wrap {{margin-bottom:0.8rem; border-bottom:1px solid {BORDER}; padding-bottom:0.7rem;}}
    .snav-title {{font-size:0.8rem; font-weight:700; color:{TEXT}; margin-bottom:0.45rem;}}
    .snav-item {{padding:0.22rem 0;}}
    .snav-active {{border-left:2px solid {BLUE}; padding-left:0.45rem; margin-left:-0.45rem;}}
    .snav-head {{display:flex; align-items:center; gap:0.4rem; flex-wrap:wrap;}}
    .snav-link {{font-size:0.82rem; font-weight:600; color:{NAVY}; text-decoration:none;}}
    .snav-link:hover {{text-decoration:underline; color:{BLUE};}}
    .snav-cur {{font-size:0.82rem; font-weight:700; color:{BLUE};}}
    .snav-off {{font-size:0.82rem; color:{MUTED};}}
    .snav-stage {{font-size:0.6rem; font-weight:700; text-transform:uppercase;
                  letter-spacing:0.04em; color:#5b6677; background:#eef1f5;
                  padding:0.05rem 0.4rem; border-radius:6px;}}
    .snav-desc {{font-size:0.7rem; color:{MUTED}; line-height:1.2; margin-top:0.05rem;}}
</style>
"""

# Suite registry, in production-decision-loop order. Each entry:
#   (key, display name, stage, live Streamlit Community Cloud url, one-line desc, is_live)
# Hosted on Streamlit Community Cloud (auto-deploys from GitHub main; unlimited public apps).
SUITE_APPS = [
    ("wps", "Well Performance Studio", "Design",
     "https://well-performance-studio.streamlit.app",
     "PVT · nodal · lift design · physics curves", True),
    ("pe-digest", "Daily Production Digest", "Monitor",
     "https://daily-pe-digest.streamlit.app",
     "Daily SCADA scan → anomaly brief", True),
    ("pe-copilot", "Production Engineer Copilot", "Diagnose",
     "https://pe-copilot.streamlit.app",
     "AI well review → one-page diagnosis", True),
    ("esp", "ESP Failure-Risk", "Predict",
     "https://esp-failure-risk.streamlit.app",
     "30-day ESP failure ML + SHAP drivers", True),
    ("deferment", "Deferment IQ", "Quantify",
     "https://deferment-iq.streamlit.app",
     "Lost-oil accounting + $-Pareto by cause", True),
    ("afe", "AFE Copilot", "Authorize",
     "https://afe-copilot.streamlit.app",
     "Drafts AFEs w/ net economics + routing", True),
    ("capital", "Capital Optimizer", "Allocate",
     "https://capital-optimizer.streamlit.app",
     "MILP capital allocation under limits", True),
    ("pipeline", "PE Pipeline", "Orchestrate",
     "https://pe-pipeline.streamlit.app",
     "Fleet triage → detect·predict·authorize", True),
]


def setup_page(title: str, icon: str = "🛢️", layout: str = "wide") -> None:
    """``st.set_page_config`` + inject the shared dark CSS. Call once, first."""
    st.set_page_config(
        page_title=title, page_icon=icon, layout=layout,
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)


def _chip_html(text: str, kind: str = "ver") -> str:
    style = _CHIP_STYLE.get(kind, _CHIP_STYLE["ver"])
    return f'<span class="suite-chip" style="{style}">{escape(str(text))}</span>'


def header(title: str, subtitle: str = "", chips=None) -> None:
    """Render the standardized flex header.

    chips: list of (text, kind) where kind ∈ {ver, eval, info, warn}.
    """
    chips_html = ""
    if chips:
        chips_html = '<div class="app-chips">' + "".join(
            _chip_html(t, k) for t, k in chips
        ) + "</div>"
    sub = f'<div class="app-subtitle">{escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="app-header"><div>'
        f'<div class="app-title">{escape(title)}</div>{sub}'
        f'</div>{chips_html}</div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, height: int | None = None, legend: bool = True):
    """Apply the suite's light Plotly template, colorway, and spacing.

    Spacing is tuned so a chart **title never overlaps the legend or the axis
    labels**: the title sits top-left, the (horizontal) legend sits top-right in the
    same band, the top margin grows when either is present, and both axes use
    ``automargin`` + a title standoff so axis titles can't collide with tick labels.

    Returns the same figure for chaining into ``st.plotly_chart``.
    """
    has_title = bool(getattr(getattr(fig.layout, "title", None), "text", None))
    top = 56 if (has_title or legend) else 30
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=12, family=FONT),
        colorway=COLORWAY,
        margin=dict(l=14, r=18, t=top, b=14),
        hoverlabel=dict(font_size=12),
    )
    if has_title:
        # title top-left, in its own band above the plot
        fig.update_layout(title=dict(
            x=0.0, xanchor="left", y=0.98, yanchor="top",
            font=dict(size=14, color=TEXT, family=FONT)))
    if legend:
        # legend top-RIGHT so it shares the top band with the title without overlap
        fig.update_layout(legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
        ))
    # automargin + standoff keep axis titles clear of tick labels
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, automargin=True, title_standoff=10)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, automargin=True, title_standoff=10)
    if height:
        fig.update_layout(height=height)
    return fig


def suite_nav(current: str = "") -> None:
    """Render the cross-app 'Upstream Copilot Suite' navigator into the sidebar.

    Pass the current app's key (see SUITE_APPS) so it renders as "you are here"
    instead of a link. Links open in a new tab; the paused orchestrator shows no
    link. Safe to call anywhere — it writes to ``st.sidebar`` directly.
    """
    rows = []
    for key, name, stage, url, desc, live in SUITE_APPS:
        if key == current:
            head = f'<span class="snav-cur">● {escape(name)}</span>'
            active = " snav-active"
        elif live and url:
            head = (f'<a class="snav-link" href="{escape(url)}" target="_blank" '
                    f'rel="noopener">{escape(name)}</a>')
            active = ""
        else:
            head = f'<span class="snav-off">{escape(name)} <em>(on-demand)</em></span>'
            active = ""
        rows.append(
            f'<div class="snav-item{active}"><div class="snav-head">{head}'
            f'<span class="snav-stage">{escape(stage)}</span></div>'
            f'<div class="snav-desc">{escape(desc)}</div></div>'
        )
    st.sidebar.markdown(
        '<div class="snav-wrap"><div class="snav-title">⛏️ Upstream Copilot Suite</div>'
        + "".join(rows) + "</div>",
        unsafe_allow_html=True,
    )


# Apps with per-well pages at <app-url>/<well_id> (st.Page url_path = well id).
# AFE is per-AFE and Capital has no per-well pages, so they're excluded.
_WELL_APP_KEYS = ("pe-digest", "pe-copilot", "esp", "deferment", "pipeline")


def well_cross_links(current: str, well_id: str) -> None:
    """Render a one-line 'open this well in sibling apps' deep-link row.

    Each well-based app exposes per-well pages at ``<app-url>/<well_id>``, so the
    same well opens (pre-selected) in every sibling app — turning the suite into one
    navigable system. Call it from a per-well page, passing the app's own key.
    """
    urls = {k: u for k, _n, _s, u, _d, _l in SUITE_APPS}
    names = {k: n for k, n, _s, _u, _d, _l in SUITE_APPS}
    wid = escape(str(well_id))
    links = []
    for k in _WELL_APP_KEYS:
        if k == current:
            continue
        u = urls.get(k)
        if not u:
            continue
        links.append(f'<a href="{escape(u)}/{wid}" target="_blank" '
                     f'rel="noopener">{escape(names[k])}</a>')
    if not links:
        return
    st.markdown('<div class="xwell">🔗 Open <b>' + wid + "</b> in: "
                + " · ".join(links) + "</div>", unsafe_allow_html=True)


def data_badge(source: str = "synthetic", detail: str = "") -> None:
    """Render a data-provenance badge under the header.

    source: 'real' → green "REAL DATA", anything else → amber "SYNTHETIC DATA".
    detail: short provenance note, e.g. "North Dakota (NDIC) public filings — Bakken"
    or "modeled fleet with known ground truth". Keeps every app honest about what a
    visitor is actually looking at.
    """
    if source == "real":
        label = "🟢 REAL DATA"
        style = "background:#e7f6ec; color:#1b7a3d; border:1px solid #b7e0c4;"
    else:
        label = "🟡 SYNTHETIC DATA"
        style = "background:#fdf3e2; color:#9a6a16; border:1px solid #f0d9a8;"
    d = f" — {escape(detail)}" if detail else ""
    st.markdown(f'<div class="data-badge" style="{style}">{label}{d}</div>',
                unsafe_allow_html=True)


def flag(text: str, kind: str = "ok") -> None:
    """Render an inline status flag. kind ∈ {ok, high, warn}."""
    cls = {"ok": "flag-ok", "high": "flag-high", "warn": "flag-warn"}.get(kind, "flag-ok")
    st.markdown(f'<div class="{cls}">{escape(str(text))}</div>', unsafe_allow_html=True)


# ---- sourced methods + instructions ----------------------------------------
# Canonical references for the deterministic engineering/ML methods used across the
# suite. Authored once HERE so every app cites identically and correctly — a PE or
# recruiter sees the math is annotated and sourced, not hand-waved. Use via
# ``theme.references([...keys...])``.
CITATIONS = {
    "arps": "Arps, J.J. (1945). “Analysis of Decline Curves.” Trans. AIME, 160, 228–247. "
            "(Exponential / hyperbolic / harmonic rate-decline models.)",
    "fetkovich": "Fetkovich, M.J. (1980). “Decline Curve Analysis Using Type Curves.” "
                 "JPT, 32(6), 1065–1077.",
    "dca_lib": "Decline-curve fitting implemented with prodpy (open-source DCA library, "
               "MIT-licensed): non-linear least-squares fit of the Arps models.",
    "monte_carlo": "Probabilistic forecast: Monte-Carlo sampling of the decline-fit "
                   "parameter uncertainty to produce a P90/P50/P10 rate fan and EUR.",
    "prms": "Reserves percentiles P90 (proved) / P50 / P10 follow the SPE-PRMS Petroleum "
            "Resources Management System (SPE/WPC/AAPG/SPEE, rev. 2018): P90 = 90% probability "
            "of ≥ that volume (conservative), P10 = optimistic.",
    "npv": "Discounted-cash-flow NPV = Σ CFₜ / (1+i)ᵗ. Standard petroleum project economics "
           "(e.g., Mian, M.A., “Project Economics and Decision Analysis,” PennWell, 2011).",
    "vogel": "Vogel, J.V. (1968). “Inflow Performance Relationships for Solution-Gas-Drive "
             "Wells.” JPT, 20(1), 83–92. (Dimensionless IPR curve.)",
    "hagedorn_brown": "Hagedorn, A.R. & Brown, K.E. (1965). “Experimental Study of Pressure "
                      "Gradients … in Small-Diameter Vertical Conduits.” JPT, 17(4). (VLP / "
                      "multiphase vertical-lift correlation.)",
    "beggs_brill": "Beggs, H.D. & Brill, J.P. (1973). “A Study of Two-Phase Flow in Inclined "
                   "Pipes.” JPT, 25(5), 607–617. (Flow-regime-based pressure-gradient model.)",
    "nodal": "Nodal (systems) analysis: the operating point is the intersection of inflow "
             "(IPR) and outflow (VLP/tubing) curves at the bottom-hole node. See Brown, K.E. "
             "(1984), “The Technology of Artificial Lift Methods,” and Beggs (1991), "
             "“Production Optimization Using Nodal Analysis.”",
    "esp_affinity": "ESP sizing via centrifugal-pump affinity laws (Q ∝ N, H ∝ N², P ∝ N³) "
                    "and total-dynamic-head staging. See Takács, G. (2017), “Electrical "
                    "Submersible Pumps Manual,” 2nd ed., Gulf Professional.",
    "pvt": "PVT (black-oil) correlations for Bo, Rs, μ, Z. See Standing (1947), Vázquez & "
           "Beggs (1980), and McCain, “The Properties of Petroleum Fluids” (1990).",
    "bluebonnet": "PVT, scaling-solution production curves, and rate-transient analysis via "
                  "bluebonnet (F. Male et al.; open-source, BSD-licensed).",
    "milp": "Capital selection as a 0/1 mixed-integer linear program (maximize risked NPV "
            "s.t. per-period budget + rig-day limits), solved by branch-and-bound (CBC) via "
            "PuLP; an LP relaxation gives the optimality-gap bound.",
    "shap": "Model explanations via SHAP. Lundberg, S.M. & Lee, S.-I. (2017). “A Unified "
            "Approach to Interpreting Model Predictions.” NeurIPS 30.",
    "survival": "Run-life / remaining-useful-life from survival (time-to-event) analysis. "
                "Foundational: Kaplan & Meier (1958), JASA 53; Cox (1972), J. R. Stat. Soc. B.",
    "pareto": "Loss attribution ranked by the Pareto principle (the vital-few causes that "
              "drive most deferred volume); cause split via a deterministic keyword classifier.",
    "deferment": "Deferment = well potential − actual. Potential is modeled from the well’s "
                 "full-uptime months (P75, decline-aware); the gap is split into downtime "
                 "(from days-produced / runtime) vs. underperformance (rate).",
}


def references(keys, title: str = "Methods & References") -> None:
    """Render a collapsible 'Methods & References' panel citing the canonical sources
    for the calculations on the page. ``keys`` are CITATIONS keys (unknown keys are
    skipped). Keeps sourcing consistent and correct across every app."""
    items = [CITATIONS[k] for k in keys if k in CITATIONS]
    if not items:
        return
    with st.expander(f"📚 {title}"):
        for c in items:
            st.markdown(f"- {c}")


def how_to(body: str, title: str = "How to use this", expanded: bool = False) -> None:
    """Render a consistent, collapsible instructions panel. ``body`` is markdown
    (use ``-`` bullets). Standardizes the 'what am I looking at / how do I drive it'
    note across the suite."""
    with st.expander(f"ℹ️ {title}", expanded=expanded):
        st.markdown(body)


def source_note(text: str) -> None:
    """A small annotation/source caption to sit directly under a chart or table
    (e.g., the method + units + data provenance for that specific graph)."""
    st.caption(f"📐 {text}")
