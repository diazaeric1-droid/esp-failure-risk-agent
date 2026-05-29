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

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_fleet
from src.explainer import explain_well, top_drivers
from src.features import featurize_fleet
from src.model import ESPRiskModel


st.set_page_config(page_title="ESP Failure Risk Agent", page_icon="⚙️", layout="wide")

st.title("ESP Failure Risk Agent")
st.caption("30-day failure probability + plain-English explanations. Built by an ex-OXY / ex-Shell Staff Production Engineer.")

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

    # Time-series plot of the well
    scada = fleet[selected]
    fig = go.Figure()
    for col, color in [("bfpd", "#1f77b4"), ("intake_pressure_psi", "#ff7f0e"),
                        ("motor_temp_f", "#d62728"), ("motor_amps", "#2ca02c")]:
        fig.add_trace(go.Scatter(x=scada["date"], y=scada[col], name=col, line=dict(color=color)))
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    drivers = top_drivers(contribs.loc[selected], k=5)
    st.subheader("Top drivers")
    drv_df = pd.DataFrame(drivers, columns=["Feature", "Contribution"])
    drv_df["Current value"] = drv_df["Feature"].map(features.loc[selected].to_dict())
    st.dataframe(drv_df, use_container_width=True)

    if explain_selected:
        with st.spinner("Generating explanation..."):
            explanation = explain_well(
                well_id=selected,
                risk_score=risk,
                feature_values=features.loc[selected].to_dict(),
                top_drivers=drivers,
            )
        st.info(explanation)
