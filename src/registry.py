"""Model registry, input validation, and score-drift monitoring.

Three small, dependency-light (stdlib + pandas/numpy) production hygiene tools:

1. ``register_model`` — append a versioned, timestamped row of model metrics so
   you have an audit trail of what shipped (the "model registry").
2. ``input_range_check`` — guard against garbage-in: validate that incoming
   feature columns sit inside physically/operationally plausible ranges before
   scoring, returning a list of violations.
3. ``score_drift`` — compare the distribution of live risk scores against a
   reference (training) distribution via the Population Stability Index (PSI),
   flagging drift past a configurable threshold.

No scikit-learn / xgboost dependency, so this is safe on the live app path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Model registry
# ---------------------------------------------------------------------------

def register_model(
    metrics: dict,
    feature_names,
    path: str | Path = "artifacts/registry.json",
    version: str | None = None,
) -> dict:
    """Append a versioned entry (timestamp, app version, metrics) to the registry.

    Stored as a JSON list of entries. Returns the entry that was appended.
    """
    if version is None:
        try:
            from . import __version__ as version  # type: ignore
        except Exception:
            version = "unknown"

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": version,
        "n_features": len(list(feature_names)),
        "feature_names": list(feature_names),
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in metrics.items()},
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history: list = []
    if path.exists():
        try:
            history = json.loads(path.read_text())
            if not isinstance(history, list):
                history = [history]
        except (json.JSONDecodeError, ValueError):
            history = []
    history.append(entry)
    path.write_text(json.dumps(history, indent=2))
    return entry


# ---------------------------------------------------------------------------
# 2. Input-range validation
# ---------------------------------------------------------------------------

# Sane operating ranges for the engineered feature columns. (min, max), inclusive.
# Chosen from ESP operating physics + the synthetic generator's clip bounds with
# headroom — values outside these usually mean a unit error or a sensor fault.
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "bfpd_last7_mean": (0.0, 6000.0),
    "intake_p_last7_mean": (0.0, 5000.0),
    "motor_temp_last7_mean": (60.0, 500.0),
    "motor_amps_last7_mean": (0.0, 200.0),
    "runtime_last7_mean": (0.0, 100.0),
    "bfpd_slope_30d": (-500.0, 500.0),
    "intake_p_slope_30d": (-200.0, 200.0),
    "motor_temp_slope_30d": (-50.0, 50.0),
    "motor_amps_slope_30d": (-50.0, 50.0),
    "bfpd_cv_30d": (0.0, 5.0),
    "intake_p_cv_30d": (0.0, 5.0),
    "high_amps_days_30d": (0.0, 30.0),
    "low_intake_days_30d": (0.0, 30.0),
    "high_temp_days_30d": (0.0, 30.0),
    "downtime_days_30d": (0.0, 30.0),
    "amps_to_bfpd_ratio_last7": (0.0, 10.0),
}


@dataclass
class RangeViolation:
    well_id: str
    feature: str
    value: float
    low: float
    high: float

    def __str__(self) -> str:
        return (f"{self.well_id}: {self.feature}={self.value:.3g} "
                f"outside [{self.low:g}, {self.high:g}]")


def input_range_check(
    features_df: pd.DataFrame,
    ranges: dict[str, tuple[float, float]] | None = None,
) -> list[RangeViolation]:
    """Validate feature columns are within plausible ranges.

    Returns a list of RangeViolation (empty list == all clean). NaNs are flagged
    as violations. Unknown columns are ignored; missing expected columns are not
    fabricated (that's a schema concern handled upstream).
    """
    ranges = ranges or FEATURE_RANGES
    violations: list[RangeViolation] = []
    for feat, (low, high) in ranges.items():
        if feat not in features_df.columns:
            continue
        col = features_df[feat]
        for well_id, value in col.items():
            v = float(value) if pd.notna(value) else np.nan
            if np.isnan(v) or v < low or v > high:
                violations.append(RangeViolation(str(well_id), feat, v, low, high))
    return violations


# ---------------------------------------------------------------------------
# 3. Score drift (Population Stability Index)
# ---------------------------------------------------------------------------

@dataclass
class DriftResult:
    psi: float
    drift: bool
    threshold: float
    n_reference: int
    n_live: int

    def label(self) -> str:
        if self.psi < 0.1:
            return "no significant drift"
        if self.psi < 0.25:
            return "moderate drift"
        return "major drift"


def score_drift(
    reference_scores,
    live_scores,
    n_bins: int = 10,
    threshold: float = 0.25,
) -> DriftResult:
    """Population Stability Index between a reference and live score distribution.

    PSI = sum_i (live_i - ref_i) * ln(live_i / ref_i), over `n_bins` quantile bins
    defined on the reference distribution. Conventional reading:
      PSI < 0.10  -> no significant shift
      0.10–0.25   -> moderate shift (monitor)
      > 0.25      -> major shift (investigate / retrain)

    `drift` is True when PSI >= `threshold`. A small epsilon prevents log(0).
    """
    ref = np.asarray(list(reference_scores), dtype=float)
    live = np.asarray(list(live_scores), dtype=float)
    ref = ref[~np.isnan(ref)]
    live = live[~np.isnan(live)]
    if ref.size == 0 or live.size == 0:
        return DriftResult(psi=float("nan"), drift=False, threshold=threshold,
                           n_reference=int(ref.size), n_live=int(live.size))

    # Quantile bin edges from the reference distribution; dedupe degenerate edges.
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if edges.size < 2:                       # reference is (near) constant
        edges = np.array([ref.min() - 1e-6, ref.max() + 1e-6])
    edges[0], edges[-1] = -np.inf, np.inf    # capture the tails

    ref_counts, _ = np.histogram(ref, bins=edges)
    live_counts, _ = np.histogram(live, bins=edges)

    eps = 1e-6
    ref_pct = ref_counts / max(ref.size, 1) + eps
    live_pct = live_counts / max(live.size, 1) + eps
    psi = float(np.sum((live_pct - ref_pct) * np.log(live_pct / ref_pct)))

    return DriftResult(
        psi=psi, drift=bool(psi >= threshold), threshold=threshold,
        n_reference=int(ref.size), n_live=int(live.size),
    )
