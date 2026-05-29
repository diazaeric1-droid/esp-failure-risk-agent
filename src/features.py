"""Feature engineering from SCADA time series.

Each well's 60-day time series is collapsed to a single feature row capturing
the things a production engineer actually looks at: levels, trends, anomalies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    # Levels (last 7 days)
    "bfpd_last7_mean",
    "intake_p_last7_mean",
    "motor_temp_last7_mean",
    "motor_amps_last7_mean",
    "runtime_last7_mean",
    # Trends (slope over last 30 days, units per day)
    "bfpd_slope_30d",
    "intake_p_slope_30d",
    "motor_temp_slope_30d",
    "motor_amps_slope_30d",
    # Volatility
    "bfpd_cv_30d",
    "intake_p_cv_30d",
    # Anomaly counts (days exceeding thresholds in last 30 days)
    "high_amps_days_30d",
    "low_intake_days_30d",
    "high_temp_days_30d",
    "downtime_days_30d",
    # Ratios
    "amps_to_bfpd_ratio_last7",
]


def _slope(series: pd.Series) -> float:
    """Linear-regression slope (per day) ignoring NaNs."""
    s = series.dropna()
    if len(s) < 2:
        return 0.0
    x = np.arange(len(s))
    return float(np.polyfit(x, s.values, 1)[0])


def _cv(series: pd.Series) -> float:
    """Coefficient of variation; 0 if mean is 0."""
    s = series.dropna()
    if len(s) == 0 or s.mean() == 0:
        return 0.0
    return float(s.std() / abs(s.mean()))


def featurize_well(scada: pd.DataFrame) -> dict[str, float]:
    """Collapse one well's SCADA history into a feature dict."""
    last7 = scada.tail(7)
    last30 = scada.tail(30)

    return {
        "bfpd_last7_mean": float(last7["bfpd"].mean()),
        "intake_p_last7_mean": float(last7["intake_pressure_psi"].mean()),
        "motor_temp_last7_mean": float(last7["motor_temp_f"].mean()),
        "motor_amps_last7_mean": float(last7["motor_amps"].mean()),
        "runtime_last7_mean": float(last7["runtime_pct"].mean()),
        "bfpd_slope_30d": _slope(last30["bfpd"]),
        "intake_p_slope_30d": _slope(last30["intake_pressure_psi"]),
        "motor_temp_slope_30d": _slope(last30["motor_temp_f"]),
        "motor_amps_slope_30d": _slope(last30["motor_amps"]),
        "bfpd_cv_30d": _cv(last30["bfpd"]),
        "intake_p_cv_30d": _cv(last30["intake_pressure_psi"]),
        "high_amps_days_30d": int((last30["motor_amps"] > 80).sum()),
        "low_intake_days_30d": int((last30["intake_pressure_psi"] < 50).sum()),
        "high_temp_days_30d": int((last30["motor_temp_f"] > 320).sum()),
        "downtime_days_30d": int((last30["runtime_pct"] < 80).sum()),
        "amps_to_bfpd_ratio_last7": float(
            last7["motor_amps"].mean() / max(last7["bfpd"].mean(), 1)
        ),
    }


def featurize_fleet(fleet: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Featurize a fleet of wells into a single DataFrame indexed by well_id."""
    rows = []
    for well_id, scada in fleet.items():
        row = {"well_id": well_id, **featurize_well(scada)}
        rows.append(row)
    df = pd.DataFrame(rows).set_index("well_id")
    return df[FEATURE_NAMES]
