"""Feature engineering from SCADA time series.

Each well's 60-day time series is collapsed to a single feature row capturing
the things a production engineer actually looks at: levels, trends, anomalies,
and the two electrical/VSD signals an ESP analyst checks first (current
imbalance, drive frequency).
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
    "current_imbalance_last7_mean",
    "drive_freq_last7_mean",
    # Trends (slope per day over last 30 days)
    "bfpd_slope_30d",
    "intake_p_slope_30d",
    "motor_temp_slope_30d",
    "motor_amps_slope_30d",
    "drive_freq_slope_30d",
    # Volatility
    "bfpd_cv_30d",
    "intake_p_cv_30d",
    # Peaks (electrical failure shows up as a peak, not a mean)
    "current_imbalance_max_30d",
    # Anomaly counts (days exceeding thresholds in last 30 days)
    "high_amps_days_30d",
    "low_intake_days_30d",
    "high_temp_days_30d",
    "downtime_days_30d",
    "high_imbalance_days_30d",
    # Ratios
    "amps_to_bfpd_ratio_last7",
]


def _slope(series: pd.Series, days: pd.Series | None = None) -> float:
    """Linear-regression slope **per day**, ignoring NaNs.

    Uses the actual elapsed days as the x-axis when a date series is supplied, so
    the slope is per calendar day rather than per sample — correct even when real
    SCADA has gaps / dropped days. Falls back to the sample index otherwise.
    """
    s = series.dropna()
    if len(s) < 2:
        return 0.0
    if days is not None:
        x = days.loc[s.index].to_numpy(dtype=float)
        if np.ptp(x) == 0:                      # all same day → no meaningful slope
            return 0.0
    else:
        x = np.arange(len(s), dtype=float)
    return float(np.polyfit(x, s.values, 1)[0])


def _cv(series: pd.Series) -> float:
    """Coefficient of variation; 0 if mean is 0."""
    s = series.dropna()
    if len(s) == 0 or s.mean() == 0:
        return 0.0
    return float(s.std() / abs(s.mean()))


def featurize_well(scada: pd.DataFrame) -> dict[str, float]:
    """Collapse one well's SCADA history into a feature dict.

    Tolerates absence of the v0.5.0 channels (drive_freq_hz, current_imbalance_pct)
    by substituting healthy-baseline defaults, so the returned schema is always the
    full ``FEATURE_NAMES`` set regardless of historian vintage.
    """
    last7 = scada.tail(7)
    last30 = scada.tail(30)

    # Elapsed-days x-axis for per-day slopes (robust to gaps in real data).
    if "date" in last30.columns:
        d0 = last30["date"].min()
        days30 = (last30["date"] - d0).dt.total_seconds() / 86400.0
        days30.index = last30.index
    else:
        days30 = None

    def col30(name: str, default: float) -> pd.Series:
        return last30[name] if name in last30.columns else pd.Series(
            [default] * len(last30), index=last30.index)

    def col7(name: str, default: float) -> pd.Series:
        return last7[name] if name in last7.columns else pd.Series(
            [default] * len(last7), index=last7.index)

    imbalance30 = col30("current_imbalance_pct", 3.0)

    return {
        "bfpd_last7_mean": float(last7["bfpd"].mean()),
        "intake_p_last7_mean": float(last7["intake_pressure_psi"].mean()),
        "motor_temp_last7_mean": float(last7["motor_temp_f"].mean()),
        "motor_amps_last7_mean": float(last7["motor_amps"].mean()),
        "runtime_last7_mean": float(last7["runtime_pct"].mean()),
        "current_imbalance_last7_mean": float(col7("current_imbalance_pct", 3.0).mean()),
        "drive_freq_last7_mean": float(col7("drive_freq_hz", 58.0).mean()),
        "bfpd_slope_30d": _slope(last30["bfpd"], days30),
        "intake_p_slope_30d": _slope(last30["intake_pressure_psi"], days30),
        "motor_temp_slope_30d": _slope(last30["motor_temp_f"], days30),
        "motor_amps_slope_30d": _slope(last30["motor_amps"], days30),
        "drive_freq_slope_30d": _slope(col30("drive_freq_hz", 58.0), days30),
        "bfpd_cv_30d": _cv(last30["bfpd"]),
        "intake_p_cv_30d": _cv(last30["intake_pressure_psi"]),
        "current_imbalance_max_30d": float(imbalance30.max()),
        "high_amps_days_30d": int((last30["motor_amps"] > 80).sum()),
        "low_intake_days_30d": int((last30["intake_pressure_psi"] < 50).sum()),
        "high_temp_days_30d": int((last30["motor_temp_f"] > 320).sum()),
        "downtime_days_30d": int((last30["runtime_pct"] < 80).sum()),
        "high_imbalance_days_30d": int((imbalance30 > 8).sum()),
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
