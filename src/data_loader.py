"""Load SCADA time series + failure labels."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


# Core channels every SCADA file MUST carry.
SCADA_COLUMNS = ["date", "bfpd", "intake_pressure_psi", "motor_temp_f", "motor_amps", "runtime_pct"]

# Channels added in v0.5.0. Optional for backward compatibility with older 5-channel
# exports: when a historian doesn't supply them they are filled with healthy-baseline
# defaults so the engineered feature schema stays fixed and the model never sees a
# missing column. (drive_freq_hz ≈ operator VSD setpoint; current_imbalance_pct ≈ a
# few percent on a healthy three-phase motor.)
OPTIONAL_COLUMNS: dict[str, float] = {
    "drive_freq_hz": 58.0,
    "current_imbalance_pct": 3.0,
}

ALL_COLUMNS = SCADA_COLUMNS + list(OPTIONAL_COLUMNS)


def load_well_scada(path: str | Path) -> pd.DataFrame:
    """Load a single well's SCADA CSV into a sorted DataFrame indexed by date."""
    df = pd.read_csv(path, parse_dates=["date"])
    missing = set(SCADA_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"SCADA file missing required columns: {missing}")
    # Backfill optional channels with healthy defaults so downstream featurization
    # always sees a complete, fixed schema.
    for col, default in OPTIONAL_COLUMNS.items():
        if col not in df.columns:
            df[col] = default
    df = df.sort_values("date").reset_index(drop=True)
    return df[ALL_COLUMNS]


def load_fleet(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load every well CSV under data_dir/*.csv (excluding labels.csv)."""
    data_dir = Path(data_dir)
    fleet = {}
    for csv in sorted(data_dir.glob("well_*.csv")):
        well_id = csv.stem
        fleet[well_id] = load_well_scada(csv)
    return fleet


def load_labels(labels_path: str | Path) -> pd.DataFrame:
    """Failure-within-30-days labels. Columns: well_id, failed_within_30d
    (and, from v0.5.0, an optional ``failure_mode`` tag for eval/traceability)."""
    return pd.read_csv(labels_path)
