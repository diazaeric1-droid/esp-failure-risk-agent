"""Load SCADA time series + failure labels."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


SCADA_COLUMNS = ["date", "bfpd", "intake_pressure_psi", "motor_temp_f", "motor_amps", "runtime_pct"]


def load_well_scada(path: str | Path) -> pd.DataFrame:
    """Load a single well's SCADA CSV into a sorted DataFrame indexed by date."""
    df = pd.read_csv(path, parse_dates=["date"])
    missing = set(SCADA_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"SCADA file missing columns: {missing}")
    df = df.sort_values("date").reset_index(drop=True)
    return df[SCADA_COLUMNS]


def load_fleet(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load every well CSV under data_dir/*.csv (excluding labels.csv)."""
    data_dir = Path(data_dir)
    fleet = {}
    for csv in sorted(data_dir.glob("well_*.csv")):
        well_id = csv.stem
        fleet[well_id] = load_well_scada(csv)
    return fleet


def load_labels(labels_path: str | Path) -> pd.DataFrame:
    """Failure-within-30-days labels. Columns: well_id, failed_within_30d."""
    return pd.read_csv(labels_path)
