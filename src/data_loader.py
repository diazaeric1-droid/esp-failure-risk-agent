"""Load SCADA time series + failure labels."""
from __future__ import annotations

import tempfile
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

# For an UPLOADED *fleet* SCADA export (the "bring your own SCADA" path) the rows for
# every well live in one long/tidy CSV, so a ``well_id`` column is required on top of
# the per-well channels above. This is the exact, strict schema surfaced in the app's
# downloadable template — the feature pipeline (src/features.py) consumes nothing else.
WELL_ID_COLUMN = "well_id"
UPLOAD_REQUIRED_COLUMNS = [WELL_ID_COLUMN] + SCADA_COLUMNS  # date + the 5 core channels
UPLOAD_OPTIONAL_COLUMNS = list(OPTIONAL_COLUMNS)            # backfilled if absent


def validate_scada_schema(df: pd.DataFrame) -> list[str]:
    """Return the REQUIRED upload columns that are missing from ``df`` (empty = valid).

    Strict by design: the app couples a fixed feature schema to a trained model, so an
    uploaded fleet SCADA CSV must carry ``well_id`` plus the core SCADA channels
    (``UPLOAD_REQUIRED_COLUMNS``). The two v0.5.0 channels (drive_freq_hz,
    current_imbalance_pct) stay OPTIONAL — backfilled with healthy defaults — so they
    are never reported here. Order is preserved so the caller can show a stable list.
    """
    present = set(df.columns)
    return [c for c in UPLOAD_REQUIRED_COLUMNS if c not in present]


def scada_template_frame() -> pd.DataFrame:
    """One-row example DataFrame in the exact uploadable fleet-SCADA schema.

    Header = the required ``well_id`` + core channels, then the two optional v0.5.0
    channels, with one healthy-baseline example row so a user can see the precise
    format (used by the app's 'download a template CSV' button)."""
    cols = UPLOAD_REQUIRED_COLUMNS + UPLOAD_OPTIONAL_COLUMNS
    example = {
        "well_id": "well_001",
        "date": "2026-01-01",
        "bfpd": 2400.0,
        "intake_pressure_psi": 130.0,
        "motor_temp_f": 290.0,
        "motor_amps": 62.0,
        "runtime_pct": 99.0,
        "drive_freq_hz": OPTIONAL_COLUMNS["drive_freq_hz"],
        "current_imbalance_pct": OPTIONAL_COLUMNS["current_imbalance_pct"],
    }
    return pd.DataFrame([{c: example[c] for c in cols}])


def load_fleet_from_frame(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split one long/tidy fleet SCADA frame into the SAME ``{well_id: DataFrame}`` shape
    that ``load_fleet`` returns, reusing the EXISTING per-well loader for each group.

    Each well's rows are written to a temp CSV and parsed back through
    ``load_well_scada`` so they get the identical treatment as on-disk synthetic wells
    (required-column check, optional-channel backfill, date-sort, fixed column order) —
    no parallel loading path. Assumes ``validate_scada_schema(df)`` already passed.
    """
    fleet: dict[str, pd.DataFrame] = {}
    for well_id, group in df.groupby(WELL_ID_COLUMN, sort=True):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=True, newline=""
        ) as tmp:
            group.drop(columns=[WELL_ID_COLUMN]).to_csv(tmp.name, index=False)
            fleet[str(well_id)] = load_well_scada(tmp.name)
    return fleet


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
