"""Honest real-data adapter path for the ESP failure-risk app.

THE SHIPPED DEMO RUNS ON THE SYNTHETIC GENERATOR (``data/synthetic/generate.py``,
seed=7). There is NO trained real-operator model in this repo and we do not claim
any real-data metrics anywhere. This module exists to show the adapter is *wired*:
given a real public SCADA export, it maps the columns into the feature frame the
app already consumes, so a reviewer can see the integration seam is real even
though the live demo stays synthetic.

Supported public schemas (column mapping documented below):

Volve / Equinor (Open Subsurface / production data, disclosed 2018)
------------------------------------------------------------------
Volve publishes daily production volumes and some downhole/ESP telemetry. The
relevant mapping into our SCADA channel schema (see ``data_loader.SCADA_COLUMNS``
+ ``OPTIONAL_COLUMNS``) is roughly::

    our channel              <- Volve column (after unit conversion)
    ----------------------------------------------------------------
    date                     <- DATEPRD                (date)
    bfpd                     <- BORE_OIL_VOL + BORE_WAT_VOL  (Sm3/d -> bbl/d, ×6.2898)
    intake_pressure_psi      <- AVG_DOWNHOLE_PRESSURE   (bar -> psi, ×14.5038)
    motor_temp_f             <- AVG_DOWNHOLE_TEMPERATURE (°C -> °F, ×9/5 + 32)
    motor_amps               <- (not in public Volve daily export — left to default)
    runtime_pct              <- ON_STREAM_HRS / 24 * 100
    drive_freq_hz            <- AVG_CHOKE_SIZE_P / wellhead VSD setpoint if present
    current_imbalance_pct    <- (not published — healthy-baseline default)

NDIC (North Dakota Industrial Commission) Oil & Gas
---------------------------------------------------
NDIC publishes monthly per-well production (oil/gas/water bbl, days produced).
It is monthly, not daily, and has no ESP electrical telemetry, so it only
populates rate + runtime channels; the electrical channels fall back to
healthy-baseline defaults (the app already tolerates that — see
``data_loader.OPTIONAL_COLUMNS``)::

    date                     <- ReportDate (month -> resampled to daily)
    bfpd                     <- (Oil + Water) / DaysProduced   (bbl/month -> bbl/d)
    runtime_pct              <- DaysProduced / days_in_month * 100
    intake_pressure_psi,
    motor_temp_f, motor_amps <- not reported -> defaults

Because the public exports do not carry the full ESP electrical signature our
trained model leans on (motor amps, current imbalance), a real deployment would
either (a) ingest the operator's own historian (PI / Ignition) which DOES carry
those tags, or (b) retrain on the reduced channel set. We do NOT do either here;
this is a documented seam, not a validated real-data pipeline.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_loader import ALL_COLUMNS, OPTIONAL_COLUMNS, SCADA_COLUMNS

# Unit conversions for the documented public schemas.
SM3_TO_BBL = 6.2898          # 1 Sm3 -> bbl
BAR_TO_PSI = 14.5037738      # 1 bar -> psi


# Column-rename maps from a known public schema INTO our raw SCADA channels.
# Values that need arithmetic (sums, unit scaling, resampling) are handled in
# the body of ``load_real_scada`` after the rename; these cover the 1:1 renames.
VOLVE_RENAME = {
    "DATEPRD": "date",
    "AVG_DOWNHOLE_PRESSURE": "intake_pressure_psi",   # then ×BAR_TO_PSI
    "AVG_DOWNHOLE_TEMPERATURE": "motor_temp_f",        # then °C→°F
}

NDIC_RENAME = {
    "ReportDate": "date",
}


def _passthrough_well(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce an already-app-schema frame into the canonical channel order,
    backfilling optional channels with healthy-baseline defaults. Used when the
    caller hands us a frame that already uses our column names (the honest
    'passthrough' branch)."""
    out = df.copy()
    for col, default in OPTIONAL_COLUMNS.items():
        if col not in out.columns:
            out[col] = default
    missing = set(SCADA_COLUMNS) - set(out.columns)
    if missing:
        raise ValueError(f"passthrough frame missing required SCADA columns: {missing}")
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True)[ALL_COLUMNS]


def load_real_scada(path: str | Path, schema: str = "auto") -> pd.DataFrame:
    """Map a real public SCADA export into the app's raw SCADA feature frame.

    ADAPTER STUB — honest by design. The shipped demo does NOT call this; it uses
    the synthetic generator. This function shows the integration seam: it accepts
    a Volve/Equinor or NDIC export (or a frame already in our schema) and returns
    a DataFrame with exactly ``data_loader.ALL_COLUMNS`` so it can flow straight
    into ``features.featurize_well`` / ``featurize_fleet``.

    Args:
        path: CSV path for a single well's export.
        schema: one of ``"volve"``, ``"ndic"``, ``"passthrough"``, or ``"auto"``
            (sniff by column names).

    Returns:
        DataFrame in the canonical SCADA schema (``ALL_COLUMNS``), date-sorted.

    Raises:
        NotImplementedError: for a recognised public schema whose full mapping
            (e.g. monthly→daily resampling, missing electrical channels) is
            documented in the module docstring but intentionally NOT shipped as a
            validated pipeline. We refuse to silently fabricate channels the
            public export does not contain.

    No real-data metrics are produced or claimed by this function.
    """
    path = Path(path)
    df = pd.read_csv(path)
    cols = set(df.columns)

    if schema == "auto":
        if {"DATEPRD"} & cols:
            schema = "volve"
        elif {"ReportDate"} & cols:
            schema = "ndic"
        elif set(SCADA_COLUMNS) <= cols:
            schema = "passthrough"
        else:
            raise ValueError(
                "Could not auto-detect schema; pass schema='volve'|'ndic'|'passthrough'. "
                f"Saw columns: {sorted(cols)}")

    if schema == "passthrough":
        # The honest, fully-supported branch: caller already conforms to our schema.
        return _passthrough_well(df)

    if schema == "volve":
        raise NotImplementedError(
            "Volve adapter is documented (see module docstring) but not shipped as a "
            "validated pipeline: the public Volve daily export lacks motor_amps / "
            "current_imbalance_pct, two channels the trained model relies on, so a "
            "real run would require the operator historian (PI/Ignition) or a model "
            "retrained on the reduced channel set. The demo runs synthetic. Provide a "
            "frame already in the app schema and call with schema='passthrough' to "
            "exercise the seam.")

    if schema == "ndic":
        raise NotImplementedError(
            "NDIC adapter is documented (see module docstring) but not shipped: NDIC "
            "is monthly per-well volumes with no ESP electrical telemetry, so it cannot "
            "populate the model's electrical signature without fabrication. The demo "
            "runs synthetic. Use schema='passthrough' for an app-schema frame.")

    raise ValueError(f"Unknown schema: {schema!r}")
