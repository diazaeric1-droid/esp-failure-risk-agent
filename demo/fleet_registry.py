"""Canonical Upstream Copilot Suite fleet registry — the single source of truth
for well IDENTITY + METADATA shared across all eight apps.

Vendored byte-identical into each app (like ``theme.py``). Apps join this metadata
onto their OWN production / SCADA / event data by ``well_id`` — every app already
uses the shared ``well_0NN`` convention, so this is purely ADDITIVE enrichment:

- It does NOT generate production data.
- It does NOT touch any app's eval / label / holdout datasets.
- So eval gates (ESP calibration, Deferment reason-code accuracy, PE Copilot blind
  holdout) are completely unaffected.

The fleet is synthetic and Permian-flavored (Midland + Delaware basins, onshore).
Operator / API are ILLUSTRATIVE placeholders for a demo asset — NOT real proprietary
data. Real public production arrives later via the NDIC / Texas-RRC adapter milestone,
which populates this same interface (``get`` / ``as_frame`` / ``enrich``) behind the
scenes so the apps don't change.

A handful of curated "hero wells" carry a consistent cross-app storyline so the same
well reads coherently end-to-end (Monitor → Diagnose → Predict → Quantify → Authorize).
Every other ``well_0NN`` gets deterministic, stable metadata derived from its number,
so a lookup never fails and never changes between runs.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

META_COLUMNS = ("basin", "area", "formation", "lift", "lateral_length_ft", "peer_group", "hero")

# Onshore Permian geology (public/generic — not proprietary).
_MIDLAND_FM = ["Wolfcamp A", "Wolfcamp B", "Spraberry (Lower)", "Spraberry (Jo Mill)", "Dean"]
_DELAWARE_FM = ["Bone Spring (2nd)", "Bone Spring (3rd)", "Wolfcamp A", "Avalon", "Wolfcamp X/Y"]
_MIDLAND_AREAS = ["Martin Co., TX", "Midland Co., TX", "Howard Co., TX", "Glasscock Co., TX"]
_DELAWARE_AREAS = ["Reeves Co., TX", "Loving Co., TX", "Ward Co., TX", "Culberson Co., TX"]
# Permian artificial-lift mix skews ESP / rod pump.
_LIFTS = ["ESP", "ESP", "ESP", "Rod pump", "Rod pump", "Gas lift", "Gas lift", "Flowing"]


@dataclass(frozen=True)
class WellMeta:
    well_id: str
    name: str
    basin: str          # "Midland" | "Delaware"
    area: str           # county-level locator
    formation: str
    lift: str           # ESP | Rod pump | Gas lift | Flowing
    api14: str          # synthetic Texas API (state 42) — illustrative only
    first_prod: str     # YYYY-MM
    peer_group: str     # for type-curve peer comparisons
    storyline: str = "" # cross-app narrative ("" unless a hero well)
    hero: bool = False
    lateral_length_ft: int = 9500   # completion lateral length (ft); curated per hero / derived

    def as_dict(self) -> dict:
        return asdict(self)


# --- curated hero wells (storyline consistent with the apps' demo behavior) ----
# Keyed by the shared well_0NN id. Stories are written to match what each app
# actually surfaces for these wells (e.g. well_013 is the top triage opportunity).
_HERO: dict[str, WellMeta] = {
    "well_007": WellMeta(
        "well_007", "Garza 7H", "Midland", "Howard Co., TX", "Spraberry (Lower)",
        "Rod pump", "42-227-30007", "2022-08", "Midland · Spraberry",
        "Base-decline rod-pump producer with a recent rate step-down — the Daily "
        "Digest's flagged rate-loss event and a Deferment IQ underperformance case.",
        hero=True, lateral_length_ft=7800),
    "well_008": WellMeta(
        "well_008", "Loving 8H", "Delaware", "Loving Co., TX", "Avalon",
        "ESP", "42-301-30008", "2023-01", "Delaware · Avalon",
        "ESP producer with rising intake pressure — an ESP-swap candidate across "
        "the ESP model, PE Copilot, and AFE Copilot.", hero=True, lateral_length_ft=9600),
    "well_013": WellMeta(
        "well_013", "Martin 13H", "Midland", "Martin Co., TX", "Wolfcamp A",
        "ESP", "42-317-30013", "2023-04", "Midland · Wolfcamp A",
        "Gas-interference ESP well — the Fleet Triage Board's top risked-NPV "
        "opportunity (gas-lift-optimization candidate) and the pipeline's flagship.",
        hero=True, lateral_length_ft=10200),
    "well_022": WellMeta(
        "well_022", "Reeves 22H", "Delaware", "Reeves Co., TX", "Bone Spring (3rd)",
        "ESP", "42-389-30022", "2022-11", "Delaware · Bone Spring",
        "ESP wear with current imbalance — high 30-day failure risk and a near-term "
        "remaining-useful-life well.", hero=True, lateral_length_ft=9900),
    "well_041": WellMeta(
        "well_041", "Ward 41H", "Delaware", "Ward Co., TX", "Wolfcamp A",
        "ESP", "42-475-30041", "2023-06", "Delaware · Wolfcamp A",
        "Late-life ESP failure signature — an ESP-swap authorization case feeding "
        "AFE Copilot and a Capital Optimizer workover candidate.", hero=True, lateral_length_ft=10500),
    "well_048": WellMeta(
        "well_048", "Midland 48H", "Midland", "Midland Co., TX", "Wolfcamp B",
        "ESP", "42-329-30048", "2023-02", "Midland · Wolfcamp B",
        "ESP-swap candidate with degrading thrust — recurring across the predict "
        "and authorize stages.", hero=True, lateral_length_ft=9800),
}


def _suffix(well_id: str) -> int:
    """Numeric suffix of a well_0NN id; 0 if unparseable (deterministic, no raise)."""
    tail = well_id.rsplit("_", 1)[-1]
    digits = "".join(c for c in tail if c.isdigit())
    return int(digits) if digits else 0


def _derive(well_id: str) -> WellMeta:
    """Deterministic, stable metadata for any non-curated well_0NN."""
    n = _suffix(well_id)
    is_midland = (n % 2 == 0)
    basin = "Midland" if is_midland else "Delaware"
    fms = _MIDLAND_FM if is_midland else _DELAWARE_FM
    areas = _MIDLAND_AREAS if is_midland else _DELAWARE_AREAS
    formation = fms[n % len(fms)]
    area = areas[(n // 2) % len(areas)]
    lift = _LIFTS[n % len(_LIFTS)]
    county = 200 + (n % 99)
    api14 = f"42-{county:03d}-{30000 + n:05d}"
    year = 2021 + (n % 4)
    month = 1 + (n % 12)
    lateral = 7500 + (n * 311) % 5001   # deterministic 7,500–12,500 ft
    return WellMeta(
        well_id=well_id, name=f"{area.split(' ')[0]} {n}H", basin=basin, area=area,
        formation=formation, lift=lift, api14=api14,
        first_prod=f"{year}-{month:02d}", peer_group=f"{basin} · {formation.split(' (')[0]}",
        lateral_length_ft=lateral,
    )


def get(well_id: str) -> WellMeta:
    """Registry metadata for a well_id — curated hero data if present, else derived.

    Never raises and never changes between runs (deterministic)."""
    return _HERO.get(well_id) or _derive(well_id)


def hero_wells() -> list[WellMeta]:
    """The curated hero wells, in id order."""
    return [_HERO[k] for k in sorted(_HERO)]


def as_frame(well_ids):
    """Return a pandas DataFrame of metadata for the given well_ids (pandas lazy-imported)."""
    import pandas as pd
    return pd.DataFrame([get(w).as_dict() for w in well_ids])


def enrich(df, id_col: str = "well_id", columns=META_COLUMNS):
    """Left-join registry metadata columns onto a DataFrame keyed by ``id_col``.

    Additive only — returns a copy with the requested metadata columns added.
    Unknown / non-well_0NN ids get deterministically-derived metadata, so no row
    is dropped and no NaNs are introduced.
    """
    out = df.copy()
    for col in columns:
        out[col] = out[id_col].map(lambda w: getattr(get(str(w)), col))
    return out
