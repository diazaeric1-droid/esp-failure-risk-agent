"""Generate a synthetic SCADA dataset for ESP failure prediction.

100 wells × 60 days × 5 channels (bfpd, intake_p, motor_temp, motor_amps,
runtime_pct), with ~12% labeled as having failed within the 30 days following
the observation window. Failure-bound wells exhibit one of several signature
degradation patterns (gradual amps creep + temp creep = scale; intake collapse
+ amps jitter = gas interference; rate drop below POR floor = downthrust).

Realism notes (deliberate, so the eval is not trivially separable):
- Failure signatures vary in ONSET DAY and SEVERITY — some are subtle.
- A fraction of HEALTHY wells get sub-threshold degradation ("degrading but
  survives") so the healthy class is not a tight Gaussian.
- ~5% LABEL NOISE is injected to mimic real mislabeling / surprise failures.
Together these push a real model into the ~0.75-0.90 AUROC band rather than 1.0.

Every pattern draws from its OWN per-well rng (seed=well index), so output does
not depend on the order patterns are evaluated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).parent
N_WELLS = 100
N_DAYS = 60
FAILURE_RATE = 0.12
LABEL_NOISE_RATE = 0.05
MASTER_SEED = 7
RNG = np.random.default_rng(MASTER_SEED)

DATE_END = pd.Timestamp("2026-05-25")
DATES = pd.date_range(end=DATE_END, periods=N_DAYS)


def healthy(rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "date": DATES,
        "bfpd": np.clip(rng.normal(2400, 120, N_DAYS), 1800, 3400),
        "intake_pressure_psi": np.clip(rng.normal(130, 15, N_DAYS), 90, 200),
        "motor_temp_f": np.clip(rng.normal(288, 6, N_DAYS), 270, 320),
        "motor_amps": np.clip(rng.normal(62, 3, N_DAYS), 55, 72),
        "runtime_pct": np.clip(rng.normal(99, 0.8, N_DAYS), 95, 100),
    })


def scale_failure(rng: np.random.Generator) -> pd.DataFrame:
    df = healthy(rng)
    onset = int(rng.integers(0, 25))                 # creep can start early or mid-window
    severity = rng.uniform(12, 26)                   # some mild (overlap with mild_degradation)
    creep = np.concatenate([np.zeros(onset),
                            np.linspace(0, severity, N_DAYS - onset)])
    df["motor_amps"] = np.clip(df["motor_amps"] + creep, 55, 95)
    df["motor_temp_f"] = np.clip(df["motor_temp_f"] + creep * 0.9, 270, 345)
    return df


def gas_interference_failure(rng: np.random.Generator) -> pd.DataFrame:
    df = healthy(rng)
    onset = int(rng.integers(25, 46))                # collapse onset varies
    floor = rng.uniform(15, 45)                      # some only dip partway (mild)
    p = df["intake_pressure_psi"].to_numpy(copy=True)
    p[onset:] = np.linspace(p[onset], floor, N_DAYS - onset)
    df["intake_pressure_psi"] = p
    df["motor_amps"] = (df["motor_amps"] + rng.normal(0, rng.uniform(4, 8), N_DAYS)).clip(40, 95)
    return df


def downthrust_failure(rng: np.random.Generator) -> pd.DataFrame:
    df = healthy(rng)
    end_rate = rng.uniform(1250, 1900)               # some only mild slumps
    df["bfpd"] = np.linspace(2400, end_rate, N_DAYS) + rng.normal(0, 60, N_DAYS)
    df["runtime_pct"] = np.clip(99 - np.linspace(0, rng.uniform(12, 25), N_DAYS), 60, 100)
    return df


def mild_degradation(rng: np.random.Generator) -> pd.DataFrame:
    """Healthy well with sub-threshold degradation — should NOT be flagged, but
    deliberately overlaps the early part of a real failure signature."""
    df = healthy(rng)
    creep = np.linspace(0, rng.uniform(4, 10), N_DAYS)
    df["motor_amps"] = np.clip(df["motor_amps"] + creep, 55, 95)
    df["intake_pressure_psi"] = np.clip(
        df["intake_pressure_psi"] - np.linspace(0, rng.uniform(10, 30), N_DAYS), 60, 200)
    return df


def normal_with_noise(rng: np.random.Generator) -> pd.DataFrame:
    """Healthy but noisier — model should NOT flag this."""
    df = healthy(rng)
    df["motor_amps"] += rng.normal(0, 2, N_DAYS)
    df["bfpd"] += rng.normal(0, 80, N_DAYS)
    return df


FAILURE_PATTERNS = [scale_failure, gas_interference_failure, downthrust_failure]


def main():
    labels = []
    n_failures = int(N_WELLS * FAILURE_RATE)
    failure_indices = set(RNG.choice(N_WELLS, size=n_failures, replace=False))
    # ~25% of healthy wells get sub-threshold degradation (overlap with failures).
    healthy_pool = [i for i in range(N_WELLS) if i not in failure_indices]
    mild_indices = set(RNG.choice(healthy_pool, size=int(0.25 * len(healthy_pool)), replace=False))

    for i in range(N_WELLS):
        well_id = f"well_{i+1:03d}"
        rng = np.random.default_rng(i)               # per-well, order-independent
        if i in failure_indices:
            pattern = FAILURE_PATTERNS[i % len(FAILURE_PATTERNS)]
            df = pattern(rng)
            failed = 1
        elif i in mild_indices:
            df = mild_degradation(rng)
            failed = 0
        else:
            df = normal_with_noise(rng)
            failed = 0
        df.to_csv(OUT / f"{well_id}.csv", index=False)
        labels.append({"well_id": well_id, "failed_within_30d": failed})

    # Inject label noise: flip ~5% of labels (real datasets have mislabeled /
    # surprise outcomes; this caps achievable AUROC below 1.0).
    n_flip = max(1, int(LABEL_NOISE_RATE * N_WELLS))
    flip_idx = RNG.choice(N_WELLS, size=n_flip, replace=False)
    for j in flip_idx:
        labels[j]["failed_within_30d"] = 1 - labels[j]["failed_within_30d"]

    pd.DataFrame(labels).to_csv(OUT / "labels.csv", index=False)
    n_pos = sum(l["failed_within_30d"] for l in labels)
    print(f"Wrote {N_WELLS} wells ({n_pos} failures incl. {n_flip} noise-flipped labels). "
          f"Labels in labels.csv.")


if __name__ == "__main__":
    main()
