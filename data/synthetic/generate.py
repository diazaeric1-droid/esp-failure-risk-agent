"""Generate a synthetic SCADA dataset for ESP failure prediction.

100 wells × 60 days × 5 channels (bfpd, intake_p, motor_temp, motor_amps,
runtime_pct), with ~12% labeled as having failed within the 30 days following
the observation window. Failure-bound wells exhibit one of several signature
degradation patterns (gradual amps creep + temp creep = scale; intake collapse
+ amps jitter = gas interference; rate drop below POR floor = downthrust).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).parent
N_WELLS = 100
N_DAYS = 60
FAILURE_RATE = 0.12
RNG = np.random.default_rng(7)

DATE_END = pd.Timestamp("2026-05-25")
DATES = pd.date_range(end=DATE_END, periods=N_DAYS)


def healthy(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "date": DATES,
        "bfpd": np.clip(rng.normal(2400, 120, N_DAYS), 1800, 3400),
        "intake_pressure_psi": np.clip(rng.normal(130, 15, N_DAYS), 90, 200),
        "motor_temp_f": np.clip(rng.normal(288, 6, N_DAYS), 270, 320),
        "motor_amps": np.clip(rng.normal(62, 3, N_DAYS), 55, 72),
        "runtime_pct": np.clip(rng.normal(99, 0.8, N_DAYS), 95, 100),
    })


def scale_failure(seed: int) -> pd.DataFrame:
    df = healthy(seed)
    creep = np.linspace(0, 22, N_DAYS)            # amps + temp creep
    df["motor_amps"] = np.clip(df["motor_amps"] + creep, 55, 95)
    df["motor_temp_f"] = np.clip(df["motor_temp_f"] + creep * 0.9, 270, 345)
    return df


def gas_interference_failure(seed: int) -> pd.DataFrame:
    df = healthy(seed)
    # Intake collapses in the last 25 days; amps get jittery.
    p = df["intake_pressure_psi"].to_numpy(copy=True)
    p[35:] = np.linspace(p[35], 18, N_DAYS - 35)
    df["intake_pressure_psi"] = p
    df["motor_amps"] += RNG.normal(0, 8, N_DAYS)
    df["motor_amps"] = df["motor_amps"].clip(40, 95)
    return df


def downthrust_failure(seed: int) -> pd.DataFrame:
    df = healthy(seed)
    # Rate slumps below POR floor (1800 bfpd); runtime degrades.
    df["bfpd"] = np.linspace(2400, 1300, N_DAYS) + RNG.normal(0, 60, N_DAYS)
    df["runtime_pct"] = np.clip(99 - np.linspace(0, 25, N_DAYS), 60, 100)
    return df


def normal_with_noise(seed: int) -> pd.DataFrame:
    """Healthy but noisier — model should NOT flag this."""
    df = healthy(seed)
    df["motor_amps"] += RNG.normal(0, 2, N_DAYS)
    df["bfpd"] += RNG.normal(0, 80, N_DAYS)
    return df


FAILURE_PATTERNS = [scale_failure, gas_interference_failure, downthrust_failure]


def main():
    labels = []
    n_failures = int(N_WELLS * FAILURE_RATE)
    failure_indices = set(RNG.choice(N_WELLS, size=n_failures, replace=False))

    for i in range(N_WELLS):
        well_id = f"well_{i+1:03d}"
        if i in failure_indices:
            pattern = FAILURE_PATTERNS[i % len(FAILURE_PATTERNS)]
            df = pattern(seed=i)
            failed = 1
        else:
            df = normal_with_noise(seed=i)
            failed = 0
        df.to_csv(OUT / f"{well_id}.csv", index=False)
        labels.append({"well_id": well_id, "failed_within_30d": failed})

    pd.DataFrame(labels).to_csv(OUT / "labels.csv", index=False)
    n_pos = sum(l["failed_within_30d"] for l in labels)
    print(f"Wrote {N_WELLS} wells ({n_pos} failures). Labels in labels.csv.")


if __name__ == "__main__":
    main()
