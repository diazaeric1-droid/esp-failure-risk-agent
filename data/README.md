# Data

All data is **synthetic**. No proprietary operator data is included or should be committed.

`synthetic/generate.py` (deterministic, seed=7) produces:
- 100 wells × 60 days of daily SCADA (7 channels: bfpd, intake_pressure_psi, motor_temp_f,
  motor_amps, runtime_pct, drive_freq_hz, current_imbalance_pct)
- `labels.csv` with `well_id, failed_within_30d, failure_mode, time_to_event_days, event_observed`
- ~12% failure rate across **five** signature failure modes (scale, gas interference,
  downthrust, gas lock, electrical), with varying onset/severity, sub-threshold degradation
  in ~25% of healthy wells, and ~5% label noise (so the classes overlap and AUROC < 1.0)
- **Run-life ground truth** for the survival model: `time_to_event_days` (failure day for
  failure-bound wells; right-censoring day for healthy wells) and `event_observed` (1 = failed,
  0 = censored). The run-life draws use an independent RNG, so they never perturb the SCADA
  channels — the classifier feature data is byte-identical with or without them.

```
python data/synthetic/generate.py
```

CSVs and `labels.csv` are `.gitignore`d; regenerate deterministically as above.
