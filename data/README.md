# Data

All data is **synthetic**. No proprietary operator data is included or should be committed.

`synthetic/generate.py` produces:
- 100 wells × 60 days of daily SCADA (5 channels)
- `labels.csv` with `well_id, failed_within_30d`
- ~12% failure rate, three signature failure patterns (scale, gas interference, downthrust)

```
python data/synthetic/generate.py
```
