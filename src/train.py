"""Train the baseline XGBoost model on synthetic data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rich.console import Console

from .data_loader import load_fleet, load_labels
from .features import featurize_fleet
from .model import ESPRiskModel


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/synthetic")
    parser.add_argument("--labels", default="data/synthetic/labels.csv")
    parser.add_argument("--out", default="artifacts/esp_risk_model.joblib")
    args = parser.parse_args()

    console = Console()
    console.print(f"[bold]Loading fleet from {args.data_dir}...[/]")
    fleet = load_fleet(args.data_dir)
    features = featurize_fleet(fleet)
    labels = load_labels(args.labels).set_index("well_id")["failed_within_30d"]

    aligned = features.join(labels, how="inner")
    X = aligned[features.columns]
    y = aligned["failed_within_30d"]

    console.print(f"[bold]Training on {len(X)} wells ({int(y.sum())} positives)...[/]")
    model = ESPRiskModel()
    result = model.fit(X, y)
    model.save(args.out)

    console.print(f"\n[bold green]Training complete.[/] Saved to {args.out}")
    console.print(f"  AUROC (out-of-fold CV): {result.auroc_cv_mean:.3f} ± {result.auroc_cv_std:.3f}  ← trust this")
    console.print(f"  Precision @ top-10%:    {result.precision_at_top10pct:.3f}  "
                  f"(alert list = {result.n_flagged_top10pct} wells)")
    console.print(f"  Recall @ top-10%:       {result.recall_at_top10pct:.3f}")
    console.print(f"  Brier score (OOF):      {result.brier:.3f}  (lower = better calibrated)")
    console.print(f"  Calibrated probs:       {result.calibrated}")

    top_features = sorted(result.feature_importance.items(), key=lambda x: -x[1])[:6]
    console.print("\n[bold]Top features by importance:[/]")
    for feat, imp in top_features:
        console.print(f"  {feat:<30} {imp:.3f}")

    Path("artifacts").mkdir(exist_ok=True)
    report = {
        "auroc_cv_mean": result.auroc_cv_mean,
        "auroc_cv_std": result.auroc_cv_std,
        "precision_at_top10pct": result.precision_at_top10pct,
        "recall_at_top10pct": result.recall_at_top10pct,
        "n_flagged_top10pct": result.n_flagged_top10pct,
        "brier": result.brier,
        "n_wells": result.n_wells,
        "n_positives": result.n_positives,
        "calibrated": result.calibrated,
        "reliability": result.reliability,
        "feature_importance": result.feature_importance,
    }
    with open("artifacts/training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Append a versioned entry to the model registry (audit trail of what shipped),
    # fingerprinted by the saved artifact's hash so a registry row ties to a file.
    try:
        from .registry import register_model
        metrics = {k: v for k, v in report.items()
                   if k not in ("feature_importance", "reliability")}
        metrics["model_sha256"] = _sha256(args.out)
        register_model(metrics=metrics, feature_names=model.feature_names)
        console.print("  Registered run in artifacts/registry.json")
    except Exception as e:  # registry is best-effort; never fail training over it
        console.print(f"  [yellow]Registry update skipped:[/] {e}")


if __name__ == "__main__":
    main()
