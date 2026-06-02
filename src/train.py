"""Train the baseline XGBoost model on synthetic data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

from .data_loader import load_fleet, load_labels
from .features import featurize_fleet
from .model import ESPRiskModel


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
    console.print(f"  AUROC (single split): {result.auroc:.3f}  "
                  f"(n_test={result.n_test}, positives={result.n_test_positives})")
    console.print(f"  AUROC (K-fold CV):    {result.auroc_cv_mean:.3f} ± {result.auroc_cv_std:.3f}  ← trust this")
    console.print(f"  Precision @ top-10%:  {result.precision_at_top10pct:.3f}")
    console.print(f"  Recall @ top-10%:     {result.recall_at_top10pct:.3f}")
    console.print(f"  Calibrated probs:     {result.calibrated}")

    top_features = sorted(result.feature_importance.items(), key=lambda x: -x[1])[:5]
    console.print("\n[bold]Top features by importance:[/]")
    for feat, imp in top_features:
        console.print(f"  {feat:<30} {imp:.3f}")

    Path("artifacts").mkdir(exist_ok=True)
    report = {
        "auroc": result.auroc,
        "auroc_cv_mean": result.auroc_cv_mean,
        "auroc_cv_std": result.auroc_cv_std,
        "precision_at_top10pct": result.precision_at_top10pct,
        "recall_at_top10pct": result.recall_at_top10pct,
        "n_test": result.n_test,
        "n_test_positives": result.n_test_positives,
        "calibrated": result.calibrated,
        "feature_importance": result.feature_importance,
    }
    with open("artifacts/training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Append a versioned entry to the model registry (audit trail of what shipped).
    try:
        from .registry import register_model
        metrics = {k: v for k, v in report.items() if k != "feature_importance"}
        register_model(metrics=metrics, feature_names=model.feature_names)
        console.print("  Registered run in artifacts/registry.json")
    except Exception as e:  # registry is best-effort; never fail training over it
        console.print(f"  [yellow]Registry update skipped:[/] {e}")


if __name__ == "__main__":
    main()
