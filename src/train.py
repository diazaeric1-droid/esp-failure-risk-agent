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
    console.print(f"  AUROC:                {result.auroc:.3f}")
    console.print(f"  Precision @ top-10%:  {result.precision_at_top10:.3f}")
    console.print(f"  Recall @ top-10%:     {result.recall_at_top10:.3f}")

    top_features = sorted(result.feature_importance.items(), key=lambda x: -x[1])[:5]
    console.print("\n[bold]Top features by importance:[/]")
    for feat, imp in top_features:
        console.print(f"  {feat:<30} {imp:.3f}")

    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/training_report.json", "w") as f:
        json.dump({
            "auroc": result.auroc,
            "precision_at_top10": result.precision_at_top10,
            "recall_at_top10": result.recall_at_top10,
            "feature_importance": result.feature_importance,
        }, f, indent=2)


if __name__ == "__main__":
    main()
