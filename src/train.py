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

    # Oracle / Bayes ceiling: the best AUROC/precision/Brier ANY model could reach on
    # this generator given its irreducible label noise. Reframes ~0.85 honestly — is
    # it a defect, or is the model already near the noise floor? (See src/oracle.py.)
    oracle = None
    try:
        from .oracle import compute_oracle_ceiling, signal_capture
        oracle = compute_oracle_ceiling(labels)
        cap = signal_capture(result.auroc_cv_mean, oracle.auroc)
        console.print("\n[bold]Oracle ceiling (best attainable given label noise):[/]")
        console.print(f"  AUROC ceiling:          {oracle.auroc:.3f}   "
                      f"→ model captures {cap['above_chance']*100:.0f}% of attainable signal "
                      f"(above-chance), {cap['ratio']*100:.0f}% of raw AUROC")
        console.print(f"  Precision@top-10% ceil: {oracle.precision_at_top10pct:.3f}")
        console.print(f"  Brier ceiling:          {oracle.brier:.3f}   (lowest attainable)")
        console.print(f"  Irreducible noise:      {oracle.n_label_flips} flipped labels "
                      f"of {oracle.n_wells} wells (p_flip={oracle.p_flip:.2f}); "
                      f"{oracle.n_true_failures} true-failures, "
                      f"{oracle.n_observed_positives} observed positives")
    except Exception as e:  # never fail training over the ceiling computation
        console.print(f"  [yellow]Oracle ceiling skipped:[/] {e}")

    # Genuine time-to-event model (discrete-time hazard) on the run-life ground truth —
    # a REAL survival model, evaluated OOF with proper survival metrics (C-index, IBS),
    # not the constant-hazard projection. Best-effort: never fail training over it.
    survival_eval = None
    try:
        from .survival_model import evaluate_oof
        srv_labels = load_labels(args.labels).set_index("well_id")
        if {"time_to_event_days", "event_observed"} <= set(srv_labels.columns):
            joined = features.join(
                srv_labels[["time_to_event_days", "event_observed"]], how="inner")
            survival_eval = evaluate_oof(
                joined[features.columns],
                joined["time_to_event_days"].to_numpy(),
                joined["event_observed"].to_numpy())
            impr = (1 - survival_eval.ibs / survival_eval.ibs_km_baseline) * 100
            console.print("\n[bold]Survival model (discrete-time hazard, OOF):[/]")
            console.print(f"  C-index (concordance):  {survival_eval.c_index:.3f}  (0.5 = chance)")
            console.print(f"  Integrated Brier (IBS): {survival_eval.ibs:.4f}  "
                          f"(KM baseline {survival_eval.ibs_km_baseline:.4f} → {impr:+.0f}% vs KM)")
            console.print(f"  Run-life:               {survival_eval.n_events} events, "
                          f"{survival_eval.n_censored} censored, horizon {survival_eval.max_horizon}d")
        else:
            console.print("  [yellow]Survival eval skipped:[/] labels lack run-life columns "
                          "(regenerate data).")
    except Exception as e:
        console.print(f"  [yellow]Survival eval skipped:[/] {e}")

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
    # Persist the oracle ceiling + signal-capture next to the metrics so CI can assert
    # "model is within X of the attainable ceiling" instead of an arbitrary AUROC floor.
    if oracle is not None:
        from .oracle import signal_capture
        report["oracle_ceiling"] = oracle.as_dict()
        report["signal_capture"] = signal_capture(result.auroc_cv_mean, oracle.auroc)
    if survival_eval is not None:
        report["survival_eval"] = survival_eval.as_dict()
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
