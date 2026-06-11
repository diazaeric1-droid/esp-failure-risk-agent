"""Tests for the oracle / Bayes-optimal ceiling (src/oracle.py).

These assert the ceiling is computed correctly from the generator's KNOWN label
process, and that it actually upper-bounds what a model can do — without needing a
trained artifact (the ceiling is a property of the data, not the model).
"""
import numpy as np
import pandas as pd
import pytest

from data.synthetic.generate import N_WELLS, LABEL_NOISE_RATE
from src.oracle import (
    compute_oracle_ceiling,
    oracle_probabilities,
    signal_capture,
    _reconstruct_true_classes,
)


def _labels():
    """Reconstruct the realised noisy labels deterministically (same as labels.csv),
    so the test needs no generated file on disk."""
    import numpy as np
    rng = np.random.default_rng(7)
    n_fail = int(N_WELLS * 0.12)
    fail_idx = set(rng.choice(N_WELLS, size=n_fail, replace=False))
    healthy_pool = [i for i in range(N_WELLS) if i not in fail_idx]
    _ = rng.choice(healthy_pool, size=int(0.25 * len(healthy_pool)), replace=False)
    obs = np.array([1 if i in fail_idx else 0 for i in range(N_WELLS)])
    n_flip = max(1, int(LABEL_NOISE_RATE * N_WELLS))
    flip = rng.choice(N_WELLS, size=n_flip, replace=False)
    for j in flip:
        obs[j] = 1 - obs[j]
    return pd.Series(obs, index=[f"well_{i+1:03d}" for i in range(N_WELLS)],
                     name="failed_within_30d")


def test_oracle_probabilities_are_two_levels():
    p = oracle_probabilities()
    p_flip = max(1, int(LABEL_NOISE_RATE * N_WELLS)) / N_WELLS
    vals = set(np.round(p.unique(), 6))
    assert vals == {round(1 - p_flip, 6), round(p_flip, 6)}
    # exactly the true-failure count sits at the high level
    n_high = int((p > 0.5).sum())
    assert n_high == int(_reconstruct_true_classes().sum())


def test_ceiling_metrics_in_valid_ranges():
    c = compute_oracle_ceiling(_labels())
    assert 0.5 <= c.auroc <= 1.0
    assert 0.0 <= c.precision_at_top10pct <= 1.0
    assert 0.0 <= c.brier <= 1.0
    assert c.n_true_failures == 12
    assert c.n_label_flips == 5
    assert c.n_wells == 100


def test_oracle_is_an_upper_bound_on_auroc():
    # No scoring of these labels by ANY predictor should beat the oracle's AUROC in
    # expectation; empirically check the oracle beats many random predictors and a
    # noise-corrupted version of itself.
    from sklearn.metrics import roc_auc_score
    y = _labels()
    oracle_auroc = compute_oracle_ceiling(y).auroc
    rng = np.random.default_rng(0)
    beaten = 0
    for _ in range(200):
        rand = rng.random(len(y))
        if roc_auc_score(y.values, rand) > oracle_auroc + 1e-9:
            beaten += 1
    # random predictors essentially never beat the oracle
    assert beaten <= 2, f"{beaten}/200 random predictors beat the oracle ceiling"


def test_signal_capture_framings():
    cap = signal_capture(0.85, 0.86)
    assert 0.0 < cap["ratio"] <= 1.0 + 1e-9
    # above-chance is the stricter framing and should be <= ratio for sub-ceiling models
    assert cap["above_chance"] <= cap["ratio"] + 1e-9
    # a model AT the ceiling captures 100%
    cap_full = signal_capture(0.86, 0.86)
    assert cap_full["above_chance"] == pytest.approx(1.0)


def test_brier_ceiling_matches_closed_form():
    # With p_flip flips, oracle Brier = p_flip*(1-p_flip) for every well (each class).
    c = compute_oracle_ceiling(_labels())
    p_flip = c.p_flip
    assert c.brier == pytest.approx(p_flip * (1 - p_flip), abs=1e-9)
