"""Oracle / Bayes-optimal ceiling for the synthetic ESP failure benchmark.

WHY THIS EXISTS
---------------
The synthetic data (``data/synthetic/generate.py``) is drawn from a *known* label
process, so there is an information-theoretic CEILING on how well **any** model can
score wells — the Bayes-optimal performance given the irreducible noise the
generator injects. Reporting a raw OOF AUROC of ~0.85 without that ceiling leaves a
reader unable to judge whether 0.85 is "good": it could look like a model defect, or
like there's lots of headroom left, when in fact the noise floor may sit right there.

THE GENERATOR'S LABEL PROCESS (the only source of irreducible noise)
--------------------------------------------------------------------
1. A deterministic subset of wells is made *failure-bound* (true class = 1); each
   gets a physical degradation signature the features encode almost perfectly.
2. The rest are healthy (true class = 0); ~25% get *sub-threshold* degradation.
3. **~5% of labels are then flipped uniformly at random** ("surprise failures" /
   mislabels). The flip is INDEPENDENT of the features, so no feature carries any
   information about which labels were flipped.

Because the flips are independent of the features, a model cannot recover them: the
features identify the *true class*, but the *observed* label of a flipped well is
unpredictable from data. The Bayes-optimal ("oracle") predictor therefore outputs

    P(observed_label = 1 | true class) =  1 - p_flip   if the well is failure-bound,
                                          p_flip        if the well is healthy,

where ``p_flip = n_flip / n_wells`` is each well's marginal flip probability (the
generator flips ``n_flip`` labels chosen uniformly without replacement). Scoring those
oracle probabilities against the *realised, noisy* labels — the same labels the model
is graded on — gives the attainable ceiling for AUROC, precision@top-k and Brier.

This module reconstructs the generator's true classes deterministically (same seed,
same RNG draw order) and reports that ceiling. It is honest in both directions: it
neither flatters the model (the ceiling can be well below 1.0) nor implies a defect
when ~0.85 is in fact at or near the noise floor.

Pure numpy / sklearn-metrics; deterministic. Safe to import on the live path.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

# Reuse the EXACT generator constants so the ceiling tracks the data, not a copy that
# can silently drift. (Importing the module does not regenerate any files.)
from data.synthetic.generate import (
    N_WELLS,
    FAILURE_RATE,
    LABEL_NOISE_RATE,
    MASTER_SEED,
    FAILURE_PATTERNS,
)


@dataclass
class OracleCeiling:
    """Bayes-optimal attainable metrics on the synthetic benchmark.

    All metrics are computed by scoring the oracle probabilities (the true-class
    Bayes posteriors) against the *realised noisy labels* — the same labels the
    model's OOF metrics are graded on — so model-vs-ceiling is apples-to-apples.
    """
    auroc: float                 # best attainable OOF-comparable AUROC
    precision_at_top10pct: float # best attainable precision@top-10%
    brier: float                 # best attainable (lowest) Brier score
    n_wells: int
    n_true_failures: int         # failure-bound wells before label noise
    n_observed_positives: int    # positives after ~p_flip label noise
    n_label_flips: int           # labels flipped (the irreducible noise)
    p_flip: float                # per-well marginal flip probability
    k_top: int                   # size of the top-10% alert list

    def as_dict(self) -> dict:
        return asdict(self)


def _reconstruct_true_classes() -> np.ndarray:
    """Recover the generator's per-well TRUE (pre-noise) class, deterministically.

    Mirrors ``data/synthetic/generate.main`` exactly: same master seed and the same
    sequence of RNG draws, so ``true_label[i] == 1`` iff well ``i`` was made
    failure-bound. Does NOT touch disk or regenerate any CSVs.
    """
    rng = np.random.default_rng(MASTER_SEED)
    n_failures = int(N_WELLS * FAILURE_RATE)
    failure_indices = set(rng.choice(N_WELLS, size=n_failures, replace=False))
    # The generator draws the mild-degradation subset next; we must advance the RNG
    # identically even though we don't use the result, so a later flip draw (if we
    # needed it) would line up. mild wells are still TRUE class 0.
    healthy_pool = [i for i in range(N_WELLS) if i not in failure_indices]
    _ = rng.choice(healthy_pool, size=int(0.25 * len(healthy_pool)), replace=False)
    true_label = np.array([1 if i in failure_indices else 0 for i in range(N_WELLS)],
                          dtype=int)
    return true_label


def oracle_probabilities() -> pd.Series:
    """Per-well Bayes-optimal probability P(observed=1 | true class), indexed by well_id.

    ``1 - p_flip`` for failure-bound wells, ``p_flip`` for healthy wells, where
    ``p_flip = n_flip / N_WELLS`` is each well's marginal label-flip probability.
    """
    true_label = _reconstruct_true_classes()
    n_flip = max(1, int(LABEL_NOISE_RATE * N_WELLS))
    p_flip = n_flip / N_WELLS
    p = np.where(true_label == 1, 1.0 - p_flip, p_flip)
    well_ids = [f"well_{i + 1:03d}" for i in range(N_WELLS)]
    return pd.Series(p, index=well_ids, name="oracle_p")


def compute_oracle_ceiling(labels: pd.Series, frac: float = 0.1) -> OracleCeiling:
    """Compute the attainable ceiling for the realised labels.

    Args:
        labels: realised ``failed_within_30d`` Series indexed by well_id (the noisy
            labels the model is also graded on — pass ``load_labels(...)`` output).
        frac: top-fraction for precision@top-k (0.1 → top-10%, matching the model eval).

    Returns:
        OracleCeiling with AUROC / precision@top-10% / Brier computed from the oracle
        probabilities scored against ``labels``.
    """
    oracle_p = oracle_probabilities()
    # Align to the labels actually present (defensive against subsetting).
    y = labels.reindex(oracle_p.index).dropna().astype(int)
    p = oracle_p.reindex(y.index)
    yv = y.to_numpy()
    pv = p.to_numpy()

    auroc = float(roc_auc_score(yv, pv)) if len(np.unique(yv)) > 1 else float("nan")
    brier = float(brier_score_loss(yv, pv))

    # Precision@top-k: ties at the high oracle score are broken by well order, which
    # is the *honest worst-ordering-independent* read because every top-scored well is
    # a true-failure; precision is whatever fraction of the top-k truly carry obs=1.
    k = max(int(frac * len(yv)), 1)
    order = np.argsort(-pv, kind="stable")[:k]
    prec = float(yv[order].mean())

    n_flip = max(1, int(LABEL_NOISE_RATE * N_WELLS))
    true_label = _reconstruct_true_classes()
    return OracleCeiling(
        auroc=auroc,
        precision_at_top10pct=prec,
        brier=brier,
        n_wells=int(len(yv)),
        n_true_failures=int(true_label.sum()),
        n_observed_positives=int(yv.sum()),
        n_label_flips=n_flip,
        p_flip=n_flip / N_WELLS,
        k_top=int(k),
    )


def signal_capture(model_auroc: float, ceiling_auroc: float) -> dict:
    """Fraction of the *attainable* ranking signal the model captures.

    Two honest framings (we report both — neither alone is complete):

    - ``ratio``      = model_auroc / ceiling_auroc. Simple, but flattering because
      AUROC's floor is 0.5 (chance), not 0.
    - ``above_chance`` = (model_auroc - 0.5) / (ceiling_auroc - 0.5). Measures the
      share of *discriminating* signal (AUROC above coin-flip) the model recovers —
      the more demanding, more meaningful number.
    """
    ratio = model_auroc / ceiling_auroc if ceiling_auroc else float("nan")
    denom = ceiling_auroc - 0.5
    above = (model_auroc - 0.5) / denom if denom > 0 else float("nan")
    return {"ratio": float(ratio), "above_chance": float(above)}
