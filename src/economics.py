"""Decision economics: turn failure probabilities into a dollar-optimal alert threshold.

A risk score is only useful if it drives a decision. This module answers the
question a production supervisor actually asks: *"at what risk score is it worth
spending on a proactive intervention?"*

The model is a standard expected-cost decision rule, computed per well and
aggregated across the fleet:

- If we DON'T act on a well, expected cost = p * failure_cost
  (with probability p the ESP fails: workover + deferred production loss).
- If we DO act (proactive intervention/workover), cost = intervention_cost
  plus the residual failure risk that the intervention does not remove
  (`residual_failure_rate`, default 0 — assume a perfect fix).

For a given alert threshold ``t`` we act on every well with p >= t. The fleet's
expected total cost is the sum of the per-well minimum-decision costs *under that
policy*, and we pick the threshold that minimises it (equivalently, maximises
expected savings vs. a never-act baseline).

Pure numpy/stdlib — safe to import on the live Streamlit path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_FAILURE_COST = 350_000.0       # workover + deferred production, USD
DEFAULT_INTERVENTION_COST = 60_000.0   # proactive intervention / planned workover, USD


@dataclass
class ThresholdRecommendation:
    """Result of the fleet-wide expected-cost optimisation."""
    recommended_threshold: float        # alert threshold minimising expected fleet cost
    expected_savings: float             # USD saved vs. never-act baseline at that threshold
    expected_cost_at_threshold: float   # expected fleet cost (USD) under that policy
    baseline_cost_no_action: float      # expected fleet cost if we never intervene
    n_wells_flagged: int                # wells with p >= recommended_threshold
    failure_cost: float
    intervention_cost: float
    curve: list[tuple[float, float]]    # (threshold, expected_savings) sweep


def _expected_fleet_cost(
    probs: np.ndarray,
    threshold: float,
    failure_cost: float,
    intervention_cost: float,
    residual_failure_rate: float,
) -> float:
    """Expected total fleet cost (USD) when we intervene on every well p >= threshold."""
    probs = np.asarray(probs, dtype=float)
    act = probs >= threshold
    # Acted wells: pay the intervention plus any residual (un-fixed) failure risk.
    cost_act = intervention_cost + residual_failure_rate * probs * failure_cost
    # Un-acted wells: bear the full expected failure cost.
    cost_no_act = probs * failure_cost
    return float(np.where(act, cost_act, cost_no_act).sum())


def recommend_threshold(
    probs,
    failure_cost: float = DEFAULT_FAILURE_COST,
    intervention_cost: float = DEFAULT_INTERVENTION_COST,
    residual_failure_rate: float = 0.0,
    n_candidates: int = 101,
) -> ThresholdRecommendation:
    """Find the alert threshold that minimises expected fleet cost.

    Args:
        probs: iterable of per-well 30-day failure probabilities in [0, 1].
        failure_cost: cost of a reactive failure (workover + deferred prod).
        intervention_cost: cost of a planned proactive intervention.
        residual_failure_rate: fraction of acted-on wells that still fail anyway
            (0.0 = intervention fully removes the risk).
        n_candidates: number of candidate thresholds swept over [0, 1].

    Returns:
        ThresholdRecommendation with the optimal threshold, expected savings,
        and the full (threshold, expected_savings) curve.
    """
    probs = np.asarray(list(probs), dtype=float)
    if probs.size == 0:
        return ThresholdRecommendation(
            recommended_threshold=1.0, expected_savings=0.0,
            expected_cost_at_threshold=0.0, baseline_cost_no_action=0.0,
            n_wells_flagged=0, failure_cost=failure_cost,
            intervention_cost=intervention_cost, curve=[],
        )

    baseline = _expected_fleet_cost(
        probs, threshold=1.0001, failure_cost=failure_cost,
        intervention_cost=intervention_cost, residual_failure_rate=residual_failure_rate,
    )  # threshold > 1 => act on nobody

    # The per-well decision rule is monotone in p, so the cost-minimising fleet
    # threshold is exactly the analytic break-even probability — no need to trust a
    # grid search to find it. We still sweep to render the savings *curve*, but the
    # recommended threshold IS the break-even, so the dashboard's two numbers
    # ("recommended threshold" and "break-even probability") can never disagree.
    be = break_even_probability(failure_cost, intervention_cost, residual_failure_rate)

    candidates = np.linspace(0.0, 1.0, n_candidates)
    curve: list[tuple[float, float]] = []
    for t in candidates:
        cost = _expected_fleet_cost(
            probs, threshold=t, failure_cost=failure_cost,
            intervention_cost=intervention_cost, residual_failure_rate=residual_failure_rate,
        )
        curve.append((float(t), float(baseline - cost)))

    best_cost = _expected_fleet_cost(
        probs, threshold=be, failure_cost=failure_cost,
        intervention_cost=intervention_cost, residual_failure_rate=residual_failure_rate,
    )
    best_savings = baseline - best_cost
    n_flagged = int((probs >= be).sum())
    return ThresholdRecommendation(
        recommended_threshold=float(be),
        expected_savings=float(best_savings),
        expected_cost_at_threshold=float(best_cost),
        baseline_cost_no_action=float(baseline),
        n_wells_flagged=n_flagged,
        failure_cost=failure_cost,
        intervention_cost=intervention_cost,
        curve=curve,
    )


def break_even_probability(
    failure_cost: float = DEFAULT_FAILURE_COST,
    intervention_cost: float = DEFAULT_INTERVENTION_COST,
    residual_failure_rate: float = 0.0,
) -> float:
    """Closed-form break-even probability where acting and not-acting cost the same.

    Act when:  intervention_cost + residual*p*failure_cost  <=  p*failure_cost
    =>  p >= intervention_cost / ((1 - residual) * failure_cost)

    Because the per-well rule is monotone in p, this closed-form value is also the
    fleet cost-minimising threshold, so ``recommend_threshold`` returns exactly this
    (the swept curve is only for visualising the savings landscape around it).
    """
    denom = (1.0 - residual_failure_rate) * failure_cost
    if denom <= 0:
        return 1.0
    return float(min(1.0, max(0.0, intervention_cost / denom)))
