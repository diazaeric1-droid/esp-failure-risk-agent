"""Remaining-useful-life (RUL) / survival layer on top of the 30-day failure model.

WHAT THIS IS — and what it is NOT
---------------------------------
This module is a *model-derived projection*, not a trained time-to-event model.
The ESP risk model emits one number per well: ``p30``, the calibrated probability
that the well fails within a 30-day window. We turn that single point into a
forward survival curve S(t) by making ONE explicit assumption:

    **Constant hazard within the window.** We assume the per-day failure hazard
    ``h`` is constant over the projection horizon. The 30-day failure probability
    then implies::

        p30 = 1 - (1 - h)**30          (prob. of >=1 failure in 30 days)
        =>  h = 1 - (1 - p30)**(1/30)  (per-day discrete-time hazard)

    and the survival function (probability of surviving past day t) is::

        S(t) = (1 - h)**t,   S(0) = 1,   S(30) = 1 - p30

This is a discrete-time hazard transform of the existing calibrated probability.
It is honest about uncertainty in exactly one way: it does NOT pretend to know
the *shape* of the hazard (early-life vs wear-out / bathtub). A real time-to-event
model (Cox PH, Weibull AFT, discrete-time survival NN) trained on censored
run-life data would estimate that shape from data. We do not have labelled
time-to-failure data for the synthetic demo, so we project the flat-hazard curve
and SAY SO. Median RUL here means "day at which projected survival crosses 50%
under the constant-hazard assumption," not an empirically validated lifetime.

Pure numpy/pandas, deterministic — safe to import on the live Streamlit path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


WINDOW_DAYS = 30  # the model's native failure-probability window (p30)


def daily_hazard(p30: float) -> float:
    """Per-day discrete-time hazard implied by a 30-day failure probability.

    Inverts ``p30 = 1 - (1 - h)**WINDOW_DAYS`` under the constant-hazard-within-
    window assumption. Clips p30 into [0, 1) so the math is always defined (a
    literal p30 == 1.0 would imply infinite hazard / immediate failure).
    """
    p = float(np.clip(p30, 0.0, 1.0 - 1e-12))
    return 1.0 - (1.0 - p) ** (1.0 / WINDOW_DAYS)


def survival_curve(p30: float, horizon_days: int = 180):
    """Forward survival curve S(t) = (1 - h)**t over a daily grid [0, horizon].

    Args:
        p30: calibrated 30-day failure probability in [0, 1].
        horizon_days: projection horizon (last day on the grid).

    Returns:
        (days, S) where ``days`` is ``np.arange(0, horizon_days + 1)`` and ``S``
        is the matching survival probability array. S(0) == 1; S is monotonically
        non-increasing and bounded in [0, 1].

    NOTE: model-derived projection under a constant-hazard assumption — see module
    docstring. Not a trained time-to-event model.
    """
    h = daily_hazard(p30)
    days = np.arange(0, int(horizon_days) + 1)
    surv = (1.0 - h) ** days
    return days, surv


def expected_rul(p30: float, horizon_days: int = 180):
    """Median remaining-useful-life in days under the constant-hazard projection.

    Median RUL = the day at which S(t) first crosses 0.5. If survival never drops
    to 0.5 within the horizon (low-risk well), we cap and FLAG it as the string
    ``">{horizon_days}d"`` rather than extrapolating a number we can't see.

    Args:
        p30: calibrated 30-day failure probability in [0, 1].
        horizon_days: projection horizon / cap.

    Returns:
        int day index in [0, horizon_days] where S crosses 0.5, OR the string
        ``">180d"`` (with the actual horizon) when it never crosses within the
        horizon. The closed-form crossing is ``ln(0.5) / ln(1 - h)``.
    """
    h = daily_hazard(p30)
    if h <= 0.0:                          # zero risk → never fails in horizon
        return f">{int(horizon_days)}d"
    # Closed-form: S(t) = (1-h)**t = 0.5  =>  t = ln(0.5) / ln(1-h).
    t_cross = np.log(0.5) / np.log(1.0 - h)
    if t_cross > horizon_days:
        return f">{int(horizon_days)}d"
    # Discrete median: first integer day with S(day) <= 0.5.
    return int(np.ceil(t_cross))


def median_rul_days(p30: float, horizon_days: int = 180) -> float:
    """Numeric median RUL for sorting/plotting: the day S crosses 0.5, or the
    horizon itself when it never crosses (so 'never crosses' sorts as latest).

    Unlike :func:`expected_rul` (which returns a flag string for ranking displays
    that want the honest '>180d'), this always returns a float so it can be sorted
    and plotted. ``capped`` semantics: a returned value == horizon_days means the
    curve did not cross 0.5 within the horizon.
    """
    h = daily_hazard(p30)
    if h <= 0.0:
        return float(horizon_days)
    t_cross = np.log(0.5) / np.log(1.0 - h)
    return float(min(np.ceil(t_cross), horizon_days))


def fleet_rul(prob_series, horizon_days: int = 180) -> pd.DataFrame:
    """Fleet-wide median RUL ranking, soonest-failure first.

    Args:
        prob_series: pandas Series of per-well p30 indexed by well_id (the exact
            object the app already builds from ``model.predict_proba``), or any
            mapping well_id -> p30.
        horizon_days: projection horizon / cap.

    Returns:
        DataFrame with columns ``well_id``, ``p30``, ``median_rul_days`` sorted
        ascending by ``median_rul_days`` (soonest projected failure at the top).
        ``median_rul_days == horizon_days`` flags a curve that never crossed 0.5.

    Projection under a constant-hazard assumption — see module docstring.
    """
    s = pd.Series(prob_series)
    rows = []
    for well_id, p30 in s.items():
        rows.append({
            "well_id": well_id,
            "p30": float(p30),
            "median_rul_days": median_rul_days(float(p30), horizon_days),
        })
    df = pd.DataFrame(rows, columns=["well_id", "p30", "median_rul_days"])
    return df.sort_values("median_rul_days", ascending=True).reset_index(drop=True)
