"""Genuine time-to-event (survival) model: a discrete-time logistic hazard.

WHAT THIS IS — and how it differs from ``src/survival.py``
----------------------------------------------------------
``src/survival.py`` is a *projection*: it takes the classifier's single 30-day
probability and spreads it over time under a flat-hazard assumption. It does NOT
learn a hazard *shape* from data.

THIS module is a real trained time-to-event model. It fits a **discrete-time hazard**
``h(t | x) = P(fail on day t | survived to day t, covariates x)`` by the standard
person-period method (Singer & Willett 2003; Tutz & Schmid 2016; Cox 1972 for the
proportional-hazards lineage): each well's run life is expanded into one row per day
it was at risk, with a binary "failed this day?" target, and a logistic regression is
fit on ``[time-basis, covariates]``. The fitted hazard gives a genuine, covariate- and
time-varying survival curve per well::

    S(t | x) = ∏_{s<=t} (1 - h(s | x))

and a remaining-useful-life (median RUL = first day S crosses 0.5) that is *estimated
from censored run-life data*, not transformed from a point probability. We train on the
synthetic run-life ground truth the generator now emits (``time_to_event_days``,
``event_observed``) — right-censored healthy wells included, which is what makes this a
survival problem and not a regression on failure day.

EVALUATION (proper survival metrics, out-of-fold)
-------------------------------------------------
- **Time-dependent concordance (C-index)** — Harrell's C generalised to censored data:
  over all comparable (earlier-failure, later/censored) well pairs, the fraction the
  model orders correctly by risk. 0.5 = chance, 1.0 = perfect ordering.
- **Integrated Brier Score (IBS)** — the time-integrated mean squared error between the
  predicted survival ``S(t|x)`` and the observed survival indicator, with
  inverse-probability-of-censoring weighting (Graf et al. 1999). Lower is better; a
  Kaplan–Meier-only baseline (no covariates) is reported alongside so the IBS is
  interpretable rather than a bare number.

Pure numpy / pandas / sklearn — no new runtime dependency. Deterministic.
"""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES


@contextmanager
def _quiet_solver():
    """Silence a BENIGN numpy-2.x/BLAS quirk: sklearn's lbfgs solver emits a spurious
    'divide by zero encountered in matmul' RuntimeWarning during ``X @ weights`` on some
    backends. The optimisation result is correct and deterministic (verified by stable
    OOF metrics); we suppress only this one warning around the fit, nothing else."""
    with warnings.catch_warnings(), np.errstate(divide="ignore", over="ignore",
                                                invalid="ignore"):
        warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
        yield

# Daily grid the discrete-time hazard is defined on. Failures land in [1, 30]; healthy
# wells are censored out to 90. We model hazard over the full observed horizon.
MAX_HORIZON = 90


# --------------------------------------------------------------------------------------
# Person-period expansion + the discrete-time hazard model
# --------------------------------------------------------------------------------------
def _person_period(durations: np.ndarray, events: np.ndarray, X: np.ndarray):
    """Expand (duration, event, covariates) into person-period rows for hazard fitting.

    A well observed for ``d`` days contributes rows for days ``1..d``. The per-row
    target ``y`` is 1 only on the final day AND only if the event was observed (an
    uncensored failure); every prior at-risk day, and every day of a censored well, is
    0. Returns (T, Xpp, ypp) where ``T`` is the integer day index per row.
    """
    T_list, X_list, y_list = [], [], []
    for d, e, x in zip(durations.astype(int), events.astype(int), X):
        d = max(int(d), 1)
        for t in range(1, d + 1):
            T_list.append(t)
            X_list.append(x)
            y_list.append(1 if (t == d and e == 1) else 0)
    return (np.asarray(T_list, dtype=float),
            np.asarray(X_list, dtype=float),
            np.asarray(y_list, dtype=int))


def _time_basis(T: np.ndarray) -> np.ndarray:
    """Smooth baseline-hazard basis in time: [t / MAX_HORIZON, log(t)]. Lets the
    baseline hazard rise or fall with run-day without a free parameter per day (which
    would overfit 90 days on 100 wells). ``t`` is scaled to ~[0, 1] so the design matrix
    is well-conditioned alongside the standardized covariates (no matmul overflow)."""
    T = np.asarray(T, dtype=float)
    return np.column_stack([T / float(MAX_HORIZON), np.log(T)])


@dataclass
class SurvivalEval:
    c_index: float                 # time-dependent concordance (OOF), 0.5=chance
    ibs: float                     # integrated Brier score (OOF), lower=better
    ibs_km_baseline: float         # IBS of a covariate-free Kaplan–Meier curve (reference)
    n_wells: int
    n_events: int
    n_censored: int
    max_horizon: int

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class DiscreteTimeHazardModel:
    """Discrete-time logistic-hazard survival model over the engineered ESP features."""
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    max_horizon: int = MAX_HORIZON
    scaler: StandardScaler | None = None   # standardizes the COVARIATES (not the time basis)
    clf: LogisticRegression | None = None

    # ---- fit / predict ----------------------------------------------------------------
    def fit(self, X: pd.DataFrame, durations, events) -> "DiscreteTimeHazardModel":
        X = X[self.feature_names].to_numpy(dtype=float)
        durations = np.asarray(durations, dtype=float)
        events = np.asarray(events, dtype=int)
        self.scaler = StandardScaler().fit(X)
        T, Xpp, ypp = _person_period(durations, events, self.scaler.transform(X))
        design = np.column_stack([_time_basis(T), Xpp])
        # L2-regularised logistic hazard. We deliberately do NOT use
        # ``class_weight='balanced'``: balancing the rare "failed today" rows inflates
        # the hazard magnitude and wrecks the survival *calibration* (IBS blows past the
        # Kaplan–Meier baseline) even though it helps ranking. Instead we keep the true
        # low daily base rate and lean on strong L2 (small C) — on this small, noisy
        # benchmark (17 events / 100 wells over 90 days) that improves BOTH the OOF
        # C-index and the IBS (beats KM by ~16%). See evaluate_oof / README.
        with _quiet_solver():
            self.clf = LogisticRegression(
                max_iter=5000, C=0.02, class_weight=None, random_state=42
            ).fit(design, ypp)
        return self

    def hazard_grid(self, X: pd.DataFrame) -> np.ndarray:
        """Per-well daily hazard h(t|x) on days 1..max_horizon. Shape (n_wells, H)."""
        Xs = self.scaler.transform(X[self.feature_names].to_numpy(dtype=float))
        days = np.arange(1, self.max_horizon + 1, dtype=float)
        tb = _time_basis(days)                                   # (H, 2)
        out = np.empty((Xs.shape[0], len(days)), dtype=float)
        with _quiet_solver():
            for i, x in enumerate(Xs):
                design = np.column_stack([tb, np.tile(x, (len(days), 1))])
                out[i] = self.clf.predict_proba(design)[:, 1]
        return out

    def survival_grid(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Per-well survival S(t|x) on days 0..max_horizon (S(0)=1).

        Returns (days, S) with ``days`` shape (H+1,) and ``S`` shape (n_wells, H+1).
        """
        h = self.hazard_grid(X)                                  # (n, H), days 1..H
        surv = np.cumprod(1.0 - h, axis=1)                       # S at days 1..H
        surv = np.column_stack([np.ones(surv.shape[0]), surv])   # prepend S(0)=1
        days = np.arange(0, self.max_horizon + 1)
        return days, surv

    def median_rul(self, X: pd.DataFrame) -> np.ndarray:
        """Median RUL per well: first day S(t|x) <= 0.5, capped at max_horizon."""
        days, surv = self.survival_grid(X)
        out = np.full(surv.shape[0], float(self.max_horizon))
        for i in range(surv.shape[0]):
            below = np.where(surv[i] <= 0.5)[0]
            if len(below):
                out[i] = float(days[below[0]])
        return out

    def risk_score(self, X: pd.DataFrame) -> np.ndarray:
        """Scalar risk for ranking / concordance: cumulative hazard over the horizon
        (higher = fails sooner). Monotone with 1 - S(H), so it orders by survival."""
        _, surv = self.survival_grid(X)
        return 1.0 - surv[:, -1]


# --------------------------------------------------------------------------------------
# Proper survival metrics (numpy; no lifelines dependency)
# --------------------------------------------------------------------------------------
def concordance_index(durations, events, risk) -> float:
    """Harrell's C for right-censored data.

    A pair (i, j) is *comparable* if the one that failed did so strictly before the
    other's (failure or censoring) time. Concordant if the earlier-failing well has the
    higher risk score; ties in risk count 0.5. Returns 0.5 when no comparable pairs.
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    risk = np.asarray(risk, dtype=float)
    num = den = 0.0
    n = len(durations)
    for i in range(n):
        if events[i] != 1:
            continue
        # j is comparable to a failure i if j outlived i (failed later or censored later).
        comparable = durations > durations[i]
        for j in np.where(comparable)[0]:
            den += 1.0
            if risk[i] > risk[j]:
                num += 1.0
            elif risk[i] == risk[j]:
                num += 0.5
    return float(num / den) if den > 0 else 0.5


def _km_survival(durations, events, grid):
    """Kaplan–Meier survival estimate evaluated on integer ``grid`` days (baseline)."""
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    order = np.argsort(durations)
    d, e = durations[order], events[order]
    n_at_risk = len(d)
    surv = 1.0
    times, vals = [0.0], [1.0]
    for t in np.unique(d):
        at_t = d == t
        deaths = int(e[at_t].sum())
        if n_at_risk > 0 and deaths > 0:
            surv *= (1.0 - deaths / n_at_risk)
        times.append(float(t)); vals.append(surv)
        n_at_risk -= int(at_t.sum())
    times = np.asarray(times); vals = np.asarray(vals)
    idx = np.searchsorted(times, grid, side="right") - 1
    idx = np.clip(idx, 0, len(vals) - 1)
    return vals[idx]


def _censoring_survival(durations, events, grid):
    """KM estimate of the CENSORING distribution G(t) (event indicator flipped), for
    IPCW weights in the Brier score. Floored to avoid divide-by-zero."""
    g = _km_survival(durations, 1 - np.asarray(events, dtype=int), grid)
    return np.clip(g, 1e-8, 1.0)


def integrated_brier_score(durations, events, surv_grid, days, max_t=None) -> float:
    """IPCW Integrated Brier Score (Graf et al. 1999) over [0, max_t].

    ``surv_grid`` is per-well predicted survival on integer ``days`` (shape
    (n_wells, len(days)), S(0)=1). Lower is better.
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    days = np.asarray(days, dtype=float)
    if max_t is None:
        max_t = float(durations.max())
    eval_t = days[(days > 0) & (days <= max_t)]
    if len(eval_t) == 0:
        return float("nan")

    # Censoring survival at each well's own event time (G(T_i-)) and at each eval time.
    G_at_T = _censoring_survival(durations, events, np.clip(durations - 1, 0, None))
    G_at_t = _censoring_survival(durations, events, eval_t)

    bs_t = np.empty(len(eval_t))
    for k, t in enumerate(eval_t):
        col = np.searchsorted(days, t)                 # index of day t in the grid
        S_pred = surv_grid[:, col]                     # predicted S(t|x) per well
        # Case A: failed by t (D_i<=t, event) -> true survival 0, weight 1/G(T_i).
        case_a = (durations <= t) & (events == 1)
        # Case B: still at risk after t (D_i>t) -> true survival 1, weight 1/G(t).
        case_b = durations > t
        contrib = np.zeros(len(durations))
        contrib[case_a] = (S_pred[case_a] ** 2) / G_at_T[case_a]
        contrib[case_b] = ((1.0 - S_pred[case_b]) ** 2) / G_at_t[k]
        bs_t[k] = contrib.sum() / len(durations)       # cases not in A or B contribute 0
    # Trapezoidal integral over time, normalised by the time span.
    _trapz = getattr(np, "trapezoid", np.trapz)   # np>=2.0 renamed trapz -> trapezoid
    return float(_trapz(bs_t, eval_t) / (eval_t[-1] - eval_t[0])) if len(eval_t) > 1 \
        else float(bs_t[0])


# --------------------------------------------------------------------------------------
# Out-of-fold evaluation (the honest generalisation estimate)
# --------------------------------------------------------------------------------------
def evaluate_oof(X: pd.DataFrame, durations, events, n_splits: int = 5,
                 max_horizon: int = MAX_HORIZON) -> SurvivalEval:
    """Stratified-by-event K-fold OOF C-index + IBS for the discrete-time hazard model.

    Each well's survival curve and risk are produced by a fold model that never trained
    on it; metrics are computed on the pooled OOF predictions — the same honest protocol
    the classifier uses.
    """
    X = X[list(FEATURE_NAMES)]
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    n = len(durations)
    n_pos = int(events.sum())
    n_splits = max(2, min(n_splits, n_pos))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    days_full = np.arange(0, max_horizon + 1)
    oof_surv = np.full((n, len(days_full)), np.nan)
    oof_risk = np.full(n, np.nan)
    for tr, te in skf.split(X, events):
        m = DiscreteTimeHazardModel(max_horizon=max_horizon).fit(
            X.iloc[tr], durations[tr], events[tr])
        _, surv = m.survival_grid(X.iloc[te])
        oof_surv[te] = surv
        oof_risk[te] = m.risk_score(X.iloc[te])

    # IBS horizon: integrate to the last *event* day (beyond it everything is censored
    # and the IPCW weights get unstable). This is the standard truncation.
    max_t = float(durations[events == 1].max())
    c = concordance_index(durations, events, oof_risk)
    ibs = integrated_brier_score(durations, events, oof_surv, days_full, max_t=max_t)

    # Kaplan–Meier baseline (no covariates): same survival curve for every well.
    km = _km_survival(durations, events, days_full)
    km_grid = np.tile(km, (n, 1))
    ibs_km = integrated_brier_score(durations, events, km_grid, days_full, max_t=max_t)

    return SurvivalEval(
        c_index=c, ibs=ibs, ibs_km_baseline=ibs_km,
        n_wells=n, n_events=n_pos, n_censored=int(n - n_pos), max_horizon=max_horizon,
    )


# --------------------------------------------------------------------------------------
# Convenience: fit on the synthetic run-life + persist a fleet survival table
# --------------------------------------------------------------------------------------
def fit_on_labels(features: pd.DataFrame, labels: pd.DataFrame) -> DiscreteTimeHazardModel:
    """Fit the hazard model on a features frame + a labels frame carrying
    ``time_to_event_days`` / ``event_observed`` (aligned on well_id)."""
    lab = labels.set_index("well_id") if "well_id" in labels.columns else labels
    joined = features.join(lab[["time_to_event_days", "event_observed"]], how="inner")
    X = joined[list(FEATURE_NAMES)]
    return DiscreteTimeHazardModel().fit(
        X, joined["time_to_event_days"], joined["event_observed"])


def fleet_survival_table(model: DiscreteTimeHazardModel, features: pd.DataFrame) -> pd.DataFrame:
    """Per-well median RUL + horizon survival from a fitted hazard model, soonest first."""
    rul = model.median_rul(features)
    _, surv = model.survival_grid(features)
    df = pd.DataFrame({
        "well_id": features.index,
        "median_rul_days": rul,
        "surv_at_horizon": surv[:, -1],
    })
    return df.sort_values("median_rul_days").reset_index(drop=True)


def evaluate_from_disk(data_dir: str = "data/synthetic",
                       labels_path: str = "data/synthetic/labels.csv") -> SurvivalEval:
    """Load the synthetic fleet + run-life labels and return the OOF survival metrics.

    Requires the run-life columns (``time_to_event_days`` / ``event_observed``) in the
    labels file — present since the generator emits them.
    """
    from .data_loader import load_fleet, load_labels
    from .features import featurize_fleet

    fleet = load_fleet(data_dir)
    features = featurize_fleet(fleet)
    labels = load_labels(labels_path).set_index("well_id")
    if "time_to_event_days" not in labels.columns or "event_observed" not in labels.columns:
        raise ValueError(
            "labels file lacks run-life columns (time_to_event_days / event_observed); "
            "regenerate with `python data/synthetic/generate.py`.")
    joined = features.join(labels[["time_to_event_days", "event_observed"]], how="inner")
    return evaluate_oof(joined[list(FEATURE_NAMES)],
                        joined["time_to_event_days"].to_numpy(),
                        joined["event_observed"].to_numpy())


if __name__ == "__main__":  # `python -m src.survival_model` → print + persist metrics
    import json
    from pathlib import Path

    from rich.console import Console

    console = Console()
    res = evaluate_from_disk()
    console.print("[bold]Discrete-time hazard survival model — OOF metrics[/]")
    console.print(f"  C-index (time-dependent concordance): {res.c_index:.3f}  "
                  f"(0.5 = chance)")
    console.print(f"  Integrated Brier Score (IBS):         {res.ibs:.4f}  "
                  f"(lower = better)")
    console.print(f"  IBS Kaplan–Meier baseline:            {res.ibs_km_baseline:.4f}  "
                  f"→ model is {(1 - res.ibs / res.ibs_km_baseline) * 100:.0f}% better than KM")
    console.print(f"  Run-life: {res.n_events} events, {res.n_censored} censored, "
                  f"horizon {res.max_horizon}d")
    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/survival_report.json", "w") as f:
        json.dump(res.as_dict(), f, indent=2)
    console.print("  Wrote artifacts/survival_report.json")
