"""XGBoost wrapper: train, save, load, predict, feature contributions.

Validation methodology (read this before quoting a number):

- The data is a **cross-sectional snapshot** — one engineered feature row per well
  at a fixed observation date, labelled "failed within the next 30 days." There is
  no within-well time ordering across rows, so the honest generalisation estimate
  is **stratified K-fold CV** (capped by the positive count), and all reported
  metrics — AUROC, precision@k, recall@k, Brier — come from **out-of-fold (OOF)**
  predictions, i.e. each well is scored by a model that never saw it. When this
  pipeline later ingests *rolling* observation windows per well (multiple snapshots
  over time), the correct upgrade is forward-chaining / grouped-by-well splits so a
  well's adjacent windows can't straddle train and validation — see README.

- The **shipped** model and the **reported** metrics use the *same* procedure
  (class-weighted XGBoost + Platt calibration), so the headline number actually
  describes what's on disk. The shipped raw booster is trained on all data except a
  stratified calibration hold-out; the calibrator wraps *that* booster (cv='prefit'),
  so the calibrated probability is a monotone transform of exactly the score Tree
  SHAP decomposes — explanation and prediction reconcile.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from .features import FEATURE_NAMES


def _prefit_calibrator(estimator, method: str = "sigmoid") -> CalibratedClassifierCV:
    """Build a CalibratedClassifierCV that calibrates an ALREADY-FIT estimator
    without refitting it. sklearn >=1.6 does this via FrozenEstimator (the legacy
    ``cv='prefit'`` was removed in 1.8); older sklearn uses ``cv='prefit'``."""
    try:
        from sklearn.frozen import FrozenEstimator
        return CalibratedClassifierCV(FrozenEstimator(estimator), method=method)
    except ImportError:                       # sklearn < 1.6
        return CalibratedClassifierCV(estimator, method=method, cv="prefit")


@dataclass
class TrainResult:
    auroc_cv_mean: float               # out-of-fold AUROC (the number to trust)
    auroc_cv_std: float
    precision_at_top10pct: float       # OOF: of the top-10% flagged, fraction that fail
    recall_at_top10pct: float          # OOF: of all failures, fraction in the top-10%
    n_flagged_top10pct: int            # how many wells "top 10%" is, on this fleet
    brier: float                       # OOF Brier score (lower = better calibrated)
    n_wells: int
    n_positives: int
    calibrated: bool
    feature_importance: dict[str, float]
    reliability: list[dict] = field(default_factory=list)  # OOF calibration curve bins


class ESPRiskModel:
    """Gradient-boosted (XGBoost) classifier for 30-day failure risk.

    ``predict_proba`` returns **Platt-calibrated** probabilities when a calibrator
    is present (the default after ``fit``); the calibrator wraps the raw booster so
    ``feature_contributions`` (Tree SHAP on that same booster) stays consistent with
    the score shown to the user.
    """

    def __init__(self, **xgb_kwargs):
        defaults = dict(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
        )
        defaults.update(xgb_kwargs)
        self._xgb_kwargs = defaults
        self.model = XGBClassifier(**defaults)
        self.calibrator: CalibratedClassifierCV | None = None
        self.feature_names = FEATURE_NAMES
        # Stored at fit time for monitoring / display (tolerated-absent on old artifacts).
        self.reference_scores: np.ndarray | None = None   # training-time score distribution (PSI reference)
        self.reliability: list[dict] = []                 # OOF calibration curve

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _pos_weight(y) -> float:
        y = np.asarray(y)
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        return (n_neg / n_pos) if n_pos > 0 else 1.0

    def _new_xgb(self, scale_pos_weight: float) -> XGBClassifier:
        return XGBClassifier(**{**self._xgb_kwargs, "scale_pos_weight": scale_pos_weight})

    def _cross_validate(self, X: pd.DataFrame, y: pd.Series):
        """Stratified K-fold producing OOF predictions + per-fold AUROC (mean, std).

        Returns (auroc_mean, auroc_std, oof_probs) where oof_probs[i] is the
        prediction for row i from a fold model that did NOT train on it.
        """
        n_pos = int(y.sum())
        n_splits = max(2, min(5, n_pos))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof = np.full(len(y), np.nan)
        aucs = []
        yv = y.to_numpy()
        for tr, te in skf.split(X, y):
            m = self._new_xgb(self._pos_weight(y.iloc[tr]))
            m.fit(X.iloc[tr], y.iloc[tr])
            p = m.predict_proba(X.iloc[te])[:, 1]
            oof[te] = p
            if len(np.unique(yv[te])) > 1:    # AUROC undefined on single-class fold
                aucs.append(roc_auc_score(yv[te], p))
        mean = float(np.mean(aucs)) if aucs else float("nan")
        std = float(np.std(aucs)) if aucs else float("nan")
        return mean, std, oof

    @staticmethod
    def _precision_recall_at_k(y, p, frac: float = 0.1):
        """Precision@k and recall@k for an alert list of the top ``frac`` by score."""
        y = np.asarray(y); p = np.asarray(p)
        mask = ~np.isnan(p)
        y, p = y[mask], p[mask]
        k = max(int(frac * len(y)), 1)
        idx = np.argsort(-p)[:k]
        tp = float(y[idx].sum())
        prec = tp / k
        rec = tp / max(float(y.sum()), 1.0)
        return float(prec), float(rec), int(k)

    @staticmethod
    def _reliability(y, p, n_bins: int = 10):
        """Reliability-diagram bins (mean predicted vs observed frequency) + Brier."""
        y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
        mask = ~np.isnan(p)
        y, p = y[mask], p[mask]
        if len(y) == 0:
            return [], float("nan")
        brier = float(np.mean((p - y) ** 2))
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bins: list[dict] = []
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
            if m.sum() == 0:
                continue
            bins.append({"mean_pred": float(p[m].mean()),
                         "obs_freq": float(y[m].mean()),
                         "count": int(m.sum())})
        return bins, brier

    # ---- fit --------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series, calibrate: bool = True,
            cal_size: float = 0.25) -> TrainResult:
        X = X[self.feature_names]
        y = y.astype(int)

        # 1) Honest generalisation metrics from out-of-fold predictions.
        cv_mean, cv_std, oof = self._cross_validate(X, y)
        prec, rec, k = self._precision_recall_at_k(y, oof, frac=0.1)
        reliability, brier = self._reliability(y, oof)

        # 2) Build the SHIPPED artifact. Train the raw booster on all data except a
        #    stratified calibration hold-out, then Platt-calibrate THAT booster
        #    (cv='prefit') so SHAP and the calibrated score reconcile.
        self.calibrator = None
        calibrated_ok = False
        if calibrate:
            try:
                X_fit, X_cal, y_fit, y_cal = train_test_split(
                    X, y, test_size=cal_size, random_state=42, stratify=y
                )
                if int(y_cal.sum()) >= 2 and int((1 - y_cal).sum()) >= 2:
                    self.model = self._new_xgb(self._pos_weight(y_fit))
                    self.model.fit(X_fit, y_fit)
                    cal = _prefit_calibrator(self.model, method="sigmoid")
                    cal.fit(X_cal, y_cal)
                    self.calibrator = cal
                    calibrated_ok = True
            except Exception:
                self.calibrator = None  # fall back to raw probabilities

        if not calibrated_ok:
            # No calibration: use ALL data for the ranking model (nothing held out).
            self.model = self._new_xgb(self._pos_weight(y))
            self.model.fit(X, y)

        # 3) Reference score distribution for drift monitoring (PSI baseline).
        self.reference_scores = self.predict_proba(X)
        self.reliability = reliability

        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        return TrainResult(
            auroc_cv_mean=cv_mean, auroc_cv_std=cv_std,
            precision_at_top10pct=prec, recall_at_top10pct=rec, n_flagged_top10pct=k,
            brier=brier, n_wells=int(len(y)), n_positives=int(y.sum()),
            calibrated=self.calibrator is not None,
            feature_importance={kk: float(v) for kk, v in importance.items()},
            reliability=reliability,
        )

    # ---- inference --------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_names]
        if self.calibrator is not None:
            return self.calibrator.predict_proba(X)[:, 1]
        return self.model.predict_proba(X)[:, 1]

    def feature_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-well feature contributions via XGBoost pred_contribs (Tree SHAP values),
        in log-odds (margin) space on the raw booster.

        The shipped calibrated probability is a monotone (sigmoid) transform of this
        booster's margin, so the SIGN and RANK of each driver carry directly over to
        the displayed risk score (magnitudes are log-odds, not probability deltas).
        Returns a DataFrame indexed by well, columns = feature names + 'bias'.
        """
        import xgboost as xgb
        dmat = xgb.DMatrix(X[self.feature_names])
        contribs = self.model.get_booster().predict(dmat, pred_contribs=True)
        cols = self.feature_names + ["bias"]
        return pd.DataFrame(contribs, index=X.index, columns=cols)

    # ---- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "calibrator": self.calibrator,
                     "features": self.feature_names,
                     "reference_scores": self.reference_scores,
                     "reliability": self.reliability}, path)

    @classmethod
    def load(cls, path: str | Path) -> "ESPRiskModel":
        bundle = joblib.load(path)
        obj = cls()
        obj.model = bundle["model"]
        obj.calibrator = bundle.get("calibrator")  # tolerate older artifacts
        obj.feature_names = bundle["features"]
        obj.reference_scores = bundle.get("reference_scores")
        obj.reliability = bundle.get("reliability", [])
        return obj
