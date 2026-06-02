"""XGBoost wrapper: train, save, load, predict, feature contributions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from .features import FEATURE_NAMES


@dataclass
class TrainResult:
    auroc: float                       # single held-out split (high variance on small data)
    auroc_cv_mean: float               # stratified K-fold mean (the number to trust)
    auroc_cv_std: float
    precision_at_top10pct: float
    recall_at_top10pct: float
    n_test: int
    n_test_positives: int
    calibrated: bool
    feature_importance: dict[str, float]


class ESPRiskModel:
    """Gradient-boosted (XGBoost) classifier for 30-day failure risk.

    NOTE: outputs raw XGBoost probabilities (not calibrated). Wrap in
    CalibratedClassifierCV if calibrated probabilities are required.
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
        # Optional probability calibrator fit on a held-out slice (cv='prefit').
        # Kept SEPARATE from self.model so the raw booster stays available for
        # Tree SHAP contributions / feature importance.
        self.calibrator: CalibratedClassifierCV | None = None
        self.feature_names = FEATURE_NAMES

    @staticmethod
    def _pos_weight(y) -> float:
        y = np.asarray(y)
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        return (n_neg / n_pos) if n_pos > 0 else 1.0

    def _cv_auroc(self, X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
        """Stratified K-fold AUROC (mean, std) — the honest metric on small,
        imbalanced data. n_splits is capped by the positive count."""
        n_pos = int(y.sum())
        n_splits = max(2, min(5, n_pos))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        aucs = []
        for tr, te in skf.split(X, y):
            m = XGBClassifier(**{**self._xgb_kwargs,
                                 "scale_pos_weight": self._pos_weight(y.iloc[tr])})
            m.fit(X.iloc[tr], y.iloc[tr])
            p = m.predict_proba(X.iloc[te])[:, 1]
            if len(np.unique(y.iloc[te])) > 1:   # AUROC undefined on single-class fold
                aucs.append(roc_auc_score(y.iloc[te], p))
        if not aucs:
            return float("nan"), float("nan")
        return float(np.mean(aucs)), float(np.std(aucs))

    def fit(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.3,
            calibrate: bool = True) -> TrainResult:
        X = X[self.feature_names]

        # Class-imbalance handling via scale_pos_weight (≈ n_neg/n_pos).
        self.model.set_params(scale_pos_weight=self._pos_weight(y))

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        self.model.set_params(scale_pos_weight=self._pos_weight(y_tr))
        self.model.fit(X_tr, y_tr)

        # Probability calibration on a held-out slice of the TRAIN set (cv='prefit'),
        # guarded for the small-sample regime so training never crashes.
        self.calibrator = None
        if calibrate:
            try:
                X_fit, X_cal, y_fit, y_cal = train_test_split(
                    X_tr, y_tr, test_size=0.33, random_state=42, stratify=y_tr
                )
                if int(y_cal.sum()) >= 2 and int((1 - y_cal).sum()) >= 2:
                    base = XGBClassifier(**{**self._xgb_kwargs,
                                            "scale_pos_weight": self._pos_weight(y_fit)})
                    base.fit(X_fit, y_fit)
                    cal = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
                    cal.fit(X_cal, y_cal)
                    self.calibrator = cal
            except Exception:
                self.calibrator = None  # fall back to raw probabilities

        probs = self.predict_proba(X_te)
        auroc = float(roc_auc_score(y_te, probs)) if len(np.unique(y_te)) > 1 else float("nan")
        cv_mean, cv_std = self._cv_auroc(X, y)

        order = np.argsort(-probs)
        top_k = max(int(0.1 * len(y_te)), 1)
        top_idx = order[:top_k]
        top_truth = y_te.iloc[top_idx]
        prec = float(precision_score(top_truth, np.ones(top_k), zero_division=0))
        rec = float(top_truth.sum() / max(y_te.sum(), 1))

        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        return TrainResult(
            auroc=auroc, auroc_cv_mean=cv_mean, auroc_cv_std=cv_std,
            precision_at_top10pct=prec, recall_at_top10pct=rec,
            n_test=int(len(y_te)), n_test_positives=int(y_te.sum()),
            calibrated=self.calibrator is not None,
            feature_importance={k: float(v) for k, v in importance.items()},
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_names]
        if self.calibrator is not None:
            return self.calibrator.predict_proba(X)[:, 1]
        return self.model.predict_proba(X)[:, 1]

    def feature_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-well feature contributions via XGBoost pred_contribs (Tree SHAP values).

        Returns a DataFrame indexed by well, columns = feature names + 'bias'.
        """
        import xgboost as xgb
        dmat = xgb.DMatrix(X[self.feature_names])
        contribs = self.model.get_booster().predict(dmat, pred_contribs=True)
        cols = self.feature_names + ["bias"]
        return pd.DataFrame(contribs, index=X.index, columns=cols)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "calibrator": self.calibrator,
                     "features": self.feature_names}, path)

    @classmethod
    def load(cls, path: str | Path) -> "ESPRiskModel":
        bundle = joblib.load(path)
        obj = cls()
        obj.model = bundle["model"]
        obj.calibrator = bundle.get("calibrator")  # tolerate older artifacts
        obj.feature_names = bundle["features"]
        return obj
