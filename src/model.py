"""XGBoost wrapper: train, save, load, predict, feature contributions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .features import FEATURE_NAMES


@dataclass
class TrainResult:
    auroc: float
    precision_at_top10: float
    recall_at_top10: float
    feature_importance: dict[str, float]


class ESPRiskModel:
    """Calibrated gradient-boosted classifier for 30-day failure risk."""

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
        self.model = XGBClassifier(**defaults)
        self.feature_names = FEATURE_NAMES

    def fit(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.3) -> TrainResult:
        X = X[self.feature_names]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        self.model.fit(X_tr, y_tr)
        probs = self.model.predict_proba(X_te)[:, 1]
        auroc = float(roc_auc_score(y_te, probs))

        order = np.argsort(-probs)
        top_k = max(int(0.1 * len(y_te)), 1)
        top_idx = order[:top_k]
        top_truth = y_te.iloc[top_idx]
        prec = float(precision_score(top_truth, np.ones(top_k), zero_division=0))
        rec = float(top_truth.sum() / max(y_te.sum(), 1))

        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        return TrainResult(auroc=auroc, precision_at_top10=prec, recall_at_top10=rec,
                          feature_importance={k: float(v) for k, v in importance.items()})

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_names])[:, 1]

    def feature_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-well feature contributions via SHAP-style pred_contribs from XGBoost.

        Returns a DataFrame indexed by well, columns = feature names + 'bias'.
        """
        import xgboost as xgb
        dmat = xgb.DMatrix(X[self.feature_names])
        contribs = self.model.get_booster().predict(dmat, pred_contribs=True)
        cols = self.feature_names + ["bias"]
        return pd.DataFrame(contribs, index=X.index, columns=cols)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.feature_names}, path)

    @classmethod
    def load(cls, path: str | Path) -> "ESPRiskModel":
        bundle = joblib.load(path)
        obj = cls()
        obj.model = bundle["model"]
        obj.feature_names = bundle["features"]
        return obj
