"""Experimental sequence model: does end-to-end temporal modelling beat features?

The shipped pipeline collapses each well's 60-day series into a single engineered
feature row and scores it with gradient-boosted trees (see ``src/features.py`` +
``src/model.py``). That is a strong, interpretable baseline. This module exists to
*honestly test* whether learning directly from the raw multivariate time series
buys us anything — a baseline-vs-sequence comparison.

It implements a small **Temporal Convolutional Network (Temporal-CNN)** binary
classifier over the 5 SCADA channels (bfpd, intake pressure, motor temp, motor
amps, runtime %). On a 100-well synthetic set with ~12% positives it will almost
certainly *not* beat the feature-based XGBoost model (too little data for a deep
net) — and demonstrating that you know that, and measured it, is the point.

IMPORTANT — this is opt-in / experimental and is deliberately NOT imported by the
Streamlit app, the ranker, or any deployed code path. ``torch`` is an optional
dependency: this module imports cleanly even when torch is absent. The clear
``RuntimeError`` is raised only when you actually call into the model, so importing
the module can never break the live app.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:                       # torch is optional; absence must not break import
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except Exception:          # pragma: no cover - exercised only without torch
    torch = None           # type: ignore
    nn = None              # type: ignore
    _TORCH_AVAILABLE = False


_TORCH_MISSING_MSG = (
    "sequence_model requires PyTorch, which is not installed. This is an "
    "experimental, opt-in baseline-vs-sequence comparison and is intentionally "
    "kept off the deployed path. Install it explicitly with:  pip install torch"
)


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise RuntimeError(_TORCH_MISSING_MSG)


SCADA_CHANNELS = ["bfpd", "intake_pressure_psi", "motor_temp_f", "motor_amps", "runtime_pct"]


@dataclass
class SequenceTrainResult:
    auroc: float
    n_epochs: int
    n_params: int
    note: str = ("Experimental Temporal-CNN baseline. Expect it to UNDER-perform "
                 "the feature-based XGBoost model on this small, imbalanced dataset.")


def _build_tcn(n_channels: int, hidden: int = 16):
    """A deliberately small Temporal-CNN: too few params to overfit 100 wells badly,
    but enough to learn a degradation shape. Built lazily so torch stays optional."""
    _require_torch()

    class TemporalCNN(nn.Module):
        def __init__(self, n_ch: int, h: int):
            super().__init__()
            # Causal-ish 1D conv stack with increasing dilation over the time axis.
            self.net = nn.Sequential(
                nn.Conv1d(n_ch, h, kernel_size=3, padding=1, dilation=1),
                nn.ReLU(),
                nn.Conv1d(h, h, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(),
                nn.Conv1d(h, h, kernel_size=3, padding=4, dilation=4),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),     # global temporal pooling -> [B, h, 1]
            )
            self.head = nn.Linear(h, 1)

        def forward(self, x):                # x: [B, channels, time]
            z = self.net(x).squeeze(-1)      # [B, h]
            return self.head(z).squeeze(-1)  # [B] logits

    return TemporalCNN(n_channels, hidden)


def fleet_to_tensor(fleet: dict, channels=None, n_days: int = 60):
    """Stack a {well_id: scada_df} fleet into a [n_wells, channels, time] tensor.

    Each channel is z-normalised across the fleet. Series shorter than ``n_days``
    are left-padded with their first value; longer ones are tail-truncated.
    """
    _require_torch()
    channels = channels or SCADA_CHANNELS
    well_ids = list(fleet.keys())
    arr = np.zeros((len(well_ids), len(channels), n_days), dtype=np.float32)
    for i, wid in enumerate(well_ids):
        df = fleet[wid]
        for c, ch in enumerate(channels):
            series = df[ch].to_numpy(dtype=np.float32)
            if len(series) >= n_days:
                series = series[-n_days:]
            else:
                pad = np.full(n_days - len(series), series[0] if len(series) else 0.0,
                              dtype=np.float32)
                series = np.concatenate([pad, series])
            arr[i, c, :] = series
    # Per-channel z-norm across the fleet.
    mu = arr.mean(axis=(0, 2), keepdims=True)
    sd = arr.std(axis=(0, 2), keepdims=True) + 1e-6
    arr = (arr - mu) / sd
    return torch.from_numpy(arr), well_ids


def train_sequence_model(
    fleet: dict,
    labels: dict,
    n_epochs: int = 40,
    lr: float = 1e-3,
    test_frac: float = 0.3,
    seed: int = 42,
) -> SequenceTrainResult:
    """Train + evaluate the experimental Temporal-CNN. Raises RuntimeError if torch
    is unavailable. This is a depth/credibility artifact, not a production path.

    Args:
        fleet: {well_id: scada DataFrame with SCADA_CHANNELS columns}.
        labels: {well_id: 0/1 failed-within-30d}.
    """
    _require_torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    X, well_ids = fleet_to_tensor(fleet)
    y = torch.tensor([float(labels[w]) for w in well_ids], dtype=torch.float32)

    idx = rng.permutation(len(well_ids))
    n_test = max(1, int(test_frac * len(well_ids)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    model = _build_tcn(n_channels=X.shape[1])
    n_pos = float(y[train_idx].sum())
    n_neg = float(len(train_idx) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)])  # class imbalance handling
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        logits = model(X[train_idx])
        loss = loss_fn(logits, y[train_idx])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X[test_idx])).numpy()
    auroc = _auroc(y[test_idx].numpy(), probs)
    n_params = sum(p.numel() for p in model.parameters())
    return SequenceTrainResult(auroc=float(auroc), n_epochs=n_epochs, n_params=int(n_params))


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based AUROC (Mann–Whitney U), numpy-only so we don't pull in sklearn."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # Average ties.
    _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    tie_mean = np.array([ranks[y_score == v].mean() for v in np.unique(y_score)])
    ranks = tie_mean[inv]
    rank_sum_pos = ranks[y_true == 1].sum()
    auc = (rank_sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)
