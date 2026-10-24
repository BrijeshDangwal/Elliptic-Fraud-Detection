"""Metrics focused on the illicit class, which is what a risk team actually acts on."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def illicit_metrics(y: np.ndarray, s: np.ndarray, threshold: float) -> dict:
    pred = (s >= threshold).astype(int)
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, s)) if y.sum() else float("nan"),
        "flagged": int(pred.sum()),
        "threshold": float(threshold),
    }


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Threshold that maximises illicit F1. Only ever called on the validation period."""
    p, r, t = precision_recall_curve(y, s)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)[:-1]
    return float(t[int(np.argmax(f1))]) if len(t) else 0.5


def per_step(steps: np.ndarray, y: np.ndarray, s: np.ndarray, threshold: float, k: int) -> pd.DataFrame:
    """Per time step metrics, including precision at k: of the k highest scored
    transactions in a step (an analyst's review queue), how many are illicit."""
    rows = []
    for t in np.unique(steps):
        m = steps == t
        yt, st = y[m], s[m]
        pred = (st >= threshold).astype(int)
        top = np.argsort(-st)[: min(k, len(st))]
        rows.append({
            "time_step": int(t),
            "illicit": int(yt.sum()),
            "labelled": int(m.sum()),
            "recall": float(recall_score(yt, pred, zero_division=0)) if yt.sum() else np.nan,
            "precision": float(precision_score(yt, pred, zero_division=0)) if pred.sum() else np.nan,
            "f1": float(f1_score(yt, pred, zero_division=0)) if yt.sum() else np.nan,
            f"precision_at_{k}": float(yt[top].mean()) if len(top) else np.nan,
        })
    return pd.DataFrame(rows)
