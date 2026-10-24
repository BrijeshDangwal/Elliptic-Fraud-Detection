"""Model factories. Each returns an unfitted estimator with ``fit`` and a score method.

Class imbalance is handled with class weights rather than resampling, so the model sees
the real data distribution and the decision threshold can be tuned explicitly.
"""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def logistic_regression(seed: int = 0):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )


def random_forest(seed: int = 0):
    # Same family as the strongest baseline in Weber et al. (2019).
    return RandomForestClassifier(
        n_estimators=300, max_features=50, class_weight="balanced_subsample",
        n_jobs=-1, random_state=seed,
    )


def lightgbm(seed: int = 0, pos_weight: float = 1.0):
    return LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.5, scale_pos_weight=pos_weight, random_state=seed, n_jobs=-1,
        verbose=-1,
    )


def isolation_forest(seed: int = 0):
    return IsolationForest(n_estimators=300, contamination="auto", random_state=seed, n_jobs=-1)


SUPERVISED = {
    "Logistic regression": logistic_regression,
    "Random forest": random_forest,
    "LightGBM": lightgbm,
}


def score(model, x) -> np.ndarray:
    """Higher means more likely illicit, for every model type."""
    if isinstance(model, IsolationForest):
        return -model.score_samples(x)
    return model.predict_proba(x)[:, 1]
