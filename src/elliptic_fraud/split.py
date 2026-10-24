"""Temporal splits. Fraud patterns change over time, so we never shuffle across steps.

Standard Elliptic protocol (Weber et al., 2019): train on time steps 1 to 34 and test
on 35 to 49. For threshold tuning we hold out the last training steps (30 to 34) as a
validation period and fit on 1 to 29, so the test period is never touched while tuning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import UNKNOWN

TRAIN_END = 34
VAL_START = 30


@dataclass(frozen=True)
class Split:
    fit: np.ndarray  # labelled rows used to fit models during tuning (steps 1..29)
    val: np.ndarray  # labelled rows used to pick the decision threshold (steps 30..34)
    train: np.ndarray  # labelled rows for the final fit (steps 1..34)
    test: np.ndarray  # labelled rows for evaluation (steps 35..49)
    train_all: np.ndarray  # all rows, labelled or not, in steps 1..34 (for unsupervised models)


def temporal_split(nodes: pd.DataFrame, train_end: int = TRAIN_END, val_start: int = VAL_START) -> Split:
    step = nodes["time_step"].to_numpy()
    labelled = nodes["label"].to_numpy() != UNKNOWN
    idx = np.arange(len(nodes))
    return Split(
        fit=idx[labelled & (step < val_start)],
        val=idx[labelled & (step >= val_start) & (step <= train_end)],
        train=idx[labelled & (step <= train_end)],
        test=idx[labelled & (step > train_end)],
        train_all=idx[step <= train_end],
    )
