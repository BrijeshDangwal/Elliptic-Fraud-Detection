"""A small synthetic dataset in the exact Elliptic file format, for tests and CI.

It is not meant to be realistic: illicit transactions get a shifted feature mean and
tend to link to each other, so models have something learnable. Real results in the
README come only from the real dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import CLASSES_FILE, EDGES_FILE, FEATURES_FILE, N_AGG, N_LOCAL


def write_synthetic(out_dir: str | Path, n_per_step: int = 60, steps: int = 49, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, classes, edges = [], [], []
    tx = 1000
    for t in range(1, steps + 1):
        ids = np.arange(tx, tx + n_per_step)
        tx += n_per_step
        cls = rng.choice(["1", "2", "unknown"], size=n_per_step, p=[0.1, 0.3, 0.6])
        shift = np.where(cls == "1", 1.5, 0.0)[:, None]
        x = rng.normal(size=(n_per_step, N_LOCAL + N_AGG)) + shift * (rng.random(N_LOCAL + N_AGG) > 0.7)
        for i, c, f in zip(ids, cls, x):
            rows.append([i, t, *f])
            classes.append([i, c])
        for _ in range(n_per_step):
            a, b = rng.choice(n_per_step, size=2, replace=False)
            if cls[a] == "1" and rng.random() < 0.5:
                pool = np.flatnonzero(cls == "1")
                b = rng.choice(pool) if len(pool) > 1 and rng.choice(pool) != a else b
            edges.append([ids[a], ids[b]])

    pd.DataFrame(rows).to_csv(out_dir / FEATURES_FILE, header=False, index=False)
    pd.DataFrame(classes, columns=["txId", "class"]).to_csv(out_dir / CLASSES_FILE, index=False)
    pd.DataFrame(edges, columns=["txId1", "txId2"]).to_csv(out_dir / EDGES_FILE, index=False)
    return out_dir
