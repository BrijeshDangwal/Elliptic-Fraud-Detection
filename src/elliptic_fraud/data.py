"""Loading and validating the Elliptic Bitcoin transaction dataset.

The Kaggle release contains three CSV files:

* ``elliptic_txs_features.csv``: no header. Column 0 is the transaction id, column 1 is
  the time step (1 to 49), columns 2 to 94 are 93 local features of the transaction and
  columns 95 to 166 are 72 features aggregated over its one hop neighbourhood.
* ``elliptic_txs_classes.csv``: ``txId,class`` where class is ``"1"`` (illicit),
  ``"2"`` (licit) or ``"unknown"``.
* ``elliptic_txs_edgelist.csv``: ``txId1,txId2``, directed payment flows between
  transactions. Edges only connect transactions inside the same time step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES_FILE = "elliptic_txs_features.csv"
CLASSES_FILE = "elliptic_txs_classes.csv"
EDGES_FILE = "elliptic_txs_edgelist.csv"

N_LOCAL = 93
N_AGG = 72

# Label encoding used everywhere in this project.
ILLICIT, LICIT, UNKNOWN = 1, 0, -1


@dataclass
class EllipticData:
    """All nodes (labelled and unlabelled) with features, labels and the edge list."""

    nodes: pd.DataFrame  # columns: txId, time_step, label, local_*, agg_*
    edges: pd.DataFrame  # columns: src, dst as row positions into ``nodes``

    @property
    def local_cols(self) -> list[str]:
        return [c for c in self.nodes.columns if c.startswith("local_")]

    @property
    def agg_cols(self) -> list[str]:
        return [c for c in self.nodes.columns if c.startswith("agg_")]


def find_file(data_dir: Path, name: str) -> Path:
    """Find a dataset file anywhere under ``data_dir`` (Kaggle zips nest a folder)."""
    direct = data_dir / name
    if direct.exists():
        return direct
    matches = sorted(data_dir.rglob(name))
    if not matches:
        raise FileNotFoundError(
            f"{name} not found under {data_dir}. Download the dataset from "
            "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set and unzip it there."
        )
    return matches[0]


def load_elliptic(data_dir: str | Path) -> EllipticData:
    data_dir = Path(data_dir)
    feats = pd.read_csv(find_file(data_dir, FEATURES_FILE), header=None)
    if feats.shape[1] != 2 + N_LOCAL + N_AGG:
        raise ValueError(f"expected {2 + N_LOCAL + N_AGG} feature columns, got {feats.shape[1]}")
    feats.columns = (
        ["txId", "time_step"]
        + [f"local_{i:02d}" for i in range(N_LOCAL)]
        + [f"agg_{i:02d}" for i in range(N_AGG)]
    )

    classes = pd.read_csv(find_file(data_dir, CLASSES_FILE), dtype={"class": str})
    label_map = {"1": ILLICIT, "2": LICIT, "unknown": UNKNOWN}
    unexpected = set(classes["class"].unique()) - set(label_map)
    if unexpected:
        raise ValueError(f"unexpected class values: {unexpected}")
    classes["label"] = classes["class"].map(label_map).astype(np.int8)

    nodes = feats.merge(classes[["txId", "label"]], on="txId", how="left", validate="one_to_one")
    if nodes["label"].isna().any():
        raise ValueError("some transactions have no class row")
    nodes["label"] = nodes["label"].astype(np.int8)
    nodes["time_step"] = nodes["time_step"].astype(np.int16)
    nodes = nodes.reset_index(drop=True)

    edges_raw = pd.read_csv(find_file(data_dir, EDGES_FILE))
    edges_raw.columns = ["txId1", "txId2"]
    pos = pd.Series(np.arange(len(nodes)), index=nodes["txId"].to_numpy())
    src = pos.reindex(edges_raw["txId1"].to_numpy()).to_numpy()
    dst = pos.reindex(edges_raw["txId2"].to_numpy()).to_numpy()
    if np.isnan(src).any() or np.isnan(dst).any():
        raise ValueError("edge list references unknown transaction ids")
    edges = pd.DataFrame({"src": src.astype(np.int64), "dst": dst.astype(np.int64)})

    return EllipticData(nodes=nodes, edges=edges)


def summarise(data: EllipticData) -> dict:
    """Headline counts used in the README and as a sanity check after loading."""
    n = data.nodes
    steps = n["time_step"].to_numpy()
    same_step = steps[data.edges["src"].to_numpy()] == steps[data.edges["dst"].to_numpy()]
    return {
        "transactions": int(len(n)),
        "edges": int(len(data.edges)),
        "illicit": int((n["label"] == ILLICIT).sum()),
        "licit": int((n["label"] == LICIT).sum()),
        "unknown": int((n["label"] == UNKNOWN).sum()),
        "time_steps": int(n["time_step"].nunique()),
        "edges_within_same_step": float(same_step.mean()) if len(same_step) else 1.0,
        "illicit_share_of_labelled": float(
            (n["label"] == ILLICIT).sum() / max(1, (n["label"] != UNKNOWN).sum())
        ),
    }
