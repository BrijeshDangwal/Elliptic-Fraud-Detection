"""Graph features: neighbourhood propagation in the style of SGC (Wu et al., 2019).

Elliptic's 72 aggregated features already summarise each transaction's one hop
neighbourhood. Here we add two things the raw features do not contain:

* degree features (how many transactions feed into and out of this one), and
* propagated local features: the local features averaged over the k hop neighbourhood
  using the symmetric normalised adjacency with self loops,
  ``A_hat = D^-1/2 (A + I) D^-1/2``, applied ``k`` times.

This is the feature side of a graph convolutional network without the learned weights,
so any tabular model can use graph structure. Unlabelled transactions take part in the
propagation (they are real neighbours), but their labels are never used, and because
Elliptic edges only connect transactions within the same time step, no information
flows across the temporal train and test boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import EllipticData


def adjacency(data: EllipticData, symmetric: bool = True) -> sp.csr_matrix:
    n = len(data.nodes)
    src = data.edges["src"].to_numpy()
    dst = data.edges["dst"].to_numpy()
    a = sp.coo_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(n, n)).tocsr()
    if symmetric:
        a = ((a + a.T) > 0).astype(np.float32)
    return a


def normalised_adjacency(a: sp.csr_matrix) -> sp.csr_matrix:
    a_self = a + sp.identity(a.shape[0], dtype=np.float32, format="csr")
    deg = np.asarray(a_self.sum(axis=1)).ravel()
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
    return (d_inv_sqrt @ a_self @ d_inv_sqrt).tocsr()


def degree_features(data: EllipticData) -> pd.DataFrame:
    a = adjacency(data, symmetric=False)
    in_deg = np.asarray(a.sum(axis=0)).ravel()
    out_deg = np.asarray(a.sum(axis=1)).ravel()
    return pd.DataFrame(
        {"in_degree": in_deg, "out_degree": out_deg, "degree": in_deg + out_deg},
        index=data.nodes.index,
    )


def propagated_features(data: EllipticData, cols: list[str], hops: int = 2) -> pd.DataFrame:
    """Return ``A_hat^k X`` for k = 1..hops, column names suffixed with the hop."""
    x = data.nodes[cols].to_numpy(dtype=np.float32)
    a_hat = normalised_adjacency(adjacency(data, symmetric=True))
    out = {}
    h = x
    for k in range(1, hops + 1):
        h = a_hat @ h
        for j, c in enumerate(cols):
            out[f"{c}_hop{k}"] = h[:, j]
    return pd.DataFrame(out, index=data.nodes.index)


def build_feature_sets(data: EllipticData, hops: int = 2) -> dict[str, pd.DataFrame]:
    """The feature sets compared in the experiments.

    * ``LF``: 93 local features only.
    * ``AF``: local plus Elliptic's 72 one hop aggregates (165 features).
    * ``AF+Graph``: AF plus degree features and local features propagated 1 and 2 hops.

    The time step column is never used as a feature, so models cannot shortcut on time.
    """
    local = data.nodes[data.local_cols]
    agg = data.nodes[data.agg_cols]
    af = pd.concat([local, agg], axis=1)
    graph = pd.concat([degree_features(data), propagated_features(data, data.local_cols, hops)], axis=1)
    return {"LF": local, "AF": af, "AF+Graph": pd.concat([af, graph], axis=1)}
