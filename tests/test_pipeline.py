import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from elliptic_fraud.data import ILLICIT, UNKNOWN, load_elliptic, summarise  # noqa: E402
from elliptic_fraud.evaluate import best_f1_threshold, illicit_metrics, per_step  # noqa: E402
from elliptic_fraud.graph import adjacency, build_feature_sets, normalised_adjacency  # noqa: E402
from elliptic_fraud.split import temporal_split  # noqa: E402
from elliptic_fraud.synthetic import write_synthetic  # noqa: E402


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = write_synthetic(tmp_path_factory.mktemp("elliptic"), n_per_step=30)
    return load_elliptic(d)


def test_loader_shapes_and_labels(data):
    n = data.nodes
    assert len(data.local_cols) == 93 and len(data.agg_cols) == 72
    assert set(n["label"].unique()) <= {ILLICIT, 0, UNKNOWN}
    assert n["time_step"].between(1, 49).all()
    s = summarise(data)
    assert s["transactions"] == len(n)
    assert s["illicit"] + s["licit"] + s["unknown"] == len(n)
    assert s["edges_within_same_step"] == 1.0


def test_temporal_split_has_no_overlap_or_leakage(data):
    sp = temporal_split(data.nodes)
    steps = data.nodes["time_step"].to_numpy()
    assert steps[sp.train].max() <= 34 < steps[sp.test].min()
    assert steps[sp.fit].max() < 30 <= steps[sp.val].min()
    assert len(np.intersect1d(sp.train, sp.test)) == 0
    assert (data.nodes["label"].to_numpy()[sp.test] != UNKNOWN).all()


def test_feature_sets_exclude_time_step(data):
    fs = build_feature_sets(data)
    assert fs["LF"].shape[1] == 93
    assert fs["AF"].shape[1] == 165
    assert fs["AF+Graph"].shape[1] == 165 + 3 + 2 * 93
    for x in fs.values():
        assert "time_step" not in x.columns
        assert np.isfinite(x.to_numpy()).all()


def test_normalised_adjacency_is_symmetric_with_unit_spectrum(data):
    a_hat = normalised_adjacency(adjacency(data))
    assert abs(a_hat - a_hat.T).max() < 1e-6
    # A_hat = D^-1/2 (A+I) D^-1/2 has largest eigenvalue 1
    v = np.ones(a_hat.shape[0])
    for _ in range(200):
        v = a_hat @ v
        v /= np.linalg.norm(v)
    assert abs(v @ (a_hat @ v) - 1.0) < 1e-3


def test_metrics_and_threshold():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.8, 0.7, 0.4, 0.9])
    m = illicit_metrics(y, s, 0.5)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0
    assert 0.4 < best_f1_threshold(y, s) <= 0.7
    df = per_step(np.array([35, 35, 35, 36, 36, 36]), y, s, 0.5, k=2)
    assert list(df["time_step"]) == [35, 36]
    assert df["precision_at_2"].between(0, 1).all()
