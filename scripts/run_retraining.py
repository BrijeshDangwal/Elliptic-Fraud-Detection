"""Does retraining every time step recover detection after the drift at step 43?

Compares, for each test step t in 35..49:
* static:     LightGBM trained once on steps 1-34 (the standard protocol), and
* retrained:  LightGBM retrained on all labelled steps before t (expanding window).

Retraining assumes labels for a step are available before the next step is scored. Real
labels arrive with a delay, so the retrained numbers are an upper bound on what a weekly
retraining cadence could achieve.

Usage: python scripts/run_retraining.py --data-dir data/raw
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from elliptic_fraud import models as M  # noqa: E402
from elliptic_fraud.data import UNKNOWN, load_elliptic  # noqa: E402
from elliptic_fraud.graph import build_feature_sets  # noqa: E402

THRESHOLD = 0.5
DARK_MARKET_STEP = 43


def fit(x, y, seed=0):
    pw = (y == 0).sum() / max(1, y.sum())
    return M.lightgbm(seed, pw).fit(x, y)


def step_metrics(y, s):
    pred = (s >= THRESHOLD).astype(int)
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, s)) if y.sum() else np.nan,
        "caught": int((pred & y).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures")
    ap.add_argument("--features", default="AF")
    a = ap.parse_args()

    data = load_elliptic(a.data_dir)
    x = build_feature_sets(data)[a.features].to_numpy(dtype=np.float32)
    steps = data.nodes["time_step"].to_numpy()
    lab = data.nodes["label"].to_numpy() != UNKNOWN
    y = (data.nodes["label"].to_numpy() == 1).astype(int)

    static = fit(x[lab & (steps <= 34)], y[lab & (steps <= 34)])
    rows = []
    for t in range(35, 50):
        test = lab & (steps == t)
        s_static = M.score(static, x[test])
        train = lab & (steps < t)
        s_retrain = M.score(fit(x[train], y[train]), x[test])
        rs, rr = step_metrics(y[test], s_static), step_metrics(y[test], s_retrain)
        rows.append({"time_step": t, "illicit": int(y[test].sum()),
                     "static_f1": rs["f1"], "retrained_f1": rr["f1"],
                     "static_pr_auc": rs["pr_auc"], "retrained_pr_auc": rr["pr_auc"],
                     "static_caught": rs["caught"], "retrained_caught": rr["caught"]})
        print(f"step {t}: illicit {rows[-1]['illicit']:4d}  static F1 {rs['f1']:.2f}  retrained F1 {rr['f1']:.2f}")
    df = pd.DataFrame(rows)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out_dir / "retraining.csv", index=False)

    after = df["time_step"] >= DARK_MARKET_STEP
    summary = {
        "features": a.features, "threshold": THRESHOLD,
        "illicit_caught_steps_43_49": {"static": int(df.loc[after, "static_caught"].sum()),
                                      "retrained": int(df.loc[after, "retrained_caught"].sum()),
                                      "total": int(df.loc[after, "illicit"].sum())},
        "illicit_caught_steps_35_42": {"static": int(df.loc[~after, "static_caught"].sum()),
                                      "retrained": int(df.loc[~after, "retrained_caught"].sum()),
                                      "total": int(df.loc[~after, "illicit"].sum())},
        "mean_step_f1_43_49": {"static": round(df.loc[after, "static_f1"].mean(), 4),
                              "retrained": round(df.loc[after, "retrained_f1"].mean(), 4)},
        "mean_step_pr_auc_43_49": {"static": round(df.loc[after, "static_pr_auc"].mean(), 4),
                                  "retrained": round(df.loc[after, "retrained_pr_auc"].mean(), 4)},
    }
    (a.out_dir / "retraining.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    a.fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["time_step"], df["static_f1"], marker="o", ms=3, label="trained once on steps 1-34")
    ax.plot(df["time_step"], df["retrained_f1"], marker="o", ms=3, label="retrained before every step")
    ax.axvline(DARK_MARKET_STEP - 0.5, color="#555", ls="--", lw=1)
    ax.text(DARK_MARKET_STEP - 0.4, 0.98, "dark market\nshutdown", fontsize=8, va="top")
    ax.set_ylim(0, 1)
    ax.set_xlabel("time step")
    ax.set_ylabel("illicit F1 (threshold 0.5)")
    ax.set_title(f"LightGBM ({a.features}): static model vs retraining every step")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(a.fig_dir / "retraining.png", dpi=160)


if __name__ == "__main__":
    main()
