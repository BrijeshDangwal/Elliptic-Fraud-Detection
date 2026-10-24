"""Run every experiment and write results/ and figures/.

Usage:
    python scripts/run_experiments.py --data-dir data/raw
    python scripts/run_experiments.py --synthetic        # quick check on fake data
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from elliptic_fraud import models as M  # noqa: E402
from elliptic_fraud.data import load_elliptic, summarise  # noqa: E402
from elliptic_fraud.evaluate import best_f1_threshold, illicit_metrics, per_step  # noqa: E402
from elliptic_fraud.graph import build_feature_sets  # noqa: E402
from elliptic_fraud.split import temporal_split  # noqa: E402
from elliptic_fraud.synthetic import write_synthetic  # noqa: E402

DARK_MARKET_STEP = 43  # Weber et al. (2019): models fail after a dark market shutdown here.
K_REVIEW = 30  # alerts an analyst could review per time step


def make_model(name: str, seed: int, pos_weight: float):
    if name == "LightGBM":
        return M.lightgbm(seed, pos_weight)
    return M.SUPERVISED[name](seed)


def run(data_dir: Path, out_dir: Path, fig_dir: Path, seeds: int) -> dict:
    t0 = time.time()
    data = load_elliptic(data_dir)
    summary = summarise(data)
    print("dataset:", summary)
    feature_sets = build_feature_sets(data)
    split = temporal_split(data.nodes)
    y = (data.nodes["label"].to_numpy() == 1).astype(int)
    steps = data.nodes["time_step"].to_numpy()

    results, test_scores = [], {}
    for fs_name, X in feature_sets.items():
        Xv = X.to_numpy(dtype=np.float32)
        for model_name in M.SUPERVISED:
            seed_list = [0] if model_name == "Logistic regression" else list(range(seeds))
            for seed in seed_list:
                # 1) tune the threshold: fit on steps 1-29, choose threshold on 30-34
                pw_fit = (y[split.fit] == 0).sum() / max(1, y[split.fit].sum())
                tuner = make_model(model_name, seed, pw_fit).fit(Xv[split.fit], y[split.fit])
                thr = best_f1_threshold(y[split.val], M.score(tuner, Xv[split.val]))
                # 2) final model on steps 1-34, evaluate once on 35-49
                pw = (y[split.train] == 0).sum() / max(1, y[split.train].sum())
                model = make_model(model_name, seed, pw).fit(Xv[split.train], y[split.train])
                s = M.score(model, Xv[split.test])
                for thr_name, t in (("default_0.5", 0.5), ("tuned_on_val", thr)):
                    results.append({"features": fs_name, "model": model_name, "seed": seed,
                                    "threshold_rule": thr_name, **illicit_metrics(y[split.test], s, t)})
                if seed == 0:
                    test_scores[(fs_name, model_name)] = (s, thr, model)
                print(f"{fs_name:9s} {model_name:20s} seed {seed}: test F1 {results[-1]['f1']:.3f} "
                      f"(tuned thr {thr:.3f}) [{time.time() - t0:.0f}s]")

    # Unsupervised baseline: Isolation Forest never sees labels.
    Xaf = feature_sets["AF"].to_numpy(dtype=np.float32)
    fit_all = split.train_all[steps[split.train_all] < 30]
    tuner = M.isolation_forest(0).fit(Xaf[fit_all])
    thr = best_f1_threshold(y[split.val], M.score(tuner, Xaf[split.val]))
    iso = M.isolation_forest(0).fit(Xaf[split.train_all])
    s_iso = M.score(iso, Xaf[split.test])
    for thr_name, t in (("tuned_on_val", thr),):
        results.append({"features": "AF", "model": "Isolation Forest (unsupervised)", "seed": 0,
                        "threshold_rule": thr_name, **illicit_metrics(y[split.test], s_iso, t)})
    test_scores[("AF", "Isolation Forest (unsupervised)")] = (s_iso, thr, iso)

    res = pd.DataFrame(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "all_runs.csv", index=False)

    agg = (res.groupby(["features", "model", "threshold_rule"])
           [["precision", "recall", "f1", "pr_auc"]].agg(["mean", "std"]).round(4))
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out_dir / "summary.csv", index=False)

    # Per time step analysis on the test period for the headline models.
    focus = [("AF", "Logistic regression"), ("AF", "Random forest"), ("AF", "LightGBM"),
             ("AF+Graph", "LightGBM"), ("AF+Graph", "Random forest")]
    test_steps = steps[split.test]
    yt = y[split.test]
    per = []
    for key in focus:
        s, thr, _ = test_scores[key]
        df = per_step(test_steps, yt, s, thr, K_REVIEW)
        df.insert(0, "model", f"{key[1]} ({key[0]})")
        per.append(df)
    per_df = pd.concat(per, ignore_index=True)
    per_df.to_csv(out_dir / "per_step.csv", index=False)

    drift = {}
    for key in focus:
        s, thr, _ = test_scores[key]
        before = test_steps < DARK_MARKET_STEP
        drift[f"{key[1]} ({key[0]})"] = {
            "f1_steps_35_42": illicit_metrics(yt[before], s[before], thr)["f1"],
            "f1_steps_43_49": illicit_metrics(yt[~before], s[~before], thr)["f1"],
            "illicit_steps_35_42": int(yt[before].sum()),
            "illicit_steps_43_49": int(yt[~before].sum()),
        }

    # Feature importance for the best graph model.
    lgb_graph = test_scores[("AF+Graph", "LightGBM")][2]
    names = feature_sets["AF+Graph"].columns
    imp = pd.Series(lgb_graph.booster_.feature_importance("gain"), index=names).sort_values(ascending=False)
    group = imp.groupby(lambda c: "graph propagated" if "_hop" in c else
                        ("degree" if "degree" in c else ("aggregated" if c.startswith("agg_") else "local")))
    imp_share = (group.sum() / imp.sum()).round(4).to_dict()

    make_figures(per_df, test_scores, yt, fig_dir, imp)

    out = {
        "dataset": summary,
        "protocol": {"train_steps": "1-34", "threshold_tuning": "fit 1-29, validate 30-34",
                     "test_steps": "35-49", "seeds": seeds, "review_queue_k": K_REVIEW},
        "summary": agg.to_dict(orient="records"),
        "drift_around_step_43": drift,
        "lightgbm_graph_importance_share": imp_share,
        "top_features_lightgbm_graph": imp.head(15).round(1).to_dict(),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    print(f"done in {out['runtime_seconds']}s")
    return out


def make_figures(per_df, test_scores, yt, fig_dir: Path, imp: pd.Series) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # 1) Per step F1 with illicit counts: shows the collapse after the dark market shutdown.
    fig, ax1 = plt.subplots(figsize=(9, 4.2))
    counts = per_df.drop_duplicates("time_step").set_index("time_step")["illicit"]
    ax2 = ax1.twinx()
    ax2.bar(counts.index, counts.values, color="#d9d9d9", width=0.7, label="illicit transactions")
    ax2.set_ylabel("illicit transactions in step")
    ax2.spines["top"].set_visible(False)
    for name, g in per_df.groupby("model", sort=False):
        if "Logistic" in name or "Random forest (AF+Graph)" in name:
            continue
        ax1.plot(g["time_step"], g["f1"], marker="o", ms=3, label=name)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    ax1.axvline(DARK_MARKET_STEP - 0.5, color="#555", ls="--", lw=1)
    ax1.text(DARK_MARKET_STEP - 0.4, 0.95, "dark market\nshutdown", fontsize=8, va="top")
    ax1.set_xlabel("time step (test period)")
    ax1.set_ylabel("illicit F1")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="lower left", fontsize=8, frameon=False)
    ax1.set_title("Illicit F1 per time step on the test period")
    fig.tight_layout()
    fig.savefig(fig_dir / "f1_per_step.png", dpi=160)
    plt.close(fig)

    # 2) Precision recall curves on the test period.
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for key in [("AF", "Logistic regression"), ("AF", "Random forest"), ("AF", "LightGBM"),
                ("AF+Graph", "LightGBM"), ("AF", "Isolation Forest (unsupervised)")]:
        s = test_scores[key][0]
        p, r, _ = precision_recall_curve(yt, s)
        ax.plot(r, p, label=f"{key[1]} ({key[0]})")
    ax.set_xlabel("recall (illicit)")
    ax.set_ylabel("precision (illicit)")
    ax.set_title("Precision and recall, test steps 35-49")
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "pr_curves.png", dpi=160)
    plt.close(fig)

    # 3) Top features by gain for LightGBM with graph features.
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.4))
    colors = ["#2b6cb0" if "_hop" in c or "degree" in c else "#888" for c in top.index]
    ax.barh(top.index, top.values, color=colors)
    ax.set_xlabel("total gain")
    ax.set_title("LightGBM (AF+Graph): top 15 features\nblue = graph features added in this project")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_importance.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--synthetic", action="store_true", help="run on a small synthetic dataset")
    a = ap.parse_args()
    if a.synthetic:
        with tempfile.TemporaryDirectory() as tmp:
            write_synthetic(tmp)
            run(Path(tmp), a.out_dir, a.fig_dir, seeds=1)
    else:
        run(a.data_dir, a.out_dir, a.fig_dir, a.seeds)


if __name__ == "__main__":
    main()
