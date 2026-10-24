# Walkthrough: how the project works and why

A guide to every design decision, written so each one can be explained in an interview.

## 1. The data

The Elliptic dataset is a graph of Bitcoin transactions. Each node is a transaction, each
directed edge is a flow of bitcoin from one transaction to another, and transactions are
grouped into 49 time steps roughly two weeks apart. About 2% of transactions are labelled
illicit (scams, ransomware, dark markets and similar), 21% licit, and the rest are unknown.

* `src/elliptic_fraud/data.py` loads the three CSVs, names the columns (93 local features,
  72 aggregated neighbourhood features), maps labels to 1 (illicit), 0 (licit), -1 (unknown)
  and validates everything: column count, class values, that every edge points to a known
  transaction. `summarise()` reproduces the published counts as a sanity check.
* Every edge connects two transactions in the same time step. The loader checks this,
  because the leakage argument in section 3 depends on it.

## 2. Why a temporal split and not a random one

Fraud changes over time. A random split lets a model see the future and makes results look
better than they would be in production. Following Weber et al. (2019), models train on time
steps 1 to 34 and are tested once on 35 to 49 (`split.py`).

The decision threshold also has to be chosen without touching the test period. So each model
is first fitted on steps 1 to 29, the threshold that maximises illicit F1 on steps 30 to 34 is
picked, and only then is the final model refitted on 1 to 34 and scored on 35 to 49.

## 3. Graph features (`graph.py`)

Elliptic's 72 aggregated features already describe each transaction's direct neighbours. The
project adds:

* degree features (in, out, total), and
* local features propagated over the graph with the GCN normalised adjacency
  `A_hat = D^-1/2 (A + I) D^-1/2`, one and two hops. This is the feature side of a graph
  convolutional network without learned weights (Simplified Graph Convolution, Wu et al., 2019),
  so a gradient boosted tree can use graph structure. It also runs on a laptop without PyTorch.

Unlabelled transactions take part in propagation because they are real neighbours, but labels
are never propagated. Since edges never cross time steps, test period information cannot reach
training rows through the graph.

The time step itself is never used as a feature, so a model cannot learn "later means safer".

## 4. Models (`models.py`)

* Logistic regression with standardised inputs: a simple linear baseline.
* Random forest: the strongest model in the original Elliptic paper.
* LightGBM: gradient boosted trees.
* Isolation Forest: an unsupervised anomaly detector that never sees labels, to test whether
  illicit transactions are simply unusual.

Imbalance is handled with class weights (`balanced`, `scale_pos_weight`) rather than resampling,
so the model sees the real data and the threshold is tuned explicitly.

## 5. Metrics (`evaluate.py`)

Accuracy is useless here: predicting "licit" for everything is about 90% accurate. Everything is
measured on the illicit class: precision (of the flagged transactions, how many are illicit),
recall (of the illicit transactions, how many were caught), F1, and the area under the
precision recall curve, which does not depend on a threshold.

Precision at k per time step models an analyst's review queue: if a risk analyst can only check
the 30 highest scored transactions in a period, how many of them are real fraud?

## 6. Drift

The paper reports that every model stops detecting new illicit activity after a dark market
was shut down around time step 43: the fraud that came after looked different. The project
measures F1 before and after step 43 to check this on its own models. This is the most
important finding to discuss with a risk team, because it is what happens in production when
attackers adapt.

## 7. Reproducing

```
pip install -r requirements.txt
python scripts/run_experiments.py --data-dir data/raw      # real data, about 30 minutes on 2 cores
python scripts/run_experiments.py --synthetic             # fake data, under a minute
pytest -q
```

Outputs: `results/metrics.json`, `results/summary.csv`, `results/per_step.csv`,
`results/all_runs.csv` and the figures in `figures/`.
