"""Random Forest net: its floating-point scores against scikit-learn's.

Replays, on 200 random forest and extra-trees classifiers and 1,000,000 test
samples, the arithmetic of two versions of the coloured net's decision:

* before 2.2.1: each leaf's row normalised from the weighted counts, the rows
  summed, no division by the number of trees;
* from 2.2.1: each leaf emits scikit-learn's predict_proba row, the rows are
  summed in tree order and divided by the number of trees.

and counts the samples whose scores differ from ``predict_proba`` in any bit,
and whose predicted class differs from ``predict``.

Run it::

    python examples/rf_net_float_exactness.py
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
tot = old_bits = new_bits = old_pred = new_pred = 0
for seed in range(200):
    rng = np.random.default_rng(seed)
    K = int(rng.integers(2, 5)); B = int(rng.integers(2, 30))
    X, y = make_classification(n_samples=600, n_features=6, n_informative=4, n_redundant=0, n_classes=K,
                               n_clusters_per_class=1, flip_y=0.2, random_state=seed)
    Est = RandomForestClassifier if seed % 2 == 0 else ExtraTreesClassifier
    m = Est(n_estimators=B, min_samples_leaf=int(rng.integers(1, 12)), max_depth=int(rng.integers(2, 7)),
            random_state=seed).fit(X, y)
    Xt = rng.normal(size=(5000, 6)) * 2
    ref_p = m.predict_proba(Xt); ref = m.predict(Xt); L = m.apply(Xt)
    old = np.zeros((len(Xt), K)); new = np.zeros((len(Xt), K))
    for b, t in enumerate(m.estimators_):
        v = t.tree_.value[:, 0, :]; cnt = v * t.tree_.weighted_n_node_samples[:, None]
        oldv = np.array([[x / sum(r.tolist()) for x in r.tolist()] for r in cnt])
        old = old + oldv[L[:, b]]; new = new + v[L[:, b]]
    new /= B
    tot += len(Xt)
    old_bits += int((old / B != ref_p).any(axis=1).sum()); new_bits += int((new != ref_p).any(axis=1).sum())
    old_pred += int((m.classes_[old.argmax(1)] != ref).sum()); new_pred += int((m.classes_[new.argmax(1)] != ref).sum())
print(f"samples={tot}\nold net: scores differ in {old_bits}, predictions differ in {old_pred}\nnew net: scores differ in {new_bits}, predictions differ in {new_pred}")
