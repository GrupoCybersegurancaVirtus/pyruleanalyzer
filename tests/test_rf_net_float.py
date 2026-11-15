"""The Random Forest net decides on scikit-learn's own floats.

scikit-learn sums the trees' predict_proba rows in tree order, divides by the
number of trees and takes the lowest-index argmax. Over the reals the division
changes nothing; in floating point it can turn two sums one ulp apart into a
tie, which the lowest index then wins. These tests pin the net to that
arithmetic: the leaf rows are scikit-learn's rows bit for bit, and a forest
built to hit such a tie is decided as scikit-learn decides it.
"""
import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pyruleanalyzer import PyRuleAnalyzer  # noqa: E402
from pyruleanalyzer._extraction import _sklearn_leaf_proba  # noqa: E402
from pyruleanalyzer.model_checker import CPNModelChecker  # noqa: E402

COLS = ["f0", "f1", "f2"]


def net_predictions(analyzer, X, tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        paths = analyzer.export_hcpn(str(tmp_path / "net"), which="initial",
                                     sample=X.iloc[0])
    mc = CPNModelChecker(paths["initial"], feature_names=COLS)
    out = []
    for x in X.to_numpy(np.float32).astype(np.float64):
        tokens = mc.cnet.run(mc.cnet.initial_marking(tuple(x), places=mc.input_places))
        pred = tokens[mc.structure.pred]
        assert len(pred) == 1
        out.append(int(pred[0]))
    return np.array(out)


@pytest.mark.parametrize("Est", [RandomForestClassifier, ExtraTreesClassifier])
def test_leaf_rows_are_predict_proba_rows(Est):
    X, y = make_classification(n_samples=300, n_features=3, n_informative=3,
                               n_redundant=0, n_classes=3, flip_y=0.2, random_state=0)
    model = Est(n_estimators=5, max_depth=4, min_samples_leaf=7, random_state=0).fit(
        pd.DataFrame(X, columns=COLS), y)
    analyzer = PyRuleAnalyzer.from_sklearn(model)
    rows = {tuple(r.leaf_proba) for r in analyzer.classifier.initial_rules}
    for tree in model.estimators_:
        leaves = np.flatnonzero(tree.tree_.children_left == -1)
        probas = tree.predict_proba(X)
        for leaf in leaves:
            row = tuple(_sklearn_leaf_proba(tree.tree_.value[leaf]))
            assert row in rows
        reached = tree.apply(X)
        for i in range(0, len(X), 17):
            assert tuple(probas[i]) == tuple(_sklearn_leaf_proba(
                tree.tree_.value[reached[i]]))


def test_a_tie_made_by_the_division_is_decided_like_scikit_learn(tmp_path):
    # Every sample reaches these three rows. Summed in tree order they give
    # [0.9999999999999999, 1.0, 1.0] (argmax 1); divided by 3 the three
    # entries are equal, so scikit-learn predicts class 0.
    rows = [[1 / 3, 1 / 3, 1 / 3], [0.5, 0.25, 0.25], [2 / 12, 5 / 12, 5 / 12]]
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 3)), columns=COLS)
    y = np.arange(30) % 3
    model = RandomForestClassifier(n_estimators=3, max_depth=1, random_state=0).fit(X, y)
    for tree, row in zip(model.estimators_, rows):
        value = tree.tree_.value
        if value.sum(axis=2).max() > 1.0 + 1e-9:
            pytest.skip("scikit-learn < 1.4 stores counts in tree_.value")
        value[:, 0, :] = row
    total = np.zeros(3)
    for row in rows:
        total = total + np.array(row)
    assert np.argmax(total) == 1 and np.argmax(total / 3) == 0
    assert (model.predict(X) == 0).all()

    analyzer = PyRuleAnalyzer.from_sklearn(model)
    np.testing.assert_array_equal(net_predictions(analyzer, X, tmp_path),
                                  model.predict(X))
