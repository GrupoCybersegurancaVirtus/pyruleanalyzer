"""The vectorized rule-usage count of GBDT refinement matches the per-sample loop."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier

from pyruleanalyzer import PyRuleAnalyzer


def _signature(rules):
    return [(r.name, tuple((c[0], c[1], float(c[2])) for c in r.parsed_conditions),
             None if r.contribution is None else float(r.contribution),
             int(r.usage_count), int(r.error_count)) for r in rules]


def _refined(sk, X, y, engine, **kw):
    names = [f'f{i}' for i in range(X.shape[1])]
    an = PyRuleAnalyzer.from_sklearn(sk, names, [str(c) for c in sk.classes_])
    an.execute_rule_refinement(X=X, y=np.asarray(y).astype(str), usage_engine=engine, **kw)
    return an


@pytest.mark.parametrize('n_classes,params', [
    (2, dict(n_estimators=20, max_depth=3)),
    (3, dict(n_estimators=15, max_depth=4)),
    (4, dict(n_estimators=10, max_leaf_nodes=12, max_depth=None)),
])
@pytest.mark.parametrize('threshold', [-1, 0, 2])
def test_batch_equals_loop(n_classes, params, threshold):
    X, y = make_classification(n_samples=600, n_features=6, n_informative=4,
                               n_classes=n_classes, random_state=n_classes)
    sk = GradientBoostingClassifier(random_state=0, **params).fit(X[:400], y[:400])
    X_ref, y_ref = X[400:], y[400:]
    loop = _refined(sk, X_ref, y_ref, 'loop', remove_below_n_classifications=threshold)
    batch = _refined(sk, X_ref, y_ref, 'batch', remove_below_n_classifications=threshold)
    assert _signature(batch.classifier.final_rules) == _signature(loop.classifier.final_rules)
    assert (batch.predict(X) == loop.predict(X)).all()


def test_batch_counts_every_leaf_once_per_sample():
    X, y = make_classification(n_samples=300, n_features=5, n_classes=3,
                               n_informative=3, random_state=1)
    sk = GradientBoostingClassifier(n_estimators=8, max_depth=3, random_state=0).fit(X, y)
    an = _refined(sk, X, y, 'batch')
    clf = an.classifier
    by_tree = {}
    for r in clf.final_rules:
        if r.parsed_conditions:
            tid = r.name.split('_')[0]
            by_tree[tid] = by_tree.get(tid, 0) + r.usage_count
    assert set(by_tree.values()) == {len(X)}


@pytest.mark.parametrize('n_classes', [2, 3])
def test_batch_recount_equals_evaluator(n_classes):
    X, y = make_classification(n_samples=500, n_features=6, n_informative=4,
                               n_classes=n_classes, random_state=7)
    sk = GradientBoostingClassifier(n_estimators=12, max_leaf_nodes=10, max_depth=None,
                                    random_state=0).fit(X[:300], y[:300])
    an = _refined(sk, X[300:], y[300:], 'loop', remove_below_n_classifications=3)
    clf = an.classifier
    names = [f'f{i}' for i in range(X.shape[1])]
    labels = np.asarray(y).astype(str)
    clf._refinement_evaluator(X, labels, names, clf.final_rules).recount(clf.final_rules)
    expected = _signature(clf.final_rules)
    for r in clf.final_rules:
        r.usage_count = r.error_count = -1
    for r in clf.final_rules:   # init scores are not recounted by either path
        if not r.parsed_conditions:
            r.usage_count, r.error_count = expected[clf.final_rules.index(r)][3:]
    assert clf._recount_gbdt_batch(X, labels, names, clf.final_rules)
    assert _signature(clf.final_rules) == expected


def test_auto_and_bad_engine():
    X, y = make_classification(n_samples=200, n_features=4, random_state=2)
    sk = GradientBoostingClassifier(n_estimators=5, max_depth=2, random_state=0).fit(X, y)
    auto = _refined(sk, X, y, 'auto', remove_below_n_classifications=1)
    loop = _refined(sk, X, y, 'loop', remove_below_n_classifications=1)
    assert _signature(auto.classifier.final_rules) == _signature(loop.classifier.final_rules)
    with pytest.raises(ValueError):
        _refined(sk, X, y, 'fast')
