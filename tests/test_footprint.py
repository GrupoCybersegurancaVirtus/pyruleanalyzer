"""
Tests for ``pyruleanalyzer.footprint``: the size of a model before training.

* The node and tree counts of the formula are those of the compiled arrays
  that the C/Arduino exports write.
* The bound of a depth is never below the size of a model trained with it.
* The deepest depth for a board is the last one whose bound fits flash and RAM,
  and the closed form agrees with the search.
"""

import os
import sys

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyruleanalyzer import PyRuleAnalyzer
from pyruleanalyzer.footprint import (
    BOARDS, Board, estimate_footprint, fit_to_board, make_board,
    max_depth_for_budget, max_leaves_for_budget,
    model_bytes, n_trees, node_bytes, probe_leaves, suggest_complexity,
)

FEATS = [f"f{i}" for i in range(6)]


def _data(n_classes=3):
    X, y = make_classification(n_samples=800, n_features=6, n_informative=4,
                               n_redundant=0, n_classes=n_classes, random_state=0)
    return X, y


def _compiled(model, X, y):
    """(trees, nodes) of the arrays the C export writes for a fitted model."""
    model.fit(X, y)
    clf = PyRuleAnalyzer.from_sklearn(model, FEATS).classifier
    clf.compile_tree_arrays(clf.initial_rules, feature_names=FEATS)
    trees = [t for t in clf._tree_arrays if 'init_score' not in t]
    return len(trees), sum(len(t['feature_idx']) for t in trees)


@pytest.mark.parametrize("algo,model,n_classes", [
    ('gbdt', GradientBoostingClassifier(n_estimators=6, max_depth=3, random_state=0), 3),
    ('gbdt', GradientBoostingClassifier(n_estimators=6, max_depth=3, random_state=0), 2),
    ('rf', RandomForestClassifier(n_estimators=5, max_depth=4, random_state=0), 3),
    ('dt', DecisionTreeClassifier(max_depth=5, random_state=0), 3),
])
def test_counts_match_the_compiled_arrays(algo, model, n_classes):
    X, y = _data(n_classes)
    trees, nodes = _compiled(model, X, y)
    n_est = getattr(model, 'n_estimators', 1)
    assert trees == n_trees(algo, n_est, n_classes)
    leaves = np.mean([t.tree_.n_leaves for t in np.ravel(getattr(model, 'estimators_', [model]))])
    est = estimate_footprint(algo, n_est, n_classes, model.max_depth, leaves_per_tree=leaves)
    assert est['nodes'] == nodes
    bound = estimate_footprint(algo, n_est, n_classes, model.max_depth)
    assert bound['bound'] and bound['model_bytes'] >= est['model_bytes']
    assert bound['model_bytes'] >= model_bytes(trees, nodes, algo, n_classes)


def test_node_bytes_follow_the_exported_types():
    assert node_bytes('gbdt') == 28           # int32 x3 + double x2
    assert node_bytes('dt') == 24             # int32 value
    assert node_bytes('rf', n_classes=4) == 20 + 32
    assert node_bytes('rf', n_classes=4, missing_routing=True) == 53
    assert node_bytes('gbdt', missing_routing=True) == 28  # GBDT never routes NaN
    assert node_bytes('gbdt', double_bytes=4) == 20        # AVR
    with pytest.raises(ValueError):
        node_bytes('svm')


def test_bound_is_capped_by_leaf_limit_and_rows():
    a = estimate_footprint('gbdt', 10, 2, max_depth=8, max_leaf_nodes=16)
    assert a['leaves'] == 10 * 16
    b = estimate_footprint('gbdt', 10, 2, max_depth=8, n_samples=40)
    assert b['leaves'] == 10 * 40
    with pytest.raises(ValueError):
        estimate_footprint('gbdt', 10, 2)


def test_without_board_only_sizes_are_reported():
    r = suggest_complexity(n_classes=3, n_estimators=10, depths=range(1, 5))
    assert r.max_depth is None and r.board is None
    assert [row['max_depth'] for row in r.rows] == [1, 2, 3, 4]
    assert all('fits' not in row for row in r.rows)
    sizes = [row['max_bytes'] for row in r.rows]
    assert sizes == sorted(sizes)
    assert 'max_bytes' in str(r)


@pytest.mark.parametrize("n_classes", [2, 8, 15, 19])
def test_deepest_depth_fits_and_next_does_not(n_classes):
    r = suggest_complexity(n_classes=n_classes, n_estimators=50, n_features=40,
                           board='esp32_huge_app', depths=range(1, 14))
    by_depth = {row['max_depth']: row for row in r.rows}
    d = r.max_depth
    assert by_depth[d]['fits'] and not by_depth[d + 1]['fits']
    assert by_depth[d]['flash'] <= BOARDS['esp32_huge_app'].flash_bytes
    assert d == max_depth_for_budget(r.flash_budget, 'gbdt', 50, n_classes)


def test_custom_memory_and_overrides():
    small = suggest_complexity(n_classes=2, n_estimators=10, n_features=4,
                               flash_bytes=60_000, ram_bytes=10_000, depths=range(1, 10))
    large = suggest_complexity(n_classes=2, n_estimators=10, n_features=4,
                               flash_bytes=600_000, ram_bytes=10_000, depths=range(1, 10))
    assert small.max_depth < large.max_depth
    b = make_board('esp32', ram_bytes=1_000)
    assert b.flash_bytes == BOARDS['esp32'].flash_bytes and b.ram_bytes == 1_000
    r = suggest_complexity(n_classes=2, n_estimators=10, n_features=4, board=b,
                           depths=range(1, 4))
    assert r.max_depth is None  # the firmware alone exceeds 1,000 bytes of RAM
    assert make_board() is None
    with pytest.raises(ValueError):
        make_board('pdp11')


def test_avr_counts_the_model_in_ram():
    uno = suggest_complexity(n_classes=2, n_estimators=1, n_features=4,
                             algorithm='dt', board='uno', depths=range(1, 12))
    rows = {r['max_depth']: r for r in uno.rows}
    assert rows[1]['ram'] > BOARDS['uno'].firmware_ram + rows[1]['max_bytes'] - 1
    assert uno.max_depth is not None and not rows[uno.max_depth + 1]['fits']
    assert not rows[uno.max_depth + 1]['ram'] <= BOARDS['uno'].ram_bytes


def test_probe_estimates_leaves_within_the_bound():
    X, y = _data(3)
    leaves = probe_leaves(X, y, 'gbdt', max_depth=4, n_estimators=3)
    assert 1 <= leaves <= 16
    r = suggest_complexity(X, y, n_estimators=20, depths=[2, 4],
                           board=Board(flash_bytes=200_000, ram_bytes=100_000))
    for row in r.rows:
        assert row['expected_bytes'] <= row['max_bytes']
        assert 'fits_expected' in row
    assert r.max_depth_expected >= r.max_depth


def test_grid_over_trees_and_leaves():
    r = suggest_complexity(n_classes=8, n_estimators=[25, 50, 100], max_leaf_nodes=[None, 16],
                           n_features=40, board='esp32_huge_app', depths=range(3, 9))
    assert len(r.rows) == 3 * 2 * 6
    assert {row['n_estimators'] for row in r.rows} == {25, 50, 100}
    best = r.largest
    assert best['fits'] and all(best['max_bytes'] >= row['max_bytes'] for row in r.fitting())
    # fewer trees leave room for deeper ones
    deepest = {n: max(row['max_depth'] for row in r.fitting()
                      if row['n_estimators'] == n and row['max_leaf_nodes'] is None)
               for n in (25, 100)}
    assert deepest[25] > deepest[100]
    assert 'n_estimators' in str(r)


def test_max_leaves_fills_the_budget():
    budget = 3_000_000
    for n, k in [(50, 2), (50, 15), (200, 8)]:
        leaves = max_leaves_for_budget(budget, 'gbdt', n, k)
        fit = estimate_footprint('gbdt', n, k, max_leaf_nodes=leaves)['model_bytes']
        over = estimate_footprint('gbdt', n, k, max_leaf_nodes=leaves + 1)['model_bytes']
        assert fit <= budget < over
        # never fewer leaves than the depth bound allows
        assert leaves >= 2 ** max_depth_for_budget(budget, 'gbdt', n, k)
    assert max_leaves_for_budget(1_000, 'gbdt', 50, 2) is None


def test_fit_to_board_picks_a_fitting_candidate():
    X, y = _data(3)
    board = Board(flash_bytes=120_000, ram_bytes=100_000, firmware_flash=20_000)
    r = fit_to_board(X, y, board=board, n_estimators=(5, 10, 20), sample_rows=None,
                     scoring='accuracy')
    assert len(r.candidates) == 3
    assert all(c['flash'] <= board.flash_bytes for c in r.candidates)
    best = max(r.candidates, key=lambda c: c['score'])
    assert r.params['n_estimators'] == best['n_estimators']
    assert r.params['max_depth'] is None and r.params['max_leaf_nodes'] >= 2
    # the chosen parameters really give a model within the guaranteed size
    m = GradientBoostingClassifier(random_state=0, **r.params).fit(X, y)
    trees = np.ravel(m.estimators_)
    nodes = sum(t.tree_.node_count for t in trees)
    assert model_bytes(len(trees), nodes) <= r.model_budget
    assert 'best' in str(r)


def test_fit_to_board_on_avr_bounds_ram_and_needs_a_board():
    X, y = _data(2)
    r = fit_to_board(X, y, board='uno', algorithm='dt', sample_rows=None)
    (c,) = r.candidates
    assert c['ram'] <= BOARDS['uno'].ram_bytes and c['flash'] <= BOARDS['uno'].flash_bytes
    with pytest.raises(ValueError):
        fit_to_board(X, y)
