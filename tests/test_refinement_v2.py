"""Low-usage refinement without holes, and error-aware refinement.

1. Removing a low-usage leaf hands its region to the rest of its tree: a
   sibling leaf is promoted (drops its last condition); a sibling sub-branch is
   grafted into the parent's place (its leaves drop the parent's condition).
   For DT and RF every tree therefore stays a partition of the space: no
   sample falls through to `default_class`, no two rules of a tree overlap.
   Gradient Boosting keeps the additive "abstain" semantics: a tree adds 0.0
   on a removed region whose sibling is a sub-branch.
2. `max_error_increase` (opt-in) makes the low-usage stage error-aware: a
   removal happens only while the misclassified refinement samples stay within
   that budget (0 = reduced-error pruning).
"""
import contextlib
import functools
import io
import os
import sys
import warnings

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pyruleanalyzer import PyRuleAnalyzer  # noqa: E402
from pyruleanalyzer.dt_analyzer import DTAnalyzer  # noqa: E402
from pyruleanalyzer.gbdt_analyzer import GBDTAnalyzer  # noqa: E402
from pyruleanalyzer.rf_analyzer import RFAnalyzer  # noqa: E402
from pyruleanalyzer.rule_classifier import Rule, RuleClassifier  # noqa: E402

OPS = {"<=": np.less_equal, ">": np.greater, "<": np.less, ">=": np.greater_equal}
NAMES = [f"f{i}" for i in range(8)]
ANALYZERS = {"dt": DTAnalyzer, "rf": RFAnalyzer, "gbdt": GBDTAnalyzer}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Function to build data splits.
def _data(seed=0):
    """Noisy 3-class data: train / refinement / held-out evaluation."""
    X, y = make_classification(n_samples=1000, n_features=8, n_informative=5,
                               n_redundant=0, n_classes=3, flip_y=0.1,
                               random_state=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=seed)
    X_ref, X_ev, y_ref, y_ev = train_test_split(X_te, y_te, test_size=0.5, random_state=seed)
    return X_tr, y_tr, X_ref, y_ref, X_ev, y_ev


# Function to train, extract and refine.
def _refine(kind, tau, seed=0, fill_holes=None, **kwargs):
    """Unpruned model refined on the refinement split.

    Returns:
        tuple: (classifier, X_ref, y_ref, X_ev, y_ev, rules after merging).
    """
    X_tr, y_tr, X_ref, y_ref, X_ev, y_ev = _data(seed)
    if kind == "dt":
        model = DecisionTreeClassifier(random_state=seed)
    elif kind == "rf":
        model = RandomForestClassifier(n_estimators=10, random_state=seed)
    else:
        model = GradientBoostingClassifier(n_estimators=15, max_depth=4, random_state=seed)
    model.fit(X_tr, y_tr)
    clf = PyRuleAnalyzer.from_sklearn(model, NAMES).classifier
    if fill_holes is not None:
        # The behaviour before hole filling: a removed rule whose sibling is
        # a sub-branch is dropped and its region left uncovered.
        clf._promote_siblings = functools.partial(
            RuleClassifier._promote_siblings, clf, fill_holes=fill_holes)
    merged = {}
    original_merge = clf.merge_boundary_redundancy

    def merge_and_record():
        pairs = original_merge()
        merged["n"] = len(clf.final_rules)
        return pairs

    clf.merge_boundary_redundancy = merge_and_record
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ANALYZERS[kind](clf).execute_rule_refinement(
            X=X_ref, y=y_ref, remove_below_n_classifications=tau,
            save_final_model=False, save_report=False, **kwargs)
    return clf, X_ref, y_ref, X_ev, y_ev, merged["n"]


# Function to evaluate a rule set with the compiled engine.
def _predict(clf, rules, X):
    """Batch predictions of `rules` and the fallbacks they needed."""
    with contextlib.redirect_stdout(io.StringIO()):
        clf.compile_tree_arrays(rules, feature_names=NAMES)
    clf.fallback_activations = 0
    pred = clf.predict_batch(X, feature_names=NAMES)
    return pred, clf.fallback_activations


# Function to count, per tree, the rules matching each sample.
def _matches_per_tree(clf, X):
    """{tree: (n_samples,) number of rules of that tree matching the sample}."""
    Xq = np.asarray(X, dtype=np.float32).astype(np.float64)
    out = {}
    for r in clf.final_rules:
        if clf.algorithm_type == "Gradient Boosting Decision Trees" and not r.parsed_conditions:
            continue
        tid = "_all" if clf.algorithm_type == "Decision Tree" else r.name.split("_")[0]
        m = np.ones(len(Xq), dtype=bool)
        for var, op, thr in r.parsed_conditions:
            m &= OPS[op](Xq[:, NAMES.index(var)], thr)
        out[tid] = out.get(tid, 0) + m.astype(int)
    return out


# Function to build a rule.
def _rule(name, cls, conds, dist=None):
    r = Rule(name, cls, [f"{v} {op} {t}" for v, op, t in conds], class_distribution=dist)
    r.parsed_conditions = list(conds)
    return r


# ---------------------------------------------------------------------------
# 1. No holes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ("dt", "rf"))
@pytest.mark.parametrize("tau", (0, 2, 5))
def test_low_usage_removal_leaves_no_holes(kind, tau):
    clf, X_ref, _y, X_ev, _ye, _n = _refine(kind, tau)
    X_all = np.vstack([X_ref, X_ev])
    # Every tree answers every sample exactly once: a partition.
    for tid, count in _matches_per_tree(clf, X_all).items():
        assert (count == 1).all(), (
            f"tree {tid}: {int((count == 0).sum())} uncovered, "
            f"{int((count > 1).sum())} overlapping samples")
    assert clf.find_overlapping_rules(clf.final_rules)["n_overlaps"] == 0
    _pred, fallbacks = _predict(clf, clf.final_rules, X_all)
    assert fallbacks == 0


def test_dt_holes_existed_before_and_accuracy_does_not_drop():
    """The fixture used to open holes; filling them keeps the rule count and
    answers the removed regions with the tree instead of default_class."""
    old, *_ = _refine("dt", 0, fill_holes=False)
    new, X_ref, y_ref, X_ev, y_ev, _n = _refine("dt", 0)
    pred_old, fb_old = _predict(old, old.final_rules, X_ev)
    pred_new, fb_new = _predict(new, new.final_rules, X_ev)
    pred_init, _ = _predict(new, new.initial_rules, X_ev)
    assert fb_old > 0 and fb_new == 0
    assert len(new.final_rules) == len(old.final_rules) < len(new.initial_rules)
    acc_old = float((pred_old == y_ev).mean())
    acc_new = float((pred_new == y_ev).mean())
    acc_init = float((pred_init == y_ev).mean())
    assert acc_new >= acc_old, (acc_init, acc_old, acc_new)


@pytest.mark.parametrize("kind", ("dt", "rf", "gbdt"))
def test_usage_counts_are_recounted_after_grafting(kind):
    clf, X_ref, _y, _xe, _ye, _n = _refine(kind, 2)
    Xq = np.asarray(X_ref, dtype=np.float32).astype(np.float64)
    for r in clf.final_rules:
        if kind == "gbdt" and not r.parsed_conditions:
            continue
        m = np.ones(len(Xq), dtype=bool)
        for var, op, thr in r.parsed_conditions:
            m &= OPS[op](Xq[:, NAMES.index(var)], thr)
        assert r.usage_count == int(m.sum()), r.name
        assert r.usage_count > 2, r.name


def test_subtree_sibling_is_grafted_into_the_parent():
    """The docstring example: removing A alone grafts the [z <= 7] branch."""
    rules = [
        _rule("A_Class0", "0", [("x", "<=", 5.0), ("y", "<=", 3.0)], dist=[4, 0]),
        _rule("B_Class0", "0", [("x", "<=", 5.0), ("y", ">", 3.0), ("z", "<=", 7.0)], dist=[3, 1]),
        _rule("C_Class1", "1", [("x", "<=", 5.0), ("y", ">", 3.0), ("z", ">", 7.0)], dist=[0, 5]),
        _rule("D_Class1", "1", [("x", ">", 5.0)], dist=[1, 9]),
    ]
    clf = RuleClassifier(rules, algorithm_type="Decision Tree")
    clf.class_labels = ["0", "1"]
    with contextlib.redirect_stdout(io.StringIO()):
        out = clf._promote_siblings([rules[0]], rules)
    got = {r.name: (r.parsed_conditions, r.class_) for r in out}
    assert got == {
        "B_Class0_grafted": ([("x", "<=", 5.0), ("z", "<=", 7.0)], "0"),
        "C_Class1_grafted": ([("x", "<=", 5.0), ("z", ">", 7.0)], "1"),
        "D_Class1": ([("x", ">", 5.0)], "1"),
    }
    # Inputs untouched.
    assert rules[1].parsed_conditions[1] == ("y", ">", 3.0)


def test_rule_alone_in_its_tree_is_kept():
    rules = [_rule("A_Class0", "0", [("x", "<=", 5.0)], dist=[3, 0]),
             _rule("B_Class1", "1", [("x", ">", 5.0)], dist=[0, 2])]
    clf = RuleClassifier(rules, algorithm_type="Decision Tree")
    clf.class_labels = ["0", "1"]
    with contextlib.redirect_stdout(io.StringIO()):
        out = clf._promote_siblings(rules, rules)
    assert len(out) == 1 and out[0].parsed_conditions == []


def test_gbdt_keeps_abstaining_on_removed_regions():
    """GBDT: a removed leaf whose sibling is a sub-branch contributes 0.0."""
    rules = [
        _rule("T1_A", "1", [("x", "<=", 5.0), ("y", "<=", 3.0)]),
        _rule("T1_B", "1", [("x", "<=", 5.0), ("y", ">", 3.0), ("z", "<=", 7.0)]),
        _rule("T1_C", "1", [("x", "<=", 5.0), ("y", ">", 3.0), ("z", ">", 7.0)]),
        _rule("T1_D", "1", [("x", ">", 5.0)]),
    ]
    for i, r in enumerate(rules):
        r.contribution = r.leaf_value = float(i + 1)
        r.class_group = "1"
    clf = RuleClassifier(rules, algorithm_type="Gradient Boosting Decision Trees")
    with contextlib.redirect_stdout(io.StringIO()):
        out = clf._promote_siblings([rules[0]], rules)
    assert sorted(r.name for r in out) == ["T1_B", "T1_C", "T1_D"]
    assert all(len(r.parsed_conditions) == len(o.parsed_conditions)
               for r, o in zip(out, rules[1:]))


# ---------------------------------------------------------------------------
# 2. Error-aware refinement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ("dt", "rf", "gbdt"))
@pytest.mark.parametrize("tau", (2, 10 ** 9))
def test_error_aware_refinement_never_adds_refinement_errors(kind, tau):
    clf, X_ref, y_ref, _xe, _ye, n_merged = _refine(kind, tau, max_error_increase=0)
    before, _ = _predict(clf, clf.initial_rules, X_ref)
    after, _ = _predict(clf, clf.final_rules, X_ref)
    assert int((after != y_ref).sum()) <= int((before != y_ref).sum())
    assert len(clf.final_rules) < n_merged
    for tid, count in _matches_per_tree(clf, X_ref).items():
        assert (count <= 1).all(), tid
        if kind != "gbdt":
            assert (count == 1).all(), tid


@pytest.mark.parametrize("budget", (0, 3))
def test_error_budget_is_respected(budget):
    clf, X_ref, y_ref, *_ = _refine("dt", 10 ** 9, max_error_increase=budget)
    before, _ = _predict(clf, clf.initial_rules, X_ref)
    after, _ = _predict(clf, clf.final_rules, X_ref)
    assert int((after != y_ref).sum()) <= int((before != y_ref).sum()) + budget


@pytest.mark.parametrize("kind", ("dt", "rf", "gbdt"))
def test_default_is_the_usage_criterion(kind):
    """max_error_increase=None (default) and an unbounded budget remove the
    same rules: the error-aware path only ever vetoes removals."""
    default, *_ = _refine(kind, 3)
    unbounded, *_ = _refine(kind, 3, max_error_increase=10 ** 9)
    explicit_none, *_ = _refine(kind, 3, max_error_increase=None)
    key = lambda c: sorted((r.name, tuple(r.parsed_conditions), str(r.class_))  # noqa: E731
                           for r in c.final_rules)
    assert key(default) == key(unbounded) == key(explicit_none)


def test_negative_budget_is_rejected():
    with pytest.raises(ValueError, match="max_error_increase"):
        _refine("dt", 0, max_error_increase=-1)
