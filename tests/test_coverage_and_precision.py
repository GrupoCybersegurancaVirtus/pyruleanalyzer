"""Tests for uncovered regions, rule-set structure and input precision.

These cover the guarantees the rule set is supposed to give once refinement has
touched it:

* a sample matching no rule resolves to ``default_class`` in *every* engine,
  and is counted rather than silently absorbed into the metrics;
* an input containing ``NaN`` is rejected instead of being answered wrongly;
* a rule set that is no longer a tree is rejected instead of being compiled
  into a structure that routes rules to the wrong leaf;
* rules whose regions overlap are detected, since accuracy cannot reveal them;
* refinement never mutates the unrefined rule set;
* single-precision quantization reproduces scikit-learn on threshold-adjacent
  samples.
"""

import io
import os
import sys
import contextlib

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyruleanalyzer.rule_classifier import Rule, RuleClassifier


FEATS = ["v1", "v2", "v3"]


def _dt_classifier(max_depth=3, seed=0):
    """Train a small DT and wrap it in a RuleClassifier.

    Args:
        max_depth (int): Tree depth.
        seed (int): RNG seed for the synthetic data.

    Returns:
        tuple: (RuleClassifier, sklearn model, X, y).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(400, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    m = DecisionTreeClassifier(max_depth=max_depth, random_state=0).fit(X, y)
    rules = RuleClassifier.get_tree_rules(m, FEATS, ["0", "1"],
                                          algorithm_type="Decision Tree")
    with contextlib.redirect_stdout(io.StringIO()):
        clf = RuleClassifier.generate_classifier_model(
            rules, {"0": 0, "1": 1}, "Decision Tree")
    return clf, m, X, y


def _matches(rule, sample):
    """Whether a sample satisfies every condition of a rule.

    Args:
        rule (Rule): The rule to test.
        sample (dict): Feature name -> value.

    Returns:
        bool: True when all conditions hold.
    """
    return all((sample[v] <= t) if o in ("<=", "<") else (sample[v] > t)
               for v, o, t in rule.parsed_conditions)


# ---------------------------------------------------------------------------
# Uncovered regions
# ---------------------------------------------------------------------------

def test_uncovered_region_resolves_to_default_class_in_every_engine():
    clf, _m, X, _y = _dt_classifier()
    victim = clf.initial_rules[0]
    clf.final_rules = clf.initial_rules[1:]          # open a hole

    hole = [x for x in X if _matches(victim, dict(zip(FEATS, x)))]
    assert hole, "expected samples inside the removed rule's region"
    x0 = hole[0]
    d0 = dict(zip(FEATS, x0))
    expected = clf._default_class_int()

    clf._arrays_compiled = False
    clf.reset_fallback_counter()
    assert clf.classify(d0, final=True)[0] == expected
    assert clf.fallback_activations == 1

    with contextlib.redirect_stdout(io.StringIO()):
        clf.compile_tree_arrays(clf.final_rules, feature_names=FEATS)
    clf.reset_fallback_counter()
    assert int(clf.predict_batch(x0.reshape(1, -1), feature_names=FEATS)[0]) == expected
    assert clf.fallback_activations == 1

    # The array path is the one `classify` prefers; it must agree.
    clf.reset_fallback_counter()
    assert clf.classify(d0, final=True)[0] == expected


def test_rf_total_abstention_resolves_to_default_class():
    rng = np.random.RandomState(0)
    X = rng.randn(400, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    m = RandomForestClassifier(n_estimators=3, max_depth=3,
                               random_state=0).fit(X, y)
    rules = RuleClassifier.get_tree_rules(m, FEATS, ["0", "1"],
                                          algorithm_type="Random Forest")
    with contextlib.redirect_stdout(io.StringIO()):
        clf = RuleClassifier.generate_classifier_model(
            rules, {"0": 0, "1": 1}, "Random Forest")

    x0 = X[0]
    d0 = dict(zip(FEATS, x0))
    by_tree = {}
    for r in clf.initial_rules:
        by_tree.setdefault(r.name.split("_")[0], []).append(r)
    drop = {id(next(r for r in rs if _matches(r, d0))) for rs in by_tree.values()}
    clf.final_rules = [r for r in clf.initial_rules if id(r) not in drop]

    expected = clf._default_class_int()

    clf._arrays_compiled = False
    assert clf.classify(d0, final=True)[0] == expected

    with contextlib.redirect_stdout(io.StringIO()):
        clf.compile_tree_arrays(clf.final_rules, feature_names=FEATS)
    assert int(clf.predict_batch(x0.reshape(1, -1), feature_names=FEATS)[0]) == expected


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

def test_nan_input_is_rejected():
    clf, _m, X, _y = _dt_classifier()
    with contextlib.redirect_stdout(io.StringIO()):
        clf.compile_tree_arrays(clf.initial_rules, feature_names=FEATS)

    bad = X[:1].copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        clf.predict_batch(bad, feature_names=FEATS)

    # Must not be swallowed by the fast-path try/except in classify().
    with pytest.raises(ValueError, match="NaN"):
        clf.classify({"v1": float("nan"), "v2": 0.0, "v3": 0.0})


# ---------------------------------------------------------------------------
# Rule-set structure
# ---------------------------------------------------------------------------

def test_non_prefix_closed_rules_are_rejected():
    # Two rules that disagree on the split at the root: not a tree.
    a = Rule("R0_Class0", "0", ["v1 <= 1.0"])
    a.parsed_conditions = [("v1", "<=", 1.0)]
    b = Rule("R1_Class1", "1", ["v1 <= 2.0"])
    b.parsed_conditions = [("v1", "<=", 2.0)]

    with pytest.raises(ValueError, match="prefix-closed"):
        RuleClassifier._build_single_tree_arrays(
            [a, b], {"v1": 0}, "Decision Tree", n_classes=2)


def test_overlap_detector_flags_nested_rules_and_clears_a_tree():
    clf, _m, _X, _y = _dt_classifier()

    # Rules straight out of a tree partition the space: no overlap.
    assert clf.find_overlapping_rules(clf.initial_rules)["n_overlaps"] == 0

    # A rule nested inside another (what sibling promotion can produce).
    outer = Rule("Rule90_Class0", "0", ["v1 <= 1.0"])
    outer.parsed_conditions = [("v1", "<=", 1.0)]
    inner = Rule("Rule91_Class1", "1", ["v1 <= 1.0", "v2 <= 0.5"])
    inner.parsed_conditions = [("v1", "<=", 1.0), ("v2", "<=", 0.5)]

    res = clf.find_overlapping_rules([outer, inner])
    assert res["n_overlaps"] == 1
    assert res["n_conflicting"] == 1          # they disagree on the class

    # Touching but disjoint boxes must not be reported.
    left = Rule("Rule92_Class0", "0", ["v1 <= 1.0"])
    left.parsed_conditions = [("v1", "<=", 1.0)]
    right = Rule("Rule93_Class0", "0", ["v1 > 1.0"])
    right.parsed_conditions = [("v1", ">", 1.0)]
    assert clf.find_overlapping_rules([left, right])["n_overlaps"] == 0


def test_refinement_does_not_mutate_initial_rules():
    clf, _m, X, _y = _dt_classifier(max_depth=6)
    before = [(r.name, list(r.parsed_conditions), r.class_)
              for r in clf.initial_rules]

    # Drive usage counts, then remove the low-usage rules (this is the path
    # that promotes siblings).
    for row in X[:200]:
        clf.classify(dict(zip(FEATS, row)))
    with contextlib.redirect_stdout(io.StringIO()):
        clf.final_rules = clf._promote_siblings(
            [r for r in clf.initial_rules if r.usage_count <= 1],
            list(clf.initial_rules))

    after = [(r.name, list(r.parsed_conditions), r.class_)
             for r in clf.initial_rules]
    assert before == after, "refinement rewrote the unrefined rule set in place"
    assert clf.find_overlapping_rules(clf.initial_rules)["n_overlaps"] == 0


# ---------------------------------------------------------------------------
# Input precision
# ---------------------------------------------------------------------------

def test_float32_quantization_matches_sklearn_near_thresholds():
    clf, m, X, _y = _dt_classifier(max_depth=6, seed=1)
    with contextlib.redirect_stdout(io.StringIO()):
        clf.compile_tree_arrays(clf.initial_rules, feature_names=FEATS)

    # scikit-learn compares float32(x) against a float64 threshold; build
    # samples that sit in the resulting disagreement band.
    feat, thr = m.tree_.feature, m.tree_.threshold
    splits = [(int(f), float(t)) for f, t in zip(feat, thr) if f != -2]
    rng = np.random.RandomState(7)
    rows = []
    for _ in range(4000):
        j, t = splits[rng.randint(len(splits))]
        x = X[rng.randint(len(X))].copy()
        x[j] = t + rng.uniform(-1, 1) * 1e-7
        if (float(np.float32(x[j])) <= t) != (x[j] <= t):
            rows.append(x)
    if not rows:
        pytest.skip("no threshold-adjacent samples generated")

    Xz = np.array(rows)
    ref = m.predict(Xz)

    clf.input_dtype = "float32"
    got = clf.predict_batch(Xz, feature_names=FEATS)
    assert (got == ref).all(), (
        f"{int((got != ref).sum())}/{len(Xz)} disagree with sklearn even with "
        "float32 quantization enabled"
    )
