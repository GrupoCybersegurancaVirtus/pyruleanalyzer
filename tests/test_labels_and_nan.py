"""Tests for non-integer labels, missing values (NaN) and ExtraTrees refits.

* Labels that are not integers ('cat'/'dog', 0.5) are encoded as their codes
  0..K-1 (position in ``classes_``); ``predict``, ``classes_``, ``score``,
  ``compare_initial_final_results`` and the Python export answer the
  original labels, the C/Arduino/binary exports answer codes and document the
  table, JSON and binary store the table. Integer labels are unchanged.
* A missing value is routed at each split as scikit-learn routes it
  (``tree_.missing_go_to_left``), by every engine and export, for trees fitted
  with NaN and for trees fitted without (which still route NaN at predict
  time). Gradient Boosting refuses NaN, as scikit-learn does.
* Refinement keeps the routing consistent: a merged split takes its routing
  with it, a promoted/grafted rule keeps the routing of the splits it keeps.
* ``predict_proba`` of a Decision Tree uses the label's column, not the label.
* An analyzer wrapping an ExtraTreesClassifier refits an ExtraTreesClassifier.
"""

import contextlib
import io
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyruleanalyzer import PyRuleAnalyzer
from pyruleanalyzer.rule_classifier import (Condition, MissingCondition, Rule,
                                            RuleClassifier, condition_signature)
from cpn_semantics import CPNNet, evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEATS = ["f0", "f1", "f2", "f3"]

ESTIMATORS = {
    "DT": lambda: DecisionTreeClassifier(max_depth=6, random_state=0),
    "RF": lambda: RandomForestClassifier(n_estimators=5, max_depth=5, random_state=0),
    "ET": lambda: ExtraTreesClassifier(n_estimators=5, max_depth=5, random_state=0),
}
GBDT = lambda: GradientBoostingClassifier(n_estimators=8, max_depth=3, random_state=0)  # noqa: E731


def _data(n_classes=3, n=300, seed=0, nan_rate=0.15):
    """Features with NaN at `nan_rate`, plus labels 0..n_classes-1.

    Returns:
        tuple: (DataFrame with NaN, DataFrame without NaN, y).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, len(FEATS))
    score = X[:, 0] + X[:, 1] - 0.5 * X[:, 2]
    y = (score > 0).astype(int)
    if n_classes == 3:
        y = y + (X[:, 3] > 0.6).astype(int)
    clean = pd.DataFrame(X, columns=FEATS)
    Xn = X.copy()
    Xn[rng.rand(*X.shape) < nan_rate] = np.nan
    return pd.DataFrame(Xn, columns=FEATS), clean, y


def _quiet(fn, *args, **kwargs):
    """Call fn with its progress prints silenced."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _load_module(path):
    """Execute a generated .py file and return its namespace."""
    ns = {}
    with open(path) as f:
        exec(compile(f.read(), path, "exec"), ns)  # noqa: S102
    return ns


def _rows(df, n=None):
    """The first `n` rows of `df` as dicts."""
    return [df.iloc[i].to_dict() for i in range(len(df) if n is None else n)]


def _engines(a, df, tmp_path, use_refined=True, n_single=60):
    """Predictions of every engine of analyzer `a` on `df`, as labels.

    Returns:
        dict: engine name -> array of predicted labels (single-sample
        engines on the first `n_single` rows only).
    """
    clf = a.classifier
    rules = clf._rules_for(use_refined)
    out = {"batch": np.asarray(a.predict(df, use_refined=use_refined))}
    rows = _rows(df, n_single)
    out["single"] = np.array([a.predict(r, use_refined=use_refined) for r in rows])

    # Native exec-compiled function, for the rule set it was compiled from.
    clf.update_native_model(rules)
    f32 = [{k: float(np.float32(v)) for k, v in r.items()} for r in rows]
    out["native"] = np.array([clf._decode_labels(clf.native_fn(r)[0]) for r in f32])

    # Iterative engine: no arrays, no native function.
    saved = (clf.native_fn, clf._arrays_compiled)
    clf.native_fn, clf._arrays_compiled = None, False
    try:
        out["iterative"] = np.array([a.predict(r, use_refined=use_refined) for r in rows])
    finally:
        clf.native_fn, clf._arrays_compiled = saved
        clf.update_native_model(rules)

    # Standalone Python export.
    py = str(tmp_path / "model.py")
    _quiet(clf.export_to_native_python, filename=py, rules=rules)
    ns = _load_module(py)
    out["to_python"] = np.array([ns["predict"](r) for r in _rows(df)])

    # Binary round trip (codes, decoded with the table it stores).
    binp = str(tmp_path / "model.bin")
    state = clf._array_state_for_rules(rules, list(a.feature_names))
    saved_state = clf._current_array_state() if clf._arrays_compiled else None
    clf._set_array_state(state)
    try:
        _quiet(clf.export_to_binary, binp)
    finally:
        if saved_state is not None:
            clf._set_array_state(saved_state)
    b = PyRuleAnalyzer.load_binary(binp)
    out["binary"] = np.asarray(b.predict(df[list(b.feature_names)].values))

    # JSON round trip.
    jp = str(tmp_path / "model.json")
    a.save_json(jp)
    j = PyRuleAnalyzer.load_json(jp)
    out["json"] = np.asarray(j.predict(df, use_refined=use_refined))
    return out


def _assert_engines_equal(results, expected):
    """Every engine answers `expected` (on the rows it was run on)."""
    for name, pred in results.items():
        np.testing.assert_array_equal(pred, expected[:len(pred)], err_msg=name)


# ---------------------------------------------------------------------------
# Missing values: every engine reproduces scikit-learn
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("fit_with_nan", [True, False], ids=["fit_nan", "fit_clean"])
@pytest.mark.parametrize("est", sorted(ESTIMATORS))
def test_nan_every_engine_matches_sklearn(est, fit_with_nan, n_classes, tmp_path):
    Xn, Xc, y = _data(n_classes)
    sk = ESTIMATORS[est]().fit(Xn if fit_with_nan else Xc, y)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    expected = sk.predict(Xn)

    _assert_engines_equal(_engines(a, Xn, tmp_path), expected)
    proba = a.predict_proba(Xn)
    if est == "DT":
        np.testing.assert_array_equal(sk.classes_[proba.argmax(axis=1)], expected)
    else:
        np.testing.assert_allclose(proba, sk.predict_proba(Xn), atol=1e-12)


def test_conditions_record_the_routing_of_every_split():
    Xn, _, y = _data()
    sk = ESTIMATORS["DT"]().fit(Xn, y)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    tree = sk.tree_
    for rule in a.classifier.initial_rules:
        node = 0
        for cond in rule.parsed_conditions:
            assert isinstance(cond, Condition)
            left = cond[1] == "<="
            # NaN satisfies exactly the side the split sends it to.
            assert cond.missing == (bool(tree.missing_go_to_left[node]) == left)
            node = tree.children_left[node] if left else tree.children_right[node]
        assert tree.children_left[node] == -1


def test_gbdt_refuses_nan_like_sklearn(tmp_path):
    Xn, Xc, y = _data(2)
    sk = GBDT().fit(Xc, y)
    with pytest.raises(ValueError):
        sk.predict(Xn)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    with pytest.raises(ValueError, match="GradientBoosting"):
        a.predict(Xn)
    with pytest.raises(ValueError, match="GradientBoosting"):
        a.predict_proba(Xn)
    with pytest.raises(ValueError, match="GradientBoosting"):
        a.predict(_rows(Xn, 40)[next(i for i in range(40) if Xn.iloc[i].isna().any())])
    with pytest.raises(ValueError, match="GradientBoosting"):
        a.execute_rule_refinement(X=Xn, y=y)
    # NaN-free inputs keep working, and the exports carry no NaN routing.
    np.testing.assert_array_equal(a.predict(Xc), sk.predict(Xc))
    path = str(tmp_path / "gbdt.py")
    _quiet(a.to_python, path)
    assert "not (" not in open(path).read()
    hdr = str(tmp_path / "gbdt.h")
    _quiet(a.to_c_header, hdr)
    assert "missing_left" not in open(hdr).read()


def test_pickle_keeps_routing_and_labels():
    import pickle
    Xn, _, y0 = _data(3)
    labels = np.array(["cat", "dog", "fox"])[y0]
    sk = ESTIMATORS["ET"]().fit(Xn, labels)
    a = pickle.loads(pickle.dumps(_quiet(PyRuleAnalyzer.from_sklearn, sk)))
    conds = [c for r in a.classifier.initial_rules for c in r.parsed_conditions]
    assert all(isinstance(c, Condition) for c in conds)
    assert any(isinstance(c, MissingCondition) for c in conds)
    np.testing.assert_array_equal(a.predict(Xn), sk.predict(Xn))
    assert a.get_params()["estimator"] == "ExtraTreesClassifier"


def test_rules_without_recorded_routing_refuse_nan():
    Xn, Xc, y = _data()
    a = _quiet(PyRuleAnalyzer.from_sklearn, ESTIMATORS["RF"]().fit(Xc, y))
    clf = a.classifier
    for rule in clf.initial_rules:
        rule.parsed_conditions = [tuple(c) for c in rule.parsed_conditions]
    clf._route_missing_cache = None
    clf._array_states = []
    clf._arrays_compiled = False
    clf.update_native_model(clf.initial_rules)
    with pytest.raises(ValueError, match="NaN"):
        a.predict(Xn)
    with pytest.raises(ValueError, match="NaN"):
        a.predict({"f0": np.nan, "f1": 0.0, "f2": 0.0, "f3": 0.0})
    np.testing.assert_array_equal(a.predict(Xc), a.classifier.predict_batch(
        Xc.values, feature_names=FEATS, use_final=True))


# ---------------------------------------------------------------------------
# Refinement with missing values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("est", sorted(ESTIMATORS))
def test_refinement_with_nan_keeps_engines_consistent(est, tmp_path):
    Xn, _, y = _data(3, n=400)
    train, test = Xn.iloc[:250], Xn.iloc[250:]
    sk = ESTIMATORS[est]().fit(train, y[:250])
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    initial_signatures = {condition_signature(c) for r in a.classifier.initial_rules
                          for c in r.parsed_conditions}

    _quiet(a.execute_rule_refinement, X=test, y=y[250:],
           remove_below_n_classifications=1)
    clf = a.classifier
    assert len(clf.final_rules) < len(clf.initial_rules)

    # The initial rules still reproduce scikit-learn ...
    np.testing.assert_array_equal(a.predict(Xn, use_refined=False), sk.predict(Xn))
    # ... and every engine answers the same refined model.
    refined = _engines(a, Xn, tmp_path, use_refined=True)
    _assert_engines_equal(refined, refined["batch"])
    # Refinement never invents or flips a routing: every refined condition
    # is one of the initial ones, routing included.
    for rule in clf.final_rules:
        for cond in rule.parsed_conditions:
            assert isinstance(cond, Condition)
            assert condition_signature(cond) in initial_signatures


def _hand_tree():
    """root: x <= 0 (NaN left) -> A; else y <= 1 (NaN right) -> B | C."""
    rules = [
        Rule("Rule0_Class0", "0", ["x <= 0.0"], class_distribution=[5.0, 0.0, 0.0]),
        Rule("Rule1_Class1", "1", ["x > 0.0", "y <= 1.0"], class_distribution=[0.0, 5.0, 0.0]),
        Rule("Rule2_Class2", "2", ["x > 0.0", "y > 1.0"], class_distribution=[0.0, 0.0, 5.0]),
    ]
    rules[0].parsed_conditions = [MissingCondition(("x", "<=", 0.0))]
    rules[1].parsed_conditions = [Condition(("x", ">", 0.0)), Condition(("y", "<=", 1.0))]
    rules[2].parsed_conditions = [Condition(("x", ">", 0.0)), MissingCondition(("y", ">", 1.0))]
    clf = RuleClassifier(rules, algorithm_type="Decision Tree")
    clf.class_labels = ["0", "1", "2"]
    clf.num_classes = 3
    return clf, rules


def _predict_all_engines(clf, samples):
    """Labels of `samples` by the arrays, the native function and the rules."""
    rules = clf._rules_for(True)
    X = np.array([[s["x"], s["y"]] for s in samples])
    batch = clf.predict_batch(X, feature_names=["x", "y"], use_final=True).tolist()
    clf.update_native_model(rules)
    native = [clf.native_fn(s)[0] for s in samples]
    iterative = [int(clf.classify_dt(s, rules).class_) for s in samples]
    return batch, native, iterative


def test_grafting_drops_the_routing_of_the_removed_split():
    clf, rules = _hand_tree()
    nan_x = [{"x": np.nan, "y": 0.5}, {"x": np.nan, "y": np.nan}, {"x": np.nan, "y": 3.0}]
    # Before: x is NaN -> left -> class 0, whatever y is.
    for engine in _predict_all_engines(clf, nan_x):
        assert engine == [0, 0, 0]

    # Remove A: the y-split is grafted into the root's place; it routes
    # every sample, and a NaN y goes where that split sends it (right: C).
    clf.final_rules = _quiet(clf._promote_siblings, [rules[0]], list(rules))
    assert [len(r.parsed_conditions) for r in clf.final_rules] == [1, 1]
    for r in clf.final_rules:
        assert all(isinstance(c, Condition) for c in r.parsed_conditions)
    for engine in _predict_all_engines(clf, nan_x):
        assert engine == [1, 2, 2]
    # The unrefined model is untouched.
    assert clf.classify({"x": np.nan, "y": 0.5}, final=False)[0] == 0


def test_promotion_and_merge_keep_routing_consistent():
    clf, rules = _hand_tree()
    # Promote: remove C, its sibling leaf B takes the parent's place (drops
    # y <= 1, and with it that split's routing).
    clf.final_rules = _quiet(clf._promote_siblings, [rules[2]], list(rules))
    promoted = next(r for r in clf.final_rules if r.name.endswith("_promoted"))
    assert promoted.parsed_conditions == [("x", ">", 0.0)]
    assert promoted.parsed_conditions[0].missing is False
    samples = [{"x": np.nan, "y": 3.0}, {"x": 1.0, "y": np.nan}, {"x": 1.0, "y": 3.0}]
    for engine in _predict_all_engines(clf, samples):
        assert engine == [0, 1, 1]

    # Merge: B and C made the same class -> merged into "x > 0"; the merged
    # split's routing disappears with it, the root's stays.
    clf, rules = _hand_tree()
    rules[2].class_ = "1"
    rules[2].class_distribution = [0.0, 5.0, 0.0]
    clf.final_rules = []
    pairs = clf.merge_boundary_redundancy()
    assert len(pairs) == 1
    merged = [r for r in clf.final_rules if "_&_" in r.name][0]
    assert merged.parsed_conditions == [("x", ">", 0.0)]
    assert merged.parsed_conditions[0].missing is False
    for engine in _predict_all_engines(clf, samples):
        assert engine == [0, 1, 1]


def test_inconsistent_routing_is_refused():
    clf, rules = _hand_tree()
    # B says NaN satisfies "x > 0" too: two sides claim the missing values.
    rules[1].parsed_conditions[0] = MissingCondition(("x", ">", 0.0))
    with pytest.raises(ValueError, match="missing values"):
        RuleClassifier._tree_from_rules(rules)


# ---------------------------------------------------------------------------
# Non-integer labels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("est", sorted(ESTIMATORS) + ["GBDT"])
def test_string_labels_every_engine(est, n_classes, tmp_path):
    _, Xc, y0 = _data(n_classes)
    names = ["cat", "dog", "fox"][:n_classes]
    labels = np.array(names)[y0]
    sk = (GBDT() if est == "GBDT" else ESTIMATORS[est]()).fit(Xc, labels)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    clf = a.classifier

    assert clf.label_table == names
    assert clf.class_labels == [str(i) for i in range(n_classes)]
    np.testing.assert_array_equal(a.classes_, sk.classes_)
    expected = sk.predict(Xc)
    _assert_engines_equal(_engines(a, Xc, tmp_path), expected)
    assert a.score(Xc, labels) == pytest.approx(sk.score(Xc, labels))
    if est != "DT":
        np.testing.assert_allclose(a.predict_proba(Xc), sk.predict_proba(Xc), atol=1e-9)

    # C and Arduino answer the codes and document the table.
    hdr = str(tmp_path / "m.h")
    _quiet(a.to_c_header, hdr)
    text = open(hdr).read()
    assert "0 = cat" in text and f"{n_classes - 1} = {names[-1]}" in text
    ino = str(tmp_path / "m.ino")
    _quiet(clf.export_to_arduino_ino, ino)
    assert "//   1 = dog" in open(ino, encoding="utf-8").read()
    # The binary model stores the table: a reloaded model answers labels.
    b = PyRuleAnalyzer.load_binary(str(tmp_path / "model.bin"))
    assert b.classifier.label_table == names
    np.testing.assert_array_equal(b.classifier.predict_batch(Xc.values),
                                  clf._encode_labels(expected))


def test_float_labels_and_refinement(tmp_path, monkeypatch):
    # scikit-learn refuses non-integral float targets ("continuous"), but
    # class_names may name the classes with any labels.
    Xn, _, y0 = _data(3, n=400)
    table = np.array([0.5, 1.5, 2.25])
    labels = table[y0]
    sk = ESTIMATORS["RF"]().fit(Xn.iloc[:250], y0[:250])
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk, class_names=list(table))
    assert a.classifier.label_table == [0.5, 1.5, 2.25]
    np.testing.assert_array_equal(a.classes_, table)
    np.testing.assert_array_equal(a.predict(Xn), table[sk.predict(Xn)])

    _quiet(a.execute_rule_refinement, X=Xn.iloc[250:], y=labels[250:],
           remove_below_n_classifications=1)
    pred = a.predict(Xn)
    assert set(np.unique(pred)) <= {0.5, 1.5, 2.25}
    assert 0.0 <= a.score(Xn, labels) <= 1.0

    # JSON keeps the labels' type; the refined model answers the same.
    path = str(tmp_path / "m.json")
    a.save_json(path)
    b = PyRuleAnalyzer.load_json(path)
    assert b.classifier.label_table == [0.5, 1.5, 2.25]
    assert b.class_names == [0.5, 1.5, 2.25]
    np.testing.assert_array_equal(b.predict(Xn), pred)

    # The comparison report shows the original labels.
    monkeypatch.chdir(tmp_path)
    os.makedirs("files", exist_ok=True)
    _quiet(a.compare_initial_final_results, X=Xn.iloc[250:], y=labels[250:])
    report = open("files/output_final_classifier_rf.txt").read()
    assert "1.5" in report


def test_fit_with_string_labels_and_nan(tmp_path):
    Xn, _, y0 = _data(3)
    labels = np.array(["low", "mid", "high"])[y0]
    a = PyRuleAnalyzer.new_model("Decision Tree", {"max_depth": 5, "random_state": 0})
    _quiet(a.fit, Xn, labels)
    sk = DecisionTreeClassifier(max_depth=5, random_state=0).fit(Xn, labels)
    np.testing.assert_array_equal(a.predict(Xn), sk.predict(Xn))
    assert a.predict(Xn.iloc[0].to_dict()) == sk.predict(Xn.iloc[[0]])[0]
    np.testing.assert_array_equal(a.classes_, sk.classes_)


def test_string_labels_cpn_uses_codes(tmp_path):
    _, Xc, y0 = _data(3)
    labels = np.array(["cat", "dog", "fox"])[y0]
    sk = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xc, labels)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    paths = _quiet(a.export_hcpn, str(tmp_path / "dt"), which="initial",
                   sample=Xc.iloc[0])
    net = CPNNet(paths["initial"])
    for i in range(30):
        x = {f"f{j}": float(np.float32(Xc.iloc[i, j])) for j in range(4)}
        code = evaluate(net, x)
        assert a.classifier._decode_labels(code) == sk.predict(Xc.iloc[[i]])[0]
    result = _quiet(a.model_check, which="initial", cpn_path=paths["initial"],
                    samples=Xc.iloc[:1], test_samples=Xc.iloc[:20], verbose=False)
    assert result.passed, result.report()


def test_integer_labels_are_unchanged(tmp_path):
    _, Xc, y0 = _data(3)
    sk = ESTIMATORS["RF"]().fit(Xc, y0 + 10)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    assert a.classifier.label_table is None
    assert a.class_names == ["10", "11", "12"]
    np.testing.assert_array_equal(a.classes_, [10, 11, 12])
    path = str(tmp_path / "m.py")
    _quiet(a.to_python, path)
    assert "_LABELS" not in open(path).read()


def test_legacy_models_are_written_as_before(tmp_path):
    """No routing, integer labels: JSON version 1, binary version <= 3,
    C header without missing-value arrays."""
    _, Xc, y0 = _data(3)
    a = _quiet(PyRuleAnalyzer.from_sklearn, ESTIMATORS["DT"]().fit(Xc, y0))
    clf = a.classifier
    for rule in clf.initial_rules:
        rule.parsed_conditions = [tuple(c) for c in rule.parsed_conditions]
    clf._array_states, clf._arrays_compiled = [], False
    data = clf.to_dict()
    assert data["version"] == 1
    assert "label_table" not in data and "sklearn_estimator" not in data
    assert all(len(c) == 3 for r in data["rules"] for c in r["parsed_conditions"])
    clf._set_array_state(clf._array_state_for_rules(clf.initial_rules, FEATS))
    binp = str(tmp_path / "m.bin")
    _quiet(clf.export_to_binary, binp)
    assert open(binp, "rb").read()[4] <= 3
    hdr = str(tmp_path / "m.h")
    _quiet(clf.export_to_c_header, hdr)
    assert "missing_left" not in open(hdr).read()


def test_json_version_1_still_loads(tmp_path):
    import json
    Xn, Xc, y0 = _data(3)
    a = _quiet(PyRuleAnalyzer.from_sklearn, ESTIMATORS["DT"]().fit(Xc, y0))
    path = str(tmp_path / "m.json")
    a.save_json(path)
    data = json.load(open(path))
    assert data["classifier"]["version"] == 2
    data["classifier"]["version"] = 1
    del data["classifier"]["label_table"], data["classifier"]["sklearn_estimator"]
    for r in data["classifier"]["rules"]:
        r["parsed_conditions"] = [c[:3] for c in r["parsed_conditions"]]
    json.dump(data, open(path, "w"))
    b = PyRuleAnalyzer.load_json(path)
    np.testing.assert_array_equal(b.predict(Xc), a.predict(Xc))
    with pytest.raises(ValueError, match="NaN"):
        b.predict(Xn)


# ---------------------------------------------------------------------------
# Decision Tree predict_proba column mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("offset,step", [(1, 1), (10, 10)])
def test_dt_predict_proba_uses_the_label_column(offset, step, tmp_path):
    _, Xc, y0 = _data(3)
    y = offset + step * y0
    sk = ESTIMATORS["DT"]().fit(Xc, y)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    proba = a.predict_proba(Xc)
    assert proba.shape == (len(Xc), 3)
    np.testing.assert_array_equal(a.classes_[proba.argmax(axis=1)], sk.predict(Xc))
    np.testing.assert_array_equal(proba.sum(axis=1), 1.0)
    # A reloaded binary model keeps the column labels of a tree too.
    binp = str(tmp_path / "m.bin")
    _quiet(a.to_binary, binp)
    b = PyRuleAnalyzer.load_binary(binp)
    np.testing.assert_array_equal(b.predict_proba(Xc.values), proba)


# ---------------------------------------------------------------------------
# ExtraTrees refit
# ---------------------------------------------------------------------------

def test_extra_trees_refit_trains_extra_trees(tmp_path):
    _, Xc, y = _data(3)
    et = ExtraTreesClassifier(n_estimators=4, max_depth=4, random_state=3).fit(Xc, y)
    a = _quiet(PyRuleAnalyzer.from_sklearn, et)
    assert a.get_params()["estimator"] == "ExtraTreesClassifier"
    assert a.get_params()["model"] == "Random Forest"

    b = _quiet(clone(a).fit, Xc, y)
    assert b.classifier.sklearn_estimator == "ExtraTreesClassifier"
    np.testing.assert_array_equal(b.predict(Xc), et.predict(Xc))
    np.testing.assert_allclose(b.predict_proba(Xc), et.predict_proba(Xc))

    c = _quiet(a.fit, Xc, y)
    np.testing.assert_array_equal(c.predict(Xc), et.predict(Xc))

    # The class survives JSON; a RandomForest analyzer does not report one.
    path = str(tmp_path / "m.json")
    a.save_json(path)
    assert PyRuleAnalyzer.load_json(path).get_params()["estimator"] == "ExtraTreesClassifier"
    rf = _quiet(PyRuleAnalyzer.from_sklearn, ESTIMATORS["RF"]().fit(Xc, y))
    assert "estimator" not in rf.get_params()

    # set_params(model=...) keeps its semantics: a new type, its default class.
    a.set_params(model="Decision Tree")
    assert "estimator" not in a.get_params()
    a.set_params(model="Random Forest", estimator="ExtraTreesClassifier")
    assert a.get_params()["estimator"] == "ExtraTreesClassifier"
    with pytest.raises(ValueError):
        a.set_params(estimator="GradientBoostingClassifier")


def test_extra_tree_classifier_refit():
    _, Xc, y = _data(3)
    et = ExtraTreeClassifier(max_depth=5, random_state=1).fit(Xc, y)
    a = _quiet(PyRuleAnalyzer.from_sklearn, et)
    assert a.get_params()["estimator"] == "ExtraTreeClassifier"
    b = _quiet(clone(a).fit, Xc, y)
    np.testing.assert_array_equal(b.predict(Xc), et.predict(Xc))


# ---------------------------------------------------------------------------
# CPN: real-valued inputs exactly, NaN refused
# ---------------------------------------------------------------------------

def test_cpn_handles_missing_only_splits_and_refuses_nan(tmp_path):
    Xn, Xc, y = _data(2)
    sk = DecisionTreeClassifier(max_depth=4, random_state=0).fit(Xn, y)
    assert np.isinf(sk.tree_.threshold).any()   # a split isolating NaN
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    paths = _quiet(a.export_hcpn, str(tmp_path / "dt"), which="initial",
                   sample=Xc.iloc[0])
    net = CPNNet(paths["initial"])
    for i in range(40):
        x = {f"f{j}": float(np.float32(Xc.iloc[i, j])) for j in range(4)}
        assert evaluate(net, x) == sk.predict(Xc.iloc[[i]])[0]
    with pytest.raises(ValueError, match="NaN"):
        evaluate(net, {"f0": float("nan"), "f1": 0.0, "f2": 0.0, "f3": 0.0})
    nan_row = Xn[Xn.isna().any(axis=1)].iloc[0]
    with pytest.raises(ValueError, match="NaN"):
        _quiet(a.export_hcpn, str(tmp_path / "dt2"), which="initial", sample=nan_row)


# ---------------------------------------------------------------------------
# C header, compiled (when a C compiler is on PATH)
# ---------------------------------------------------------------------------

def _c_compiler():
    """A C compiler command on PATH (gcc/cc/clang), or None."""
    for name in ("gcc", "cc", "clang"):
        path = shutil.which(name)
        if path:
            return path
    return None


@pytest.mark.parametrize("est", ["DT", "RF"])
def test_c_header_routes_nan_like_sklearn(est, tmp_path):
    compiler = _c_compiler()
    if compiler is None:
        pytest.skip("no C compiler on PATH")
    Xn, _, y = _data(3)
    sk = ESTIMATORS[est]().fit(Xn, y)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    hdr = tmp_path / "model.h"
    _quiet(a.to_c_header, str(hdr))
    rows = Xn.values[:80]
    lit = lambda v: "NAN" if np.isnan(v) else repr(float(v))  # noqa: E731
    data = ",\n".join("{" + ", ".join(lit(v) for v in r) + "}" for r in rows)
    (tmp_path / "main.c").write_text(
        '#include <stdio.h>\n#include <math.h>\n#include "model.h"\n'
        f"static const double X[{len(rows)}][{rows.shape[1]}] = {{{data}}};\n"
        "int main(void) { int i; for (i = 0; i < "
        f"{len(rows)}; i++) printf(\"%d\\n\", (int)predict(X[i])); return 0; }}\n")
    exe = str(tmp_path / "main")
    subprocess.run([compiler, "-O1", "-o", exe, str(tmp_path / "main.c"), "-lm"],
                   check=True, cwd=str(tmp_path))
    out = subprocess.run([exe], check=True, capture_output=True, text=True).stdout
    np.testing.assert_array_equal([int(v) for v in out.split()], sk.predict(Xn.iloc[:80]))
