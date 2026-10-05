"""Fidelity of rule addressing and refinement, across every engine.

The rule set is the model: a sample is routed, in each tree, to the one rule
whose region contains it (none after refinement: the tree abstains). This file
checks that every engine computes exactly that, for the initial and for the
refined rule set, and that the refinement does what the paper describes.

The reference semantics below is written from the definitions, not from the
engines, and is the same function the CPN computes (catch-all transitions
abstain):

- DT: the label of the matching rule, `default_class` when none matches;
- RF: sum over trees of the matching leaf's normalised distribution,
  argmax (lowest index on ties), `default_class` when every tree abstains;
- GBDT: per class, init score + sum of the matching contributions; binary
  decides on score >= 0 (as scikit-learn does), multiclass on argmax.

Engines compared: the public API (`predict`, `classify`), the compiled tree
arrays (`predict_batch`), the exec-compiled native function, the iterative
engine, the standalone Python export, the binary format (`load_binary`), the
C header (compiled with gcc when available) and the coloured-net engine of the
model checker.
"""
import contextlib
import importlib.util
import io
import os
import shutil
import subprocess
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
from pyruleanalyzer.rule_classifier import Rule, RuleClassifier  # noqa: E402

OPS = {"<=": np.less_equal, ">": np.greater, "<": np.less, ">=": np.greater_equal}
NAMES = [f"f{i}" for i in range(6)]
GCC = shutil.which("gcc")


# ---------------------------------------------------------------------------
# Reference semantics
# ---------------------------------------------------------------------------

# Function to tell the tree a rule belongs to.
def tree_of(clf, rule):
    """Tree id: a DT is one tree; ensembles use the name prefix."""
    if clf.algorithm_type == "Decision Tree":
        return "_all"
    return rule.name.split("_")[0] if "_" in rule.name else rule.name


# Function to evaluate every rule on every sample.
def match_matrix(rules, X):
    """Boolean (n_samples, n_rules): does the sample satisfy the rule?"""
    idx = {n: i for i, n in enumerate(NAMES)}
    M = np.ones((len(X), len(rules)), bool)
    for j, r in enumerate(rules):
        for var, op, thr in r.parsed_conditions:
            M[:, j] &= OPS[op](X[:, idx[var]], thr)
    return M


# Function to compute the reference prediction.
def reference(clf, rules, X):
    """Predictions of `rules` on X from the definitions (see module doc).

    Returns:
        tuple: (predictions, number of (sample, tree) pairs matching >1 rule).
    """
    M = match_matrix(rules, X)
    trees = {}
    for j, r in enumerate(rules):
        if r.parsed_conditions or clf.algorithm_type != "Gradient Boosting Decision Trees":
            trees.setdefault(tree_of(clf, r), []).append(j)
    overlaps = sum(int((M[:, cols].sum(1) > 1).sum()) for cols in trees.values())
    default = clf._default_class_int()
    algo = clf.algorithm_type
    if algo == "Decision Tree":
        out = []
        for i in range(len(X)):
            hit = np.flatnonzero(M[i])
            out.append(int(rules[hit[0]].class_) if len(hit) else default)
        return np.array(out), overlaps
    if algo == "Random Forest":
        out = []
        for i in range(len(X)):
            s = np.zeros(clf.num_classes)
            for cols in trees.values():
                hit = [j for j in cols if M[i, j]]
                if hit:
                    d = np.asarray(rules[hit[0]].class_distribution, float)
                    s += d / d.sum()
            out.append(default if not s.any() else int(np.argmax(s)))
        return np.array(out), overlaps
    classes = clf._gbdt_classes
    S = {c: np.full(len(X), clf._gbdt_init_scores.get(c, 0.0)) for c in classes}
    for j, r in enumerate(rules):
        if r.parsed_conditions:
            S[r.class_group] = S[r.class_group] + np.where(M[:, j], r.contribution, 0.0)
    if clf._gbdt_is_binary:
        return np.where(S[classes[1]] >= 0, int(classes[1]), int(classes[0])), overlaps
    best = np.argmax(np.stack([S[c] for c in classes], 1), 1)
    return np.array([int(classes[k]) for k in best]), overlaps


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

KINDS = ("dt", "rf", "gbdt2", "gbdt3")


# Function to train a model and extract its rules.
def build(kind, seed=0, tau=-1, between=False):
    """Train, extract, refine; return (analyzer, sklearn model, X_ref, y_ref, X_eval)."""
    n_classes = 3 if kind in ("rf", "gbdt3") else 2
    X, y = make_classification(n_samples=900, n_features=6, n_informative=4,
                               n_redundant=0, n_classes=n_classes,
                               n_clusters_per_class=1, random_state=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=seed)
    if kind == "dt":
        model = DecisionTreeClassifier(max_depth=6, random_state=seed)
    elif kind == "rf":
        model = RandomForestClassifier(n_estimators=12, max_depth=5, random_state=seed)
    else:
        model = GradientBoostingClassifier(n_estimators=12, max_depth=3, random_state=seed)
    model.fit(X_tr, y_tr)
    analyzer = PyRuleAnalyzer.from_sklearn(model, NAMES)
    # Refinement data and evaluation data are kept apart.
    X_ref, X_ev, y_ref, _ = train_test_split(X_te, y_te, test_size=0.5, random_state=seed)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analyzer.execute_rule_refinement(X=X_ref, y=y_ref,
                                         remove_below_n_classifications=tau,
                                         refine_between_trees=between)
    return analyzer, model, X_ref, y_ref, np.vstack([X_ev, boundary_samples(analyzer, X_ev)])


# Function to put samples on both sides of every threshold.
def boundary_samples(analyzer, X, limit=400):
    """Samples at, just above and just below the thresholds of the rules.

    Just above a threshold (`nextafter`) is where float64 and float32
    comparisons disagree; the threshold itself is where `<=` vs `<` matters.
    """
    rng = np.random.default_rng(0)
    rows = []
    thr = sorted({(v, t) for r in analyzer.classifier.initial_rules
                  for v, _, t in r.parsed_conditions})
    for v, t in thr:
        for value in (t, np.nextafter(t, np.inf), np.nextafter(t, -np.inf),
                      float(np.float32(t)), float(np.nextafter(np.float32(t), np.float32(np.inf)))):
            row = X[rng.integers(len(X))].copy()
            row[NAMES.index(v)] = value
            rows.append(row)
    rows = np.array(rows)
    return rows[rng.permutation(len(rows))[:limit]]


# Function to quantize like the classifier does.
def f32(X):
    """Inputs as the classifier compares them (float32-rounded)."""
    return np.asarray(X, dtype=np.float32).astype(np.float64)


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

# Function to predict with the iterative (rule-by-rule) engine.
def iterative(clf, rules, X):
    """The rule-by-rule engine on the float32-rounded samples."""
    out = []
    for x in f32(X):
        d = dict(zip(NAMES, x))
        if clf.algorithm_type == "Decision Tree":
            r = RuleClassifier.classify_dt(d, rules)
            out.append(int(r.class_) if r else clf._default_class_int())
        elif clf.algorithm_type == "Random Forest":
            p = RuleClassifier.classify_rf(d, rules)[0]
            out.append(clf._default_class_int() if p is None else int(p))
        else:
            out.append(int(RuleClassifier.classify_gbdt(
                d, rules, clf._gbdt_init_scores, clf._gbdt_is_binary,
                clf._gbdt_classes)[0]))
    return np.array(out)


# Function to predict with the exec-compiled native function.
def native(clf, rules, X):
    """The exec-compiled function, compiled from `rules`."""
    clf.update_native_model(rules)
    assert clf.native_fn is not None
    return np.array([int(clf.native_fn(dict(zip(NAMES, x)))[0]) for x in f32(X)])


# Function to predict with the standalone Python export.
def python_export(clf, rules, X, tmp_path):
    """The standalone .py file, imported and run."""
    path = tmp_path / f"exported_{id(rules)}.py"
    with contextlib.redirect_stdout(io.StringIO()):
        clf.export_to_native_python(filename=str(path), rules=rules)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return np.array([int(module.predict(dict(zip(NAMES, x)))) for x in X])


# Function to predict with the binary format.
def binary(analyzer, use_final, X, tmp_path):
    """export(formats=['binary']) then load_binary().predict_batch."""
    base = str(tmp_path / f"model_{int(use_final)}")
    with contextlib.redirect_stdout(io.StringIO()):
        analyzer.export(base_name=base, formats=["binary"], use_refined=use_final)
    loaded = RuleClassifier.load_binary(base + ".bin")
    return np.asarray(loaded.predict_batch(X)).astype(int)


# Function to predict with the C header.
def c_header(analyzer, use_final, X, tmp_path):
    """export(formats=['c']), compiled with gcc, fed the samples on stdin."""
    base = str(tmp_path / f"model_{int(use_final)}")
    with contextlib.redirect_stdout(io.StringIO()):
        analyzer.export(base_name=base, formats=["c"], use_refined=use_final)
    driver = tmp_path / f"driver_{int(use_final)}.c"
    driver.write_text(
        '#include <stdio.h>\n'
        f'#include "{os.path.basename(base)}.h"\n'
        'int main(void) {\n'
        '    double f[N_FEATURES];\n'
        '    for (;;) {\n'
        '        for (int i = 0; i < N_FEATURES; i++)\n'
        '            if (scanf("%lf", &f[i]) != 1) return 0;\n'
        '        printf("%d\\n", (int)predict(f));\n'
        '    }\n'
        '}\n')
    exe = tmp_path / f"driver_{int(use_final)}.exe"
    subprocess.run([GCC, "-O1", "-o", str(exe), str(driver), "-lm"], check=True,
                   cwd=str(tmp_path), capture_output=True)
    feed = "\n".join(" ".join(repr(float(v)) for v in row) for row in X) + "\n"
    out = subprocess.run([str(exe)], input=feed, capture_output=True, text=True,
                         check=True).stdout.split()
    return np.array([int(v) for v in out])


# Function to predict with the coloured-net engine.
def cpn(analyzer, use_final, X, tmp_path):
    """The generated .cpn run by the model checker's coloured engine."""
    from pyruleanalyzer.model_checker import CPNModelChecker
    which = "final" if use_final else "initial"
    base = str(tmp_path / "net")
    with contextlib.redirect_stdout(io.StringIO()):
        paths = analyzer.export_hcpn(base, which=which, sample=X[0])
    mc = CPNModelChecker(paths[which], feature_names=NAMES)
    out = []
    for x in f32(X):
        # Which places carry the input is the profile's to say, not the net's.
        start = mc.cnet.initial_marking(tuple(x), places=mc.input_places)
        tokens = mc.cnet.run(start)[mc.structure.pred]
        assert len(tokens) == 1
        out.append(int(tokens[0]))
    return np.array(out)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
def test_initial_rules_equal_sklearn_on_boundaries(kind):
    """Theorem 1 in practice: the extracted rules are the model, bit for bit,
    including samples on and one ULP around every threshold."""
    analyzer, model, _, _, X = build(kind)
    clf = analyzer.classifier
    ref, overlaps = reference(clf, clf.initial_rules, f32(X))
    assert overlaps == 0
    np.testing.assert_array_equal(ref, model.predict(X))
    np.testing.assert_array_equal(analyzer.predict(X, use_refined=False), model.predict(X))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("tau", (-1, 0, 3))
def test_every_engine_routes_like_the_reference(kind, tau, tmp_path):
    """Initial and refined rule sets, every in-process engine and export."""
    between = kind.startswith("gbdt") and tau == 3
    analyzer, model, _, _, X = build(kind, seed=1, tau=tau, between=between)
    clf = analyzer.classifier
    for use_final, rules in ((False, clf.initial_rules), (True, clf.final_rules)):
        ref, overlaps = reference(clf, rules, f32(X))
        assert overlaps == 0, "a tree's rules overlap: not a partition"
        engines = {
            "predict": analyzer.predict(X, use_refined=use_final),
            "classify": [clf.classify(dict(zip(NAMES, x)), final=use_final)[0] for x in X],
            "predict_batch": clf.predict_batch(X, feature_names=NAMES, use_final=use_final),
            "iterative": iterative(clf, rules, X),
            "native": native(clf, rules, X),
            "python_export": python_export(clf, rules, X, tmp_path),
            "binary": binary(analyzer, use_final, X, tmp_path),
        }
        if GCC:
            engines["c_header"] = c_header(analyzer, use_final, X, tmp_path)
        for name, pred in engines.items():
            bad = np.flatnonzero(np.asarray(pred).astype(int) != ref)
            assert not len(bad), (f"{name} ({'final' if use_final else 'initial'}) "
                                  f"disagrees on {len(bad)}/{len(X)} samples")
        if not use_final:
            np.testing.assert_array_equal(ref, model.predict(X))


@pytest.mark.parametrize("kind", KINDS)
def test_coloured_net_routes_like_the_reference(kind, tmp_path):
    """The CPN (with catch-all transitions) computes the same function."""
    analyzer, _, _, _, X = build(kind, seed=2, tau=2)
    clf = analyzer.classifier
    X = X[:120]
    for use_final, rules in ((False, clf.initial_rules), (True, clf.final_rules)):
        ref, _ = reference(clf, rules, f32(X))
        np.testing.assert_array_equal(cpn(analyzer, use_final, X, tmp_path), ref)


@pytest.mark.parametrize("kind", KINDS)
def test_uncovered_regions_resolve_the_same_everywhere(kind, tmp_path):
    """Refinement can leave regions no rule covers (Remark 3.3). Remove the
    leaves of a few samples from *every* tree, with a default class that is
    not 0 (so an engine that falls back to argmax of nothing shows), and
    check that every engine, the CPN included, abstains the same way."""
    analyzer, _, _, _, X = build(kind, seed=7)
    clf = analyzer.classifier
    X = X[:150]
    clf.default_class = str(clf.num_classes - 1)
    clf._array_states, clf._arrays_compiled = [], False
    M = match_matrix(clf.initial_rules, f32(X[:3]))
    drop = {j for j, r in enumerate(clf.initial_rules)
            if M[:, j].any() and r.parsed_conditions}
    clf.final_rules = [r for j, r in enumerate(clf.initial_rules) if j not in drop]
    ref, _ = reference(clf, clf.final_rules, f32(X))
    if kind in ("dt", "rf"):
        assert (ref[:3] == clf.num_classes - 1).all()
    engines = {
        "predict": analyzer.predict(X, use_refined=True),
        "classify": [clf.classify(dict(zip(NAMES, x)), final=True)[0] for x in X],
        "iterative": iterative(clf, clf.final_rules, X),
        "native": native(clf, clf.final_rules, X),
        "python_export": python_export(clf, clf.final_rules, X, tmp_path),
        "binary": binary(analyzer, True, X, tmp_path),
        "cpn": cpn(analyzer, True, X, tmp_path),
    }
    if GCC:
        engines["c_header"] = c_header(analyzer, True, X, tmp_path)
    for name, pred in engines.items():
        np.testing.assert_array_equal(np.asarray(pred).astype(int), ref, err_msg=name)


@pytest.mark.parametrize("kind", KINDS)
def test_predict_honours_the_rule_set_flag(kind):
    """use_refined=False is the unrefined model even after arrays of the
    refined one were compiled, and vice versa (the reported defect)."""
    analyzer, model, _, _, X = build(kind, seed=3, tau=3)
    clf = analyzer.classifier
    ref_final, _ = reference(clf, clf.final_rules, f32(X))
    assert (ref_final != model.predict(X)).any(), "refinement changed nothing"
    for _ in range(2):   # order of the calls must not matter
        clf.compile_tree_arrays(clf.final_rules, feature_names=NAMES)
        np.testing.assert_array_equal(analyzer.predict(X, use_refined=False), model.predict(X))
        clf.compile_tree_arrays(clf.initial_rules, feature_names=NAMES)
        np.testing.assert_array_equal(analyzer.predict(X, use_refined=True), ref_final)
    for x, want_i, want_f in zip(X[:60], model.predict(X[:60]), ref_final[:60]):
        d = dict(zip(NAMES, x))
        assert clf.classify(d, final=False)[0] == want_i
        assert clf.classify(d, final=True)[0] == want_f


@pytest.mark.parametrize("kind", KINDS)
def test_boundary_merge_preserves_the_model_and_reaches_a_fixpoint(kind):
    """Merging sibling leaves with the same output changes no prediction and
    repeats until no pair is left (paper, Sec. IV-A)."""
    analyzer, model, _, _, X = build(kind, seed=4, tau=-1)
    clf = analyzer.classifier
    ref, overlaps = reference(clf, clf.final_rules, f32(X))
    assert overlaps == 0
    np.testing.assert_array_equal(ref, model.predict(X))
    assert clf.find_duplicated_rules() == []


def test_boundary_merge_cascades_to_the_root():
    """A merge that makes the parent redundant with its sibling is followed up:
    three leaves of one class under two splits collapse into one rule."""
    mk = lambda name, cls, conds: _rule(name, cls, conds, dist=[1.0, 0.0] if cls == "0" else [0.0, 1.0])
    rules = [mk("R0_Class0", "0", [("f0", "<=", 1.0), ("f1", "<=", 2.0)]),
             mk("R1_Class0", "0", [("f0", "<=", 1.0), ("f1", ">", 2.0)]),
             mk("R2_Class0", "0", [("f0", ">", 1.0)])]
    clf = _classifier(rules, "Decision Tree")
    pairs = clf.merge_boundary_redundancy()
    assert len(pairs) == 2
    assert len(clf.final_rules) == 1 and clf.final_rules[0].parsed_conditions == []


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("tau", (0, 3))
def test_usage_counts_are_the_coverage_of_the_final_rules(kind, tau):
    """Low-usage refinement keeps a rule iff it matches more than tau of the
    refinement samples (paper: FilterLowUsage). Merged and promoted rules must
    carry the usage of the regions they absorbed; before the fix merged rules
    started at 0 and were deleted by the very stage that reads the counts."""
    analyzer, _, X_ref, _, _ = build(kind, seed=5, tau=tau)
    clf = analyzer.classifier
    M = match_matrix(clf.final_rules, f32(X_ref))
    for j, r in enumerate(clf.final_rules):
        if not r.parsed_conditions and r.contribution is not None:
            continue   # GBDT init score: never a removal candidate
        assert r.usage_count == int(M[:, j].sum()), r.name
        assert r.usage_count > tau, f"{r.name} kept with usage {r.usage_count} <= {tau}"


def test_merged_rules_survive_low_usage_filtering():
    """A merged parent that is used must not be removed as 'unused'."""
    analyzer, _, X_ref, _, _ = build("dt", seed=0, tau=0)
    merged = [r for r in analyzer.classifier.final_rules if "_&_" in r.name]
    assert merged, "the fixture should produce boundary merges"
    assert all(r.usage_count > 0 for r in merged)


def test_gbdt_semantic_merge_preserves_scores():
    """Inter-tree merge of rules with identical regions: the merged rule adds
    what the separate rules added, so predictions are unchanged."""
    analyzer, model, _, _, X = build("gbdt3", seed=6, tau=-1, between=True)
    clf = analyzer.classifier
    across = [r for r in clf.final_rules
              if len({part.split("_")[0] for part in r.name.split("_&_")}) > 1]
    assert across, "the fixture should merge rules of different trees"
    ref, overlaps = reference(clf, clf.final_rules, f32(X))
    assert overlaps == 0
    np.testing.assert_array_equal(ref, model.predict(X))


# ---------------------------------------------------------------------------
# Structural guarantees of the compiled engines
# ---------------------------------------------------------------------------

def _rule(name, cls, conds, dist=None, value=None, lr=None, group=None):
    r = Rule(name, cls, [f"{v} {op} {t}" for v, op, t in conds],
             class_distribution=dist, leaf_value=value, learning_rate=lr,
             class_group=group)
    r.parsed_conditions = list(conds)
    return r


def _classifier(rules, algo):
    clf = RuleClassifier(rules, algorithm_type=algo)
    clf.class_labels = ["0", "1"]
    clf.num_classes = 2
    return clf


@pytest.mark.parametrize("rules", [
    # a rule whose region contains another rule's region
    [_rule("A_Class0", "0", [("f0", "<=", 1.0)], dist=[3, 1]),
     _rule("B_Class1", "1", [("f0", "<=", 1.0), ("f1", "<=", 2.0)], dist=[0, 2]),
     _rule("C_Class1", "1", [("f0", ">", 1.0)], dist=[0, 5])],
    # same, shorter rule listed last
    [_rule("B_Class1", "1", [("f0", "<=", 1.0), ("f1", "<=", 2.0)], dist=[0, 2]),
     _rule("C_Class1", "1", [("f0", ">", 1.0)], dist=[0, 5]),
     _rule("A_Class0", "0", [("f0", "<=", 1.0)], dist=[3, 1])],
    # two rules for one region
    [_rule("A_Class0", "0", [("f0", "<=", 1.0)], dist=[3, 1]),
     _rule("B_Class1", "1", [("f0", "<=", 1.0)], dist=[0, 2]),
     _rule("C_Class1", "1", [("f0", ">", 1.0)], dist=[0, 5])],
])
def test_compiled_engines_refuse_rules_that_are_not_a_tree(rules):
    """Overlapping rules used to be compiled silently: one of them was lost
    and the arrays disagreed with the iterative engine."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf = _classifier(rules, "Decision Tree")
    assert clf.native_fn is None
    assert any("Native model not compiled" in str(w.message) for w in caught)
    with pytest.raises(ValueError, match="not a tree|same region"):
        clf.compile_tree_arrays(feature_names=["f0", "f1"])


@pytest.mark.parametrize("score", (-1e-17, -1e-300, 0.0))
def test_binary_gbdt_decides_on_the_sign_of_the_score(score, tmp_path):
    """scikit-learn predicts the positive class iff raw score >= 0. The old
    sigmoid(score) >= 0.5 test rounds to 0.5 for tiny negative scores."""
    analyzer, _, _, _, X = build("gbdt2", seed=0)
    clf = analyzer.classifier
    pos = clf._gbdt_classes[1]
    for r in clf.initial_rules:
        if r.parsed_conditions:
            r.contribution = 0.0
            r.leaf_value = 0.0
        else:
            r.contribution = score
            r.leaf_value = score
    clf._gbdt_init_scores[pos] = score
    clf.final_rules = []
    want = int(pos) if score >= 0 else int(clf._gbdt_classes[0])
    X = X[:20]
    clf.compile_tree_arrays(feature_names=NAMES)
    assert set(clf.predict_batch(X)) == {want}
    assert {clf.classify(dict(zip(NAMES, x)), final=False)[0] for x in X} == {want}
    assert set(native(clf, clf.initial_rules, X)) == {want}
    assert set(iterative(clf, clf.initial_rules, X)) == {want}
    assert set(python_export(clf, clf.initial_rules, X, tmp_path)) == {want}
    if GCC:
        assert set(c_header(analyzer, False, X, tmp_path)) == {want}


def test_exports_carry_the_input_precision(tmp_path):
    """A float32 model reloaded from the binary format still rounds its
    inputs; the Arduino sketch no longer reinterprets float as double."""
    analyzer, model, _, _, X = build("dt", seed=0)
    base = str(tmp_path / "m")
    with contextlib.redirect_stdout(io.StringIO()):
        analyzer.export(base_name=base, formats=["binary"])
        analyzer.classifier.compile_tree_arrays(feature_names=NAMES)
        analyzer.classifier.export_to_arduino_ino(filepath=base + ".ino")
    loaded = RuleClassifier.load_binary(base + ".bin")
    assert loaded.input_dtype == "float32"
    np.testing.assert_array_equal(loaded.predict_batch(X), model.predict(X))
    sketch = open(base + ".ino").read()
    assert "(const double*)" not in sketch
    assert "double features[N_FEATURES];" in sketch
    assert not any(line.lstrip().startswith("[") for line in sketch.splitlines())
