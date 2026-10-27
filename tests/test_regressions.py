"""Regression tests for bugs found in an audit of 2.0.0.

Each test pins one defect that the rest of the suite could not see, because
every other test trains on labels 0..K-1, passes the columns in training order
and uses plain feature names:

* a Random Forest answered the *column index* of the winning class instead of
  its label (labels 1..3 gave 0..2) in every engine and export;
* labels such as ``1.0`` or ``-1`` broke the engines or the exports, and
  string labels were silently replaced by indices;
* ``predict`` read a DataFrame by position, ignoring its column names;
* ``PyRuleAnalyzer.load`` could not read what ``save`` wrote;
* ``PyRuleAnalyzer.create(..., refine=True)`` called a method that did not
  exist;
* feature names were pasted unescaped into generated (and executed) Python
  and into C comments;
* ``full_pipeline`` silently dropped ``max_depth``, ``n_estimators`` and
  ``min_samples_split``.
"""

import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyruleanalyzer import PyRuleAnalyzer, full_pipeline
from pyruleanalyzer.rule_classifier import RuleClassifier
from cpn_semantics import CPNNet, evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEATS = [f"f{i}" for i in range(5)]


def _data(n_classes=3, n=400, seed=1):
    """A small dataset with labels 0..n_classes-1 and named columns."""
    X, y = make_classification(n_samples=n, n_features=5, n_informative=4,
                               n_redundant=0, n_classes=n_classes,
                               random_state=seed)
    return pd.DataFrame(X, columns=FEATS), y


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


ESTIMATORS = {
    "DT": lambda: DecisionTreeClassifier(max_depth=4, random_state=0),
    "RF": lambda: RandomForestClassifier(n_estimators=5, max_depth=4,
                                         random_state=0),
    "GB": lambda: GradientBoostingClassifier(n_estimators=5, max_depth=2,
                                             random_state=0),
}

# Label sets that are not 0..K-1, each built from the 0..K-1 labels of _data.
LABELINGS = {
    "1_2_3": lambda y: y + 1,
    "10_20_30": lambda y: (y + 1) * 10,
    "minus1_0_1": lambda y: y - 1,
    "float": lambda y: y.astype(float),
}


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("est", sorted(ESTIMATORS))
@pytest.mark.parametrize("labeling", sorted(LABELINGS))
def test_every_engine_answers_the_label_not_its_index(est, labeling, tmp_path):
    """Batch, single-sample and exported Python agree with scikit-learn."""
    X, y0 = _data()
    y = LABELINGS[labeling](y0)
    sk = ESTIMATORS[est]().fit(X, y)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    expected = sk.predict(X)

    np.testing.assert_array_equal(a.predict(X), expected)
    single = [a.predict(X.iloc[i].to_dict()) for i in range(40)]
    np.testing.assert_array_equal(single, expected[:40])

    path = str(tmp_path / "model.py")
    _quiet(a.to_python, path)
    predict = _load_module(path)["predict"]
    exported = [predict(X.iloc[i].to_dict()) for i in range(40)]
    np.testing.assert_array_equal(exported, expected[:40])


@pytest.mark.parametrize("labeling", ["1_2_3", "minus1_0_1"])
def test_forest_labels_survive_the_binary_export(labeling, tmp_path):
    X, y0 = _data()
    y = LABELINGS[labeling](y0)
    sk = ESTIMATORS["RF"]().fit(X, y)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    path = str(tmp_path / "model.bin")
    _quiet(a.to_binary, path)
    loaded = PyRuleAnalyzer.load_binary(path)
    np.testing.assert_array_equal(
        loaded.classifier.predict_batch(X.values, FEATS), sk.predict(X))


def test_forest_c_header_maps_the_argmax_to_the_label(tmp_path):
    X, y0 = _data()
    sk = ESTIMATORS["RF"]().fit(X, (y0 + 1) * 10)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    path = str(tmp_path / "model.h")
    _quiet(a.to_c_header, path)
    text = open(path).read()
    assert "class_map[N_CLASSES] = { 10, 20, 30 }" in text
    assert "return class_map[best];" in text


def test_forest_petri_net_answers_the_label(tmp_path):
    """The net's Vote, read back from the .cpn file, gives the label."""
    X, y0 = _data()
    sk = RandomForestClassifier(n_estimators=3, max_depth=3,
                                random_state=0).fit(X, y0 + 1)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    paths = _quiet(a.export_hcpn, str(tmp_path / "rf"), which="initial",
                   sample=X.iloc[0])
    net = CPNNet(paths["initial"])
    for i in range(30):
        x = {f"f{j}": float(np.float32(X.iloc[i, j])) for j in range(5)}
        assert evaluate(net, x) == sk.predict(X.iloc[[i]])[0]

    result = _quiet(a.model_check, which="initial", cpn_path=paths["initial"],
                    samples=X.iloc[:1], test_samples=X.iloc[:30],
                    verbose=False)
    assert result.properties["PC"].holds is True, result.report()
    assert result.passed, result.report()


def test_label_minus_one_is_not_taken_for_an_uncovered_region(tmp_path):
    """-1 used to be the 'no rule here' sentinel of the DT code generators."""
    X, y0 = _data(n_classes=2)
    y = np.where(y0 == 1, 1, -1)
    sk = ESTIMATORS["DT"]().fit(X, y)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    native = [a.classifier.native_fn(X.iloc[i].to_dict())[0] for i in range(60)]
    np.testing.assert_array_equal(native, sk.predict(X)[:60])


@pytest.mark.parametrize("est", sorted(ESTIMATORS))
def test_non_integer_labels_are_refused_not_replaced(est):
    X, y0 = _data()
    sk = ESTIMATORS[est]().fit(X, np.array(["cat", "dog", "fox"])[y0])
    with pytest.raises(ValueError, match="integer class labels"):
        PyRuleAnalyzer.from_sklearn(sk)


# ---------------------------------------------------------------------------
# Column order
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("est", sorted(ESTIMATORS))
def test_dataframe_columns_are_matched_by_name(est):
    X, y = _data()
    sk = ESTIMATORS[est]().fit(X, y)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    shuffled = X[FEATS[::-1]].assign(extra=0.0)
    np.testing.assert_array_equal(a.predict(shuffled), sk.predict(X))
    np.testing.assert_allclose(a.predict_proba(shuffled), a.predict_proba(X))


def test_a_dataframe_missing_a_feature_is_refused():
    X, y = _data()
    a = _quiet(PyRuleAnalyzer.from_sklearn, ESTIMATORS["DT"]().fit(X, y))
    with pytest.raises(ValueError, match="missing"):
        a.predict(X.drop(columns=["f0"]))


# ---------------------------------------------------------------------------
# Persistence and factories
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("est", sorted(ESTIMATORS))
def test_load_reads_what_save_wrote(est, tmp_path):
    X, y = _data()
    a = PyRuleAnalyzer.new_model(
        {"DT": "Decision Tree", "RF": "Random Forest",
         "GB": "Gradient Boosting Decision Trees"}[est],
        {"max_depth": 3, "random_state": 0}).fit(X, y)
    _quiet(a.execute_rule_refinement, X=X, y=y,
           remove_below_n_classifications=1)
    path = str(tmp_path / "model.pkl")
    a.save(path)
    b = PyRuleAnalyzer.load(path)
    np.testing.assert_array_equal(b.predict(X), a.predict(X))
    np.testing.assert_array_equal(b.predict(X, use_refined=False),
                                  a.predict(X, use_refined=False))


def test_create_with_refine(tmp_path):
    X, y = _data()
    df = X.assign(Target=y)
    tr, te = str(tmp_path / "train.csv"), str(tmp_path / "test.csv")
    df.iloc[:300].to_csv(tr, index=False)
    df.iloc[300:].to_csv(te, index=False)
    a = _quiet(PyRuleAnalyzer.create, tr, te, params={"max_depth": 5,
                                                       "random_state": 0},
               refine=True,
               refine_params={"remove_below_n_classifications": 1})
    assert a.classifier.final_rules
    assert len(a.classifier.final_rules) < len(a.classifier.initial_rules)
    test = df.iloc[300:].drop(columns=["Target"])
    np.testing.assert_array_equal(a.predict(test.values), a.predict(test))


# ---------------------------------------------------------------------------
# Generated code
# ---------------------------------------------------------------------------

def test_feature_names_are_data_not_code(tmp_path):
    """A quote no longer breaks the export; a crafted name runs nothing."""
    marker = tmp_path / "PWNED"
    evil = f"a', 0) or open({str(marker)!r}, 'w') or sample.get('a"
    names = ["patient's age", evil, "x\ny", "z", "w"]
    X, y = _data()
    X.columns = names
    for est in ("DT", "RF", "GB"):
        sk = ESTIMATORS[est]().fit(X, y)
        a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
        assert a.classifier.native_fn is not None or est == "GB"
        path = str(tmp_path / f"{est}.py")
        _quiet(a.to_python, path)
        predict = _load_module(path)["predict"]
        got = [predict(X.iloc[i].to_dict()) for i in range(30)]
        np.testing.assert_array_equal(got, sk.predict(X)[:30])
        got = [a.predict(X.iloc[i].to_dict()) for i in range(30)]
        np.testing.assert_array_equal(got, sk.predict(X)[:30])
    assert not marker.exists()


def test_feature_names_cannot_escape_a_c_comment(tmp_path):
    X, y = _data()
    X.columns = ["a */ int pwned; /* b", "c\nint pwned2;", "z", "w", "v"]
    a = _quiet(PyRuleAnalyzer.from_sklearn, ESTIMATORS["DT"]().fit(X, y))
    header = str(tmp_path / "model.h")
    _quiet(a.to_c_header, header)
    sketch = str(tmp_path / "model.ino")
    _quiet(a.classifier.export_to_arduino_ino, sketch, board_model="esp32")
    for path in (header, sketch):
        for line in open(path).read().splitlines():
            assert not line.lstrip().startswith("int pwned"), (path, line)
        assert "*/ int pwned" not in open(path).read()


# ---------------------------------------------------------------------------
# full_pipeline
# ---------------------------------------------------------------------------

def test_full_pipeline_passes_its_hyperparameters_on(tmp_path):
    X, y = _data(n_classes=2)
    csv = str(tmp_path / "train.csv")
    X.assign(Target=y).to_csv(csv, index=False)
    r = _quiet(full_pipeline, train_csv=csv, target_feature="Target",
               model_type="Random Forest", n_estimators=3, max_depth=2,
               output_dir=str(tmp_path), remove_below_n_classifications=-1)
    rules = r["model"].classifier.initial_rules
    assert max(len(rule.parsed_conditions) for rule in rules) <= 2
    assert len({rule.name.split("_")[0] for rule in rules}) == 3
