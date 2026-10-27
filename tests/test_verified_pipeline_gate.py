"""Tests for the gate of the verified pipeline and for its data handling.

* **inconclusive is not passed** -- a check whose occurrence graph exceeds the
  budget leaves the state-space properties without a verdict; the pipeline must
  not report such a model as verified;
* **small models still pass** -- every property of a tiny tree, forest or
  boosting model is decided, so the stricter gate changes nothing for them;
* **no leakage** -- refinement runs on rows held out of the training data, and
  the test rows the accuracy is reported on are never among them.
"""

import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import train_test_split

from pyruleanalyzer.verified_pipeline import VerificationError, verified_pipeline


RF_PARAMS = {"n_estimators": 4, "max_depth": 3, "random_state": 0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dataset(n_classes=2, n_features=4, seed=0):
    """Build a small dataset with a known boundary."""
    rng = np.random.RandomState(seed)
    X = rng.rand(300, n_features)
    if n_classes == 2:
        y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    else:
        y = (X[:, 0] * n_classes).astype(int).clip(0, n_classes - 1)
    cols = [f"feat_{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="target")


def _run(tmp_path, **kwargs):
    """Run the pipeline quietly (refinement prints even with verbose=False)."""
    kwargs.setdefault("verify_samples", 1)
    kwargs.setdefault("export_formats", ())
    kwargs.setdefault("output_dir", str(tmp_path))
    kwargs.setdefault("verbose", False)
    with contextlib.redirect_stdout(io.StringIO()):
        return verified_pipeline(**kwargs)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def test_inconclusive_check_raises_by_default(tmp_path):
    X, y = _dataset(seed=1)
    with pytest.raises(VerificationError) as excinfo:
        _run(tmp_path, X=X, y=y, model_type="Random Forest", params=RF_PARAMS,
             max_nodes=5, export_formats=("python",), output_name="gated")
    assert excinfo.value.stage == "initial"
    assert excinfo.value.result.skipped
    assert "max_nodes" in str(excinfo.value)
    assert "stubborn" in str(excinfo.value)
    assert not os.path.exists(os.path.join(str(tmp_path), "gated.py"))


def test_inconclusive_check_is_not_passed(tmp_path):
    X, y = _dataset(seed=1)
    result = _run(tmp_path, X=X, y=y, model_type="Random Forest",
                  params=RF_PARAMS, max_nodes=5, fail_on_inconclusive=False)
    assert result["passed"] is False
    final = result["verification"]["final"]
    # Nothing was violated: the run is undecided, not wrong.
    assert final.passed and not final.failures
    for stage in ("initial", "final"):
        undecided = result["inconclusive"][stage]
        assert {"A1", "B1", "D1"} <= set(undecided)
        # Structural analysis and conformance testing need no state space.
        assert "B4" not in undecided and "PC" not in undecided


def test_final_stage_alone_is_gated(tmp_path):
    X, y = _dataset(seed=1)
    with pytest.raises(VerificationError) as excinfo:
        _run(tmp_path, X=X, y=y, model_type="Random Forest", params=RF_PARAMS,
             max_nodes=5, verify_initial=False)
    assert excinfo.value.stage == "final"


@pytest.mark.parametrize("model,params,n_classes", [
    ("Decision Tree", {"max_depth": 4, "random_state": 0}, 2),
    ("Random Forest", {"n_estimators": 3, "max_depth": 3, "random_state": 0}, 2),
    ("Gradient Boosting Decision Trees",
     {"n_estimators": 3, "max_depth": 2, "learning_rate": 0.2,
      "random_state": 0}, 3),
], ids=["dt", "rf", "gbdt3"])
def test_small_models_still_pass(tmp_path, model, params, n_classes):
    X, y = _dataset(n_classes=n_classes, seed=2)
    result = _run(tmp_path, X=X, y=y, model_type=model, params=params)
    assert result["passed"], result["verification"]["final"].report()
    assert result["inconclusive"] == {"initial": [], "final": []}


def test_skipped_by_design_is_not_inconclusive():
    """A6 has no verdict by design when the label domain is not declared; that
    is not a budget problem and must not fail the gate, unlike a property left
    undecided by an occurrence graph over budget."""
    from pyruleanalyzer.model_checker import ModelCheckResult, PropertyResult
    from pyruleanalyzer.verified_pipeline import _inconclusive

    result = ModelCheckResult("m.cpn", "Decision Tree", {
        "A1": PropertyResult("A1", True),
        "A6": PropertyResult("A6", None,
                             "label domain not declared; pass class_labels"),
        "A8": PropertyResult("A8", None, "occurrence graph over budget"),
    }, {})
    assert _inconclusive(result) == ["A8"]


# ---------------------------------------------------------------------------
# Refinement does not see the test set
# ---------------------------------------------------------------------------

def _capture_refinement(monkeypatch):
    """Record the X passed to execute_rule_refinement."""
    from pyruleanalyzer import pyruleanalyzer as pra

    seen = {}
    real = pra.PyRuleAnalyzer.execute_rule_refinement

    def spy(self, X=None, y=None, **kwargs):
        seen["X"] = X
        return real(self, X=X, y=y, **kwargs)

    monkeypatch.setattr(pra.PyRuleAnalyzer, "execute_rule_refinement", spy)
    return seen


def test_refinement_rows_are_disjoint_from_test_rows(tmp_path, monkeypatch):
    seen = _capture_refinement(monkeypatch)
    X, y = _dataset(seed=6)
    result = _run(tmp_path, X=X, y=y, model_type="Decision Tree",
                  params={"max_depth": 4, "random_state": 0},
                  test_size=0.25, validation_size=0.25, random_seed=7)
    _X_train, X_test, _y_train, _y_test = train_test_split(
        X, y, test_size=0.25, random_state=7)
    used = seen["X"]
    assert len(used) == result["metrics"]["refinement_rows"] > 0
    assert not set(used.index) & set(X_test.index)
    # The rows themselves, not only the labels, are absent from the test set.
    test_rows = {tuple(r) for r in X_test.to_numpy()}
    assert not any(tuple(r) in test_rows for r in used.to_numpy())
    assert "accuracy_final" in result["metrics"]


def test_pretrained_model_refines_on_the_train_split(tmp_path, monkeypatch):
    from sklearn.tree import DecisionTreeClassifier

    seen = _capture_refinement(monkeypatch)
    X, y = _dataset(seed=8)
    X_train, X_test, y_train, _y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
    clf = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_train,
                                                                  y_train)
    result = _run(tmp_path, X=X, y=y, sklearn_model=clf)
    assert result["passed"]
    assert set(seen["X"].index) <= set(X_train.index)
    assert not set(seen["X"].index) & set(X_test.index)


def test_validation_size_zero_refines_on_the_whole_train_split(tmp_path,
                                                               monkeypatch):
    seen = _capture_refinement(monkeypatch)
    X, y = _dataset(seed=10)
    result = _run(tmp_path, X=X, y=y, model_type="Decision Tree",
                  params={"max_depth": 3, "random_state": 0},
                  validation_size=0)
    assert result["metrics"]["refinement_rows"] == 225
    _X_train, X_test, _y_train, _y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
    assert not set(seen["X"].index) & set(X_test.index)
