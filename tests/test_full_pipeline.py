"""
Tests for ``full_pipeline``: data splits, label encoding and test columns.

* Rules were refined on the training data itself, and a test CSV without the
  target column was replaced by a split of the training data (so the reported
  accuracy was measured on rows the model was trained on).
* Train and test labels were encoded by two independent LabelEncoders, so a
  test set missing a class got different integer codes.
* Test columns were renamed positionally, so a reordered test CSV was
  silently misaligned.
"""

import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyruleanalyzer import PyRuleAnalyzer, full_pipeline


FEATS = [f"f{i}" for i in range(4)]
N_TRAIN = 300


def _data(labels=None):
    """(train, test) frames from one well-separated 3-class dataset.

    The continuous f0 values are unique, so they identify the rows.
    """
    X, y = make_classification(n_samples=400, n_features=4, n_informative=3,
                               n_redundant=0, n_classes=3, class_sep=3.0,
                               random_state=0)
    df = pd.DataFrame(X, columns=FEATS)
    df["Target"] = y if labels is None else np.asarray(labels)[y]
    return df.iloc[:N_TRAIN].reset_index(drop=True), \
        df.iloc[N_TRAIN:].reset_index(drop=True)


def _ids(X):
    """Row identities of a frame (by its first feature column)."""
    col = X["f0"] if "f0" in X.columns else X["feature_0"]
    return set(np.round(col.to_numpy(dtype=float), 9))


def _run(tmp_path, train, test=None, **kwargs):
    """Write the CSVs and run full_pipeline quietly with a small tree."""
    train_csv = str(tmp_path / "train.csv")
    train.to_csv(train_csv, index=False)
    test_csv = None
    if test is not None:
        test_csv = str(tmp_path / "test.csv")
        test.to_csv(test_csv, index=False)
    kwargs.setdefault("model_type", "Decision Tree")
    kwargs.setdefault("max_depth", 4)
    with contextlib.redirect_stdout(io.StringIO()):
        return full_pipeline(train_csv=train_csv, test_csv=test_csv,
                             target_feature="Target", output_dir=str(tmp_path),
                             memory_report=False, **kwargs)


@pytest.fixture
def spy(monkeypatch):
    """Record the rows seen by fit, rule refinement and predict."""
    seen = {"fit": [], "refine": [], "predict": []}

    def wrap(name, key, getter):
        original = getattr(PyRuleAnalyzer, name)

        def wrapper(self, *args, **kwargs):
            seen[key].append(_ids(getter(args, kwargs)))
            return original(self, *args, **kwargs)

        monkeypatch.setattr(PyRuleAnalyzer, name, wrapper)

    wrap("fit", "fit", lambda a, k: a[0])
    wrap("execute_rule_refinement", "refine", lambda a, k: k["X"])
    wrap("predict", "predict", lambda a, k: a[0])
    return seen


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def test_refinement_uses_held_out_rows_without_test_csv(tmp_path, spy):
    train, _ = _data()
    r = _run(tmp_path, train)
    fit, refine, scored = spy["fit"][0], spy["refine"][0], spy["predict"][0]
    assert not fit & refine
    assert not fit & scored
    assert not refine & scored
    assert fit | refine | scored == _ids(train)
    assert len(scored) == 60
    assert len(r["validation"]["predictions"]) == 60


def test_refinement_data_disjoint_from_test_csv(tmp_path, spy):
    train, test = _data()
    r = _run(tmp_path, train, test)
    assert spy["predict"][0] == _ids(test)
    assert not spy["refine"][0] & _ids(test)
    assert not spy["fit"][0] & spy["refine"][0]
    # Every training row is used for fitting or refinement, none for testing.
    assert spy["fit"][0] | spy["refine"][0] == _ids(train)
    assert len(r["validation"]["predictions"]) == len(test)
    assert r["validation"]["accuracy"] > 0.8


def test_validation_size_zero_refines_on_training_data(tmp_path, spy):
    train, _ = _data()
    _run(tmp_path, train, validation_size=0)
    assert spy["fit"][0] == spy["refine"][0]


def test_invalid_validation_size_is_rejected(tmp_path):
    train, _ = _data()
    with pytest.raises(ValueError, match="validation_size"):
        _run(tmp_path, train, validation_size=1.0)


def test_test_csv_without_target_is_not_scored_on_training_rows(tmp_path, spy):
    train, test = _data()
    with pytest.warns(UserWarning, match="cannot be scored"):
        r = _run(tmp_path, train, test.drop(columns=["Target"]))
    fit, refine, scored = spy["fit"][0], spy["refine"][0], spy["predict"][0]
    # Scored on a held-out split of the training data that was excluded from
    # fitting and refinement (the unlabeled test rows cannot be scored).
    assert scored <= _ids(train)
    assert not scored & fit
    assert not scored & refine
    assert len(r["validation"]["predictions"]) == 60


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def test_test_set_missing_a_class_keeps_label_codes(tmp_path):
    labels = ["ant", "bee", "cat"]
    train, test = _data(labels)
    test = test[test["Target"] != "ant"].reset_index(drop=True)
    r = _run(tmp_path, train, test)
    assert r["label_classes"] == labels
    # With independent encoders bee/cat became 0/1 in the test set and the
    # accuracy collapsed; with the shared encoder they stay 1/2.
    expected = test["Target"].map({c: i for i, c in enumerate(labels)})
    preds = np.asarray(r["validation"]["predictions"])
    acc = float(np.mean(preds == expected.to_numpy()))
    assert r["validation"]["accuracy"] == pytest.approx(acc)
    assert acc > 0.8


def test_unseen_test_label_is_rejected(tmp_path):
    train, _ = _data(["ant", "bee", "cat"])
    _, test = _data(["ant", "bee", "dog"])
    with pytest.raises(ValueError, match="dog"):
        _run(tmp_path, train, test)


def test_numeric_labels_have_no_label_classes(tmp_path):
    train, _ = _data()
    assert _run(tmp_path, train)["label_classes"] is None


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------

def test_reordered_test_columns_give_the_same_predictions(tmp_path):
    train, test = _data()
    shuffled = test[["Target"] + FEATS[::-1]].assign(extra=1.5)
    r1 = _run(tmp_path, train, test)
    r2 = _run(tmp_path, train, shuffled)
    assert r1["validation"]["predictions"] == r2["validation"]["predictions"]
    assert r1["validation"]["accuracy"] == r2["validation"]["accuracy"]
    assert r1["validation"]["accuracy"] > 0.8


def test_missing_test_column_is_rejected(tmp_path):
    train, test = _data()
    with pytest.raises(ValueError, match="f2"):
        _run(tmp_path, train, test.drop(columns=["f2"]))


def test_missing_values_are_filled_with_a_warning(tmp_path):
    train, _ = _data()
    train.loc[:4, "f1"] = np.nan
    with pytest.warns(UserWarning, match="5 missing value"):
        _run(tmp_path, train)
