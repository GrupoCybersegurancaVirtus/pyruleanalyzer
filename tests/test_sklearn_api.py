"""PyRuleAnalyzer as a scikit-learn estimator, and the feature names of create().

* ``get_params``/``set_params`` expose ``model`` and the estimator's
  hyperparameters, so ``clone``, ``cross_val_score``, ``GridSearchCV`` and
  ``Pipeline`` work with ``new_model(...)``;
* a fitted analyzer has ``classes_`` (the labels ``predict`` returns) and
  ``score`` (accuracy);
* ``create()`` names the features after the training CSV's columns, in their
  order, so an array is read in the order of the CSV;
* the package imports without the removed ``parsers`` module.
"""

import importlib
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyruleanalyzer import PyRuleAnalyzer


@pytest.fixture(scope="module")
def iris():
    return load_iris(return_X_y=True)


def _dt(**params):
    return PyRuleAnalyzer.new_model("Decision Tree", {"random_state": 0, **params})


# ---------------------------------------------------------------------------
# get_params / set_params / clone
# ---------------------------------------------------------------------------

def test_get_params_exposes_model_and_hyperparameters():
    a = _dt(max_depth=3)
    assert a.get_params() == {"model": "Decision Tree", "random_state": 0,
                              "max_depth": 3}
    assert a.get_params(deep=False) == a.get_params()
    assert is_classifier(a)


def test_set_params_updates_and_validates():
    a = _dt(max_depth=3)
    assert a.set_params(max_depth=2) is a
    assert a.get_params()["max_depth"] == 2
    with pytest.raises(ValueError, match="Invalid parameter"):
        a.set_params(not_a_parameter=1)
    a.set_params(model="Random Forest", n_estimators=3)
    assert a.get_params() == {"model": "Random Forest", "random_state": 0,
                              "max_depth": 2, "n_estimators": 3}
    assert a.classifier.algorithm_type == "Random Forest"


def test_clone_gives_an_unfitted_copy_with_the_same_params(iris):
    X, y = iris
    a = _dt(max_depth=3).fit(X, y)
    c = clone(a)
    assert c is not a
    assert c.get_params() == a.get_params()
    assert not c.classifier.initial_rules
    with pytest.raises(AttributeError):
        c.classes_
    # The clone trains to the same model.
    np.testing.assert_array_equal(c.fit(X, y).predict(X), a.predict(X))


def test_clone_of_an_analyzer_built_from_sklearn(iris):
    X, y = iris
    clf = DecisionTreeClassifier(max_depth=2, random_state=1).fit(X, y)
    c = clone(PyRuleAnalyzer.from_sklearn(clf))
    assert c.get_params()["max_depth"] == 2
    np.testing.assert_array_equal(c.fit(X, y).predict(X), clf.predict(X))


def test_constructor_keeps_the_classifier_form():
    a = _dt()
    b = PyRuleAnalyzer(a.classifier, ["f0"], ["0", "1"])
    assert b.classifier is a.classifier
    with pytest.raises(TypeError):
        PyRuleAnalyzer(a.classifier, [], [], max_depth=2)


# ---------------------------------------------------------------------------
# classes_ / score / model selection
# ---------------------------------------------------------------------------

def test_classes_and_score(iris):
    X, y = iris
    a = _dt(max_depth=3).fit(X, y)
    check_is_fitted(a)
    np.testing.assert_array_equal(a.classes_, [0, 1, 2])
    assert set(np.unique(a.predict(X))) <= set(a.classes_)
    assert a.score(X, y) == pytest.approx(np.mean(a.predict(X) == y))


def test_cross_val_score(iris):
    X, y = iris
    scores = cross_val_score(_dt(max_depth=3), X, y, cv=3)
    assert scores.shape == (3,)
    assert (scores > 0.8).all()


def test_grid_search_over_max_depth(iris):
    X, y = iris
    search = GridSearchCV(_dt(), param_grid={"max_depth": [2, 3]}, cv=3).fit(X, y)
    assert search.best_params_["max_depth"] in (2, 3)
    assert search.best_estimator_.get_params()["max_depth"] == search.best_params_["max_depth"]
    np.testing.assert_array_equal(search.classes_, [0, 1, 2])
    assert 0.8 < search.score(X, y) <= 1.0


def test_pipeline(iris):
    X, y = iris
    rf = PyRuleAnalyzer.new_model("Random Forest",
                                  {"n_estimators": 5, "max_depth": 3, "random_state": 0})
    scores = cross_val_score(make_pipeline(StandardScaler(), rf), X, y, cv=3)
    assert (scores > 0.8).all()


# ---------------------------------------------------------------------------
# create(): feature names from the CSV
# ---------------------------------------------------------------------------

def _write_csvs(folder, columns, header=True, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(300, len(columns))
    y = ((X[:, -1] + X[:, 0]) > 1.0).astype(int)
    df = pd.DataFrame(X, columns=columns)
    df["target"] = y
    tr, te = os.path.join(folder, "train.csv"), os.path.join(folder, "test.csv")
    df.iloc[:200].to_csv(tr, index=False, header=header)
    df.iloc[200:].to_csv(te, index=False, header=header)
    return tr, te, df.iloc[200:]


def test_create_populates_feature_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Not in alphabetical order: an array must be read in the CSV's order.
    columns = ["zeta", "alpha", "mid"]
    tr, te, test = _write_csvs(str(tmp_path), columns)
    a = PyRuleAnalyzer.create(train_path=tr, test_path=te, model="Decision Tree",
                              params={"max_depth": 4, "random_state": 0})
    assert a.feature_names == columns
    assert "features=3" in repr(a)
    X_test = test[columns]
    np.testing.assert_array_equal(a.predict(X_test.values), a.predict(X_test))
    assert a.get_params()["max_depth"] == 4


def test_create_without_header_names_features_v1_to_vn(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    columns = [f"c{i}" for i in range(11)]   # v10 and v11 sort before v2
    tr, te, test = _write_csvs(str(tmp_path), columns, header=False)
    a = PyRuleAnalyzer.create(train_path=tr, test_path=te, model="Random Forest",
                              params={"n_estimators": 3, "max_depth": 4,
                                      "random_state": 0})
    names = [f"v{i + 1}" for i in range(11)]
    assert a.feature_names == names
    X_test = test[columns].values
    np.testing.assert_array_equal(a.predict(X_test),
                                  a.predict(pd.DataFrame(X_test, columns=names)))


# ---------------------------------------------------------------------------
# parsers.py is gone
# ---------------------------------------------------------------------------

def test_package_imports_without_parsers():
    import pyruleanalyzer
    assert hasattr(pyruleanalyzer, "PyRuleAnalyzer")
    with pytest.raises(ImportError):
        importlib.import_module("pyruleanalyzer.parsers")
