"""Tests for small fixes:

* ``load_binary`` accepted split nodes that do not form a tree -- a child
  pointing back to an ancestor, a node shared by two splits, a node no split
  reaches -- and a ``max_depth`` smaller than the depth of the tree, which
  stops the traversal on a split node. All stay in bounds but route samples
  wrongly;
* ``from_sklearn`` refused a fitted ``ExtraTreesClassifier``, which is a
  Random Forest once fitted.
"""

import contextlib
import io
import os
import struct
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, load_wine, make_classification
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyruleanalyzer import PyRuleAnalyzer
from pyruleanalyzer.rule_classifier import RuleClassifier


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


def _write(tmp_path, buf, name="bad.bin"):
    """Write `buf` to a file and return its path."""
    path = str(tmp_path / name)
    with open(path, "wb") as f:
        f.write(bytes(buf))
    return path


def _dt_bin(tmp_path):
    """The bytes of a valid Decision Tree .bin (depth 4) and its layout.

    Returns:
        tuple: (bytearray, dict) -- the layout has 'n_nodes', the byte offsets
        'feature_idx', 'threshold', 'left', 'right', 'max_depth', and the
        decoded 'left'/'right' arrays as 'L'/'R'.
    """
    X, y = _data()
    sk = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    path = str(tmp_path / "model.bin")
    _quiet(a.to_binary, path)
    with open(path, "rb") as f:
        buf = bytearray(f.read())

    version, algo, n_feat, n_cls, n_trees = struct.unpack_from("<BBHHH", buf, 4)
    assert algo == 0 and n_trees == 1
    # Version 4: flag bit 2 = column labels, bit 1 = per-node missing-value
    # routing (one byte per node after the right children).
    flags = buf[16] if version >= 2 else 0
    pos = 16 + (1 if version >= 2 else 0) + (4 * n_cls if version == 3 or flags & 4 else 0)
    for _ in range(n_feat):
        (ln,) = struct.unpack_from("<H", buf, pos)
        pos += 2 + ln
    (n_nodes,) = struct.unpack_from("<i", buf, pos)
    lay = {"n_nodes": n_nodes}
    pos += 4
    for key, size in ((("feature_idx", 4), ("threshold", 8), ("left", 4),
                       ("right", 4))
                      + ((("missing_left", 1),) if flags & 2 else ())
                      + (("value", 4),)):
        lay[key] = pos
        pos += n_nodes * size
    lay["max_depth"] = pos
    assert pos + 4 == len(buf)
    lay["L"] = np.frombuffer(bytes(buf[lay["left"]:lay["left"] + 4 * n_nodes]), "<i4")
    lay["R"] = np.frombuffer(bytes(buf[lay["right"]:lay["right"] + 4 * n_nodes]), "<i4")
    return buf, lay


def _is_split(lay, i):
    """Whether node i is a split (not a self-loop leaf)."""
    return lay["L"][i] != i


def _depth(lay):
    """Depth of the deepest leaf, walking from node 0."""
    best, stack = 0, [(0, 0)]
    while stack:
        n, d = stack.pop()
        if _is_split(lay, n):
            stack += [(int(lay["L"][n]), d + 1), (int(lay["R"][n]), d + 1)]
        else:
            best = max(best, d)
    return best


# ---------------------------------------------------------------------------
# load_binary: the split nodes must form a tree
# ---------------------------------------------------------------------------

def test_child_pointing_to_an_ancestor_is_refused(tmp_path):
    buf, lay = _dt_bin(tmp_path)
    # A split at depth >= 2 and the path of ancestors above it.
    path, stack = None, [[0]]
    while stack and path is None:
        p = stack.pop()
        if len(p) >= 3 and _is_split(lay, p[-1]):
            path = p
        elif _is_split(lay, p[-1]):
            stack += [p + [int(lay["L"][p[-1]])], p + [int(lay["R"][p[-1]])]]
    assert path is not None
    node, ancestors = path[-1], path[:-1]
    for side in ("left", "right"):
        for ancestor in ancestors:
            b = bytearray(buf)
            struct.pack_into("<i", b, lay[side] + 4 * node, ancestor)
            with pytest.raises(ValueError, match="reached twice"):
                RuleClassifier.load_binary(_write(tmp_path, b))


def test_root_as_a_child_is_refused(tmp_path):
    buf, lay = _dt_bin(tmp_path)
    b = bytearray(buf)
    struct.pack_into("<i", b, lay["right"], 0)    # root's right child = root
    with pytest.raises(ValueError, match="own child|reached twice"):
        RuleClassifier.load_binary(_write(tmp_path, b))
    b = bytearray(buf)
    struct.pack_into("<i", b, lay["right"] + 4 * int(lay["L"][0]), 0)
    with pytest.raises(ValueError, match="reached twice"):
        RuleClassifier.load_binary(_write(tmp_path, b))


def test_node_shared_by_two_splits_is_refused(tmp_path):
    buf, lay = _dt_bin(tmp_path)
    left, right = int(lay["L"][0]), int(lay["R"][0])
    assert _is_split(lay, left)
    # The left subtree's right child becomes the root's right child.
    b = bytearray(buf)
    struct.pack_into("<i", b, lay["right"] + 4 * left, right)
    with pytest.raises(ValueError, match="reached twice"):
        RuleClassifier.load_binary(_write(tmp_path, b))
    # Both children of the root are the same node.
    b = bytearray(buf)
    struct.pack_into("<i", b, lay["right"], left)
    with pytest.raises(ValueError, match="reached twice"):
        RuleClassifier.load_binary(_write(tmp_path, b))


def test_unreachable_node_is_refused(tmp_path):
    """Turning a split into a leaf orphans its subtree."""
    buf, lay = _dt_bin(tmp_path)
    left = int(lay["L"][0])
    assert _is_split(lay, left)
    b = bytearray(buf)
    struct.pack_into("<i", b, lay["left"] + 4 * left, left)
    struct.pack_into("<i", b, lay["right"] + 4 * left, left)
    struct.pack_into("<i", b, lay["feature_idx"] + 4 * left, 0)
    struct.pack_into("<d", b, lay["threshold"] + 8 * left, -np.inf)
    with pytest.raises(ValueError, match="not reachable"):
        RuleClassifier.load_binary(_write(tmp_path, b))


def test_max_depth_smaller_than_the_tree_is_refused(tmp_path):
    buf, lay = _dt_bin(tmp_path)
    real = _depth(lay)
    assert real >= 2
    (stored,) = struct.unpack_from("<i", buf, lay["max_depth"])
    assert stored == real
    for depth in range(real):
        b = bytearray(buf)
        struct.pack_into("<i", b, lay["max_depth"], depth)
        with pytest.raises(ValueError, match="max_depth"):
            RuleClassifier.load_binary(_write(tmp_path, b))
    # A larger max_depth only runs extra no-op steps on the leaves.
    b = bytearray(buf)
    struct.pack_into("<i", b, lay["max_depth"], real + 3)
    RuleClassifier.load_binary(_write(tmp_path, b))


ESTIMATORS = {
    "DT": lambda: DecisionTreeClassifier(max_depth=6, random_state=0),
    "RF": lambda: RandomForestClassifier(n_estimators=8, max_depth=6,
                                         random_state=0),
    "GB": lambda: GradientBoostingClassifier(n_estimators=6, max_depth=3,
                                             random_state=0),
    "ET": lambda: ExtraTreesClassifier(n_estimators=8, max_depth=6,
                                       random_state=0),
}


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("est", sorted(ESTIMATORS))
def test_valid_binaries_still_load(est, n_classes, tmp_path):
    X, y = _data(n_classes=n_classes)
    sk = ESTIMATORS[est]().fit(X, y)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    path = str(tmp_path / "model.bin")
    _quiet(a.to_binary, path)
    loaded = PyRuleAnalyzer.load_binary(path)
    np.testing.assert_array_equal(
        loaded.classifier.predict_batch(X.values, FEATS), sk.predict(X))


def test_refined_binary_still_loads(tmp_path):
    """Refinement leaves holes and merged leaves; the tree stays a tree."""
    X, y = _data()
    sk = RandomForestClassifier(n_estimators=5, max_depth=6,
                                random_state=0).fit(X, y)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    _quiet(a.execute_rule_refinement, X=X, y=y,
           remove_below_n_classifications=2)
    path = str(tmp_path / "model.bin")
    _quiet(a.to_binary, path)
    loaded = PyRuleAnalyzer.load_binary(path)
    np.testing.assert_array_equal(
        loaded.classifier.predict_batch(X.values, FEATS),
        a.classifier.predict_batch(X.values, FEATS))


# ---------------------------------------------------------------------------
# ExtraTreesClassifier through from_sklearn
# ---------------------------------------------------------------------------

def _datasets():
    """(name, X DataFrame, base labels 0..K-1) for a few datasets."""
    out = []
    for k in (2, 3):
        X, y = _data(n_classes=k, n=300, seed=k)
        out.append((f"synthetic{k}", X, y))
    iris = load_iris()
    out.append(("iris", pd.DataFrame(iris.data, columns=[f"c{i}" for i in range(4)]),
                iris.target))
    wine = load_wine()
    out.append(("wine", pd.DataFrame(wine.data, columns=[f"w{i}" for i in range(13)]),
                wine.target))
    return out


DATASETS = {name: (X, y) for name, X, y in _datasets()}

# Label sets: indices 0..K-1 mapped to these labels.
LABELS = {
    "0..K": None,
    "1..K": lambda y: y + 1,
    "-1/1 or -1/0/1": lambda y: y - 1 if y.max() >= 2 else 2 * y - 1,
    "10,20,..": lambda y: 10 * (y + 1),
}


@pytest.mark.parametrize("labels", sorted(LABELS))
@pytest.mark.parametrize("dataset", sorted(DATASETS))
def test_extra_trees_match_sklearn(dataset, labels, tmp_path):
    X, y0 = DATASETS[dataset]
    y = y0 if LABELS[labels] is None else LABELS[labels](y0)
    sk = ExtraTreesClassifier(n_estimators=10, max_depth=6,
                              random_state=0).fit(X, y)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    assert a.classifier.algorithm_type == "Random Forest"

    np.testing.assert_array_equal(a.predict(X), sk.predict(X))
    np.testing.assert_allclose(a.predict_proba(X), sk.predict_proba(X),
                               rtol=0, atol=1e-12)

    # Standalone Python export.
    path = str(tmp_path / "model.py")
    _quiet(a.to_python, path)
    predict = _load_module(path)["predict"]
    rows = X.to_dict("records")
    np.testing.assert_array_equal([predict(r) for r in rows], sk.predict(X))

    # Binary export.
    path = str(tmp_path / "model.bin")
    _quiet(a.to_binary, path)
    loaded = PyRuleAnalyzer.load_binary(path)
    np.testing.assert_array_equal(loaded.predict(X), sk.predict(X))


@pytest.mark.parametrize("bootstrap", [False, True])
def test_extra_trees_numpy_input_and_bootstrap(bootstrap):
    """No feature names, and bootstrap only changes training."""
    X, y = _data(n_classes=3, seed=7)
    Xn = X.values
    sk = ExtraTreesClassifier(n_estimators=12, bootstrap=bootstrap,
                              max_features=None, random_state=3).fit(Xn, y)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    np.testing.assert_array_equal(a.predict(Xn), sk.predict(Xn))
    np.testing.assert_allclose(a.predict_proba(Xn), sk.predict_proba(Xn),
                               rtol=0, atol=1e-12)


def test_unsupported_estimator_message_names_extra_trees():
    from sklearn.linear_model import LogisticRegression
    X, y = _data()
    with pytest.raises(ValueError, match="ExtraTreesClassifier"):
        PyRuleAnalyzer.from_sklearn(LogisticRegression(max_iter=500).fit(X, y))
