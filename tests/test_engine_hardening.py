"""Hardening tests for the prediction engines and the binary format.

Each test pins one defect:

* ``load_binary`` trusted the file: feature and child indices went straight
  into the C traversal, which does no bounds checks (out-of-bounds read);
* the ``_accel`` wrappers handed C an ``X`` with fewer columns than the tree
  reads;
* a missing feature was read as 0 by the arrays, the native function and the
  exported ``.py``, but as "rule does not match" by the iterative engine;
* a Random Forest rule without a class distribution put its one-hot vote in
  the column named by its *label* (label 10 -> column 10, i.e. nowhere);
* a GBDT with a non-constant ``init`` estimator was turned into rules with a
  single, wrong, constant initial score.
"""

import contextlib
import io
import os
import struct
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyruleanalyzer import PyRuleAnalyzer
from pyruleanalyzer import _accel
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


ESTIMATORS = {
    "DT": lambda: DecisionTreeClassifier(max_depth=4, random_state=0),
    "RF": lambda: RandomForestClassifier(n_estimators=5, max_depth=4,
                                         random_state=0),
    "GB": lambda: GradientBoostingClassifier(n_estimators=5, max_depth=2,
                                             random_state=0),
}


def _dt_bin(tmp_path):
    """A valid Decision Tree .bin, its bytes and the data it was fitted on."""
    X, y = _data()
    sk = ESTIMATORS["DT"]().fit(X, y)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    path = str(tmp_path / "model.bin")
    _quiet(a.to_binary, path)
    with open(path, "rb") as f:
        return path, bytearray(f.read()), X, sk


def _dt_layout(buf):
    """Offsets of the single tree of a Decision Tree .bin.

    Returns:
        dict: 'n_nodes', and the byte offset of 'feature_idx', 'threshold',
        'left', 'right', 'value' and 'max_depth'.
    """
    version, algo, n_feat, n_cls, n_trees = struct.unpack_from("<BBHHH", buf, 4)
    assert algo == 0 and n_trees == 1
    pos = 16 + (1 if version >= 2 else 0) + (4 * n_cls if version >= 3 else 0)
    for _ in range(n_feat):
        (ln,) = struct.unpack_from("<H", buf, pos)
        pos += 2 + ln
    (n_nodes,) = struct.unpack_from("<i", buf, pos)
    out = {"n_nodes_at": pos, "n_nodes": n_nodes}
    pos += 4
    for key, size in (("feature_idx", 4), ("threshold", 8), ("left", 4),
                      ("right", 4), ("value", 4)):
        out[key] = pos
        pos += n_nodes * size
    out["max_depth"] = pos
    assert pos + 4 == len(buf)
    return out


def _write(tmp_path, buf, name="bad.bin"):
    """Write `buf` to a file and return its path."""
    path = str(tmp_path / name)
    with open(path, "wb") as f:
        f.write(bytes(buf))
    return path


def _leaf_and_split(buf, lay):
    """Index of one leaf and one split node of the tree in `buf`."""
    n = lay["n_nodes"]
    left = np.frombuffer(bytes(buf[lay["left"]:lay["left"] + 4 * n]), "<i4")
    leaves = [i for i in range(n) if left[i] == i]
    splits = [i for i in range(n) if left[i] != i]
    return leaves[0], splits[0]


# ---------------------------------------------------------------------------
# Binary file validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("est", sorted(ESTIMATORS))
def test_valid_binary_round_trips(est, tmp_path):
    X, y = _data()
    sk = ESTIMATORS[est]().fit(X, y)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    path = str(tmp_path / "model.bin")
    _quiet(a.to_binary, path)
    loaded = PyRuleAnalyzer.load_binary(path)
    np.testing.assert_array_equal(
        loaded.classifier.predict_batch(X.values, FEATS), sk.predict(X))


def test_out_of_range_feature_index_is_refused(tmp_path):
    _, buf, _, _ = _dt_bin(tmp_path)
    lay = _dt_layout(buf)
    _, split = _leaf_and_split(buf, lay)
    for bad in (5, 9999, -1, -2, 2**31 - 1):
        b = bytearray(buf)
        struct.pack_into("<i", b, lay["feature_idx"] + 4 * split, bad)
        with pytest.raises(ValueError, match="feature_idx"):
            RuleClassifier.load_binary(_write(tmp_path, b))


@pytest.mark.parametrize("side", ["left", "right"])
def test_out_of_range_child_index_is_refused(side, tmp_path):
    _, buf, _, _ = _dt_bin(tmp_path)
    lay = _dt_layout(buf)
    _, split = _leaf_and_split(buf, lay)
    for bad in (lay["n_nodes"], 10**6, -1):
        b = bytearray(buf)
        struct.pack_into("<i", b, lay[side] + 4 * split, bad)
        with pytest.raises(ValueError, match="child"):
            RuleClassifier.load_binary(_write(tmp_path, b))


def test_malformed_leaf_is_refused(tmp_path):
    """A self-loop leaf must read column 0 and compare against -inf."""
    _, buf, _, _ = _dt_bin(tmp_path)
    lay = _dt_layout(buf)
    leaf, _ = _leaf_and_split(buf, lay)

    b = bytearray(buf)
    struct.pack_into("<i", b, lay["feature_idx"] + 4 * leaf, 10**6)
    with pytest.raises(ValueError, match="leaf"):
        RuleClassifier.load_binary(_write(tmp_path, b))

    b = bytearray(buf)
    struct.pack_into("<d", b, lay["threshold"] + 8 * leaf, 0.0)
    with pytest.raises(ValueError, match="leaf"):
        RuleClassifier.load_binary(_write(tmp_path, b))


def test_bad_max_depth_and_node_count_are_refused(tmp_path):
    _, buf, _, _ = _dt_bin(tmp_path)
    lay = _dt_layout(buf)

    for depth in (-1, lay["n_nodes"] + 1, 2**31 - 1):
        b = bytearray(buf)
        struct.pack_into("<i", b, lay["max_depth"], depth)
        with pytest.raises(ValueError, match="max_depth"):
            RuleClassifier.load_binary(_write(tmp_path, b))

    for n_nodes in (0, -5, 2**31 - 1, lay["n_nodes"] + 1):
        b = bytearray(buf)
        struct.pack_into("<i", b, lay["n_nodes_at"], n_nodes)
        with pytest.raises(ValueError):
            RuleClassifier.load_binary(_write(tmp_path, b))


def test_bad_header_is_refused(tmp_path):
    _, buf, _, _ = _dt_bin(tmp_path)
    for offset, value in ((0, ord("X")), (4, 99), (5, 7)):
        b = bytearray(buf)
        b[offset] = value
        with pytest.raises(ValueError):
            RuleClassifier.load_binary(_write(tmp_path, b))


@pytest.mark.parametrize("est", sorted(ESTIMATORS))
def test_every_truncation_raises_value_error(est, tmp_path):
    """No short read surfaces as struct.error, IndexError or a crash."""
    X, y = _data()
    sk = ESTIMATORS[est]().fit(X, y)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    path = str(tmp_path / "model.bin")
    _quiet(a.to_binary, path)
    with open(path, "rb") as f:
        buf = f.read()
    for cut in range(len(buf)):
        with pytest.raises(ValueError):
            RuleClassifier.load_binary(_write(tmp_path, buf[:cut]))


def test_trailing_bytes_are_refused(tmp_path):
    _, buf, _, _ = _dt_bin(tmp_path)
    with pytest.raises(ValueError, match="trailing"):
        RuleClassifier.load_binary(_write(tmp_path, buf + b"\x00"))


def test_random_corruption_never_crashes(tmp_path):
    """Flip random bytes; every load either works or raises ValueError, and
    any model that loads predicts without crashing. Run in a child process,
    so an out-of-bounds read would show as a non-zero exit, not a hang."""
    path, _, _, _ = _dt_bin(tmp_path)
    script = textwrap.dedent(f"""
        import sys, random
        import numpy as np
        sys.path.insert(0, {os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))!r})
        from pyruleanalyzer.rule_classifier import RuleClassifier
        buf = open({path!r}, 'rb').read()
        rng = random.Random(0)
        X = np.random.RandomState(0).normal(size=(50, 5))
        out = {str(tmp_path / 'fuzz.bin')!r}
        for _ in range(300):
            b = bytearray(buf)
            for _ in range(rng.randint(1, 4)):
                b[rng.randrange(len(b))] = rng.randrange(256)
            open(out, 'wb').write(bytes(b))
            try:
                clf = RuleClassifier.load_binary(out)
            except ValueError:
                continue
            try:
                clf.predict_batch(X)
            except (ValueError, KeyError, IndexError):
                pass
        print('OK')
    """)
    res = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, timeout=300)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout


# ---------------------------------------------------------------------------
# _accel wrappers
# ---------------------------------------------------------------------------

def _stump():
    """A depth-1 tree splitting on column 3."""
    feat = np.array([3, 0, 0], dtype=np.int32)
    thr = np.array([0.0, -np.inf, -np.inf])
    left = np.array([1, 1, 2], dtype=np.int32)
    right = np.array([2, 1, 2], dtype=np.int32)
    return feat, thr, left, right, 1


def test_accel_answers_like_numpy():
    X = np.random.RandomState(0).normal(size=(30, 4))
    feat, thr, left, right, depth = _stump()
    want = _accel._numpy_traverse_tree_batch(X, feat, thr, left, right, depth)
    np.testing.assert_array_equal(
        _accel.traverse_tree_batch(X, feat, thr, left, right, depth), want)
    np.testing.assert_array_equal(
        _accel.traverse_tree_batch_multi(X, [_stump(), _stump()]),
        np.vstack([want, want]))


@pytest.mark.skipif(not _accel.HAS_C_EXTENSION, reason="C extension only")
def test_accel_rejects_x_with_too_few_columns():
    X = np.zeros((10, 3))   # the stump reads column 3
    with pytest.raises(ValueError, match="column"):
        _accel.traverse_tree_batch(X, *_stump())
    with pytest.raises(ValueError, match="column"):
        _accel.traverse_tree_batch_multi(X, [_stump()])


@pytest.mark.skipif(not _accel.HAS_C_EXTENSION, reason="C extension only")
def test_accel_rejects_out_of_range_indices():
    X = np.zeros((10, 4))
    feat, thr, left, right, depth = _stump()
    bad_left = left.copy()
    bad_left[0] = 50
    with pytest.raises(ValueError, match="Child"):
        _accel.traverse_tree_batch(X, feat, thr, bad_left, right, depth)
    with pytest.raises(ValueError, match="Child"):
        _accel.traverse_tree_batch_multi(X, [(feat, thr, bad_left, right, depth)])
    bad_feat = feat.copy()
    bad_feat[0] = -2
    with pytest.raises(ValueError, match="negative"):
        _accel.traverse_tree_batch(X, bad_feat, thr, left, right, depth)
    with pytest.raises(ValueError, match="same length"):
        _accel.traverse_tree_batch_multi(X, [(feat, thr[:2], left, right, depth)])


def test_predict_batch_with_too_few_columns_does_not_read_out_of_bounds(tmp_path):
    X, y = _data()
    sk = ESTIMATORS["RF"]().fit(X, y)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    with pytest.raises((ValueError, IndexError)):
        a.classifier.predict_batch(X.values[:, :1], use_final=False)


# ---------------------------------------------------------------------------
# Missing features
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("est", sorted(ESTIMATORS))
def test_single_sample_missing_a_feature_raises(est, tmp_path):
    X, y = _data()
    sk = ESTIMATORS[est]().fit(X, y)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    clf = a.classifier
    used = clf._used_features(clf._rules_for(False))
    assert used
    sample = X.iloc[0].to_dict()
    del sample[used[0]]

    for final in (False, True):
        with pytest.raises(KeyError, match=used[0]):
            clf.classify(sample, final=final)

    # The native function indexes the sample: no silent 0.
    if clf.native_fn is not None:
        with pytest.raises(KeyError):
            clf.native_fn(sample)

    path = str(tmp_path / "model.py")
    _quiet(a.to_python, path)
    predict = _load_module(path)["predict"]
    with pytest.raises(KeyError):
        predict(sample)

    # The full sample still answers like scikit-learn.
    full = X.iloc[0].to_dict()
    assert clf.classify(full)[0] == sk.predict(X.iloc[[0]])[0]
    assert predict(full) == sk.predict(X.iloc[[0]])[0]


def test_a_feature_the_model_does_not_use_may_be_absent():
    X, y = _data()
    sk = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    clf = a.classifier
    used = set(clf._used_features(clf.initial_rules))
    unused = [f for f in FEATS if f not in used]
    assert unused
    sample = {k: v for k, v in X.iloc[0].to_dict().items() if k not in unused}
    assert clf.classify(sample)[0] == sk.predict(X.iloc[[0]])[0]


# ---------------------------------------------------------------------------
# Random Forest rules without class distributions
# ---------------------------------------------------------------------------

def test_rf_one_hot_fallback_uses_the_label_column():
    X, y0 = _data()
    y = (y0 + 1) * 10   # labels 10, 20, 30 -> columns 0, 1, 2
    sk = ESTIMATORS["RF"]().fit(X, y)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    clf = a.classifier
    rules = clf.initial_rules
    for r in rules:
        r.class_distribution = None
    clf.compile_tree_arrays(rules, FEATS)

    labels = [10, 20, 30]
    for tree in clf._tree_arrays:
        n = len(tree["feature_idx"])
        ids = np.arange(n)
        leaves = (tree["children_left"] == ids) & ~tree["is_hole"]
        vals = tree["value"][leaves]
        assert vals.shape[1] == 3
        np.testing.assert_array_equal(vals.sum(axis=1), 1.0)

    # The one-hot columns cover several labels (before the fix, label 10
    # pointed past the 3 columns and every leaf abstained).
    seen = set()
    for tree in clf._tree_arrays:
        cols = tree["value"].argmax(axis=1)[tree["value"].sum(axis=1) > 0]
        seen.update(labels[c] for c in cols)
    assert seen <= set(labels) and len(seen) > 1

    # Hard voting of the trees: every prediction is a label.
    pred = clf.predict_batch(X.values, FEATS)
    assert set(np.unique(pred)) <= set(labels)
    # Each tree's argmax column is the label its matched rule votes for.
    for i in range(20):
        sample = X.iloc[i].to_dict()
        _, votes, _, _ = RuleClassifier.classify_rf(sample, rules)
        counts = {lab: votes.count(lab) for lab in labels}
        best = max(counts.values())
        if list(counts.values()).count(best) == 1:
            assert pred[i] == max(counts, key=counts.get)


# ---------------------------------------------------------------------------
# GBDT init estimator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_classes", [2, 3])
def test_gbdt_with_non_constant_init_is_refused(n_classes):
    X, y = _data(n_classes=n_classes)
    sk = GradientBoostingClassifier(n_estimators=3, max_depth=2,
                                    init=LogisticRegression(max_iter=500),
                                    random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="init"):
        _quiet(PyRuleAnalyzer.from_sklearn, sk)


@pytest.mark.parametrize("init", [None, "zero"])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_gbdt_with_constant_init_still_works(init, n_classes):
    X, y = _data(n_classes=n_classes)
    sk = GradientBoostingClassifier(n_estimators=5, max_depth=2, init=init,
                                    random_state=0).fit(X, y)
    a = _quiet(PyRuleAnalyzer.from_sklearn, sk)
    np.testing.assert_array_equal(a.predict(X), sk.predict(X))
