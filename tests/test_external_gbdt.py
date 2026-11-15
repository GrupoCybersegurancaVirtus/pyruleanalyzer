"""XGBoost, LightGBM and CatBoost classifiers read as GBDT rules.

The bar is the one the scikit-learn models meet: the rules *are* the model.
For each library this file checks, on random samples and on samples placed on
and one ULP around every split threshold (with and without NaN):

- the neutral form read from the fitted model reproduces the library's raw
  score bit for bit (split operator, input precision, score arithmetic);
- every engine and export predicts exactly what the library's ``predict``
  returns: the vectorized and per-sample engines, the exec-compiled native
  function, the standalone Python export, JSON, binary and pickle round
  trips, the C header (compiled with gcc when available) and the coloured
  net of the model checker;
- training by name (``new_model("XGBoost")``), ``get_params``/``set_params``
  and ``clone`` work, string labels included;
- what the rules cannot express is refused with a clear error.

Each library is optional: its tests are skipped when it is not installed.
"""
import contextlib
import importlib.util
import io
import os
import pickle
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pyruleanalyzer import PyRuleAnalyzer  # noqa: E402
from pyruleanalyzer import _boosting_import as bi  # noqa: E402

NAMES = [f"f{i}" for i in range(6)]
GCC = shutil.which("gcc")
LIBS = ("xgboost", "lightgbm", "catboost")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_model(lib, **kw):
    """An untrained classifier of `lib`, small and deterministic."""
    if lib == "xgboost":
        xgb = pytest.importorskip("xgboost")
        return xgb.XGBClassifier(n_estimators=25, max_depth=4, random_state=0, **kw)
    if lib == "lightgbm":
        lgb = pytest.importorskip("lightgbm")
        return lgb.LGBMClassifier(n_estimators=25, num_leaves=12, random_state=0,
                                  verbose=-1, **kw)
    cb = pytest.importorskip("catboost")
    return cb.CatBoostClassifier(iterations=25, depth=4, random_seed=0, verbose=0,
                                 allow_writing_files=False, **kw)


def data(k, nan, seed=0, n=900):
    X, y = make_classification(n_samples=n, n_features=6, n_informative=5,
                               n_redundant=0, n_classes=k, random_state=seed)
    if nan:
        X[::11, 2] = np.nan
        X[5::13, 4] = np.nan
    return pd.DataFrame(X, columns=NAMES), y


def boundary_rows(form, X, limit=150):
    """Samples on and one ULP (of the comparison precision) around the
    thresholds of the first trees."""
    rng = np.random.default_rng(0)
    f32 = form["input_dtype"] == "float32"
    rows = []
    for g in range(form["n_groups"]):
        for t in form["trees"][g][:6]:
            for n, f in enumerate(t["feature"]):
                if f < 0:
                    continue
                th = t["threshold"][n]
                base = X[rng.integers(len(X))].copy()
                for d in (-1, 0, 1):
                    r = base.copy()
                    if f32:
                        v = np.float32(th)
                        v = v if d == 0 else np.nextafter(v, np.float32(np.inf * d))
                    else:
                        v = th if d == 0 else np.nextafter(th, np.inf * d)
                    r[f] = float(v)
                    rows.append(r)
    return np.array(rows[: 3 * limit])


def fitted(lib, k, nan):
    X, y = data(k, nan)
    model = make_model(lib).fit(X, y)
    form = bi.READERS[bi.estimator_kind(model)](model)
    Xa = X.to_numpy()
    Xs = np.vstack([Xa[:300], boundary_rows(form, Xa)])
    return model, form, Xs


# ---------------------------------------------------------------------------
# Reference raw scores (from the neutral form, written from its definition)
# ---------------------------------------------------------------------------

def neutral_scores(form, X):
    Xc = (X.astype(np.float32).astype(np.float64) if form["input_dtype"] == "float32"
          else X.astype(np.float64))
    dt = np.float32 if form["score_dtype"] == "float32" else np.float64
    S = np.zeros((len(X), form["n_groups"]), dtype=dt)
    for g in range(form["n_groups"]):
        S[:, g] += dt(form["init"][g])
        for t in form["trees"][g]:
            for i, x in enumerate(Xc):
                n = 0
                while t["left"][n] != -1:
                    v = x[t["feature"][n]]
                    left = t["nan_left"][n] if v != v else v <= t["threshold"][n]
                    n = t["left"][n] if left else t["right"][n]
                S[i, g] = dt(S[i, g] + dt(t["value"][n]))
    return S


def library_raw(lib, model, X):
    # Fed as the DataFrame the model was trained on: a bare array is
    # matched to the trained columns differently by some libraries.
    X = pd.DataFrame(X, columns=NAMES)
    if lib == "xgboost":
        return model.predict(X, output_margin=True)
    if lib == "lightgbm":
        return model.predict(X, raw_score=True)
    return model.predict(X, prediction_type="RawFormulaVal")


def library_predict(model, X):
    return np.asarray(model.predict(pd.DataFrame(X, columns=NAMES))).ravel().astype(int)


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

def load_module(path):
    spec = importlib.util.spec_from_file_location(f"m{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def engines(analyzer, X, tmp_path):
    clf = analyzer.classifier
    Xdf = pd.DataFrame(X, columns=NAMES)
    out = {
        "predict": analyzer.predict(Xdf),
        "classify": [clf.classify(dict(zip(NAMES, x)), final=False)[0] for x in X],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        py = tmp_path / "model.py"
        clf.export_to_native_python(filename=str(py))
        mod = load_module(py)
        out["python_export"] = [mod.predict(dict(zip(NAMES, map(float, x)))) for x in X]

        j = tmp_path / "model.json"
        analyzer.save_json(str(j))
        out["json"] = PyRuleAnalyzer.load_json(str(j)).predict(Xdf)

        b = tmp_path / "model.bin"
        analyzer.to_binary(str(b))
        out["binary"] = PyRuleAnalyzer.load_binary(str(b)).predict(X)

    out["pickle"] = pickle.loads(pickle.dumps(analyzer)).predict(Xdf)

    if GCC:
        base = str(tmp_path / "model_c")
        with contextlib.redirect_stdout(io.StringIO()):
            analyzer.export(base_name=base, formats=["c"])
        driver = tmp_path / "driver.c"
        driver.write_text(
            '#include <stdio.h>\n#include <math.h>\n'
            f'#include "{os.path.basename(base)}.h"\n'
            'int main(void) {\n    double f[N_FEATURES];\n    for (;;) {\n'
            '        for (int i = 0; i < N_FEATURES; i++) {\n'
            '            char s[64]; if (scanf("%63s", s) != 1) return 0;\n'
            '            f[i] = (s[0] == \'n\') ? NAN : strtod(s, 0);\n        }\n'
            '        printf("%d\\n", (int)predict(f));\n    }\n}\n')
        driver.write_text('#include <stdlib.h>\n' + driver.read_text())
        exe = tmp_path / "driver.exe"
        subprocess.run([GCC, "-O1", "-o", str(exe), str(driver), "-lm"], check=True,
                       cwd=str(tmp_path), capture_output=True)
        feed = "\n".join(" ".join("nan" if v != v else repr(float(v)) for v in row)
                         for row in X) + "\n"
        res = subprocess.run([str(exe)], input=feed, capture_output=True, text=True,
                             check=True).stdout.split()
        out["c_header"] = [int(v) for v in res]
    return {k: np.asarray(v).astype(int) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lib", LIBS)
@pytest.mark.parametrize("k", (2, 3))
@pytest.mark.parametrize("nan", (False, True))
def test_raw_scores_are_the_library_bit_for_bit(lib, k, nan):
    model, form, X = fitted(lib, k, nan)
    ours = neutral_scores(form, X).astype(np.float64)
    theirs = np.asarray(library_raw(lib, model, X), dtype=np.float64).reshape(len(X), -1)
    np.testing.assert_array_equal(ours, theirs)


@pytest.mark.parametrize("lib", LIBS)
@pytest.mark.parametrize("k", (2, 3))
@pytest.mark.parametrize("nan", (False, True))
def test_every_engine_predicts_like_the_library(lib, k, nan, tmp_path):
    model, _, X = fitted(lib, k, nan)
    analyzer = PyRuleAnalyzer.from_sklearn(model)
    expected = library_predict(model, X)
    for name, pred in engines(analyzer, X, tmp_path).items():
        bad = np.flatnonzero(pred != expected)
        assert not len(bad), f"{lib} K={k} nan={nan}: {name} disagrees on {len(bad)}/{len(X)}"


@pytest.mark.parametrize("lib", LIBS)
def test_coloured_net_predicts_like_the_library(lib, tmp_path):
    from pyruleanalyzer.model_checker import CPNModelChecker
    model, form, X = fitted(lib, 3, False)
    analyzer = PyRuleAnalyzer.from_sklearn(model)
    X = X[:80]
    with contextlib.redirect_stdout(io.StringIO()):
        paths = analyzer.export_hcpn(str(tmp_path / "net"), which="initial", sample=X[0])
    mc = CPNModelChecker(paths["initial"], feature_names=NAMES)
    Xc = X.astype(np.float32).astype(np.float64) if form["input_dtype"] == "float32" else X
    got = []
    for x in Xc:
        start = mc.cnet.initial_marking(tuple(x), places=mc.input_places)
        tokens = mc.cnet.run(start)[mc.structure.pred]
        assert len(tokens) == 1
        got.append(int(tokens[0]))
    np.testing.assert_array_equal(np.array(got), library_predict(model, X))


def test_lightgbm_zero_band():
    """LightGBM reads |x| <= 1e-35f as 0.0: inputs in that band, on its edges
    and on splits at +-1e-35 route exactly as LightGBM routes them."""
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    X, y = data(2, False, n=900)
    X.loc[::3, "f1"] = 0.0          # many zeros: LightGBM splits at +-1e-35
    model = lgb.LGBMClassifier(n_estimators=20, num_leaves=8, verbose=-1).fit(X, y)
    z = bi._LGBM_ZERO
    band = [0.0, -0.0, z, -z, np.nextafter(z, 1), np.nextafter(-z, -1),
            np.nextafter(z, 0), np.nextafter(-z, 0), 1e-40, -1e-40, 1e-30, -1e-30]
    rows = np.repeat(X.to_numpy()[rng.integers(len(X), size=len(band))], 1, axis=0)
    rows[:, 1] = band
    form = bi.read_lightgbm(model)
    np.testing.assert_array_equal(neutral_scores(form, rows),
                                  np.asarray(library_raw("lightgbm", model, rows)).reshape(len(rows), -1))
    a = PyRuleAnalyzer.from_sklearn(model)
    np.testing.assert_array_equal(a.predict(pd.DataFrame(rows, columns=NAMES)).astype(int),
                                  library_predict(model, rows))


def test_xgboost_refuses_a_non_nan_missing_value():
    xgb = pytest.importorskip("xgboost")
    X, y = data(2, False, n=300)
    with pytest.raises(ValueError, match="missing"):
        PyRuleAnalyzer.from_sklearn(xgb.XGBClassifier(n_estimators=3, missing=0.0).fit(X, y))


def test_binary_cutoffs():
    """The cutoff is the smallest score the library predicts as positive."""
    assert bi.lightgbm_binary_threshold() > 0.0
    p = bi.lightgbm_binary_threshold()
    q = float(np.nextafter(p, 0.0))
    import math
    assert 1 / (1 + math.exp(-p)) > 1 - 1 / (1 + math.exp(-p))
    assert not (1 / (1 + math.exp(-q)) > 1 - 1 / (1 + math.exp(-q)))
    assert bi.binary_test(bi.CATBOOST_BINARY_THRESHOLD) == (">", 0.0)
    assert bi.binary_test(0.0) == (">=", 0.0)
    pytest.importorskip("xgboost")
    t = bi.xgboost_binary_threshold()
    assert t > 0.0 and np.float32(t) == t


@pytest.mark.parametrize("lib, alias, estimator", [
    ("xgboost", "XGBoost", "XGBClassifier"),
    ("lightgbm", "LightGBM", "LGBMClassifier"),
    ("catboost", "CatBoost", "CatBoostClassifier"),
])
def test_training_by_name(lib, alias, estimator):
    pytest.importorskip(lib)
    X, y = data(3, False)
    labels = np.array(["low", "mid", "high"])[y]
    a = PyRuleAnalyzer.new_model(alias).fit(X, labels)
    assert a.get_params()["estimator"] == estimator
    assert a.get_params()["model"] == "Gradient Boosting Decision Trees"
    assert sorted(a.classes_) == ["high", "low", "mid"]
    assert np.mean(a.predict(X) == labels) > 0.7
    b = clone(a).fit(X, labels)
    assert b.get_params()["estimator"] == estimator
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    a.set_params(n_estimators=5)
    assert a.get_params()["n_estimators"] == 5
    with pytest.raises(ValueError, match="Invalid parameter"):
        a.set_params(not_a_parameter=1)


def test_set_params_switches_library():
    pytest.importorskip("xgboost")
    a = PyRuleAnalyzer.new_model("Gradient Boosting Decision Trees")
    a.set_params(model="XGBoost")
    params = a.get_params()
    assert params["estimator"] == "XGBClassifier"
    # The scikit-learn defaults XGBoost also accepts are kept.
    assert params["n_estimators"] == 100 and params["max_depth"] == 3


def test_refinement_runs_and_keeps_predicting():
    pytest.importorskip("xgboost")
    X, y = data(2, True, n=1200)
    a = PyRuleAnalyzer.from_sklearn(make_model("xgboost").fit(X[:800], y[:800]))
    with contextlib.redirect_stdout(io.StringIO()):
        a.execute_rule_refinement(X=X[800:], y=y[800:])
    pred = a.predict(X[800:])
    assert len(pred) == 400


def test_unknown_estimator_message():
    with pytest.raises(ValueError, match="XGBClassifier"):
        PyRuleAnalyzer(model="Gradient Boosting Decision Trees", estimator="Nope")


# --- refusals -------------------------------------------------------------

@pytest.mark.parametrize("k", (2, 3))
def test_xgboost_dart_weights_are_folded_into_the_leaves(k):
    xgb = pytest.importorskip("xgboost")
    X, y = data(k, True)
    m = xgb.XGBClassifier(booster="dart", rate_drop=0.3, n_estimators=20,
                          max_depth=4, random_state=0).fit(X, y)
    form = bi.read_xgboost(m)
    np.testing.assert_array_equal(
        neutral_scores(form, X.to_numpy()).astype(np.float64),
        np.asarray(library_raw("xgboost", m, X.to_numpy()), dtype=np.float64).reshape(len(X), -1))
    np.testing.assert_array_equal(PyRuleAnalyzer.from_sklearn(m).predict(X).astype(int),
                                  library_predict(m, X.to_numpy()))


def test_xgboost_refuses_linear_and_categorical():
    xgb = pytest.importorskip("xgboost")
    X, y = data(2, False, n=300)
    with pytest.raises(ValueError, match="booster"):
        PyRuleAnalyzer.from_sklearn(xgb.XGBClassifier(booster="gblinear", n_estimators=3).fit(X, y))
    Xc = X.copy()
    Xc["f0"] = pd.Categorical(np.where(Xc["f0"] > 0, "a", "b"))
    m = xgb.XGBClassifier(n_estimators=3, enable_categorical=True, tree_method="hist").fit(Xc, y)
    with pytest.raises(ValueError, match="categorical"):
        PyRuleAnalyzer.from_sklearn(m)


def test_lightgbm_refuses_zero_as_missing_and_linear_trees():
    lgb = pytest.importorskip("lightgbm")
    X, y = data(2, False, n=300)
    with pytest.raises(ValueError, match="zero_as_missing"):
        PyRuleAnalyzer.from_sklearn(lgb.LGBMClassifier(
            n_estimators=3, zero_as_missing=True, verbose=-1).fit(X, y))
    with pytest.raises(ValueError, match="linear"):
        PyRuleAnalyzer.from_sklearn(lgb.LGBMClassifier(
            n_estimators=3, linear_tree=True, verbose=-1).fit(X, y))


def test_catboost_refuses_categorical_and_non_symmetric_trees():
    cb = pytest.importorskip("catboost")
    X, y = data(2, False, n=300)
    Xc = X.copy()
    Xc["f0"] = np.where(Xc["f0"] > 0, "a", "b")
    m = cb.CatBoostClassifier(iterations=3, verbose=0, allow_writing_files=False,
                              cat_features=["f0"]).fit(Xc, y)
    with pytest.raises(ValueError, match="categorical"):
        PyRuleAnalyzer.from_sklearn(m)
    m = cb.CatBoostClassifier(iterations=3, verbose=0, allow_writing_files=False,
                              grow_policy="Depthwise").fit(X, y)
    with pytest.raises(ValueError, match="symmetric"):
        PyRuleAnalyzer.from_sklearn(m)


def test_footprint_knows_the_libraries():
    from pyruleanalyzer import footprint
    for name in ("XGBClassifier", "LGBMClassifier", "CatBoostClassifier", "xgboost"):
        assert footprint._algorithm(name) == "gbdt"


@pytest.mark.parametrize("lib, name, estimator", [
    ("xgboost", "XGBoost", "XGBClassifier"),
    ("lightgbm", "LightGBM", "LGBMClassifier"),
    ("catboost", "CatBoost", "CatBoostClassifier")])
def test_model_is_named_after_its_library(lib, name, estimator, tmp_path):
    model, _, _ = fitted(lib, 2, False)
    a = PyRuleAnalyzer.from_sklearn(model)
    assert a.classifier.model_name == name
    assert a.classifier.algorithm_type == "Gradient Boosting Decision Trees"
    assert repr(a).startswith(f"PyRuleAnalyzer(model={name},")
    assert a.summary()["model_name"] == name
    assert a.summary_report()["model_name"] == name
    # The name survives every round trip, the binary one included.
    path = str(tmp_path / "m.bin")
    a.to_binary(path)
    b = PyRuleAnalyzer.load_binary(path)
    assert b.classifier.model_name == name
    assert b.classifier.sklearn_estimator == estimator
    path = str(tmp_path / "m.json")
    a.save_json(path)
    assert PyRuleAnalyzer.load_json(path).classifier.model_name == name
    ino = str(tmp_path / "m.ino")
    a.classifier.export_to_arduino_ino(filepath=ino)
    assert f"// Model: {name}" in open(ino, encoding="utf-8").read()


def test_scikit_learn_models_keep_their_family_name():
    from sklearn.ensemble import GradientBoostingClassifier
    X, y = data(2, False)
    a = PyRuleAnalyzer.from_sklearn(
        GradientBoostingClassifier(n_estimators=5, random_state=0).fit(X, y))
    assert a.classifier.model_name == "Gradient Boosting Decision Trees"
