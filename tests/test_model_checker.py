"""Tests for the model checker, its mutants and its agreement with CPN Tools.

The checker is held to three standards:

* **soundness on correct nets** -- every generated net, of every family,
  initial and refined, satisfies every property;
* **sensitivity** -- every property is falsified by a mutant built to violate
  it (a checker that only ever answers "true" verifies nothing), and the
  mutants that separate the techniques do separate them: an overlap the input
  never reaches is caught by the structural check only, and a net that computes
  the wrong class passes every CTL property and fails conformance only;
* **agreement with CPN Tools** -- occurrence-graph sizes and every CTL verdict
  equal what CPN Tools 4.0.1 computes, when its engine is installed.
"""

import io
import os
import re
import sys
import zlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyruleanalyzer import PyRuleAnalyzer
from pyruleanalyzer.cpn_mutants import MUTANTS, make_mutants
from pyruleanalyzer.cpntools_oracle import CPNToolsOracle, compare_with_oracle
from pyruleanalyzer.model_checker import (
    METHODS,
    PROPERTY_CATALOG,
    CPNModelChecker,
    ModelCheckResult,
    PropertyResult,
    check_cpn,
    compare_results,
)
from pyruleanalyzer.verified_pipeline import VerificationError, verified_pipeline


MODELS = [
    ("Decision Tree", {"max_depth": 4, "random_state": 0}, 2),
    ("Random Forest", {"n_estimators": 4, "max_depth": 3, "random_state": 0}, 2),
    ("Gradient Boosting Decision Trees",
     {"n_estimators": 4, "max_depth": 2, "learning_rate": 0.2, "random_state": 0}, 2),
    ("Gradient Boosting Decision Trees",
     {"n_estimators": 3, "max_depth": 2, "learning_rate": 0.2, "random_state": 0}, 3),
]

IDS = ["dt", "rf", "gbdt2", "gbdt3"]


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
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="target"), cols


def _nets(tmp_path, model, params, n_classes):
    """Train, refine and export both HCPN models; return analyzer, paths, X_test."""
    # zlib.crc32, not hash(): string hashing is salted per interpreter run.
    X, y, _cols = _dataset(n_classes=n_classes,
                           seed=zlib.crc32(f"{model}{n_classes}".encode()) % 40)
    X_train, y_train = X.iloc[:220], y.iloc[:220]
    X_test, y_test = X.iloc[220:], y.iloc[220:]
    analyzer = PyRuleAnalyzer.new_model(model=model, params=params)
    analyzer.fit(X_train, y_train)
    analyzer.execute_rule_refinement(X=X_test, y=y_test,
                                     remove_below_n_classifications=1,
                                     save_final_model=False, save_report=False)
    paths = analyzer.export_hcpn(base_name=os.path.join(str(tmp_path), "net"),
                                 which="both", sample=X_test.iloc[0])
    return analyzer, paths, X_test


def _labels(analyzer):
    return [int(c) for c in analyzer.class_names]


# ---------------------------------------------------------------------------
# Correct nets satisfy every property
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,params,n_classes", MODELS, ids=IDS)
@pytest.mark.parametrize("stage", ["initial", "final"])
def test_generated_net_satisfies_every_property(tmp_path, model, params,
                                                n_classes, stage):
    analyzer, paths, X_test = _nets(tmp_path, model, params, n_classes)
    result = check_cpn(paths[stage], samples=X_test.iloc[:2],
                       classifier=analyzer.classifier,
                       feature_names=analyzer.feature_names,
                       class_labels=_labels(analyzer),
                       use_final=(stage == "final"), test_samples=X_test)
    assert result.passed, result.report()
    assert not result.skipped, result.report()
    for pid in ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
                "B1", "B2", "B3", "B4", "PC"):
        assert result.properties[pid].holds is True, (pid, result.report())
    assert result.tested == len(X_test)


@pytest.mark.parametrize("model,params,n_classes", MODELS, ids=IDS)
def test_family_specific_properties_are_checked(tmp_path, model, params, n_classes):
    analyzer, paths, _X = _nets(tmp_path, model, params, n_classes)
    result = check_cpn(paths["final"], class_labels=_labels(analyzer))
    ids = set(result.properties)
    if model == "Gradient Boosting Decision Trees":
        assert {"C1", "C2", "C3a", "C3b"} <= ids and "D1" not in ids
    elif model == "Random Forest":
        assert {"D1", "D2a", "D2b"} <= ids and "C1" not in ids
    else:
        assert not ({"C1", "D1"} & ids)


@pytest.mark.parametrize("model,params,n_classes", MODELS[2:], ids=IDS[2:])
def test_gbdt_occurrence_graph_has_the_closed_form_size(tmp_path, model, params,
                                                        n_classes):
    """(3M+3)^K + 2 nodes and K(3M+2)(3M+3)^(K-1) + 2 arcs, K channels of M
    stages: the channels interleave freely, the decision layer is serial."""
    _analyzer, paths, _X = _nets(tmp_path, model, params, n_classes)
    K = 1 if n_classes == 2 else n_classes
    M = params["n_estimators"]
    result = check_cpn(paths["initial"])
    assert result.stats["nodes"] == (3 * M + 3) ** K + 2
    assert result.stats["arcs"] == K * (3 * M + 2) * (3 * M + 3) ** (K - 1) + 2


def test_label_domain_is_required_for_a_decision_tree(tmp_path):
    _analyzer, paths, _X = _nets(tmp_path, *MODELS[0])
    assert check_cpn(paths["final"]).properties["A6"].holds is None
    assert check_cpn(paths["final"], class_labels=[0, 1]).properties["A6"].holds


# ---------------------------------------------------------------------------
# Sensitivity: every property fails on its mutant
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,params,n_classes", MODELS, ids=IDS)
def test_every_mutant_is_detected(tmp_path, model, params, n_classes):
    analyzer, paths, _X = _nets(tmp_path, model, params, n_classes)
    mutants = make_mutants(paths["final"], os.path.join(str(tmp_path), "m"))
    # Operators that edit dormant leaves need a tree with enough of them; the
    # others apply to every net of the family.
    needs_dormant = {"dead_branch", "overlap", "hidden_overlap"}
    # Mutant.applies is a predicate over the analysis, not a list of families.
    required = {m.name for m in MUTANTS
                if m.applies({"family": model}) and m.name not in needs_dormant}
    assert required <= {m["name"] for m in mutants}
    for m in mutants:
        result = check_cpn(m["path"], class_labels=_labels(analyzer),
                           classifier=analyzer.classifier,
                           feature_names=analyzer.feature_names)
        failed = {r.id for r in result.failures}
        for target in m["targets"]:
            assert target in failed, (m["name"], target, result.report())


@pytest.mark.parametrize("model,params,n_classes", MODELS, ids=IDS)
def test_stubborn_reduction_gives_the_full_graph_verdicts(tmp_path, model,
                                                          params, n_classes):
    """Partial-order reduction must not change a verdict, mutants included."""
    analyzer, paths, X_test = _nets(tmp_path, model, params, n_classes)
    mutants = make_mutants(paths["final"], os.path.join(str(tmp_path), "m"))
    for path in [paths["initial"], paths["final"]] + [m["path"] for m in mutants]:
        checker = CPNModelChecker(path, class_labels=_labels(analyzer))
        full = checker.check()
        reduced = checker.check(reduction="stubborn")
        for pid, r in full.properties.items():
            assert reduced.properties[pid].holds == r.holds, \
                (os.path.basename(path), pid, reduced.properties[pid].detail)
        assert reduced.stats["nodes"] <= full.stats["nodes"]

    result = analyzer.model_check(which="final", cpn_path=paths["final"],
                                  samples=X_test.iloc[:2], test_samples=X_test,
                                  reduction="stubborn", verbose=False)
    assert result.passed, result.report()
    assert result.stats["reduction"] == "stubborn"


def test_every_property_has_a_mutant():
    covered = {t for m in MUTANTS for t in m.targets}
    assert covered >= set(PROPERTY_CATALOG) - {"D2b"}   # D2b is A5's formula


def test_unreached_overlap_is_caught_only_structurally(tmp_path):
    """Per-input model checking cannot see an overlap the input never reaches;
    the structural analysis can."""
    analyzer, paths, _X = _nets(tmp_path, *MODELS[0])
    [m] = make_mutants(paths["final"], str(tmp_path), names=["hidden_overlap"])
    result = check_cpn(m["path"], class_labels=_labels(analyzer))
    assert result.properties["B1"].holds is True
    assert result.properties["B4"].holds is False


def test_wrong_class_passes_every_ctl_property_and_fails_conformance(tmp_path):
    """Temporal properties say the net behaves like a classifier, not that it
    is the right one."""
    analyzer, paths, X_test = _nets(tmp_path, *MODELS[0])
    [m] = make_mutants(paths["final"], str(tmp_path), names=["wrong_value"])
    result = check_cpn(m["path"], class_labels=_labels(analyzer),
                       classifier=analyzer.classifier,
                       feature_names=analyzer.feature_names)
    assert [r.id for r in result.failures] == ["PC"], result.report()


def test_askctl_ev_is_vacuous_at_a_dead_marking(tmp_path):
    """On a net that stops before predicting, ASK-CTL's EV(PRED) holds while
    the textbook AF pred does not."""
    analyzer, paths, _X = _nets(tmp_path, *MODELS[2])
    [m] = make_mutants(paths["final"], str(tmp_path), names=["deadlock"])
    result = check_cpn(m["path"])
    assert result.oracle["A3"] == "false"
    assert result.oracle["A3_ev"] == "true"
    assert result.properties["A2"].holds is False


def test_value_differences_are_distinct_markings(tmp_path):
    """Two enabled leaves producing different values lead to different
    markings: the occurrence graph keeps token values."""
    analyzer, paths, _X = _nets(tmp_path, *MODELS[2])
    [m] = make_mutants(paths["final"], str(tmp_path), names=["overlap"])
    base = check_cpn(paths["final"])
    mutant = check_cpn(m["path"])
    assert mutant.stats["nodes"] > base.stats["nodes"]


def test_budget_withholds_state_space_verdicts(tmp_path):
    _analyzer, paths, _X = _nets(tmp_path, *MODELS[3])
    result = check_cpn(paths["final"], max_nodes=10)
    ctl = [r for r in result.properties.values() if r.spec.method in ("ctl", "scc")]
    assert ctl and all(r.holds is None for r in ctl)
    assert result.properties["B4"].holds is True
    assert any("exceeded" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# ASK-CTL generation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,params,n_classes", MODELS, ids=IDS)
def test_ml_program_asks_every_question_the_checker_answers(tmp_path, model, params,
                                                            n_classes):
    analyzer, paths, _X = _nets(tmp_path, model, params, n_classes)
    checker = CPNModelChecker(paths["final"], class_labels=_labels(analyzer))
    result = checker.check()
    decl, queries = checker.ml_program()
    assert set(queries) == set(result.oracle)
    for page in re.findall(r"Mark\.(\w+)'", decl):
        assert page in checker.net.pages, page
    assert "AFmax" in decl


def test_askctl_script_is_self_contained(tmp_path):
    analyzer, paths, _X = _nets(tmp_path, *MODELS[1])
    checker = CPNModelChecker(paths["final"], class_labels=_labels(analyzer))
    text = io.open(checker.export_askctl(os.path.join(str(tmp_path), "q.sml")),
                   encoding="utf-8").read()
    assert 'ASKCTLloader.sml' in text and "CalculateOccGraph" in text
    assert "PYRA_REPORT" in text and '("D1",' in text


# ---------------------------------------------------------------------------
# Agreement with CPN Tools (only where its engine is installed)
# ---------------------------------------------------------------------------

_ORACLE = CPNToolsOracle()
needs_cpntools = pytest.mark.skipif(
    not _ORACLE.available() or os.environ.get("PYRA_CPNTOOLS") != "1",
    reason="set PYRA_CPNTOOLS=1 with CPN Tools, CPN IDE and Java 8 installed")


@needs_cpntools
@pytest.mark.parametrize("mutant", [None, "deadlock", "overlap"])
def test_agreement_with_cpn_tools(tmp_path, mutant):
    analyzer, paths, _X = _nets(tmp_path, *MODELS[2])
    net = paths["final"]
    if mutant:
        [m] = make_mutants(net, str(tmp_path), names=[mutant])
        net = m["path"]
    report = compare_with_oracle(net, _ORACLE, class_labels=_labels(analyzer))
    assert not report["errors"], report["errors"]
    assert report["agree"], [r for r in report["rows"] if not r[3]]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def test_report_and_serialisation(tmp_path):
    analyzer, paths, _X = _nets(tmp_path, *MODELS[1])
    result = check_cpn(paths["final"], class_labels=_labels(analyzer))
    text = result.report()
    assert "VERIFIED" in text and "occurrence graph" in text
    for title in ("CTL model checking", "SCC analysis", "Structural"):
        assert title in text
    data = result.to_dict()
    assert data["passed"] is True
    assert all(v["method"] in METHODS for v in data["properties"].values())
    rows = result.latex_rows()
    assert "$" not in rows and r"\texttt{" in rows
    assert "_" not in rows.replace(r"\_", "")
    table = result.latex_table(label="tab:mc")
    assert table.startswith(r"\begin{table}") and table.endswith(r"\end{table}")
    assert r"\label{tab:mc}" in table and rows in table


def test_compare_results_flags_a_regression():
    def _result(b4_holds):
        props = {"A1": PropertyResult("A1", True),
                 "B4": PropertyResult("B4", b4_holds, "" if b4_holds else "overlap")}
        return ModelCheckResult("m.cpn", "Decision Tree", props,
                                {"nodes": 2, "arcs": 1, "dead": 1})

    text = compare_results(_result(True), _result(False))
    assert "REGRESSION" in text and "broken by the refinement" in text
    assert "preserved every verified property" in compare_results(_result(True),
                                                                  _result(True))


def test_property_catalog_is_consistent():
    for pid, prop in PROPERTY_CATALOG.items():
        assert prop.id == pid
        assert prop.kind in ("desired", "undesired", "structural")
        assert prop.method in METHODS
        assert prop.ctl and prop.description
    assert PROPERTY_CATALOG["A8"].method == "scc"
    assert PROPERTY_CATALOG["B4"].method == "structural"
    assert PROPERTY_CATALOG["PC"].method == "testing"


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

def test_verified_pipeline_runs_every_stage(tmp_path):
    X, y, _cols = _dataset(n_classes=2, seed=3)
    result = verified_pipeline(
        X=X, y=y, model_type="Random Forest",
        params={"n_estimators": 4, "max_depth": 3, "random_state": 0},
        verify_samples=2, export_formats=("python", "binary"),
        output_dir=str(tmp_path), output_name="pipe", verbose=False,
    )
    assert result["passed"]
    assert os.path.exists(result["hcpn"]["initial"])
    assert os.path.exists(result["hcpn"]["final"])
    assert result["verification"]["final"].properties["PC"].holds is True
    assert result["verification"]["final"].tested > 2
    assert "comparison" in result["verification"]
    assert os.path.exists(result["exports"]["python"])


def test_verified_pipeline_gates_on_a_violation(tmp_path, monkeypatch):
    """A broken refined net must stop the pipeline before it exports anything."""
    from pyruleanalyzer import pyruleanalyzer as pra

    real_export = pra.PyRuleAnalyzer.export_hcpn

    def sabotage(self, base_name="model", which="both", **kwargs):
        paths = real_export(self, base_name=base_name, which=which, **kwargs)
        if "final" in paths:
            [m] = make_mutants(paths["final"], str(tmp_path), names=["hidden_overlap"])
            os.replace(m["path"], paths["final"])
        return paths

    monkeypatch.setattr(pra.PyRuleAnalyzer, "export_hcpn", sabotage)
    X, y, _cols = _dataset(n_classes=2, seed=5)
    with pytest.raises(VerificationError) as excinfo:
        verified_pipeline(
            X=X, y=y, model_type="Decision Tree",
            params={"max_depth": 4, "random_state": 0},
            verify_samples=1, export_formats=("python",),
            output_dir=str(tmp_path), output_name="gated", verbose=False,
        )
    assert excinfo.value.stage == "final"
    assert "B4" in str(excinfo.value)
    assert not os.path.exists(os.path.join(str(tmp_path), "gated.py"))


def test_verified_pipeline_accepts_a_pretrained_estimator(tmp_path):
    from sklearn.ensemble import RandomForestClassifier

    X, y, _cols = _dataset(n_classes=3, seed=9)
    clf = RandomForestClassifier(n_estimators=4, max_depth=3,
                                 random_state=0).fit(X, y)
    result = verified_pipeline(
        X=X, y=y, sklearn_model=clf, verify_samples=1, export_formats=(),
        output_dir=str(tmp_path), output_name="pre", verbose=False,
    )
    assert result["passed"]
    assert result["model"].classifier.algorithm_type == "Random Forest"


def test_from_sklearn_matches_fit(tmp_path):
    from sklearn.tree import DecisionTreeClassifier

    X, y, _cols = _dataset(n_classes=2, seed=11)
    clf = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
    wrapped = PyRuleAnalyzer.from_sklearn(clf, feature_names=list(X.columns))
    trained = PyRuleAnalyzer.new_model(model="Decision Tree",
                                       params={"max_depth": 4, "random_state": 0})
    trained.fit(X, y)
    assert len(wrapped.classifier.initial_rules) == \
        len(trained.classifier.initial_rules)
    assert list(wrapped.predict(X)) == list(trained.predict(X))


def test_from_sklearn_rejects_unsupported_estimators():
    from sklearn.linear_model import LogisticRegression

    with pytest.raises(ValueError, match="Unsupported estimator"):
        PyRuleAnalyzer.from_sklearn(LogisticRegression())
