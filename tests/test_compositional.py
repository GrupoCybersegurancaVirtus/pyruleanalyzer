"""Tests for compositional verification of the ensemble nets.

``reduction="compositional"`` decides the catalog from one graph per tree (or
boosting channel) and the join layer. It is held to two standards:

* **the full graph's verdicts** -- on nets small enough for the full
  occurrence graph, correct ones and every mutant, for the net's own input
  and for other inputs, every verdict the composition gives equals the full
  graph's, and with the fallback every verdict does;
* **scale** -- ensembles whose interleaved graph is out of reach get a
  verdict for every property, and the verified pipeline passes on them.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyruleanalyzer import PyRuleAnalyzer
from pyruleanalyzer.compositional import Decomposition
from pyruleanalyzer.cpn_mutants import analyse, make_mutants
from pyruleanalyzer.model_checker import (CompositionalResult, CPNModelChecker,
                                          check_cpn)
from pyruleanalyzer.verified_pipeline import verified_pipeline

NETS = os.path.join(os.path.dirname(__file__), "cpn_nets")

SMALL = [
    ("Random Forest", {"n_estimators": 3, "max_depth": 3, "random_state": 0}, 2),
    ("Random Forest", {"n_estimators": 5, "max_depth": 3, "random_state": 1}, 3),
    ("Gradient Boosting Decision Trees",
     {"n_estimators": 4, "max_depth": 2, "random_state": 0}, 2),
    ("Gradient Boosting Decision Trees",
     {"n_estimators": 3, "max_depth": 2, "random_state": 0}, 3),
]
SMALL_IDS = ["rf3", "rf5x3", "gbdt2", "gbdt3"]


def _dataset(n_classes=2, n_features=4, seed=0, rows=300):
    rng = np.random.RandomState(seed)
    X = rng.rand(rows, n_features)
    if n_classes == 2:
        y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    else:
        y = (X[:, 0] * n_classes).astype(int).clip(0, n_classes - 1)
    cols = [f"feat_{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="target")


def _nets(tmp_path, model, params, n_classes):
    X, y = _dataset(n_classes)
    analyzer = PyRuleAnalyzer.new_model(model=model, params=params)
    analyzer.fit(X.iloc[:220], y.iloc[:220])
    analyzer.execute_rule_refinement(X=X.iloc[220:], y=y.iloc[220:],
                                     remove_below_n_classifications=1,
                                     save_final_model=False, save_report=False)
    paths = analyzer.export_hcpn(base_name=os.path.join(str(tmp_path), "net"),
                                 which="both", sample=X.iloc[220])
    return analyzer, paths, X.iloc[220:]


def _labels(analyzer):
    return [int(c) for c in analyzer.class_names]


# ---------------------------------------------------------------------------
# Same verdicts as the full graph
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,params,n_classes", SMALL, ids=SMALL_IDS)
def test_compositional_verdicts_equal_the_full_graph(tmp_path, model, params,
                                                     n_classes):
    """Correct nets and every mutant, own input and two other inputs."""
    analyzer, paths, X = _nets(tmp_path, model, params, n_classes)
    mutants = make_mutants(paths["final"], os.path.join(str(tmp_path), "m"))
    composed = 0
    for path in [paths["initial"], paths["final"]] + [m["path"] for m in mutants]:
        for samples in (None, X.iloc[1:3]):
            mc = CPNModelChecker(path, class_labels=_labels(analyzer))
            full = mc.check(samples=samples)
            alone = mc.check(samples=samples, reduction="compositional",
                             fallback=None)
            backed = mc.check(samples=samples, reduction="compositional")
            assert isinstance(backed, CompositionalResult)
            for pid, r in full.properties.items():
                a, b = alone.properties[pid], backed.properties[pid]
                if a.holds is not None:
                    composed += 1
                    assert a.holds == r.holds, (os.path.basename(path), pid,
                                                a.detail)
                assert b.holds == r.holds, (os.path.basename(path), pid,
                                            b.detail)
    assert composed > 100


def test_every_mutant_is_caught_and_its_trace_replays(tmp_path):
    """The composition's counterexamples are sequences of the full net."""
    analyzer, paths, _X = _nets(tmp_path, *SMALL[3])
    replayed = 0
    for m in make_mutants(paths["final"], os.path.join(str(tmp_path), "m")):
        mc = CPNModelChecker(m["path"], class_labels=_labels(analyzer))
        result = mc.check(reduction="compositional")
        failed = {r.id for r in result.failures}
        for target in m["targets"]:
            if target != "PC":
                assert target in failed, (m["name"], target, result.report())
        for r in result.failures:
            assert r.detail == str(r.cex)
            if r.cex.kind in ("finite", "lasso"):
                assert r.cex.replay(mc.cnet), (m["name"], r.id, r.detail)
                replayed += 1
    assert replayed > 10


def test_composed_verdicts_say_so(tmp_path):
    _analyzer, paths, _X = _nets(tmp_path, *SMALL[0])
    result = check_cpn(paths["final"], class_labels=[0, 1],
                       reduction="compositional")
    assert result.passed and not result.skipped, result.report()
    for pid in ("A1", "A2", "A5", "A8", "B1", "D1", "D2a"):
        assert result.properties[pid].detail.startswith("compositional:")
        assert result.stats["plans"][pid].startswith("compositional")
    assert "reduction        : compositional" in result.report()
    assert result.stats["reduction"] == "compositional"


def test_what_cannot_be_composed_falls_back_with_a_reason(tmp_path):
    """A forest whose Vote ignores a tree: D1 is not structural any more."""
    analyzer, paths, _X = _nets(tmp_path, *SMALL[0])
    [m] = make_mutants(paths["final"], str(tmp_path), names=["partial_vote"])
    mc = CPNModelChecker(m["path"], class_labels=[0, 1])
    alone = mc.check(reduction="compositional", fallback=None)
    assert alone.properties["D1"].holds is None
    assert "not compositional" in alone.properties["D1"].detail
    backed = mc.check(reduction="compositional")
    assert backed.properties["D1"].holds is False
    assert "stubborn" in backed.stats["plans"]["D1"]


def test_a_decision_tree_is_not_decomposed(tmp_path):
    analyzer, paths, _X = _nets(tmp_path, "Decision Tree",
                                {"max_depth": 3, "random_state": 0}, 2)
    mc = CPNModelChecker(paths["final"], class_labels=[0, 1])
    dec = Decomposition(mc.cnet, mc.input_places)
    assert not dec.ok and dec.reason    # the leaves read the input directly
    result = mc.check(reduction="compositional")
    assert result.passed and not result.skipped


def test_fixture_nets_decompose():
    for name, n in (("rf_multiclass.cpn", 4), ("gbdt_binary.cpn", 1)):
        mc = CPNModelChecker(os.path.join(NETS, name), class_labels=[0, 1, 2])
        dec = Decomposition(mc.cnet, mc.input_places)
        assert dec.ok and dec.sync, dec.reason or dec.sync_reason
        assert len(dec.components) == n
        result = mc.check(reduction="compositional", fallback=None)
        assert result.passed and not result.skipped, result.report()


def test_unknown_fallback_is_refused(tmp_path):
    mc = CPNModelChecker(os.path.join(NETS, "rf_multiclass.cpn"))
    with pytest.raises(ValueError):
        mc.check(reduction="compositional", fallback="sweep")


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,params,n_classes", [
    ("Random Forest", {"n_estimators": 20, "max_depth": 4, "random_state": 0}, 2),
    ("Gradient Boosting Decision Trees",
     {"n_estimators": 12, "max_depth": 2, "random_state": 0}, 3),
], ids=["rf20", "gbdt12x3"])
def test_large_ensembles_get_every_verdict(tmp_path, model, params, n_classes):
    """2**20 + 2 and 39**3 + 2 markings interleaved; a few dozen composed."""
    analyzer, paths, X = _nets(tmp_path, model, params, n_classes)
    mc = CPNModelChecker(paths["final"], class_labels=_labels(analyzer),
                         feature_names=analyzer.feature_names, max_nodes=50_000)
    t0 = time.time()
    result = mc.check(samples=X.iloc[:2], reduction="compositional",
                      fallback=None, classifier=analyzer.classifier,
                      test_samples=X)
    assert time.time() - t0 < 30
    assert not result.skipped, result.report()
    assert result.passed, result.report()
    assert result.stats["nodes"] < 1000


def test_verified_pipeline_passes_a_20_tree_forest(tmp_path):
    X, y = _dataset(n_classes=2, seed=3)
    result = verified_pipeline(
        X=X, y=y, model_type="Random Forest",
        params={"n_estimators": 20, "max_depth": 4, "random_state": 0},
        verify_samples=2, max_nodes=50_000, reduction="compositional",
        export_formats=(), output_dir=str(tmp_path), output_name="rf20",
        verbose=False)
    assert result["passed"]
    assert result["inconclusive"] == {"initial": [], "final": []}
    final = result["verification"]["final"]
    assert final.stats["reduction"] == "compositional"
    assert final.properties["D2a"].holds is True


# ---------------------------------------------------------------------------
# The label domain read off the net
# ---------------------------------------------------------------------------

def test_mutation_analysis_reads_constant_labels_only():
    """Decide_k writes k; a forest's decide(...) is not a constant."""
    assert analyse(os.path.join(NETS, "gbdt_binary.cpn"))["labels"] == [0, 1]
    assert analyse(os.path.join(NETS, "rf_multiclass.cpn"))["labels"] == []
    mc = CPNModelChecker(os.path.join(NETS, "rf_multiclass.cpn"))
    assert mc.labels is None
