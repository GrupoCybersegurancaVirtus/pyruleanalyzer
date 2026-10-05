"""The tree profile on its fixture nets: the catalog, the mutants, the
counterexamples, the reduction and CPN Tools.

These tests moved here from CPNCheck when the tree profile did: they
check pyruleanalyzer's plugin, through CPNCheck's generic machinery.
"""

import json
import os

import pytest

from cpncheck import METHODS, compare_results
from pyruleanalyzer.model_checker import CPNModelChecker, check_cpn
from pyruleanalyzer.cpn_mutants import make_mutants
from pyruleanalyzer.cpn_profile import TREE_CATALOG
from pyruleanalyzer.cpn_mutants import TREE_MUTANTS

NETS = os.path.join(os.path.dirname(__file__), "cpn_nets")
FIXTURES = [("gbdt_binary.cpn", "Gradient Boosting Decision Trees", [0, 1]),
            ("rf_multiclass.cpn", "Random Forest", [0, 1, 2])]
IDS = [f[0] for f in FIXTURES]


@pytest.mark.parametrize("name,family,labels", FIXTURES, ids=IDS)
def test_a_correct_net_satisfies_every_property(name, family, labels):
    result = check_cpn(os.path.join(NETS, name), class_labels=labels)
    assert result.family == family
    assert result.passed, result.report()
    assert not result.skipped, result.report()

@pytest.mark.parametrize("name,family,labels", FIXTURES, ids=IDS)
def test_family_specific_properties_are_selected(name, family, labels):
    ids = set(check_cpn(os.path.join(NETS, name), class_labels=labels).properties)
    if family == "Gradient Boosting Decision Trees":
        assert {"C1", "C2", "C3a", "C3b"} <= ids and "D1" not in ids
    else:
        assert {"D1", "D2a", "D2b"} <= ids and "C1" not in ids

@pytest.mark.parametrize("name,family,labels", FIXTURES, ids=IDS)
def test_every_mutant_fails_the_property_it_targets(tmp_path, name, family, labels):
    """A checker that only ever answers "true" verifies nothing."""
    made = make_mutants(os.path.join(NETS, name), str(tmp_path))
    assert made, "no mutant applies to this net"
    for m in made:
        result = check_cpn(m["path"], class_labels=m["labels"] or labels)
        for pid in m["targets"]:
            if pid == "PC":
                continue           # PC needs the reference classifier
            assert result.properties[pid].holds is False, (
                f"{m['name']}: {pid} still holds\n{result.report()}")
            assert result.properties[pid].detail, (
                f"{m['name']}: {pid} failed without a counterexample")

def test_every_property_has_a_mutant():
    covered = {pid for m in TREE_MUTANTS for pid in m.targets}
    # D2b is A5's formula on the same place, so double_output covers it too.
    missing = set(TREE_CATALOG) - {"D2b"} - covered
    assert not missing, missing

def test_the_occurrence_graph_is_deterministic():
    mc = CPNModelChecker(os.path.join(NETS, "gbdt_binary.cpn"))
    a, b = mc.state_space(), mc.state_space()
    assert (a.n_nodes, a.n_arcs) == (b.n_nodes, b.n_arcs)
    assert a.markings == b.markings

def test_report_json_and_latex():
    result = check_cpn(os.path.join(NETS, "gbdt_binary.cpn"), class_labels=[0, 1])
    data = json.loads(json.dumps(result.to_dict()))
    assert data["passed"] is True
    assert all(v["method"] in METHODS for v in data["properties"].values())
    assert "VERIFIED" in result.report()
    assert r"\begin{table}" in result.latex_table()

def test_compare_results_flags_a_regression(tmp_path):
    good = check_cpn(os.path.join(NETS, "gbdt_binary.cpn"), class_labels=[0, 1])
    bad_path = make_mutants(os.path.join(NETS, "gbdt_binary.cpn"),
                            str(tmp_path), names=["deadlock"])[0]["path"]
    bad = check_cpn(bad_path, class_labels=[0, 1])
    text = compare_results(good, bad)
    assert "REGRESSION" in text

def test_the_askctl_script_is_written(tmp_path):
    mc = CPNModelChecker(os.path.join(NETS, "gbdt_binary.cpn"), class_labels=[0, 1])
    path = mc.export_askctl(str(tmp_path / "askctl.sml"))
    text = open(path, encoding="utf-8").read()
    decl, queries = mc.ml_program()
    assert set(queries) <= set(mc.check().oracle)
    assert "EXIST_UNTIL" in text or "EV(" in text
