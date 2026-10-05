"""Partial-order reduction: the reduced graphs must give the full graph's verdicts.

A reduction is only worth having if it is invisible in the answers. The
standard held to here is the strongest one available without a proof
assistant: on every fixture net and every one of its mutants -- nets built to
violate each property in turn -- the verdict of every property with
``reduction="stubborn"`` equals the verdict on the full occurrence graph, and
every counterexample found on a reduced graph replays on the net.
"""

import os

import pytest

from pyruleanalyzer.model_checker import CPNModelChecker, check_cpn
from cpncheck.coloured import ColouredNet
from cpncheck.formula import (AF, AG, EF, EU, Dead, Enabled, Implies, Marked,
                              MaxTokens, Not)
from pyruleanalyzer.cpn_mutants import make_mutants
from cpncheck.net import CPNNet
from cpncheck.reduction import (Dependency, ReducedChecker, build_reduced,
                                classify, split)
from cpncheck.selectors import Places, Transitions
from cpncheck.statespace import build_state_space

NETS = os.path.join(os.path.dirname(__file__), "cpn_nets")
FIXTURES = [("gbdt_binary.cpn", [0, 1]), ("rf_multiclass.cpn", [0, 1, 2])]
IDS = [f[0] for f in FIXTURES]


def _nets(tmp_path, name):
    """The fixture itself followed by every mutant of it."""
    src = os.path.join(NETS, name)
    return [{"name": "correct", "path": src, "labels": None}] + \
        make_mutants(src, str(tmp_path))


@pytest.mark.parametrize("name,labels", FIXTURES, ids=IDS)
def test_reduced_verdicts_equal_full_verdicts(tmp_path, name, labels):
    compared = 0
    for m in _nets(tmp_path, name):
        mc = CPNModelChecker(m["path"], class_labels=m["labels"] or labels)
        full = mc.check()
        reduced = mc.check(reduction="stubborn")
        assert list(full.properties) == list(reduced.properties)
        for pid, r in full.properties.items():
            assert reduced.properties[pid].holds == r.holds, \
                (m["name"], pid, reduced.properties[pid].detail)
            compared += 1
    assert compared > 150

@pytest.mark.parametrize("name,labels", FIXTURES, ids=IDS)
def test_reduced_counterexamples_replay(tmp_path, name, labels):
    replayed = 0
    for m in _nets(tmp_path, name):
        mc = CPNModelChecker(m["path"], class_labels=m["labels"] or labels)
        for r in mc.check(reduction="stubborn").failures:
            assert r.cex is not None, (m["name"], r.id)
            if r.cex.kind in ("finite", "lasso"):
                assert r.cex.replay(mc.cnet), (m["name"], r.id, str(r.cex))
                replayed += 1
    assert replayed > 5

@pytest.mark.parametrize("name,labels", FIXTURES, ids=IDS)
def test_every_dead_marking_survives_the_reduction(tmp_path, name, labels):
    for m in _nets(tmp_path, name):
        net = ColouredNet(CPNNet(m["path"]))
        full = build_state_space(net)
        reduced = build_reduced(net, Dependency(net), "deadlock")
        assert reduced.complete
        assert sorted(reduced.markings[v] for v in reduced.dead) == \
            sorted(full.markings[v] for v in full.dead), m["name"]

def test_the_report_says_which_graph_decided_each_property():
    result = check_cpn(os.path.join(NETS, "rf_multiclass.cpn"),
                       class_labels=[0, 1, 2], reduction="stubborn")
    text = result.report()
    assert "stubborn sets" in text
    assert "decided on: deadlock" in text
    assert result.stats["reduction"] == "stubborn"
    assert result.oracle == {}

def test_an_exhausted_budget_gives_no_verdict_and_says_why():
    mc = CPNModelChecker(os.path.join(NETS, "rf_multiclass.cpn"),
                         class_labels=[0, 1, 2])
    result = mc.check(reduction="stubborn", max_nodes=8)
    d2a = result.properties["D2a"]
    assert d2a.holds is None
    assert "cannot be reordered" in d2a.detail
    assert result.properties["A1"].holds is True
