"""Counterexamples: the evidence a violated property hands back.

A verdict of "false" that cannot be shown is a verdict that has to be taken on
trust. The standard held to here is that every violation the checker reports
comes with evidence, and that the evidence *replays* -- re-firing it with the
net's own firing rule, not the search that found it, has to reproduce the
violation. A trace that cannot be replayed is a trace the checker invented.
"""

import json
import os

import pytest

from pyruleanalyzer.model_checker import CPNModelChecker, check_cpn
from cpncheck.counterexample import Counterexample, Step
from pyruleanalyzer.cpn_mutants import make_mutants

NETS = os.path.join(os.path.dirname(__file__), "cpn_nets")
FIXTURES = [("gbdt_binary.cpn", [0, 1]), ("rf_multiclass.cpn", [0, 1, 2])]
IDS = [f[0] for f in FIXTURES]

#: Shapes that are a sequence and must therefore replay.
SEQUENCES = ("finite", "lasso")


def _mutants(tmp_path, name):
    return make_mutants(os.path.join(NETS, name), str(tmp_path))

@pytest.mark.parametrize("name,labels", FIXTURES, ids=IDS)
def test_every_violation_carries_a_counterexample(tmp_path, name, labels):
    for m in _mutants(tmp_path, name):
        result = check_cpn(m["path"], class_labels=m["labels"] or labels)
        for r in result.failures:
            assert r.cex is not None, (m["name"], r.id)
            assert r.detail == str(r.cex)
            assert r.cex.note, (m["name"], r.id)

@pytest.mark.parametrize("name,labels", FIXTURES, ids=IDS)
def test_every_sequence_replays(tmp_path, name, labels):
    """The certificate: re-firing the trace must reproduce the violation."""
    checked = 0
    for m in _mutants(tmp_path, name):
        mc = CPNModelChecker(m["path"], class_labels=m["labels"] or labels)
        for r in mc.check().failures:
            if r.cex.kind in SEQUENCES:
                assert r.cex.replay(mc.cnet), (m["name"], r.id, str(r.cex))
                checked += 1
    assert checked > 10, "hardly any sequences were exercised"

def test_a_tampered_trace_does_not_replay(tmp_path):
    """The certificate has to be able to fail, or it certifies nothing."""
    m = _mutants(tmp_path, "gbdt_binary.cpn")[1]        # deadlock
    mc = CPNModelChecker(m["path"], class_labels=m["labels"] or [0, 1])
    cex = next(r.cex for r in mc.check().failures if r.cex.kind == "finite")
    assert cex.replay(mc.cnet)

    swapped = Counterexample(
        cex.kind, note=cex.note, prefix=list(reversed(cex.prefix)),
        final=cex.final, input=cex.input, data=cex.data)
    assert not swapped.replay(mc.cnet)

    truncated = Counterexample(cex.kind, note=cex.note, prefix=cex.prefix[:-1],
                               final=cex.final, input=cex.input, data=cex.data)
    assert not truncated.replay(mc.cnet)

def test_a_cycle_gives_a_lasso(tmp_path):
    """A run that never terminates has no finite witness; the cycle is it."""
    m = _mutants(tmp_path, "gbdt_binary.cpn")[0]        # cycle
    assert m["name"] == "cycle"
    mc = CPNModelChecker(m["path"], class_labels=m["labels"] or [0, 1])
    cex = mc.check().properties["A1"].cex
    assert cex.kind == "lasso"
    assert cex.cycle, "a lasso without a cycle is not a lasso"
    assert "then forever [" in str(cex)
    assert cex.replay(mc.cnet)

def test_a_deadlock_gives_a_finite_trace_to_a_dead_marking(tmp_path):
    m = _mutants(tmp_path, "gbdt_binary.cpn")[1]        # deadlock
    assert m["name"] == "deadlock"
    mc = CPNModelChecker(m["path"], class_labels=m["labels"] or [0, 1])
    ss = mc.state_space()
    cex = mc.check().properties["A3"].cex
    assert cex.kind == "finite" and not cex.cycle
    assert cex.final in [ss.markings[v] for v in ss.dead]
    assert cex.replay(mc.cnet)

def test_no_home_marking_names_two_components(tmp_path):
    m = _mutants(tmp_path, "gbdt_binary.cpn")[2]        # dead_branch
    assert m["name"] == "dead_branch"
    result = check_cpn(m["path"], class_labels=m["labels"] or [0, 1])
    cex = result.properties["A8"].cex
    assert cex.kind == "witness"
    assert cex.data["terminal_components"] >= 2
    assert len(cex.data["representatives"]) == 2
    assert "no longer meet" in cex.note

def test_steps_carry_the_binding(tmp_path):
    m = _mutants(tmp_path, "gbdt_binary.cpn")[1]
    mc = CPNModelChecker(m["path"], class_labels=m["labels"] or [0, 1])
    cex = mc.check().properties["A3"].cex
    assert all(isinstance(s, Step) for s in cex.prefix)
    assert any(s.binding for s in cex.prefix), "no step bound anything"
    detailed = cex.sequence(bindings=True)
    assert "<" in detailed and "=" in detailed

def test_a_guard_overlap_yields_an_input_that_exhibits_it(tmp_path):
    """The structural check proves an overlap for *some* input; this is one.

    ``hidden_overlap`` puts the same guard on two leaves the net's own input
    never reaches, so the per-input check sees nothing and only the structural
    one fires. Feeding its witness back has to make the per-input check fail
    too, on the very pair that was reported -- which is what makes the
    structural finding reproducible rather than merely asserted.
    """
    m = next(x for x in _mutants(tmp_path, "gbdt_binary.cpn")
             if x["name"] == "hidden_overlap")
    labels = m["labels"] or [0, 1]

    before = check_cpn(m["path"], class_labels=labels)
    assert before.properties["B4"].holds is False
    assert before.properties["B1"].holds is True       # nothing reaches it yet

    cex = before.properties["B4"].cex
    assert cex.kind == "structural"
    witness = cex.input
    width = CPNModelChecker(m["path"], class_labels=labels).width
    assert witness is not None and len(witness) == width

    after = check_cpn(m["path"], samples=[witness], class_labels=labels)
    assert after.properties["B1"].holds is False
    for leaf in cex.data["leaves"]:
        assert leaf in after.properties["B1"].detail

def test_the_input_is_named_when_several_were_checked(tmp_path):
    m = _mutants(tmp_path, "gbdt_binary.cpn")[1]
    labels = m["labels"] or [0, 1]
    own = CPNModelChecker(m["path"], class_labels=labels).own_sample

    without = check_cpn(m["path"], class_labels=labels)
    assert "for input=" not in without.properties["A3"].detail

    with_sample = check_cpn(m["path"], samples=[own], class_labels=labels)
    assert with_sample.properties["A3"].detail.startswith("for input=")

def test_the_counterexample_survives_serialisation(tmp_path):
    m = _mutants(tmp_path, "gbdt_binary.cpn")[1]
    result = check_cpn(m["path"], class_labels=m["labels"] or [0, 1])
    data = json.loads(json.dumps(result.to_dict()))
    entry = data["properties"]["A3"]["counterexample"]
    assert entry["kind"] == "finite"
    assert entry["length"] == len(entry["prefix"])
    assert entry["prefix"][0]["transition"]
    assert data["properties"]["A1"]["counterexample"] is None   # A1 holds here

def test_a_passing_property_has_no_counterexample():
    result = check_cpn(os.path.join(NETS, "gbdt_binary.cpn"),
                       class_labels=[0, 1])
    assert result.passed
    assert all(r.cex is None for r in result.properties.values())

def test_the_full_sequence_is_kept_even_when_the_report_elides_it(tmp_path):
    m = _mutants(tmp_path, "gbdt_binary.cpn")[1]
    result = check_cpn(m["path"], class_labels=m["labels"] or [0, 1])
    cex = result.properties["A3"].cex
    assert cex.length > 12, "this net no longer produces a long trace"
    assert "more ..." in str(cex)
    assert "more ..." not in cex.sequence(limit=10 ** 6)
    assert len(cex.to_dict()["prefix"]) == cex.length
