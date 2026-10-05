"""Guard boxes and conformance, the two techniques that are not CTL.

Neither needs an occurrence graph, and neither is specific to any family of
nets: the first asks whether two guards can hold together for *any* input, the
second runs the net on concrete inputs and compares with a reference.
"""

import os

import pytest

from cpncheck.coloured import ColouredNet
from cpncheck.conformance import check_conformance, projection, terminal_marking
from cpncheck.net import CPNNet
from pyruleanalyzer.cpn_profile import MLTreeProfile
from cpncheck.structural import (WHOLE, boxes_intersect, guard_box,
                                 overlapping_guards)

NETS = os.path.join(os.path.dirname(__file__), "cpn_nets")
GBDT = os.path.join(NETS, "gbdt_binary.cpn")


@pytest.fixture(scope="module")
def net():
    return ColouredNet(CPNNet(GBDT))


def test_the_leaves_of_a_correct_net_are_disjoint(net):
    groups = {label: roles["leaves"] for label, roles
              in MLTreeProfile().resolve_groups(net)["trees"].items()}
    found, unparsed = overlapping_guards(
        net, groups, ignore=lambda n: n.endswith("_default"))
    assert found == []
    assert unparsed == 0

def test_pairs_are_only_formed_inside_a_group(net):
    """Two branches of different choices may of course both be enabled."""
    everything = {label: roles["leaves"] for label, roles
                  in MLTreeProfile().resolve_groups(net)["trees"].items()}
    merged = {"all of them": [ti for g in everything.values() for ti in g]}
    apart, _ = overlapping_guards(net, everything,
                                  ignore=lambda n: n.endswith("_default"))
    together, _ = overlapping_guards(net, merged,
                                     ignore=lambda n: n.endswith("_default"))
    assert apart == [] and together != []

def test_an_unparsable_guard_is_counted_not_guessed(net):
    groups = {"one tree": [ti for ti, t in enumerate(net.transitions)
                           if t["page"] == "Tree_c1_s1"]}
    _found, unparsed = overlapping_guards(net, groups)
    assert unparsed == 1                      # the catch-all, when not ignored

def test_a_run_reaches_a_marking_holding_the_result(net):
    profile = MLTreeProfile()
    structure = profile.structure(net)
    final = terminal_marking(net)
    assert projection(final, (structure.pred,))

def test_conformance_agrees_with_the_net_itself(net):
    """A reference that is the net cannot disagree with it."""
    profile = MLTreeProfile()
    structure = profile.structure(net)
    inputs = [net.initial[structure.input][0]]
    mine = [projection(terminal_marking(net, x, (structure.input,)),
                       (structure.pred,))[0] for x in inputs]
    result = check_conformance(net, inputs, lambda batch: mine,
                               input_places=(structure.input,),
                               output_places=(structure.pred,))
    assert result.holds is True
    assert "1/1 inputs agree" in result.detail

def test_a_disagreement_names_the_input_that_caused_it(net):
    profile = MLTreeProfile()
    structure = profile.structure(net)
    inputs = [net.initial[structure.input][0]]
    result = check_conformance(net, inputs, lambda batch: [999] * len(batch),
                               input_places=(structure.input,),
                               output_places=(structure.pred,))
    assert result.holds is False
    assert "input=(" in result.detail and "expected=999" in result.detail

def test_a_broken_reference_is_reported_as_not_evaluated(net):
    profile = MLTreeProfile()
    structure = profile.structure(net)
    inputs = [net.initial[structure.input][0]]

    def boom(batch):
        raise RuntimeError("the model is not loaded")

    result = check_conformance(net, inputs, boom,
                               input_places=(structure.input,),
                               output_places=(structure.pred,))
    assert result.holds is None
    assert "the model is not loaded" in result.detail

def test_a_reference_of_the_wrong_length_is_refused(net):
    profile = MLTreeProfile()
    structure = profile.structure(net)
    inputs = [net.initial[structure.input][0]]
    result = check_conformance(net, inputs, lambda batch: [],
                               input_places=(structure.input,),
                               output_places=(structure.pred,))
    assert result.holds is None
    assert "0 values for 1 inputs" in result.detail
