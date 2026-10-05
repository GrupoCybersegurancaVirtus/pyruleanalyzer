"""The generic layer: selectors, formulas, profiles and ASK-CTL rendering.

These tests use the fixture nets, but none of them go through the tree
profile's catalog: they state properties by hand, and one of them drives the
whole checker with a profile written inside the test. That is the claim the
extraction has to support -- that a caller can verify their own nets without
touching the checker.
"""

import collections
import os

import pytest

from cpncheck import METHODS
from pyruleanalyzer.model_checker import CPNModelChecker
from cpncheck.askctl import MLProgram, UnsupportedFormula
from cpncheck.coloured import ColouredNet
from cpncheck.formula import (AF, AG, EF, AllMarked, And, Context, Dead,
                              Enabled, EnabledCount, Implies, Marked,
                              MaxTokens, Not, Or, TokensIn)
from cpncheck.net import CPNNet
from cpncheck.profile import Profile, detect
from pyruleanalyzer.cpn_profile import MLTreeProfile
from cpncheck.properties import Property
from cpncheck.selectors import Explicit, Places, Transitions, outputs_of

NETS = os.path.join(os.path.dirname(__file__), "cpn_nets")
GBDT = os.path.join(NETS, "gbdt_binary.cpn")
RF = os.path.join(NETS, "rf_multiclass.cpn")


@pytest.fixture(scope="module")
def net():
    return ColouredNet(CPNNet(GBDT))

@pytest.fixture(scope="module")
def ctx(net):
    from cpncheck.statespace import build_state_space
    return Context(build_state_space(net), net)


def test_every_catalog_formula_renders_to_askctl():
    for path, labels in ((GBDT, [0, 1]), (RF, [0, 1, 2])):
        mc = CPNModelChecker(path, class_labels=labels)
        decl, queries = mc.ml_program()
        assert set(queries) >= set(mc.formulas)
        for pid in mc.formulas:
            assert queries[pid].startswith("Bool.toString (eval_node")
        assert "val DEAD" in decl and "fun AFmax" in decl

def test_the_generated_program_asks_about_every_property():
    """Every property the checker answers is also asked of CPN Tools.

    That the two engines then give the same answers is what ``cpncheck
    oracle`` establishes; see tests/test_oracle.py.
    """
    for path, labels in ((GBDT, [0, 1]), (RF, [0, 1, 2])):
        mc = CPNModelChecker(path, class_labels=labels)
        queries = mc.ml_program()[1]
        assert set(mc.property_ids) <= set(queries)
        assert set(queries) <= set(mc.check().oracle)

def test_the_tree_profile_is_detected(net):
    profile = detect(net)
    assert profile is not None and profile.name == "ml_trees"

def test_a_property_that_cannot_be_decided_is_skipped_not_hidden():
    """"Does not apply" and "could not be answered" are different verdicts.

    A net with no aggregation simply has no aggregation property, and it is
    never mentioned. A net that *has* a label domain the checker cannot read
    still has the valid-label property, and leaving it out would read as if it
    had passed.
    """
    class NoReadableDomain(MLTreeProfile):
        """As for a decision tree, whose labels the net does not write down."""

        def label_domain(self, net, declared=None):
            if declared is None:
                return None, "undeclared"
            return super().label_domain(net, declared)

    profile = NoReadableDomain()
    net = ColouredNet(CPNNet(GBDT))
    groups = profile.resolve_groups(net)
    assert profile.unavailable(net, groups, {}) == {
        "A6": "label domain not declared; pass class_labels"}
    assert profile.unavailable(net, groups, {"class_labels": [0, 1]}) == {}
    assert "A6" not in profile.ctl_formulas(net, groups, {})
    assert "A6" in profile.ctl_formulas(net, groups, {"class_labels": [0, 1]})
