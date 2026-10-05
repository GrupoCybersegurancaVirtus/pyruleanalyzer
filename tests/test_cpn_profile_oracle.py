"""Agreement with CPN Tools itself.

CPNCheck answers the questions CPN Tools answers interactively, so the answers
have to be the same. These tests put both engines on the same nets and compare
every statistic and every verdict.

They need the CPN Tools engine -- CPN Tools 4.0.1, CPN IDE's bundled Access/CPN
libraries and a 32-bit Java 8 with ``jjs`` -- and each net costs about ten
seconds, so they are skipped unless ``CPNCHECK_CPNTOOLS=1`` is set.

The full sweep, correct nets and every mutant, is what actually settles the
question; it takes a few minutes and is run from the command line:

    cpncheck oracle tests/nets/gbdt_binary.cpn --labels 0,1

Agreement on correct nets alone would also be reached by two checkers that
always answer "true", so the mutants -- where verdicts are false -- are the
part that carries the argument.
"""

import os

import pytest

from pyruleanalyzer.cpn_mutants import make_mutants
from pyruleanalyzer.cpntools_oracle import CPNToolsOracle, compare_with_oracle

NETS = os.path.join(os.path.dirname(__file__), "cpn_nets")
GBDT = os.path.join(NETS, "gbdt_binary.cpn")

pytestmark = pytest.mark.skipif(
    "1" not in (os.environ.get("CPNCHECK_CPNTOOLS"), os.environ.get("PYRA_CPNTOOLS")),
    reason="set CPNCHECK_CPNTOOLS=1 and install the CPN Tools engine")


@pytest.fixture(scope="module")
def oracle():
    engine = CPNToolsOracle()
    missing = engine.missing()
    if missing:
        pytest.skip(f"CPN Tools engine incomplete: {', '.join(missing)}")
    return engine

def test_a_correct_net_agrees(oracle):
    report = compare_with_oracle(GBDT, oracle=oracle, class_labels=[0, 1])
    assert not report["errors"], report["errors"]
    assert report["agree"], [r for r in report["rows"] if not r[3]]

@pytest.mark.parametrize("mutant", ["cycle", "deadlock", "bad_label",
                                    "two_decisions"])
def test_a_mutant_agrees_including_its_false_verdicts(tmp_path, oracle, mutant):
    """The half that matters: both engines must answer *false* in the same places."""
    made = make_mutants(GBDT, str(tmp_path), names=[mutant])
    assert made, f"the {mutant} operator did not apply"
    report = compare_with_oracle(made[0]["path"], oracle=oracle,
                                 class_labels=made[0]["labels"] or [0, 1])
    assert not report["errors"], report["errors"]
    assert report["agree"], [r for r in report["rows"] if not r[3]]
    falses = [key for key, mine, _theirs, _ok in report["rows"]
              if mine == "false"]
    assert falses, "this mutant produced no false verdict to confirm"
