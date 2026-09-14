"""Backwards-compatible shim: the reference evaluator now lives in the package.

The module moved to :mod:`pyruleanalyzer.cpn_semantics` so that the library --
and not only the test suite -- can read back a generated ``.cpn``. Scripts that
still do ``sys.path.insert(0, "tests"); import cpn_semantics`` keep working
through this re-export.
"""

from pyruleanalyzer.cpn_semantics import (  # noqa: F401
    CPNNet,
    _argmax,
    _eval,
    _parse_prob_list,
    evaluate,
    sml_to_python,
)

__all__ = ["CPNNet", "evaluate", "sml_to_python"]
