"""Backwards-compatible re-export: the occurrence graph moved to CPNCheck.

Flattening a net, firing its binding elements and exploring its occurrence
graph are not specific to the models this package generates, so they now live
in `CPNCheck <https://github.com/GrupoCybersegurancaVirtus/cpncheck>`_ and this
module only points at them.

The CPN ML helpers this module used to export -- ``sml_expr_to_python``,
``compile_guard``, ``compile_output``, ``parse_pattern``,
``parse_initial_marking`` -- are gone. CPNCheck parses CPN ML properly rather
than translating it with regular expressions; the equivalents are
:mod:`cpncheck.cpnml` and the ``compile_*`` methods of
:class:`~cpncheck.coloured.ColouredNet`.
"""

from cpncheck.coloured import ColouredNet, UnsupportedNet  # noqa: F401
from cpncheck.statespace import (  # noqa: F401
    DEFAULT_MAX_NODES,
    StateSpace,
    build_state_space,
)

__all__ = [
    "ColouredNet",
    "StateSpace",
    "UnsupportedNet",
    "build_state_space",
    "DEFAULT_MAX_NODES",
]
