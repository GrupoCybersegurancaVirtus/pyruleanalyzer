"""Backwards-compatible re-export: the mutants moved to CPNCheck.

A checker that answers "true" on correct nets has shown nothing until it is
also seen answering "false" on incorrect ones. The machinery for deriving such
nets is generic and lives in :mod:`cpncheck.mutation`; the sixteen operators
written against *these* nets are
:mod:`cpncheck.profiles.ml_trees_mutants`.

``Mutant.families`` is now ``Mutant.applies``, a predicate over the analysis
rather than a tuple of family names.
"""

from cpncheck.mutation import CPNDocument, Mutant, make_mutants  # noqa: F401
from cpncheck.profiles.ml_trees_mutants import TREE_MUTANTS, analyse  # noqa: F401

#: Every mutation operator of the generated nets, in report order.
MUTANTS = TREE_MUTANTS

__all__ = ["CPNDocument", "Mutant", "MUTANTS", "analyse", "make_mutants"]
