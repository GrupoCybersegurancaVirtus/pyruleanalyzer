"""Backwards-compatible re-export: model checking moved to CPNCheck.

The checker, its CTL engine, its report and its ASK-CTL generator are not
specific to the models this package generates, so they now live in `CPNCheck
<https://github.com/GrupoCybersegurancaVirtus/cpncheck>`_. What *is* specific
-- which place holds the prediction, which pages are trees, the twenty
properties of the catalog -- is CPNCheck's ``ml_trees`` profile, which it
selects automatically for every net this package exports.

Nothing about the public API changed: ``check_cpn``, ``CPNModelChecker``,
``PROPERTY_CATALOG`` and the rest mean what they always did.

Two things are new, and are documented in CPNCheck:

* a violated property carries a :class:`~cpncheck.counterexample.Counterexample`
  on ``result.properties[pid].cex`` -- the occurrence sequence, the binding of
  every step and the input it needs -- which ``replay()`` re-fires to prove it
  is real;
* properties are formulas over selectors, so a caller can state their own.
"""

from cpncheck import (  # noqa: F401
    METHODS,
    CPNModelChecker,
    ModelCheckResult,
    Property,
    PropertyResult,
    check_cpn,
    compare_results,
)
from cpncheck.counterexample import Counterexample, Step  # noqa: F401
from cpncheck.profiles.ml_trees import (  # noqa: F401
    TREE_CATALOG,
    MLTreeProfile,
    TreeStructure,
)
from cpncheck.statespace import DEFAULT_MAX_NODES  # noqa: F401

#: The properties of the generated nets, in report order.
PROPERTY_CATALOG = TREE_CATALOG

__all__ = [
    "METHODS",
    "PROPERTY_CATALOG",
    "CPNModelChecker",
    "Counterexample",
    "ModelCheckResult",
    "Property",
    "PropertyResult",
    "Step",
    "check_cpn",
    "compare_results",
]
