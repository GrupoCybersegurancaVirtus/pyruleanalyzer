"""Model checking of the generated nets, through CPNCheck.

The checker, its CTL engine, its report and its ASK-CTL generator are generic
and live in `CPNCheck <https://github.com/GrupoCybersegurancaVirtus/cpncheck>`_.
What is specific to these nets -- which place holds the prediction, which
pages are trees, the twenty properties of the catalog, the conformance test
against the classifier -- is the profile in :mod:`pyruleanalyzer.cpn_profile`,
which pyruleanalyzer publishes to CPNCheck as a plugin.

CPNCheck's own API knows nothing of classifiers: a profile receives its
settings as ``options``. This module keeps the signature pyruleanalyzer has
always used -- ``class_labels``, ``feature_names``, ``classifier``,
``use_final`` -- and translates it, so no caller has to change.

A violated property carries a
:class:`~cpncheck.counterexample.Counterexample` on
``result.properties[pid].cex``, which ``replay()`` re-fires to prove it real.
"""

from typing import Any, Dict, Optional, Sequence

from cpncheck import (  # noqa: F401
    METHODS,
    ModelCheckResult,
    Property,
    PropertyResult,
    compare_results,
)
from cpncheck import CPNModelChecker as _GenericChecker
from cpncheck.counterexample import Counterexample, Step  # noqa: F401
from cpncheck.statespace import DEFAULT_MAX_NODES  # noqa: F401

from .cpn_profile import (  # noqa: F401  (importing registers the profile)
    PROFILE,
    TREE_CATALOG,
    MLTreeProfile,
    TreeStructure,
)

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


# Function to put the classifier-shaped arguments into profile options.
def _options(feature_names=None, class_labels=None,
             options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The tree profile's settings, from pyruleanalyzer's arguments.

    Args:
        feature_names (list, optional): Ordered feature names.
        class_labels (list, optional): The label domain.
        options (dict, optional): Further settings, which win.

    Returns:
        dict: The options for CPNCheck.
    """
    out: Dict[str, Any] = {}
    if feature_names:
        out["feature_names"] = list(feature_names)
    if class_labels is not None:
        out["class_labels"] = list(class_labels)
    out.update(options or {})
    return out


# Class checking a generated net with pyruleanalyzer's argument names.
class CPNModelChecker(_GenericChecker):
    """CPNCheck's checker, taking the arguments pyruleanalyzer always took.

    Example:
        >>> mc = CPNModelChecker("model.cpn", class_labels=[0, 1])
        >>> mc.check().passed
        True
    """

    # Method to load a generated net.
    def __init__(self, path: str, feature_names: Optional[Sequence[str]] = None,
                 class_labels: Optional[Sequence[int]] = None,
                 max_nodes: int = DEFAULT_MAX_NODES, profile=None,
                 prelude: Optional[Dict[str, Any]] = None, properties=None,
                 options: Optional[Dict[str, Any]] = None):
        """Parse the net and choose its profile.

        Args:
            path (str): The ``.cpn`` file.
            feature_names (list, optional): Ordered feature names.
            class_labels (list, optional): Label domain for A6.
            max_nodes (int): Occurrence-graph budget.
            profile (Profile, optional): The profile; detected when omitted.
            prelude (dict, optional): Names added to the net's declarations.
            properties (optional): User properties (``cpncheck.query``).
            options (dict, optional): Further profile settings.
        """
        super().__init__(path, max_nodes=max_nodes, profile=profile,
                         prelude=prelude, properties=properties,
                         options=_options(feature_names, class_labels, options))

    # Property kept for callers that read the feature names back.
    @property
    def feature_names(self):
        """The feature names given, if any.

        Returns:
            list|None: The names.
        """
        return self.options.get("feature_names")

    # Property giving the located trees, prediction place and transitions.
    @property
    def structure(self) -> TreeStructure:
        """Where the prediction, the input and every tree are in the net.

        Returns:
            TreeStructure: The structure the tree profile reads.
        """
        return self.profile.structure(self.cnet)

    # Property giving the number of fields of the net's own input record.
    @property
    def width(self) -> int:
        """Fields of the input token in the net's initial marking.

        Returns:
            int: The width, 0 when there is no input token.
        """
        own = self.own_sample
        return len(own) if isinstance(own, tuple) else 0

    # Property giving the label domain used for A6.
    @property
    def labels(self):
        """The label domain, declared or read off the net.

        Returns:
            list|None: The labels, or None when there is none to read.
        """
        return self.profile.label_domain(
            self.cnet, self.options.get("class_labels"))[0]

    # Method to verify the net, with the classifier as the reference.
    def check(self, samples=None, classifier=None, use_final: bool = True,
              test_samples=None, max_nodes: Optional[int] = None,
              verbose: bool = False, reduction: Optional[str] = None,
              reference=None, options: Optional[Dict[str, Any]] = None,
              **reduction_options) -> ModelCheckResult:
        """Verify every applicable property.

        Args:
            samples: Inputs to model check.
            classifier: The classifier, for prediction consistency (PC).
            use_final (bool): Whether the net carries the refined rules.
            test_samples: Inputs for the conformance test.
            max_nodes (int, optional): Occurrence-graph budget.
            verbose (bool): Print progress.
            reduction (str, optional): ``"stubborn"``, ``"sweep"``,
                ``"equivalence"`` or ``"symmetry"`` (see CPNCheck).
            reference: Same as ``classifier``, under CPNCheck's name.
            options (dict, optional): Further profile settings.
            **reduction_options: ``progress``, ``equivalence`` or
                ``symmetry``, passed to CPNCheck.

        Returns:
            ModelCheckResult: The verdicts.
        """
        run_options = {"use_final": use_final}
        run_options.update(options or {})
        return super().check(samples=samples,
                             reference=classifier if classifier is not None
                             else reference,
                             test_samples=test_samples, max_nodes=max_nodes,
                             verbose=verbose, reduction=reduction,
                             options=run_options, **reduction_options)


# Function to verify a generated net in one call.
def check_cpn(path: str, samples=None, classifier=None,
              feature_names: Optional[Sequence[str]] = None,
              class_labels: Optional[Sequence[int]] = None,
              use_final: bool = True, test_samples=None,
              max_nodes: int = DEFAULT_MAX_NODES, verbose: bool = False,
              profile=None, prelude: Optional[Dict[str, Any]] = None,
              reduction: Optional[str] = None, properties=None,
              options: Optional[Dict[str, Any]] = None) -> ModelCheckResult:
    """Verify a generated ``.cpn`` model.

    Args:
        path (str): The ``.cpn`` file.
        samples: Inputs to model check (defaults to the file's own).
        classifier: Classifier for the conformance test.
        feature_names (list, optional): Ordered feature names.
        class_labels (list, optional): Label domain for A6.
        use_final (bool): Whether the net carries the refined rules.
        test_samples: Inputs for conformance testing.
        max_nodes (int): Occurrence-graph budget.
        verbose (bool): Print progress.
        profile (Profile, optional): The profile to use.
        prelude (dict, optional): Names added to the net's ML declarations.
        reduction (str, optional): ``"stubborn"`` for partial-order reduction.
        properties (optional): User properties.
        options (dict, optional): Further profile settings.

    Returns:
        ModelCheckResult: Verdicts and statistics.
    """
    checker = CPNModelChecker(path, feature_names=feature_names,
                              class_labels=class_labels, max_nodes=max_nodes,
                              profile=profile, prelude=prelude,
                              properties=properties, options=options)
    return checker.check(samples=samples, classifier=classifier,
                         use_final=use_final, test_samples=test_samples,
                         verbose=verbose, reduction=reduction)
