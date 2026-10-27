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

Besides CPNCheck's reductions, ``reduction="compositional"`` decides the
catalog of an ensemble net from one small graph per tree (or boosting channel)
and the join layer, instead of the exponential interleaving of all of them
(:mod:`pyruleanalyzer.compositional`, where the soundness argument of every
property is written down). What the composition cannot decide on a given net
is handed to ``fallback`` (``"stubborn"`` by default).
"""

import collections
import time
from typing import Any, Dict, List, Optional, Sequence

from cpncheck import (  # noqa: F401
    METHODS,
    ModelCheckResult,
    Property,
    PropertyResult,
    compare_results,
)
from cpncheck import CPNModelChecker as _GenericChecker
from cpncheck.counterexample import Counterexample, Step  # noqa: F401
from cpncheck.formula import Context
from cpncheck.reduction import Dependency, ReducedChecker
from cpncheck.statespace import DEFAULT_MAX_NODES  # noqa: F401
from cpncheck.statespace import build_state_space

from .compositional import (DEFAULT_MAX_COMBINATIONS, TAG,
                            CompositionalChecker)
from .cpn_profile import (  # noqa: F401  (importing registers the profile)
    PROFILE,
    TREE_CATALOG,
    MLTreeProfile,
    TreeStructure,
)

#: The value of ``reduction`` that selects compositional verification.
COMPOSITIONAL = TAG

#: Accepted values of ``fallback`` for ``reduction="compositional"``.
FALLBACKS = ("stubborn", "full", None)

#: The properties of the generated nets, in report order.
PROPERTY_CATALOG = TREE_CATALOG

__all__ = [
    "COMPOSITIONAL",
    "METHODS",
    "PROPERTY_CATALOG",
    "CPNModelChecker",
    "CompositionalResult",
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


# Class holding the verdicts of a compositional run.
class CompositionalResult(ModelCheckResult):
    """A :class:`ModelCheckResult` whose report names the composition.

    Everything else -- the verdicts, ``passed``, ``failures``, ``skipped``,
    the serialisation -- is the base class's, so the pipeline's gate reads it
    like any other result.
    """

    # Method to render the result as a plain-text report.
    def report(self, show_formula: bool = True) -> str:
        """The base report, with the graph line describing the composition.

        Args:
            show_formula (bool): Include the formula column.

        Returns:
            str: The report.
        """
        st = self.stats
        sizes = st.get("local_nodes") or [0]
        line = (f"  reduction        : compositional; {st.get('components', 0)} "
                f"component graph(s), {sum(sizes)} markings in all, largest "
                f"{max(sizes)}; join layer over {st.get('combinations', 0)} "
                f"output combination(s)"
                f"{'' if st.get('complete', True) else ' (SOME INCOMPLETE)'}")
        fallback = st.get("fallback")
        lines = []
        for row in super().report(show_formula).split("\n"):
            if row.startswith("  reduction        :"):
                lines.append(line)
                if fallback:
                    lines.append(f"  fallback         : {fallback}")
                continue
            if row.startswith("  deadlock graph   :"):
                continue
            lines.append(row)
        return "\n".join(lines)


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
                ``"equivalence"`` or ``"symmetry"`` (see CPNCheck), or
                ``"compositional"``: one graph per tree or boosting channel
                and the join layer instead of their interleaving
                (:mod:`pyruleanalyzer.compositional`).
            reference: Same as ``classifier``, under CPNCheck's name.
            options (dict, optional): Further profile settings.
            **reduction_options: ``progress``, ``equivalence`` or
                ``symmetry``, passed to CPNCheck; for ``"compositional"``,
                ``fallback`` (``"stubborn"``, the default, ``"full"`` or None:
                where the properties the composition cannot decide go) and
                ``max_combinations`` (output combinations of the join layer
                enumerated at most).

        Returns:
            ModelCheckResult: The verdicts.
        """
        run_options = {"use_final": use_final}
        run_options.update(options or {})
        if reduction == COMPOSITIONAL:
            return self._check_compositional(
                samples=samples,
                reference=classifier if classifier is not None else reference,
                test_samples=test_samples, max_nodes=max_nodes,
                verbose=verbose, options=run_options, **reduction_options)
        return super().check(samples=samples,
                             reference=classifier if classifier is not None
                             else reference,
                             test_samples=test_samples, max_nodes=max_nodes,
                             verbose=verbose, reduction=reduction,
                             options=run_options, **reduction_options)

    # Method to verify the net compositionally.
    def _check_compositional(self, samples=None, reference=None,
                             test_samples=None, max_nodes: Optional[int] = None,
                             verbose: bool = False,
                             options: Optional[Dict[str, Any]] = None,
                             fallback: Optional[str] = "stubborn",
                             max_combinations: int = DEFAULT_MAX_COMBINATIONS
                             ) -> ModelCheckResult:
        """:meth:`check` with ``reduction="compositional"``.

        Per input, every property the composition can decide is decided on
        the component graphs and the join layer
        (:class:`~pyruleanalyzer.compositional.CompositionalChecker`); the
        rest goes to ``fallback`` and says so in its ``detail``. The
        structural and conformance checks are run as in every mode.

        Args:
            samples: Inputs to model check.
            reference: The classifier, for PC.
            test_samples: Inputs for the conformance test.
            max_nodes (int, optional): Budget of each graph.
            verbose (bool): Print progress.
            options (dict, optional): Profile settings for this run.
            fallback (str, optional): ``"stubborn"``, ``"full"`` or None.
            max_combinations (int): Bound on enumerated output combinations.

        Returns:
            CompositionalResult: The verdicts.

        Raises:
            ValueError: For an unknown ``fallback``.
        """
        if fallback not in FALLBACKS:
            raise ValueError(f"fallback must be one of {FALLBACKS}, not "
                             f"{fallback!r}")
        t0 = time.time()
        if options:
            self.options.update(options)
        if max_nodes is not None:
            self.max_nodes = int(max_nodes)
        if not isinstance(self.profile, MLTreeProfile):
            raise ValueError("reduction='compositional' needs the tree profile")
        comp = getattr(self, "_compositional", None)
        if comp is None or comp.max_combinations != int(max_combinations):
            comp = self._compositional = CompositionalChecker(
                self, max_combinations=max_combinations)
        xs = self._sample_list(samples)
        merged: "collections.OrderedDict[str, PropertyResult]" = \
            collections.OrderedDict()
        stats: Dict[str, Any] = {}
        warnings: List[str] = []
        for i, x in enumerate(xs):
            if verbose:
                print(f"  [mc] input {i + 1}/{len(xs)} (compositional)")
            part, undecided, cstats = comp.evaluate(x)
            if not cstats.get("complete", True):
                warnings.append(f"a component graph exceeded {self.max_nodes} "
                                "markings")
            fstats: Dict[str, Any] = {}
            if undecided:
                fpart, fstats = self._fallback(x, undecided, fallback)
                part.update(fpart)
                for pid, reason in undecided.items():
                    cstats["plans"][pid] = (f"{fallback} graph ({reason})"
                                            if fallback else reason)
                if fallback and not fstats.get("complete", True):
                    warnings.append(f"the {fallback} fallback exceeded "
                                    f"{self.max_nodes} markings: the "
                                    "properties it decides have no verdict")
            for pid, reason in self.unavailable.items():
                part[pid] = PropertyResult(pid, None, reason,
                                           spec=self.catalog.get(pid))
            if not stats:
                sizes = cstats["local_nodes"] or [0]
                stats = dict(cstats)
                stats.update({
                    "nodes": sum(sizes), "nodes_total": sum(sizes),
                    "nodes_max": max(sizes), "graphs": len(sizes),
                    "arcs": sum(lc.ss.n_arcs for lc in getattr(comp, "locals", ())),
                    "dead": None, "per_kind": {"component": len(sizes)},
                    "undecided": sorted(undecided),
                    "fallback": (f"{fallback} graph for {', '.join(undecided)}"
                                 if undecided and fallback else None),
                    "fallback_stats": fstats,
                })
                stats["complete"] = cstats.get("complete", True) and \
                    fstats.get("complete", True)
            self._conjoin(merged, part)

        merged.update(self.profile.static_checks(self))
        tested = 0
        if reference is not None:
            pool = xs if test_samples is None else self._sample_list(test_samples)
            txs = [x if x is not None else self.own_sample for x in pool]
            verdict = self.profile.conformance(self, reference, txs, self.options)
            if verdict is not None:
                merged[verdict.id] = verdict
                tested = len(txs)
        ordered = collections.OrderedDict(
            (pid, merged[pid]) for pid in self.catalog if pid in merged)
        for pid, verdict in merged.items():
            ordered.setdefault(pid, verdict)
        return CompositionalResult(self.path, self.family, ordered, stats,
                                   samples=len(xs), tested=tested,
                                   warnings=warnings,
                                   elapsed=time.time() - t0, oracle={},
                                   report=None)

    # Method to conjoin the verdicts of several inputs.
    @staticmethod
    def _conjoin(target: Dict[str, PropertyResult],
                 part: Dict[str, PropertyResult]):
        """Conjoin per-input verdicts: False wins, then None, then True.

        Unlike the generic merge, a property left undecided for one input is
        not reported as passed because another input decided it: "holds for
        every input checked" needs every input.

        Args:
            target (dict): Accumulated verdicts, modified in place.
            part (dict): The verdicts of one more input.
        """
        for pid, r in part.items():
            old = target.get(pid)
            if old is None:
                target[pid] = PropertyResult(pid, r.holds, r.detail, r.elapsed,
                                             spec=r._spec, cex=r.cex)
                continue
            old.elapsed += r.elapsed
            if old.holds is False:
                continue
            if r.holds is False or (r.holds is None and old.holds is True):
                old.holds, old.detail, old.cex = r.holds, r.detail, r.cex

    # Method to decide the properties the composition left over.
    def _fallback(self, sample, undecided: Dict[str, str],
                  fallback: Optional[str]):
        """Decide ``undecided`` on the stubborn-reduced or the full graph.

        Args:
            sample (optional): The input.
            undecided (dict): ``{pid: why the composition did not decide it}``.
            fallback (str, optional): ``"stubborn"``, ``"full"`` or None.

        Returns:
            tuple: ``({pid: PropertyResult}, statistics)``.
        """
        out: Dict[str, PropertyResult] = {}
        if fallback is None:
            for pid, reason in undecided.items():
                out[pid] = PropertyResult(pid, None, reason,
                                          spec=self.catalog.get(pid))
            return out, {"complete": True}
        initial = None
        if sample is not None:
            initial = self.cnet.initial_marking(self.as_sample(sample),
                                                places=self.input_places)

        def note(pid, holds, detail):
            if holds is True:
                return (f"decided on the {fallback} graph "
                        f"({undecided[pid]})")
            if holds is None:
                return f"{detail or 'over budget'} ({undecided[pid]})"
            return detail

        if fallback == "stubborn":
            if self._dependency is None:
                self._dependency = Dependency(self.cnet)
            rc = ReducedChecker(self.cnet, self._dependency, self.groups,
                                initial=initial, max_nodes=self.max_nodes)
            plans: Dict[str, str] = {}
            for pid in undecided:
                t0 = time.perf_counter()
                if pid in self.formulas:
                    holds, cex, plan = rc.decide(self.formulas[pid])
                    if cex is not None:
                        cex.input = sample
                        cex.data["input_places"] = self.input_places
                    detail = "" if holds else (
                        str(cex) if cex is not None else
                        "; ".join(plan.reasons[:2]) or "reduced graph over budget")
                    r = PropertyResult(pid, holds, note(pid, holds, detail),
                                       time.perf_counter() - t0,
                                       spec=self.catalog.get(pid), cex=cex)
                elif self.scc_props.get(pid) == "home":
                    r = self._home_reduced(rc, pid, sample, plans)
                    r.detail = note(pid, r.holds, r.detail)
                    r.elapsed = time.perf_counter() - t0
                else:
                    r = PropertyResult(pid, None, undecided[pid],
                                       spec=self.catalog.get(pid))
                out[pid] = r
            return out, rc.stats()

        ss = build_state_space(self.cnet, self.max_nodes, initial=initial)
        if not ss.complete:
            for pid, reason in undecided.items():
                out[pid] = PropertyResult(pid, None,
                                          f"occurrence graph over budget ({reason})",
                                          spec=self.catalog.get(pid))
            return out, {"complete": False, "nodes": ss.n_nodes}
        ctx = Context(ss, self.cnet, self.groups, semantics="textbook")
        for pid in undecided:
            t0 = time.perf_counter()
            cex = None
            if pid in self.formulas:
                holds = 0 in ctx.sat(self.formulas[pid])
                if not holds:
                    cex = self.formulas[pid].counterexample(ctx)
                    cex.input = sample
                    cex.data["input_places"] = self.input_places
                detail = "" if holds else str(cex)
            elif self.scc_props.get(pid) == "home":
                holds = bool(ss.home)
                if holds:
                    detail = ""
                else:
                    cex = self._no_home_marking(ss, sample)
                    detail = str(cex)
            else:
                holds, detail = None, undecided[pid]
            out[pid] = PropertyResult(pid, holds, note(pid, holds, detail),
                                      time.perf_counter() - t0,
                                      spec=self.catalog.get(pid), cex=cex)
        return out, {"complete": True, "nodes": ss.n_nodes}


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
        reduction (str, optional): ``"stubborn"`` for partial-order
            reduction, ``"compositional"`` for per-component verification.
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
