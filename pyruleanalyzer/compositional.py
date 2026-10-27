"""Compositional verification of the ensemble nets.

The occurrence graph of an ensemble net is exponential in the number of base
learners, because the learners fire independently and every interleaving of
their occurrences is a marking of its own: a random forest of ``n`` trees has
``2**n + 2`` markings, a ``K``-class boosting model with ``M`` stages per class
``(3M + 3)**K + 2``. Nothing in the catalog is *about* those interleavings,
though. This module decides the catalog without building the product: one small
graph per learner, plus the join layer evaluated once per combination of the
learners' outputs. A forest of 50 trees costs 50 graphs of 3 markings each.

Reach for it with ``CPNModelChecker.check(reduction="compositional")``
(:mod:`pyruleanalyzer.model_checker`), ``PyRuleAnalyzer.model_check(...,
reduction="compositional")`` or ``verified_pipeline(...,
reduction="compositional")``. The verdicts are the full graph's; every one says
in its ``detail`` (and in the report's *decided on* line) that it came from the
composition, and a property the composition cannot decide on a given net is
handed to the stubborn-set reduced graph (or the full graph) instead -- never
reported as passed without the argument below.

The shape that is checked, not assumed
--------------------------------------
Every condition below is *verified on the net* before a verdict is given; a
net that does not have this shape -- a mutant, a hand-edited file, a decision
tree -- falls back, entirely or property by property.

Level 1, independence (static, :class:`Decomposition`):

* the net is untimed, has no priorities and no inhibitor or reset arcs;
* one place ``I`` holds the input. Exactly one transition ``F``
  (``Distribute``) consumes from it, ``F`` is on the top page, consumes from
  ``I`` only, and nothing produces into ``I``;
* the *components* are the transitions off the top page, grouped by the places
  they share (union-find): a tree page for a forest, a class channel and its
  stage trees for boosting. ``P_c``, the places a component touches, are
  therefore disjoint between components;
* ``F`` produces only into component places (its *input* places ``In_c``);
* a component's *output* places ``Out_c`` are its top-page places that it does
  not consume from and that ``F`` does not feed;
* every other top-page transition (a *join*: ``Vote``, ``Decide_k``) consumes
  only from output places, through ordinary input patterns, at least one
  token, and produces only into places no component touches (the *result*
  places ``R``, among them ``Prediction``).

Per input (dynamic): the initial marking holds exactly one token, in ``I``, and
exactly one binding element is enabled in it, an occurrence of ``F``.

The local graph ``G_c`` of a component is the occurrence graph of the very same
net, from the same initial marking, in which only ``F`` and the component's
transitions may occur. It is a sub-graph of the full graph ``G``: every path of
``G_c`` is an occurrence sequence of the net (**lifting**), so a counterexample
found on it replays on the full net. Conversely (**projection**), dropping from
any occurrence sequence of ``G`` the occurrences outside ``F`` and component
``c`` leaves an occurrence sequence of ``G_c`` that reaches a marking agreeing
with the original on every place of ``P_c``, except that output places may hold
*more* tokens in ``G_c`` (joins remove tokens from them; nothing else outside
``c`` touches ``P_c``, and the component never reads its output places, so its
own enabling is unaffected).

Level 2, the combination (per input): every component is *well behaved* --
``G_c`` is complete and acyclic, ``c`` has exactly one output place and never
holds more than one token in it -- and every join is *synchronising*: it takes
exactly one token from every output place of every component. Then each
component deposits at most one value in its output, at most once, and no join
can occur before all of them have; the joins see one tuple
``tau in V_1 x ... x V_n`` of output values (``V_c``: the values in ``Out_c``
over ``G_c``) and at most one join occurs per run. Every such tuple is
reachable, because the components can be run to any of their markings one
after the other. The join layer is then evaluated on each tuple (``E(tau)``:
its enabled join bindings, ``post(tau, b)``: what each puts into ``R``). A
component is *silent* when ``G_c`` has a dead marking with an empty output.

Per property (``=>`` lifting, ``<=`` projection):

``A1`` termination
    An infinite occurrence sequence of ``G`` makes infinitely many occurrences
    of some component (``F`` occurs once, and every join consumes a token some
    component produced), and its projection is an infinite path of that
    ``G_c``; a cycle of ``G_c`` lifts. So ``A1`` holds iff every ``G_c`` is
    acyclic. Level 1.
``A7`` safeness
    The initial marking is safe (one token); a place of ``P_c`` exceeds one
    token in ``G`` iff it does in ``G_c`` (projection only adds output tokens,
    and the bound is anti-monotone); a result place is filled only by the one
    join occurrence, ``post(tau, b)``. Local part Level 1, result part Level 2.
``B1``/``B3`` leaf determinism, single tree output
    ``AG`` of a proposition over one tree's leaves and places, all in one
    component: the enabling of a leaf depends only on non-output places of
    ``P_c``, and a token bound is anti-monotone, so the proposition holds at a
    marking of ``G`` iff it holds at its projection. Level 1.
``B2`` inevitable leaf selection
    ``False`` on ``G_c`` lifts (a run of ``G_c`` that stops is extended by the
    other components and joins, which cannot enable the tree's leaves). ``True``
    on ``G_c`` transfers when every *other* component is acyclic: then every
    maximal run of ``G`` projects onto a maximal run of ``G_c``.
``C1``/``C2`` stage precedence, no premature score
    ``not E[a' U b]`` with ``a`` an accumulator (not an output place) and ``b``
    an accumulator or the channel score: a witness of ``G_c`` lifts; the prefix
    of a witness of ``G`` up to the first ``b`` projects onto one of ``G_c``
    (``Marked`` is monotone and the projection only adds output tokens). Level 1.
``D1`` complete vote
    Structural: when every ``Vote`` occurrence consumes a token from every tree
    output place, ``EN(Vote)`` implies they are all marked, in any net. Otherwise
    it falls back.
``A2``/``A3``/``A4`` no spurious deadlock, inevitable decision, recoverability
    Level 2 makes ``G`` finite and acyclic and ``Prediction`` is never consumed,
    so all three are equivalent to "every reachable dead marking holds a
    prediction". The dead markings are: some component silent (no join can
    occur; no prediction); all productive with ``E(tau)`` empty; or after the
    one join occurrence, ``post(tau, b)``. Hence: no silent component and, for
    every ``tau``, ``E(tau)`` non-empty and every ``post(tau, b)`` marks
    ``Prediction``.
``A5``/``D2b``/``A6`` unique output, valid label
    ``Prediction`` changes only by the single join occurrence; checked on every
    ``post(tau, b)``. Level 2.
``A8`` home marking
    ``G`` is finite and acyclic, so a home marking exists iff there is exactly
    one dead marking. The dead markings are enumerated as above from the
    product of the components' distinct dead markings (bounded by
    ``max_combinations``). Level 2.
``C3a``/``D2a`` inevitable decision firing, inevitable aggregation
    On every maximal run, the joins become able to occur exactly when the last
    output is deposited, with some tuple ``tau``, and the outputs stay put until
    a join occurs. So ``AF EN(V)`` holds iff no component is silent and
    ``E_V(tau)`` is non-empty for every ``tau``. Level 2.
``C3b`` decision determinism
    A ``Decide`` is enabled only when every output is marked, and its bindings
    depend on the outputs alone: ``|E_Decide(tau)| <= 1`` for every ``tau``.
    Level 2.

``B4`` (structural) and ``PC`` (conformance) never needed an occurrence graph
and are run as usual. Anything else -- a user property, an ASK-CTL query, a
property whose level does not hold on this net -- falls back, with the reason.
"""

import collections
import itertools
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from cpncheck.counterexample import Counterexample, Step
from cpncheck.formula import AF, AG, Context, Dead, ForEach, MaxTokens
from cpncheck.result import PropertyResult
from cpncheck.selectors import Explicit
from cpncheck.statespace import StateSpace, build_state_space

__all__ = ["Decomposition", "CompositionalChecker", "COMPOSABLE",
           "DEFAULT_MAX_COMBINATIONS"]

#: The catalog properties the composition knows how to decide.
COMPOSABLE = ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "B1", "B2",
              "B3", "C1", "C2", "C3a", "C3b", "D1", "D2a", "D2b")

#: Most output combinations (or dead-marking combinations) evaluated before
#: the combination properties are handed to the fallback.
DEFAULT_MAX_COMBINATIONS = 4096

#: Marks every verdict the composition gave.
TAG = "compositional"


# Function to list the places a transition reads from.
def _consumes(t) -> set:
    """Places a transition takes tokens from.

    Args:
        t (dict): A transition record.

    Returns:
        set: Place indices.
    """
    return {p for p, _ in t["in"]} | {p for p, _ in t["in_expr"]}


# Function to list every place a transition is connected to.
def _touches(t) -> set:
    """Places a transition reads, writes, inhibits or resets.

    Args:
        t (dict): A transition record.

    Returns:
        set: Place indices.
    """
    return (_consumes(t) | {p for p, _ in t["out"]} | set(t.get("inhibit", ()))
            | set(t.get("reset", ())))


# Class presenting the net with only some transitions allowed to occur.
class _Restricted:
    """The coloured net, with every transition but a chosen few frozen.

    :func:`~cpncheck.statespace.build_state_space` explores it like any net;
    its graph is the sub-graph of the full occurrence graph in which only the
    allowed transitions occur. Everything but :meth:`enabled` is the net's own.
    """

    # Method to build the restricted view.
    def __init__(self, net, allowed):
        """Create the view.

        Args:
            net (ColouredNet): The net.
            allowed (iterable): Indices of the transitions that may occur.
        """
        self._net = net
        self._allowed = frozenset(allowed)
        trans = net.transitions
        self._sources = sorted(ti for ti in self._allowed
                               if not trans[ti]["in"] and not trans[ti]["in_expr"])
        watch: Dict[int, List[int]] = collections.defaultdict(list)
        for ti in sorted(self._allowed):
            for p in _consumes(trans[ti]):
                watch[p].append(ti)
        self._watch = dict(watch)
        # The net's own binding enumeration, restricted, when it is available
        # and the net is one whose enabling needs nothing else (untimed, no
        # priorities: the Decomposition requires both). Otherwise filter.
        self._direct = hasattr(net, "_bindings") and not net.timed and \
            not net.uses_priorities

    # Method to list the enabled binding elements of the allowed transitions.
    def enabled(self, marking):
        """Binding elements of the allowed transitions enabled in a marking.

        Args:
            marking (tuple): A marking.

        Returns:
            list: ``(transition index, binding key, binding)`` triples, as
            :meth:`ColouredNet.enabled` returns them.
        """
        net = self._net
        if not self._direct:
            return [e for e in net.enabled(marking) if e[0] in self._allowed]
        candidates = set(self._sources)
        for p, tis in self._watch.items():
            if marking[p]:
                candidates.update(tis)
        found = []
        for ti in sorted(candidates):
            t = net.transitions[ti]
            if any(len(marking[p]) < n for p, n in t["needs"].items()):
                continue
            for env, _ready in net._bindings(marking, t):
                found.append((ti, tuple(sorted(env.items())), env))
        return found

    # Method to delegate everything else to the net.
    def __getattr__(self, name):
        """The net's own attribute.

        Args:
            name (str): Attribute name.

        Returns:
            The attribute.
        """
        return getattr(self._net, name)


# Class holding one independent part of the net.
class Component:
    """One base learner (or boosting channel) and the places it owns.

    Attributes:
        index (int): Position among the components.
        label (str): A page name, for reports.
        transitions (frozenset): Its transitions.
        places (frozenset): ``P_c``, every place its transitions touch.
        inputs (frozenset): ``In_c``, the places ``F`` feeds.
        outputs (tuple): ``Out_c``, the output places, sorted.
    """

    # Method to build the component.
    def __init__(self, index, label, transitions, places, inputs, outputs):
        """Create the component.

        Args:
            index (int): Position.
            label (str): Name.
            transitions (iterable): Transition indices.
            places (iterable): Place indices.
            inputs (iterable): Input places.
            outputs (iterable): Output places.
        """
        self.index = index
        self.label = label
        self.transitions = frozenset(transitions)
        self.places = frozenset(places)
        self.inputs = frozenset(inputs)
        self.outputs = tuple(sorted(outputs))

    # Method to render the component for debugging.
    def __repr__(self):
        """Return a short representation.

        Returns:
            str: ``<Component Tree_0: 4 transitions, 2 places>``.
        """
        return (f"<Component {self.label}: {len(self.transitions)} transitions, "
                f"{len(self.places)} places>")


# Class establishing (or refusing) the fork / components / join shape.
class Decomposition:
    """The Level 1 conditions of the module docstring, checked on a net.

    Attributes:
        ok (bool): Whether the net has the shape.
        reason (str): Why not, when it has not.
        input (int): The input place ``I``.
        fork (int): The transition ``F``.
        components (list): The :class:`Component` objects.
        joins (tuple): The join transitions.
        results (frozenset): The result places ``R``.
        sync (bool): Whether every join is synchronising and every component
            has exactly one output place (half of Level 2).
        sync_reason (str): Why not.
    """

    # Method to analyse a net.
    def __init__(self, cnet, input_places: Sequence[int]):
        """Locate the fork, the components and the joins, and check the shape.

        Args:
            cnet (ColouredNet): The flattened net.
            input_places (sequence): The places the profile writes the input to.
        """
        self.cnet = cnet
        self.ok = False
        self.reason = ""
        self.sync = False
        self.sync_reason = ""
        self.components: List[Component] = []
        self.joins: Tuple[int, ...] = ()
        self.results = frozenset()
        self.input = self.fork = None
        self.comp_of_transition: Dict[int, int] = {}
        self.comp_of_place: Dict[int, int] = {}
        try:
            self._analyse(list(input_places))
        except _Refused as exc:
            self.reason = str(exc)

    # Method doing the analysis.
    def _analyse(self, input_places):
        """Check every Level 1 condition, raising on the first that fails.

        Args:
            input_places (list): The profile's input places.

        Raises:
            _Refused: With the condition that does not hold.
        """
        net = self.cnet
        trans, places = net.transitions, net.places
        top = net.top["name"]
        if net.timed or net.uses_priorities:
            raise _Refused("the net is timed or uses priorities")
        if getattr(net, "special_arcs", False):
            raise _Refused("the net has inhibitor or reset arcs")
        if len(input_places) != 1:
            raise _Refused("the net does not have exactly one input place")
        inp = input_places[0]
        readers = [ti for ti, t in enumerate(trans) if inp in _consumes(t)]
        if len(readers) != 1:
            raise _Refused(f"{len(readers)} transitions read the input place; "
                           "a single fork is needed")
        fork = readers[0]
        ft = trans[fork]
        if ft["page"] != top or _consumes(ft) != {inp} or ft["in_expr"]:
            raise _Refused(f"{ft['name']} is not a fork reading only the input")
        if any(inp in {p for p, _ in t["out"]} for t in trans):
            raise _Refused("a transition puts tokens back into the input place")

        inner = [ti for ti, t in enumerate(trans) if t["page"] != top]
        if not inner:
            raise _Refused("the net has no subpages: there is nothing to compose")
        parent = {ti: ti for ti in inner}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        owner: Dict[int, int] = {}
        for ti in inner:
            for p in _touches(trans[ti]):
                if p in owner:
                    a, b = find(owner[p]), find(ti)
                    if a != b:
                        parent[a] = b
                else:
                    owner[p] = ti
        groups: "collections.OrderedDict[int, List[int]]" = collections.OrderedDict()
        for ti in inner:
            groups.setdefault(find(ti), []).append(ti)

        fed = {p for p, _ in ft["out"]}
        owned = set()
        for index, members in enumerate(groups.values()):
            touched = set().union(*(_touches(trans[ti]) for ti in members))
            if inp in touched:
                raise _Refused("a subpage transition touches the input place")
            read = set().union(*(_consumes(trans[ti]) for ti in members))
            ins = touched & fed
            outs = {p for p in touched if places[p]["page"] == top
                    and p not in ins and p not in read}
            pages = sorted({trans[ti]["page"] for ti in members})
            label = next((pg for pg in pages if not pg.startswith("Tree")),
                         pages[0])
            comp = Component(index, label, members, touched, ins, outs)
            self.components.append(comp)
            for ti in members:
                self.comp_of_transition[ti] = index
            for p in touched:
                self.comp_of_place[p] = index
            owned |= touched
        if not fed <= owned:
            raise _Refused(f"{ft['name']} feeds a place no component reads")

        all_out = set().union(*(set(c.outputs) for c in self.components))
        results = frozenset(range(len(places))) - owned - {inp}
        joins = tuple(ti for ti, t in enumerate(trans)
                      if t["page"] == top and ti != fork)
        for ti in joins:
            t = trans[ti]
            if t["in_expr"]:
                raise _Refused(f"{t['name']} reads through an input expression")
            if not t["needs"]:
                raise _Refused(f"{t['name']} has no input: it can always occur")
            if not _consumes(t) <= all_out:
                raise _Refused(f"{t['name']} reads a place that is not a "
                               "component output")
            if not {p for p, _ in t["out"]} <= results:
                raise _Refused(f"{t['name']} writes into a component place")
        self.input, self.fork = inp, fork
        self.joins, self.results = joins, results
        self.ok = True

        # Half of Level 2 is static: synchronising joins, one output each.
        lonely = [c.label for c in self.components if len(c.outputs) != 1]
        partial = [trans[ti]["name"] for ti in joins
                   if set(trans[ti]["needs"]) != all_out
                   or any(n != 1 for n in trans[ti]["needs"].values())]
        if lonely:
            self.sync_reason = (f"{lonely[0]} does not have exactly one output "
                                "place")
        elif partial:
            self.sync_reason = (f"{partial[0]} does not take one token from "
                                "every component output")
        elif not joins:
            self.sync_reason = "the net has no join transition"
        else:
            self.sync = True


# Exception carrying a failed Level 1 condition.
class _Refused(Exception):
    """A condition of the decomposition does not hold on the net."""


# Class holding the local graph of one component and what is read off it.
class _Local:
    """``G_c`` and its facts.

    Attributes:
        comp (Component): The component.
        ss (StateSpace): The local graph.
        ctx (Context): An evaluation context on it.
        acyclic (bool): No cycle (complete graphs only).
        out (int|None): The output place, when there is exactly one.
        max_out (int): Most tokens ever in an output place.
        values (OrderedDict): Output value -> a dead node holding it.
        silent (int|None): A dead node with an empty output.
        all_values (set): Output values over every node.
    """

    # Method to build and analyse the local graph.
    def __init__(self, net, comp, fork, initial, max_nodes, groups):
        """Explore ``G_c``.

        Args:
            net (ColouredNet): The net.
            comp (Component): The component.
            fork (int): ``F``.
            initial (tuple): The initial marking.
            max_nodes (int): Budget.
            groups (dict): Resolved groups, for the context.
        """
        self.comp = comp
        view = _Restricted(net, comp.transitions | {fork})
        self.ss = build_state_space(view, max_nodes, initial=initial)
        self.ss.net = net
        self.ctx = Context(self.ss, net, groups, semantics="textbook")
        ss = self.ss
        self.acyclic = ss.complete and ss.n_sccs == ss.n_nodes and not any(
            dst == src for src, out in enumerate(ss.succ) for _t, _k, dst in out)
        self.out = comp.outputs[0] if len(comp.outputs) == 1 else None
        self.max_out = 0
        self.values: "collections.OrderedDict[tuple, int]" = \
            collections.OrderedDict()
        self.all_values = set()
        self.silent = None
        for v, m in enumerate(ss.markings):
            for p in comp.outputs:
                self.max_out = max(self.max_out, len(m[p]))
            if self.out is not None and m[self.out]:
                self.all_values.add(m[self.out])
        dead = set(ss.dead)
        for v in ss.dead:
            m = ss.markings[v]
            if self.out is not None and m[self.out]:
                self.values.setdefault(m[self.out], v)
            elif self.silent is None:
                self.silent = v
        self.dead = sorted(dead)

    # Property telling whether the component is well behaved (Level 2).
    @property
    def well_behaved(self) -> bool:
        """Complete, acyclic, one output place, at most one token in it.

        Returns:
            bool: The answer.
        """
        return (self.ss.complete and self.acyclic and self.out is not None
                and self.max_out <= 1
                and self.all_values == set(self.values))


# Class deciding the catalog of one net compositionally.
class CompositionalChecker:
    """Decide the tree catalog from per-component graphs and the join layer.

    Attributes:
        checker: The :class:`~pyruleanalyzer.model_checker.CPNModelChecker`.
        decomposition (Decomposition): The shape of its net.
        max_combinations (int): Bound on enumerated output combinations.
    """

    # Method to set up the compositional checker.
    def __init__(self, checker, max_combinations: int = DEFAULT_MAX_COMBINATIONS):
        """Analyse the net once.

        Args:
            checker (CPNModelChecker): The checker whose net and formulas are
                decided.
            max_combinations (int): Bound on enumerated combinations.
        """
        self.checker = checker
        self.cnet = checker.cnet
        self.max_combinations = int(max_combinations)
        self.decomposition = Decomposition(self.cnet, checker.input_places)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    # Method to decide the properties of one input.
    def evaluate(self, sample=None):
        """Decide what can be decided compositionally for one input.

        Args:
            sample (optional): The input, or None for the file's own marking.

        Returns:
            tuple: ``(verdicts, undecided, stats)`` -- ``{pid: PropertyResult}``
            for the properties decided here, ``{pid: reason}`` for the ones
            left to the fallback, and statistics.
        """
        checker = self.checker
        wanted = [pid for pid in checker.formulas] + \
            [pid for pid in checker.scc_props if pid not in checker.formulas]
        dec = self.decomposition
        stats: Dict[str, Any] = {"reduction": TAG, "components": 0,
                                 "local_nodes": [], "combinations": 0,
                                 "complete": True, "plans": {}}
        if not dec.ok:
            return {}, {pid: f"not compositional: {dec.reason}"
                        for pid in wanted}, stats
        initial = self.cnet.initial
        if sample is not None:
            initial = self.cnet.initial_marking(checker.as_sample(sample),
                                                places=checker.input_places)
        why = self._initial_ok(initial)
        if why:
            return {}, {pid: f"not compositional: {why}" for pid in wanted}, stats

        self.sample = sample
        self.initial = initial
        fork_env = self.cnet.enabled(initial)[0]
        self.fork_step = fork_env
        self.after_fork = self.cnet.fire(initial, fork_env[0], fork_env[2])
        locals_ = [_Local(self.cnet, c, dec.fork, initial, checker.max_nodes,
                          checker.groups) for c in dec.components]
        self.locals = locals_
        stats["components"] = len(locals_)
        stats["local_nodes"] = [lc.ss.n_nodes for lc in locals_]
        stats["complete"] = all(lc.ss.complete for lc in locals_)

        self._combo = None
        self._combo_reason = self._level2_reason()
        if not self._combo_reason:
            self._combo = self._combinations()
            if isinstance(self._combo, str):
                self._combo_reason, self._combo = self._combo, None
        stats["combinations"] = len(self._combo["tuples"]) if self._combo else 0

        verdicts: Dict[str, PropertyResult] = {}
        undecided: Dict[str, str] = {}
        for pid in wanted:
            t0 = time.perf_counter()
            method = getattr(self, f"_decide_{pid}", None)
            if method is None:
                undecided[pid] = "not compositional: no decomposition is " \
                                 "known for this property"
                continue
            out = method(pid)
            if isinstance(out, str):
                undecided[pid] = out
                continue
            holds, detail, cex, plan = out
            if cex is not None:
                cex.input = sample
                cex.data["input_places"] = checker.input_places
                cex.data["reduction"] = TAG
                cex.note = f"[{TAG}] {cex.note}"
                detail = str(cex)
            elif holds:
                detail = f"{TAG}: {detail}"
            verdicts[pid] = PropertyResult(pid, holds, detail,
                                           time.perf_counter() - t0,
                                           spec=checker.catalog.get(pid), cex=cex)
            stats["plans"][pid] = plan
        return verdicts, undecided, stats

    # Method to check the per-input preconditions.
    def _initial_ok(self, initial) -> str:
        """One token, in the input place, and one fork occurrence enabled.

        Args:
            initial (tuple): The initial marking.

        Returns:
            str: Empty when the conditions hold, the reason otherwise.
        """
        dec = self.decomposition
        for p, tokens in enumerate(initial):
            if p == dec.input:
                if len(tokens) != 1:
                    return "the input place does not hold exactly one token"
            elif tokens:
                return (f"place {self.cnet.places[p]['name']} is marked in the "
                        "initial marking")
        enabled = self.cnet.enabled(initial)
        if len(enabled) != 1 or enabled[0][0] != dec.fork:
            return "the initial marking does not enable exactly one fork occurrence"
        return ""

    # ------------------------------------------------------------------
    # Level 2: the join layer
    # ------------------------------------------------------------------

    # Method to say why the combination cannot be evaluated.
    def _level2_reason(self) -> str:
        """Empty when Level 2 holds, the first failing condition otherwise.

        Returns:
            str: The reason.
        """
        dec = self.decomposition
        if not dec.sync:
            return dec.sync_reason
        structure = self.checker.structure
        if structure.pred not in dec.results:
            return "Prediction is not a result place"
        for lc in self.locals:
            if not lc.ss.complete:
                return f"the local graph of {lc.comp.label} is over budget"
            if not lc.acyclic:
                return f"{lc.comp.label} has a cycle"
            if lc.max_out > 1:
                return f"{lc.comp.label} puts more than one token in its output"
            if not lc.well_behaved:
                return f"{lc.comp.label} is not well behaved"
        return ""

    # Method to evaluate the join layer on every output combination.
    def _combinations(self):
        """``E(tau)`` and ``post(tau, b)`` for every tuple of output values.

        Returns:
            dict|str: ``{"tuples": [(tau, [(b, post)])]}``, or the reason it
            could not be computed.
        """
        dec = self.decomposition
        sizes = [len(lc.values) for lc in self.locals]
        total = 1
        for s in sizes:
            total *= s
        if total > self.max_combinations:
            return (f"{total} output combinations exceed max_combinations="
                    f"{self.max_combinations}")
        joins = set(dec.joins)
        n_places = len(self.cnet.places)
        tuples = []
        for tau in itertools.product(*[list(lc.values) for lc in self.locals]):
            m = [()] * n_places
            for lc, value in zip(self.locals, tau):
                m[lc.out] = value
            m = tuple(m)
            fired = []
            for ti, key, env in self.cnet.enabled(m):
                if ti not in joins:
                    continue
                post = self.cnet.fire(m, ti, env)
                if any(post[lc.out] for lc in self.locals):
                    return (f"{self.cnet.transitions[ti]['name']} leaves a token "
                            "in a component output")
                fired.append(((ti, key, env), post))
            tuples.append((tau, fired))
        return {"tuples": tuples}

    # Method to assemble a global marking from local nodes.
    def _assemble(self, choice: Dict[int, int]) -> tuple:
        """The marking reached by running each component to a chosen node.

        Args:
            choice (dict): Component index -> node of its local graph.

        Returns:
            tuple: The marking.
        """
        m = list(self.after_fork)
        for lc in self.locals:
            node = lc.ss.markings[choice[lc.comp.index]]
            for p in lc.comp.places:
                m[p] = node[p]
        return tuple(m)

    # Method to build a replayable trace through chosen local nodes.
    def _trace(self, choice: Dict[int, int], join=None, upto=None):
        """The fork, each component's path to its node, then a join.

        Args:
            choice (dict): Component index -> node.
            join (tuple, optional): ``(ti, key, env)`` to fire at the end.
            upto (int, optional): Only the components before this index run.

        Returns:
            tuple: ``(steps, final marking)``.
        """
        net = self.cnet
        ti, key, env = self.fork_step
        steps = [Step(ti, net.transitions[ti]["name"], net.transitions[ti]["page"],
                      dict(key))]
        marking = self.after_fork
        for lc in self.locals:
            if upto is not None and lc.comp.index >= upto:
                break
            for sti, skey, _dst in lc.ss.path(choice[lc.comp.index])[1:]:
                t = net.transitions[sti]
                steps.append(Step(sti, t["name"], t["page"], dict(skey)))
                marking = net.fire(marking, sti, dict(skey))
        if join is not None:
            jti, jkey, jenv = join
            t = net.transitions[jti]
            steps.append(Step(jti, t["name"], t["page"], dict(jkey)))
            marking = net.fire(marking, jti, jenv)
        return steps, marking

    # Method to pick, for a tuple, the dead node of each component holding it.
    def _choice(self, tau) -> Dict[int, int]:
        """Component index -> a dead node whose output is ``tau``'s value.

        Args:
            tau (tuple): One output value per component.

        Returns:
            dict: The choice.
        """
        return {lc.comp.index: lc.values[v] for lc, v in zip(self.locals, tau)}

    # Method to pick dead nodes with one silent component.
    def _silent_choice(self) -> Optional[Dict[int, int]]:
        """A choice of dead nodes in which some component deposited nothing.

        Returns:
            dict|None: The choice, or None when no component is silent.
        """
        silent = next((lc for lc in self.locals if lc.silent is not None), None)
        if silent is None:
            return None
        return {lc.comp.index: (lc.silent if lc is silent else lc.dead[0])
                for lc in self.locals}

    # Method to describe a tuple of outputs.
    def _show_tau(self, tau) -> str:
        """``Tree_0=..., Tree_1=...``, shortened.

        Args:
            tau (tuple): The tuple.

        Returns:
            str: The text.
        """
        parts = [f"{lc.comp.label}={_short(v)}" for lc, v in zip(self.locals, tau)]
        if len(parts) > 4:
            parts = parts[:3] + [f"... {len(parts) - 3} more"]
        return ", ".join(parts)

    # Method to evaluate a state proposition at one marking.
    def _atom_at(self, formula, marking) -> Tuple[bool, str]:
        """Whether a token proposition holds at a marking, and why not.

        The proposition is the profile's own (the operand of ``AG``), asked
        on a one-node graph; only token propositions are asked this way.

        Args:
            formula (Formula): The proposition.
            marking (tuple): The marking.

        Returns:
            tuple: ``(holds, reason when it does not)``.
        """
        ss = StateSpace(self.cnet, [marking], [[]], True, 0.0, [None])
        ctx = Context(ss, self.cnet, self.checker.groups)
        if 0 in ctx.sat(formula):
            return True, ""
        return False, formula.at_fail(ctx, 0)

    # Method building the finite counterexample through a tuple.
    def _cex(self, note, choice, join=None, data=None, upto=None):
        """A replayable finite counterexample.

        Args:
            note (str): What is wrong.
            choice (dict): Component nodes.
            join (tuple, optional): A join occurrence to end with.
            data (dict, optional): Extras.
            upto (int, optional): See :meth:`_trace`.

        Returns:
            Counterexample: The evidence.
        """
        steps, final = self._trace(choice, join, upto)
        return Counterexample("finite", note=note, via="after:", prefix=steps,
                              final=final, data=dict(data or {}))

    # ------------------------------------------------------------------
    # Level 1 properties: decided on the local graphs
    # ------------------------------------------------------------------

    # Method to evaluate a formula on a local graph.
    def _local(self, lc: _Local, formula):
        """Whether ``formula`` holds at the initial marking of ``G_c``.

        Args:
            lc (_Local): The local graph.
            formula (Formula): The formula.

        Returns:
            tuple: ``(holds, counterexample or None)``.
        """
        if 0 in lc.ctx.sat(formula):
            return True, None
        return False, formula.counterexample(lc.ctx)

    # Method naming the local graphs for the plan line.
    def _plan_local(self) -> str:
        """The plan text of a property decided on the local graphs.

        Returns:
            str: The text.
        """
        sizes = [lc.ss.n_nodes for lc in self.locals]
        return (f"{TAG}: {len(sizes)} component graph(s), {sum(sizes)} "
                f"markings in all, largest {max(sizes)}")

    # Method naming the join layer for the plan line.
    def _plan_combo(self) -> str:
        """The plan text of a property decided on the join layer.

        Returns:
            str: The text.
        """
        return (f"{TAG}: join layer over {len(self._combo['tuples'])} output "
                f"combination(s) of {len(self.locals)} component(s)")

    # Method to report an over-budget local graph.
    def _incomplete(self, lcs) -> str:
        """The reason a local property got no verdict.

        Args:
            lcs (list): Local graphs that were needed.

        Returns:
            str: The reason, or empty when all are complete.
        """
        for lc in lcs:
            if not lc.ss.complete:
                return (f"the local graph of {lc.comp.label} exceeded "
                        f"{self.checker.max_nodes} markings")
        return ""

    # A1: termination.
    def _decide_A1(self, pid):
        """Every ``G_c`` acyclic (see the module docstring).

        Args:
            pid (str): The property id.

        Returns:
            tuple|str: The verdict, or the reason for the fallback.
        """
        for lc in self.locals:
            if lc.ss.complete and not lc.acyclic:
                holds, cex = self._local(lc, AF(Dead().phrased("termination")))
                if cex is None:          # pragma: no cover - acyclic disagrees
                    return "not compositional: inconsistent cycle detection"
                cex.note = f"{lc.comp.label}: {cex.note}"
                return False, "", cex, self._plan_local()
        why = self._incomplete(self.locals)
        if why:
            return f"not compositional: {why}"
        return (True, "every component graph is acyclic and every join "
                "consumes a token", None, self._plan_local())

    # A7: safeness.
    def _decide_A7(self, pid):
        """Local places on ``G_c``, result places on the join layer.

        Args:
            pid (str): The property id.

        Returns:
            tuple|str: The verdict, or the reason for the fallback.
        """
        for lc in self.locals:
            if not lc.ss.complete:
                continue
            sel = Explicit(sorted(lc.comp.places), "place",
                           f"{lc.comp.label} places")
            holds, cex = self._local(lc, AG(MaxTokens(sel, 1)))
            if not holds:
                cex.note = f"{lc.comp.label}: {cex.note}"
                return False, "", cex, self._plan_local()
        why = self._incomplete(self.locals)
        if why:
            return f"not compositional: {why}"
        if self._combo is None:
            return f"not compositional (result places): {self._combo_reason}"
        operand = self.checker.formulas[pid].operand
        for tau, fired in self._combo["tuples"]:
            for b, post in fired:
                holds, why = self._atom_at(operand, post)
                if not holds:
                    cex = self._cex(f"{why} (outputs {self._show_tau(tau)})",
                                    self._choice(tau), b)
                    return False, "", cex, self._plan_combo()
        return (True, "every component graph is safe and every join "
                "occurrence fills each result place at most once", None,
                self._plan_local() + "; " + self._plan_combo())

    # Method to decide one per-member property (trees, channels) locally.
    def _per_member(self, pid, group, members_of, needs_others_acyclic=False,
                    check=None):
        """Evaluate each instance of a ``ForEach`` on its component's graph.

        Args:
            pid (str): The property id.
            group (str): The group name.
            members_of (callable): ``roles -> (transitions, places)`` the
                instance mentions.
            needs_others_acyclic (bool): Whether a True verdict also needs the
                other components acyclic (``AF`` properties).
            check (callable, optional): ``(roles, component) -> reason`` for an
                extra condition on the instance.

        Returns:
            tuple|str: The verdict, or the reason for the fallback.
        """
        formula = self.checker.formulas[pid]
        if not isinstance(formula, ForEach):
            return "not compositional: unexpected formula shape"
        dec = self.decomposition
        instances = []
        for label, roles in self.checker.groups.get(group, {}).items():
            inst = formula.builder(roles, label)
            if inst is None:
                continue
            trans, places = members_of(roles)
            comps = {dec.comp_of_transition.get(t) for t in trans} | \
                {dec.comp_of_place.get(p) for p in places}
            if len(comps) != 1 or None in comps:
                return (f"not compositional: {label} is not inside a single "
                        "component")
            index = comps.pop()
            if check is not None:
                why = check(roles, dec.components[index])
                if why:
                    return f"not compositional: {label}: {why}"
            instances.append((label, inst, self.locals[index]))
        for label, inst, lc in instances:
            if not lc.ss.complete:
                continue
            holds, cex = self._local(lc, inst)
            if not holds:
                cex.note = f"{label}: {cex.note}"
                cex.data["group_member"] = label
                return False, "", cex, self._plan_local()
        why = self._incomplete([lc for _l, _i, lc in instances])
        if why:
            return f"not compositional: {why}"
        if needs_others_acyclic:
            for lc in self.locals:
                if not (lc.ss.complete and lc.acyclic):
                    return (f"not compositional: {lc.comp.label} may run "
                            "forever, so another component may never move")
        return (True, f"{len(instances)} instance(s), each on its component's "
                "graph", None, self._plan_local())

    # B1: leaf determinism.
    def _decide_B1(self, pid):
        """Per tree, on the graph of its component.

        Args:
            pid (str): The property id.

        Returns:
            tuple|str: The verdict, or the reason for the fallback.
        """
        return self._per_member(pid, "trees",
                                lambda r: (r["leaves"], r["out"]))

    # B2: inevitable leaf selection.
    def _decide_B2(self, pid):
        """Per tree, with every other component terminating.

        Args:
            pid (str): The property id.

        Returns:
            tuple|str: The verdict, or the reason for the fallback.
        """
        return self._per_member(pid, "trees",
                                lambda r: (r["leaves"], r["out"]),
                                needs_others_acyclic=True)

    # B3: single tree output.
    def _decide_B3(self, pid):
        """Each tree output place on the graph of its component.

        Args:
            pid (str): The property id.

        Returns:
            tuple|str: The verdict, or the reason for the fallback.
        """
        dec = self.decomposition
        outs = collections.defaultdict(list)
        for label, roles in self.checker.groups.get("trees", {}).items():
            for p in roles["out"]:
                index = dec.comp_of_place.get(p)
                if index is None:
                    return (f"not compositional: the output of {label} is "
                            "outside every component")
                outs[index].append(p)
        for index, places in outs.items():
            lc = self.locals[index]
            if not lc.ss.complete:
                continue
            sel = Explicit(tuple(places), "place", "tree output place")
            holds, cex = self._local(lc, AG(MaxTokens(sel, 1)))
            if not holds:
                cex.note = f"{lc.comp.label}: {cex.note}"
                return False, "", cex, self._plan_local()
        why = self._incomplete([self.locals[i] for i in outs])
        if why:
            return f"not compositional: {why}"
        return (True, "every tree output place is safe on its component's "
                "graph", None, self._plan_local())

    # Method checking that the hold part of an EU is not an output place.
    @staticmethod
    def _accs_internal(roles, comp) -> str:
        """The accumulators a channel formula holds on must not be outputs.

        Args:
            roles (dict): The channel's roles.
            comp (Component): Its component.

        Returns:
            str: The reason, or empty.
        """
        if set(roles["acc"]) & set(comp.outputs):
            return "an accumulator is a component output"
        return ""

    # C1: stage precedence.
    def _decide_C1(self, pid):
        """Per channel, on the graph of its component.

        Args:
            pid (str): The property id.

        Returns:
            tuple|str: The verdict, or the reason for the fallback.
        """
        return self._per_member(pid, "channels",
                                lambda r: ((), tuple(r["acc"])),
                                check=self._accs_internal)

    # C2: no premature score.
    def _decide_C2(self, pid):
        """Per channel, on the graph of its component.

        Args:
            pid (str): The property id.

        Returns:
            tuple|str: The verdict, or the reason for the fallback.
        """
        return self._per_member(pid, "channels",
                                lambda r: ((), tuple(r["acc"]) + tuple(r["score"])),
                                check=self._accs_internal)

    # D1: complete vote.
    def _decide_D1(self, pid):
        """Structural: every Vote occurrence consumes every tree output.

        Args:
            pid (str): The property id.

        Returns:
            tuple|str: The verdict, or the reason for the fallback.
        """
        structure = self.checker.structure
        trans = self.cnet.transitions
        socks = structure.vote_socks
        for ti in structure.vote:
            needs = trans[ti]["needs"]
            missing = [p for p in socks if needs.get(p, 0) < 1]
            if missing:
                return (f"not compositional: {trans[ti]['name']} does not "
                        f"consume from {self.cnet.places[missing[0]]['name']}")
        return (True, "every Vote occurrence consumes a token from every tree "
                "output place, so it is enabled only when all are marked",
                None, f"{TAG}: structural (input arcs of Vote)")

    # ------------------------------------------------------------------
    # Level 2 properties: decided on the join layer
    # ------------------------------------------------------------------

    # Method guarding the Level 2 properties.
    def _need_combo(self) -> str:
        """The fallback reason when Level 2 does not hold.

        Returns:
            str: The reason, or empty.
        """
        if self._combo is None:
            return f"not compositional: {self._combo_reason}"
        return ""

    # Method deciding "every dead marking holds a prediction" (A2, A3, A4).
    def _decision_reached(self, pid, how):
        """No silent component; every tuple enables a join that predicts.

        Args:
            pid (str): The property id.
            how (str): ``"dead"``, ``"af"`` or ``"recover"``: the wording.

        Returns:
            tuple|str: The verdict, or the reason for the fallback.
        """
        why = self._need_combo()
        if why:
            return why
        pred = self.checker.structure.pred
        notes = {"dead": "dead marking without a prediction",
                 "af": "the prediction never happens",
                 "recover": "no prediction is reachable from this dead marking"}
        silent = self._silent_choice()
        if silent is not None:
            label = next(lc.comp.label for lc in self.locals
                         if silent[lc.comp.index] == lc.silent)
            cex = self._cex(f"{notes[how]}: {label} ends without an output, so "
                            "no join can occur", silent,
                            data={"ends_in": "a dead marking"})
            return False, "", cex, self._plan_combo()
        for tau, fired in self._combo["tuples"]:
            if not fired:
                cex = self._cex(f"{notes[how]}: no join is enabled for outputs "
                                f"{self._show_tau(tau)}", self._choice(tau),
                                data={"ends_in": "a dead marking"})
                return False, "", cex, self._plan_combo()
            for b, post in fired:
                if not post[pred]:
                    cex = self._cex(f"{notes[how]}: "
                                    f"{self.cnet.transitions[b[0]]['name']} "
                                    "deposits no prediction", self._choice(tau), b,
                                    data={"ends_in": "a dead marking"})
                    return False, "", cex, self._plan_combo()
        return (True, "no component ends without an output and, for every "
                "combination of outputs, a join occurs and deposits a "
                "prediction; the net terminates, so every run ends with one",
                None, self._plan_combo())

    # A2: no spurious deadlock.
    def _decide_A2(self, pid):
        """See :meth:`_decision_reached`.

        Args:
            pid (str): The property id.

        Returns:
            tuple|str: The verdict, or the reason for the fallback.
        """
        return self._decision_reached(pid, "dead")

    # A3: inevitable decision.
    def _decide_A3(self, pid):
        """See :meth:`_decision_reached`.

        Args:
            pid (str): The property id.

        Returns:
            tuple|str: The verdict, or the reason for the fallback.
        """
        return self._decision_reached(pid, "af")

    # A4: prediction recoverability.
    def _decide_A4(self, pid):
        """See :meth:`_decision_reached`.

        Args:
            pid (str): The property id.

        Returns:
            tuple|str: The verdict, or the reason for the fallback.
        """
        return self._decision_reached(pid, "recover")

    # Method deciding an AG of a token proposition on Prediction.
    def _prediction_invariant(self, pid):
        """The proposition on every ``post(tau, b)`` (A5, A6, D2b).

        Args:
            pid (str): The property id.

        Returns:
            tuple|str: The verdict, or the reason for the fallback.
        """
        why = self._need_combo()
        if why:
            return why
        formula = self.checker.formulas[pid]
        if not isinstance(formula, AG):
            return "not compositional: unexpected formula shape"
        for tau, fired in self._combo["tuples"]:
            for b, post in fired:
                holds, reason = self._atom_at(formula.operand, post)
                if not holds:
                    cex = self._cex(f"{reason} (outputs {self._show_tau(tau)})",
                                    self._choice(tau), b)
                    return False, "", cex, self._plan_combo()
        return (True, "Prediction is written only by the single join "
                "occurrence, and every join occurrence satisfies it", None,
                self._plan_combo())

    _decide_A5 = _prediction_invariant
    _decide_A6 = _prediction_invariant
    _decide_D2b = _prediction_invariant

    # A8: home marking.
    def _decide_A8(self, pid):
        """Exactly one dead marking, enumerated from the components'.

        Args:
            pid (str): The property id.

        Returns:
            tuple|str: The verdict, or the reason for the fallback.
        """
        why = self._need_combo()
        if why:
            return why
        distinct = []
        for lc in self.locals:
            seen = collections.OrderedDict()
            for v in lc.dead:
                m = lc.ss.markings[v]
                seen.setdefault(tuple(m[p] for p in sorted(lc.comp.places)), v)
            distinct.append(list(seen.values()))
        total = 1
        for nodes in distinct:
            total *= len(nodes)
        if total > self.max_combinations:
            return (f"not compositional: {total} dead-marking combinations "
                    f"exceed max_combinations={self.max_combinations}")
        fired_of = {tau: fired for tau, fired in self._combo["tuples"]}
        found = collections.OrderedDict()
        for nodes in itertools.product(*distinct):
            choice = {lc.comp.index: v for lc, v in zip(self.locals, nodes)}
            outs = tuple(lc.ss.markings[v][lc.out]
                         for lc, v in zip(self.locals, nodes))
            fired = fired_of.get(outs) if all(outs) else None
            if not fired:
                found.setdefault(self._assemble(choice), (choice, None))
            else:
                base = self._assemble(choice)
                for b, _post in fired:
                    end = self.cnet.fire(base, b[0], b[2])
                    found.setdefault(end, (choice, b))
            if len(found) > 1:
                break
        if len(found) == 1:
            return (True, "terminating with one dead marking, which is the "
                    "home marking", None, self._plan_combo())
        (c1, b1), (c2, b2) = list(found.values())[:2]
        t1 = " -> ".join(s.name for s in self._trace(c1, b1)[0][-3:])
        t2 = " -> ".join(s.name for s in self._trace(c2, b2)[0][-3:])
        cex = Counterexample(
            "witness", note=(f"at least 2 dead markings: no marking is "
                             f"reachable from all others; after ... {t1} and "
                             f"after ... {t2} the runs can no longer meet"),
            data={"terminal_components": len(found),
                  "representatives": [t1, t2]})
        return False, "", cex, self._plan_combo()

    # Method deciding "AF some join of V is enabled" (C3a, D2a).
    def _join_inevitable(self, pid, group: Sequence[int], name: str):
        """No silent component, and ``E_V(tau)`` non-empty for every tuple.

        Args:
            pid (str): The property id.
            group (sequence): The transitions ``V``.
            name (str): Their name, for the evidence.

        Returns:
            tuple|str: The verdict, or the reason for the fallback.
        """
        why = self._need_combo()
        if why:
            return why
        group = set(group)
        if not group <= set(self.decomposition.joins):
            return f"not compositional: {name} is not a join"
        silent = self._silent_choice()
        if silent is not None:
            cex = self._cex(f"a {name} enabling never happens: a component "
                            "ends without an output", silent,
                            data={"ends_in": "a dead marking"})
            return False, "", cex, self._plan_combo()
        for tau, fired in self._combo["tuples"]:
            if not any(b[0] in group for b, _post in fired):
                other = fired[0][0] if fired else None
                cex = self._cex(f"a {name} enabling never happens for outputs "
                                f"{self._show_tau(tau)}", self._choice(tau),
                                other, data={"ends_in": "a dead marking"})
                return False, "", cex, self._plan_combo()
        return (True, f"no component ends without an output and every "
                f"combination of outputs enables {name}", None,
                self._plan_combo())

    # C3a: inevitable decision firing.
    def _decide_C3a(self, pid):
        """See :meth:`_join_inevitable`.

        Args:
            pid (str): The property id.

        Returns:
            tuple|str: The verdict, or the reason for the fallback.
        """
        return self._join_inevitable(pid, self.checker.structure.decides,
                                     "Decide")

    # D2a: inevitable aggregation.
    def _decide_D2a(self, pid):
        """See :meth:`_join_inevitable`.

        Args:
            pid (str): The property id.

        Returns:
            tuple|str: The verdict, or the reason for the fallback.
        """
        return self._join_inevitable(pid, self.checker.structure.vote, "Vote")

    # C3b: decision determinism.
    def _decide_C3b(self, pid):
        """At most one Decide binding per output combination.

        Args:
            pid (str): The property id.

        Returns:
            tuple|str: The verdict, or the reason for the fallback.
        """
        why = self._need_combo()
        if why:
            return why
        decides = set(self.checker.structure.decides)
        if not decides <= set(self.decomposition.joins):
            return "not compositional: a Decide is not a join"
        for tau, fired in self._combo["tuples"]:
            names = [self.cnet.transitions[b[0]]["name"] for b, _p in fired
                     if b[0] in decides]
            if len(names) > 1:
                cex = self._cex(f"{len(names)} occurrences enabled together "
                                f"({', '.join(names[:3])}) for outputs "
                                f"{self._show_tau(tau)}", self._choice(tau))
                return False, "", cex, self._plan_combo()
        return (True, "every combination of outputs enables at most one "
                "Decide occurrence", None, self._plan_combo())


# Function to shorten a token value for a note.
def _short(value) -> str:
    """A short rendering of a token tuple.

    Args:
        value (tuple): The tokens of a place.

    Returns:
        str: The text.
    """
    if len(value) == 1:
        value = value[0]
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(f"{v:.3g}" if isinstance(v, float) else str(v)
                               for v in value) + "]"
    return str(value)
