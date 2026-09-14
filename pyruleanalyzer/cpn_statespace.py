"""State-space generation for the Coloured Petri Nets written by the exporter.

This module builds the occurrence graph of a generated ``.cpn`` file under the
firing rule of Coloured Petri Nets (Jensen & Kristensen, *Coloured Petri Nets:
Modelling and Validation of Concurrent Systems*, Springer, 2009): a marking
assigns a multiset of token *values* to every place, a binding element is a
transition together with values for its variables, and it is enabled when its
input arcs evaluate to tokens present in the marking and its guard evaluates to
true. Nodes of the graph are markings reachable from the initial marking; arcs
are occurrences of binding elements. This is the same object the CPN Tools
state-space tool builds with ``CalculateOccGraph``, and the counts produced here
are compared with CPN Tools' own ``NoOfNodes``/``NoOfArcs`` in the cross-check
suite.

Tokens are real values, not an abstraction: two markings that put different
values in the same place are different nodes. Nothing has to be argued about an
abstraction being exact, because there is none.

Supported CPN ML fragment -- the one the exporter emits, and nothing more:

* colour sets ``INT``, ``REAL``, records of ``REAL`` (``SAMPLE``), products
  (``SS``) and ``list REAL`` (``PROB``);
* arc inscriptions that are variables, literals, tuples, lists, arithmetic on
  reals, the declared soft-voting functions (``vadd``, ``vsum``, ``argmax``,
  ``allzero``, ``decide``), and multisets written with ``++`` and backquote;
* input-arc inscriptions that are patterns (variables, literals, tuples);
* guards over record fields (``#f3 x``), comparisons and boolean connectives;
* hierarchy through substitution transitions and port/socket pairs.

Anything outside the fragment raises :class:`UnsupportedNet` rather than being
approximated.
"""

import collections
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .cpn_semantics import CPNNet

__all__ = [
    "UnsupportedNet",
    "ColouredNet",
    "StateSpace",
    "build_state_space",
]


# Exception raised when a net uses CPN ML outside the supported fragment.
class UnsupportedNet(Exception):
    """The net uses a construct the state-space generator does not implement."""


# ---------------------------------------------------------------------------
# CPN ML fragment -> Python
# ---------------------------------------------------------------------------

# Function to add two probability vectors element-wise, as the ML vadd does.
def _vadd(xs, ys):
    """ML ``vadd``: ``ListPair.map (op +)``, truncating to the shorter list.

    Args:
        xs (tuple): First vector.
        ys (tuple): Second vector.

    Returns:
        tuple: The element-wise sum.
    """
    return tuple(a + b for a, b in zip(xs, ys))


# Function to sum a list of vectors, as the ML vsum does.
def _vsum(vectors):
    """ML ``vsum``: ``List.foldl vadd h t`` (empty list gives the empty vector).

    Args:
        vectors (list): Vectors to add.

    Returns:
        tuple: The accumulated vector.
    """
    vectors = list(vectors)
    if not vectors:
        return ()
    acc = tuple(vectors[0])
    for vec in vectors[1:]:
        acc = _vadd(vec, acc)          # foldl f init: f (element, accumulator)
    return acc


# Function to return the index of the maximum, lowest index winning ties.
def _argmax(values):
    """ML ``argmax``: strict ``>`` scan from the left, so ties keep the first.

    Args:
        values (tuple): The vector.

    Returns:
        int: Index of the maximum (0 for the empty vector).
    """
    values = list(values)
    if not values:
        return 0
    best, best_index = values[0], 0
    for i, v in enumerate(values[1:], start=1):
        if v > best:
            best, best_index = v, i
    return best_index


# Function to test whether every entry of a vector is zero, as ML allzero does.
def _allzero(values):
    """ML ``allzero``: ``x <= 0.0 andalso x >= 0.0`` for every entry.

    Args:
        values (tuple): The vector.

    Returns:
        bool: True when every entry is exactly zero.
    """
    return all(v <= 0.0 and v >= 0.0 for v in values)


# Function to split a string on a separator at bracket depth zero.
def _split_top(text: str, sep: str) -> List[str]:
    """Split on ``sep`` only where it is not nested inside brackets.

    Args:
        text (str): Source text.
        sep (str): Separator (``","`` or ``"++"``).

    Returns:
        list: The top-level pieces.
    """
    out, depth, cur, i = [], 0, [], 0
    while i < len(text):
        ch = text[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if depth == 0 and text.startswith(sep, i):
            out.append("".join(cur))
            cur = []
            i += len(sep)
            continue
        cur.append(ch)
        i += 1
    out.append("".join(cur))
    return out


# Function to strip one pair of enclosing parentheses when they wrap the text.
def _strip_parens(text: str) -> str:
    """Remove outer parentheses only if they enclose the whole expression.

    Args:
        text (str): Source text.

    Returns:
        str: The text without a redundant enclosing pair.
    """
    text = text.strip()
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        for i, ch in enumerate(text):
            depth += ch == "("
            depth -= ch == ")"
            if depth == 0 and i < len(text) - 1:
                return text
        text = text[1:-1].strip()
    return text


# Function to translate a CPN ML expression of the supported fragment to Python.
def sml_expr_to_python(expr: str) -> str:
    """Translate one CPN ML expression into Python source.

    Args:
        expr (str): The ML expression (guard body or arc inscription term).

    Returns:
        str: Equivalent Python source over the bound variables.

    Raises:
        UnsupportedNet: For constructs outside the supported fragment.
    """
    out = expr.strip()
    if re.search(r"\b(fn|let|case|if|val|fun|div|mod|ref)\b", out):
        raise UnsupportedNet(f"unsupported CPN ML expression: {expr!r}")
    if "`" in out:
        raise UnsupportedNet(f"nested multiset in expression: {expr!r}")
    out = re.sub(r"#f(\d+)\s+([A-Za-z_]\w*)", r"\2[\1]", out)
    out = out.replace("~", "-")
    out = re.sub(r"\bandalso\b", " and ", out)
    out = re.sub(r"\borelse\b", " or ", out)
    out = re.sub(r"\bnot\b", " not ", out)
    out = re.sub(r"\btrue\b", "True", out)
    out = re.sub(r"\bfalse\b", "False", out)
    out = out.replace("<>", "!=")
    out = re.sub(r"(?<![<>=!])=(?!=)", "==", out)
    return out


# Function to compile a guard into a Python code object.
def compile_guard(guard: Optional[str]):
    """Compile a transition guard.

    CPN Tools writes guards as ``[e1, e2, ...]``, a list of conditions that must
    all hold; the brackets are removed and the conditions conjoined.

    Args:
        guard (str): Guard text, possibly empty.

    Returns:
        code: A code object, or None when the transition has no guard.
    """
    text = (guard or "").strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if not text:
        return None
    parts = [sml_expr_to_python(p) for p in _split_top(text, ",") if p.strip()]
    return compile(" and ".join(f"({p})" for p in parts), "<guard>", "eval")


# Function to compile an output arc inscription into (count, code) terms.
def compile_output(expr: str) -> List[Tuple[int, Any]]:
    """Compile an output inscription, which may be a multiset.

    Args:
        expr (str): The inscription, e.g. ``(x, sc + lv)`` or ``1`0 ++ 1`1``.

    Returns:
        list: ``(count, code)`` pairs, one per multiset term.
    """
    terms = []
    for term in _split_top(expr.strip(), "++"):
        term = term.strip()
        if not term:
            continue
        m = re.fullmatch(r"(\d+)\s*`\s*(.+)", term, flags=re.S)
        count, body = (int(m.group(1)), m.group(2)) if m else (1, term)
        terms.append((count, compile(sml_expr_to_python(body), "<arc>", "eval")))
    if not terms:
        raise UnsupportedNet(f"empty arc inscription {expr!r}")
    return terms


# Function to parse an input arc inscription into a pattern tree.
def parse_pattern(expr: str):
    """Parse an input-arc pattern: variable, literal or tuple of patterns.

    Args:
        expr (str): The inscription.

    Returns:
        tuple: ``("var", name)``, ``("lit", value)`` or ``("tuple", [..])``.

    Raises:
        UnsupportedNet: For patterns outside the fragment.
    """
    text = _strip_parens(expr)
    parts = _split_top(text, ",")
    if len(parts) > 1:
        return ("tuple", [parse_pattern(p) for p in parts])
    text = text.strip()
    if re.fullmatch(r"[a-z_][A-Za-z0-9_']*", text) and text not in ("true", "false"):
        return ("var", text)
    try:
        return ("lit", _canon(eval(sml_expr_to_python(text), {"__builtins__": {}})))  # noqa: S307
    except Exception as exc:
        raise UnsupportedNet(f"unsupported input pattern {expr!r}: {exc}")


# Function to put a token value in canonical, hashable form.
def _canon(value):
    """Convert lists to tuples recursively so tokens are hashable and ordered.

    Args:
        value: A token value.

    Returns:
        The canonical value.
    """
    if isinstance(value, (list, tuple)):
        return tuple(_canon(v) for v in value)
    if isinstance(value, bool):
        return value
    return value


# Function to match a pattern against a token under a partial binding.
def _match(pattern, token, env):
    """Extend ``env`` so that ``pattern`` evaluates to ``token``.

    Args:
        pattern (tuple): Pattern tree from :func:`parse_pattern`.
        token: The candidate token.
        env (dict): Variables bound so far.

    Returns:
        dict|None: The extended binding, or None if they do not match.
    """
    kind = pattern[0]
    if kind == "var":
        name = pattern[1]
        if name in env:
            return env if env[name] == token else None
        new = dict(env)
        new[name] = token
        return new
    if kind == "lit":
        return env if pattern[1] == token else None
    items = pattern[1]
    if not isinstance(token, tuple) or len(token) != len(items):
        return None
    for sub, tok in zip(items, token):
        env = _match(sub, tok, env)
        if env is None:
            return None
    return env


# Function to rebuild the token a pattern denotes under a complete binding.
def _instantiate(pattern, env):
    """Evaluate a pattern back into the token it consumes.

    Args:
        pattern (tuple): Pattern tree.
        env (dict): Complete binding.

    Returns:
        The token value.
    """
    kind = pattern[0]
    if kind == "var":
        return env[pattern[1]]
    if kind == "lit":
        return pattern[1]
    return tuple(_instantiate(p, env) for p in pattern[1])


# Function to parse an initial marking into a multiset of token values.
def parse_initial_marking(text: Optional[str]) -> List[Any]:
    """Parse a place's initial marking.

    Record literals ``{f0=v0, f1=v1, ...}`` become tuples ordered by field
    number, which is how guards address them (``#f3 x`` is ``x[3]``).

    Args:
        text (str): The initial-marking inscription, possibly empty.

    Returns:
        list: The tokens.
    """
    tokens: List[Any] = []
    text = (text or "").strip()
    if not text:
        return tokens
    for term in _split_top(text, "++"):
        term = term.strip()
        m = re.fullmatch(r"(\d+)\s*`\s*(.+)", term, flags=re.S)
        count, body = (int(m.group(1)), m.group(2).strip()) if m else (1, term)
        if body.startswith("{"):
            fields = re.findall(r"f(\d+)\s*=\s*\(?\s*(~?[\d.]+(?:[eE]~?\d+)?)\s*\)?",
                                body)
            value = tuple(float(v.replace("~", "-"))
                          for _, v in sorted(fields, key=lambda f: int(f[0])))
        else:
            value = _canon(eval(sml_expr_to_python(body),  # noqa: S307
                                {"__builtins__": {}}))
        tokens.extend([value] * count)
    return tokens


# ---------------------------------------------------------------------------
# The flattened coloured net
# ---------------------------------------------------------------------------

# Class holding a hierarchy-free view of a generated net, ready to fire.
class ColouredNet:
    """A flattened Coloured Petri Net with compiled guards and inscriptions.

    Attributes:
        places (list): Place records ``{gid, name, page, port}``; list index is
            the place index used in markings.
        transitions (list): Transition records with compiled guard, input
            patterns and output terms.
        initial (tuple): The initial marking.
        family (str): Model family detected from the top page.
    """

    # Method to flatten and compile a parsed net.
    def __init__(self, net: CPNNet, sample: Optional[Sequence[float]] = None,
                 default_class: int = 0):
        """Flatten the hierarchy and compile every inscription.

        Args:
            net (CPNNet): The parsed ``.cpn``.
            sample (sequence, optional): Replaces the token of the ``Input``
                place, so the same net can be explored for another input.
            default_class (int): Fallback class of the declared ``decide``
                function, read from the net's ML declarations when present.
        """
        self.net = net
        self.family = net.family()
        self.top = net.top_page()
        self.default_class = self._read_decide_default(default_class)
        self.env = {
            "__builtins__": {},
            "vadd": _vadd, "vsum": _vsum, "argmax": _argmax,
            "allzero": _allzero,
            "decide": lambda xs: self.default_class if _allzero(xs) else _argmax(xs),
        }
        self.places: List[Dict[str, Any]] = []
        self.transitions: List[Dict[str, Any]] = []
        self.port_of: Dict[Tuple[str, str], int] = {}
        self._flatten()
        self.index = {p["gid"]: i for i, p in enumerate(self.places)}
        self.consumers: Dict[int, List[int]] = collections.defaultdict(list)
        for ti, t in enumerate(self.transitions):
            for pidx in {p for p, _ in t["in"]}:
                self.consumers[pidx].append(ti)
        self.initial = self.initial_marking(sample)

    # Method to read the default class baked into the declared decide function.
    def _read_decide_default(self, fallback: int) -> int:
        """Read ``if allzero xs then <k> else argmax xs`` from the declarations.

        Args:
            fallback (int): Value used when the net declares no ``decide``.

        Returns:
            int: The default class.
        """
        import xml.etree.ElementTree as ET
        try:
            root = ET.parse(self.net.path).getroot()
        except Exception:
            return fallback
        for ml in root.iter("ml"):
            text = "".join(ml.itertext())
            m = re.search(r"fun\s+decide.*?allzero\s+\w+\s+then\s+(~?\d+)", text, re.S)
            if m:
                return int(m.group(1).replace("~", "-"))
        return fallback

    # Method to expand the page hierarchy into flat places and transitions.
    def _flatten(self):
        """Fuse port places with their sockets and compile every transition."""
        places: Dict[str, Dict[str, Any]] = collections.OrderedDict()
        trans: List[Dict[str, Any]] = []

        def expand(page, prefix, fused):
            pid_of = {}
            for lid, pl in page["places"].items():
                gid = fused.get(lid) or f"{prefix}/{pl['name']}"
                pid_of[lid] = gid
                if gid not in places:
                    places[gid] = {"gid": gid, "name": pl["name"],
                                   "page": page["name"], "port": pl.get("port"),
                                   "initmark": pl.get("initmark")}
                if pl.get("port"):
                    self.port_of[(page["name"], pl["name"])] = gid
            for tid, tr in page["transitions"].items():
                if tr["subpage"]:
                    continue
                ins, outs = [], []
                for a in page["arcs"]:
                    if a["trans"] != tid:
                        continue
                    (ins if a["orient"] == "PtoT" else outs).append(
                        (pid_of[a["place"]], a["expr"]))
                trans.append({"gid": f"{prefix}/{tr['name']}", "name": tr["name"],
                              "page": page["name"], "guard_text": tr["guard"],
                              "in_raw": ins, "out_raw": outs})
            for tid, tr in page["transitions"].items():
                if not tr["subpage"]:
                    continue
                sub = self.net.page_by_id[tr["subpage"]]
                sub_fused = {}
                for a, b in tr["portsock"]:
                    sock, port = (a, b) if a in page["places"] else (b, a)
                    sub_fused[port] = pid_of[sock]
                expand(sub, f"{prefix}/{tr['name']}", sub_fused)

        expand(self.top, self.top["name"], {})
        self.places = list(places.values())
        index = {p["gid"]: i for i, p in enumerate(self.places)}
        for t in trans:
            try:
                t["guard"] = compile_guard(t["guard_text"])
                t["in"] = [(index[g], parse_pattern(e)) for g, e in t["in_raw"]]
                t["out"] = [(index[g], compile_output(e)) for g, e in t["out_raw"]]
            except UnsupportedNet as exc:
                raise UnsupportedNet(f"{t['page']}/{t['name']}: {exc}")
            if not t["in"]:
                raise UnsupportedNet(f"{t['page']}/{t['name']} has no input arc "
                                     "(it would be enabled forever)")
            t["in_vars"] = self._pattern_vars([p for _, p in t["in"]])
            self.transitions.append(t)

    # Method to collect the variables a list of patterns binds.
    @staticmethod
    def _pattern_vars(patterns) -> set:
        """Variables bound by a list of input patterns.

        Args:
            patterns (list): Pattern trees.

        Returns:
            set: Variable names.
        """
        out = set()

        def walk(p):
            if p[0] == "var":
                out.add(p[1])
            elif p[0] == "tuple":
                for q in p[1]:
                    walk(q)

        for p in patterns:
            walk(p)
        return out

    # Method to build the initial marking, optionally with another sample.
    def initial_marking(self, sample=None) -> tuple:
        """Assemble the initial marking from the places' initial inscriptions.

        Args:
            sample (sequence, optional): Replacement value for the Input token.

        Returns:
            tuple: The initial marking, one sorted token tuple per place.
        """
        marking = []
        for pl in self.places:
            tokens = parse_initial_marking(pl["initmark"])
            if sample is not None and pl["name"] == "Input" and \
                    pl["page"] == self.top["name"]:
                tokens = [tuple(float(v) for v in sample)]
            marking.append(tuple(sorted(tokens)))
        return tuple(marking)

    # Method to list every enabled binding element of a marking.
    def enabled(self, marking: tuple) -> List[Tuple[int, Tuple, dict]]:
        """Enumerate the binding elements enabled in a marking.

        Args:
            marking (tuple): A marking.

        Returns:
            list: ``(transition index, binding key, binding)`` triples.
        """
        candidates = set()
        for pidx, tokens in enumerate(marking):
            if tokens:
                candidates.update(self.consumers.get(pidx, ()))
        found = []
        for ti in sorted(candidates):
            t = self.transitions[ti]
            if any(not marking[p] for p, _ in t["in"]):
                continue
            for env in self._bindings(marking, t):
                key = tuple(sorted(env.items()))
                found.append((ti, key, env))
        return found

    # Method to enumerate the bindings of one transition in a marking.
    def _bindings(self, marking, t) -> List[dict]:
        """Enumerate the bindings of a transition, honouring its guard.

        Args:
            marking (tuple): The marking.
            t (dict): The transition record.

        Returns:
            list: Distinct complete bindings.
        """
        results, seen = [], set()
        inputs = t["in"]

        def rec(i, env, used):
            if i == len(inputs):
                if t["guard"] is not None:
                    scope = dict(self.env)
                    scope.update(env)
                    try:
                        ok = eval(t["guard"], scope)  # noqa: S307
                    except NameError as exc:
                        raise UnsupportedNet(f"{t['name']}: guard uses an unbound "
                                             f"variable ({exc})")
                    except Exception as exc:
                        raise UnsupportedNet(f"{t['name']}: guard could not be "
                                             f"evaluated ({exc})")
                    if not ok:
                        return
                key = tuple(sorted(env.items()))
                if key not in seen:
                    seen.add(key)
                    results.append(env)
                return
            pidx, pattern = inputs[i]
            available = collections.Counter(marking[pidx])
            available.subtract(used.get(pidx, {}))
            for token in sorted(set(marking[pidx])):
                if available[token] <= 0:
                    continue
                env2 = _match(pattern, token, env)
                if env2 is None:
                    continue
                used2 = {k: collections.Counter(v) for k, v in used.items()}
                used2.setdefault(pidx, collections.Counter())[token] += 1
                rec(i + 1, env2, used2)

        rec(0, {}, {})
        return results

    # Method to follow one maximal occurrence sequence.
    def run(self, marking: Optional[tuple] = None, limit: int = 100_000) -> tuple:
        """Fire the first enabled binding element until none is enabled.

        Used for conformance testing on many inputs, where building the whole
        occurrence graph per input would be wasteful. It gives the terminal
        marking of *one* maximal run; that this is *the* result of the net
        rests on confluence, which the model-checking step establishes.

        Args:
            marking (tuple, optional): Start marking (initial marking if None).
            limit (int): Maximum number of occurrences.

        Returns:
            tuple: The marking where the run stopped.
        """
        m = self.initial if marking is None else marking
        for _ in range(limit):
            options = self.enabled(m)
            if not options:
                return m
            ti, _key, env = options[0]
            m = self.fire(m, ti, env)
        raise UnsupportedNet(f"no dead marking within {limit} occurrences")

    # Method to fire one binding element.
    def fire(self, marking: tuple, ti: int, env: dict) -> tuple:
        """Compute the marking reached by an occurrence.

        Args:
            marking (tuple): Current marking.
            ti (int): Transition index.
            env (dict): The binding.

        Returns:
            tuple: The successor marking.
        """
        t = self.transitions[ti]
        new = [list(tokens) for tokens in marking]
        for pidx, pattern in t["in"]:
            new[pidx].remove(_instantiate(pattern, env))
        scope = dict(self.env)
        scope.update(env)
        for pidx, terms in t["out"]:
            for count, code in terms:
                try:
                    value = _canon(eval(code, scope))  # noqa: S307
                except NameError as exc:
                    raise UnsupportedNet(f"{t['name']}: output arc uses an "
                                         f"unbound variable ({exc})")
                new[pidx].extend([value] * count)
        return tuple(tuple(sorted(tokens)) for tokens in new)


# ---------------------------------------------------------------------------
# The occurrence graph
# ---------------------------------------------------------------------------

# Class holding a computed occurrence graph and its SCC decomposition.
class StateSpace:
    """The occurrence graph of a net and the derived structures.

    Attributes:
        net (ColouredNet): The explored net.
        markings (list): Marking of each node; node 0 is the initial marking.
        succ (list): Per node, ``(transition index, binding key, target)``.
        pred (list): Per node, ``(transition index, source)``.
        parent (list): BFS tree parent ``(source, transition index)`` per node,
            used to print counterexample traces.
        dead (list): Nodes without successors.
        complete (bool): False when exploration stopped at the node budget.
        scc (list): SCC index of every node.
        terminal_sccs (list): SCC indices with no arc leaving them.
        home (list): Home markings (nodes of the unique terminal SCC).
        elapsed (float): Generation time in seconds.
    """

    # Method to hold the graph data.
    def __init__(self, net, markings, succ, complete, elapsed, parent):
        """Create the state space and compute its SCC decomposition.

        Args:
            net (ColouredNet): The net.
            markings (list): Node markings.
            succ (list): Successor lists.
            complete (bool): Whether exploration finished.
            elapsed (float): Generation time.
            parent (list): BFS parents.
        """
        self.net = net
        self.markings = markings
        self.succ = succ
        self.complete = complete
        self.elapsed = elapsed
        self.parent = parent
        self.pred: List[List[Tuple[int, int]]] = [[] for _ in markings]
        for src, out in enumerate(succ):
            for ti, _key, dst in out:
                self.pred[dst].append((ti, src))
        self.dead = [n for n, out in enumerate(succ) if not out]
        self._compute_sccs()

    # Property returning the number of nodes.
    @property
    def n_nodes(self) -> int:
        """Number of reachable markings.

        Returns:
            int: Node count.
        """
        return len(self.markings)

    # Property returning the number of arcs.
    @property
    def n_arcs(self) -> int:
        """Number of binding-element occurrences in the graph.

        Returns:
            int: Arc count.
        """
        return sum(len(out) for out in self.succ)

    # Method to compute strongly connected components with Tarjan's algorithm.
    def _compute_sccs(self):
        """Iterative Tarjan decomposition, then terminal SCCs and home markings.

        A home marking exists exactly when the SCC graph has a single terminal
        component, and then every marking of that component is a home marking
        (Jensen & Kristensen, 2009). This is also how CPN Tools computes
        ``ListHomeMarkings``.
        """
        n = len(self.markings)
        index = [-1] * n
        low = [0] * n
        on_stack = [False] * n
        stack: List[int] = []
        comp = [-1] * n
        counter = 0
        n_comp = 0
        for root in range(n):
            if index[root] != -1:
                continue
            work = [(root, 0)]
            while work:
                v, i = work.pop()
                if i == 0:
                    index[v] = low[v] = counter
                    counter += 1
                    stack.append(v)
                    on_stack[v] = True
                recurse = False
                out = self.succ[v]
                while i < len(out):
                    w = out[i][2]
                    i += 1
                    if index[w] == -1:
                        work.append((v, i))
                        work.append((w, 0))
                        recurse = True
                        break
                    if on_stack[w]:
                        low[v] = min(low[v], index[w])
                if recurse:
                    continue
                if low[v] == index[v]:
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        comp[w] = n_comp
                        if w == v:
                            break
                    n_comp += 1
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[v])
        self.scc = comp
        self.n_sccs = n_comp
        leaves = set(range(n_comp))
        inter_arcs = 0
        for src, out in enumerate(self.succ):
            for _ti, _key, dst in out:
                if comp[src] != comp[dst]:
                    leaves.discard(comp[src])
                    inter_arcs += 1
        self.n_scc_arcs = inter_arcs
        self.terminal_sccs = sorted(leaves)
        if len(self.terminal_sccs) == 1:
            c = self.terminal_sccs[0]
            self.home = [v for v in range(n) if comp[v] == c]
        else:
            self.home = []

    # Method to count the tokens of one place in one node.
    def tokens(self, node: int, pidx: int) -> tuple:
        """Tokens held by a place in a marking.

        Args:
            node (int): Node id.
            pidx (int): Place index.

        Returns:
            tuple: The sorted token values.
        """
        return self.markings[node][pidx]

    # Method to print the shortest firing sequence reaching a node.
    def trace(self, node: int, limit: int = 12) -> str:
        """Shortest occurrence sequence from the initial marking to a node.

        Args:
            node (int): Target node.
            limit (int): Maximum number of steps to print.

        Returns:
            str: ``t1 -> t2 -> ...``.
        """
        steps = []
        while node != 0 and self.parent[node] is not None:
            src, ti = self.parent[node]
            steps.append(self.net.transitions[ti]["name"])
            node = src
        steps.reverse()
        if not steps:
            return "(initial marking)"
        if len(steps) > limit:
            steps = steps[:3] + [f"... {len(steps) - 6} more ..."] + steps[-3:]
        return " -> ".join(steps)


# Function to build the occurrence graph of a net by breadth-first search.
def build_state_space(net: ColouredNet, max_nodes: int = 200_000,
                      initial: Optional[tuple] = None) -> StateSpace:
    """Explore every reachable marking.

    Args:
        net (ColouredNet): The net to explore.
        max_nodes (int): Stop after this many markings; the result is then
            flagged as incomplete and must not be used for verdicts.
        initial (tuple, optional): Start from this marking instead of the
            net's own initial marking (another input sample).

    Returns:
        StateSpace: The (possibly partial) occurrence graph.
    """
    t0 = time.perf_counter()
    start = net.initial if initial is None else initial
    nodes = {start: 0}
    markings = [start]
    succ: List[List[Tuple[int, Tuple, int]]] = [[]]
    parent: List[Optional[Tuple[int, int]]] = [None]
    queue = collections.deque([0])
    complete = True
    while queue:
        src = queue.popleft()
        marking = markings[src]
        for ti, key, env in net.enabled(marking):
            nxt = net.fire(marking, ti, env)
            dst = nodes.get(nxt)
            if dst is None:
                if len(markings) >= max_nodes:
                    complete = False
                    continue
                dst = len(markings)
                nodes[nxt] = dst
                markings.append(nxt)
                succ.append([])
                parent.append((src, ti))
                queue.append(dst)
            succ[src].append((ti, key, dst))
    return StateSpace(net, markings, succ, complete,
                      time.perf_counter() - t0, parent)
