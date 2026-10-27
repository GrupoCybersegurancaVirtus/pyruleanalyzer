"""A reference evaluator for generated ``.cpn`` files.

Structural validation (DTD, ids, port/socket pairs) says the net is a
well-formed CPN Tools document; it says nothing about *what the net computes*.
This module closes that gap without requiring CPN Tools: it reads the exported
``.cpn``, reconstructs the function the net denotes straight from the guards
and arc expressions found in the XML, and evaluates it on concrete samples.

The point is that nothing here is taken from the exporter's Python objects --
only from the file that CPN Tools would open. A mismatch against
``RuleClassifier`` therefore means the net and the classifier really do
disagree, which is the "prediction consistency" property.

Supported: single-page Decision Tree, Random Forest (soft voting) and GBDT
(binary and multiclass).

:mod:`pyruleanalyzer.model_checker` builds on the parser exposed here: it
flattens the hierarchy read by :class:`CPNNet` into a place/transition net and
explores its reachability graph, so both the *what the net computes* question
and the *how the net behaves* question are answered from the same file.

The parsing itself is :class:`cpncheck.net.CPNNet`, which knows nothing about
what a net is for. :class:`CPNNet` here adds back the four methods that do --
which model family a net encodes, where its sample is, what its leaves emit --
because those are facts about *these* nets, not about Coloured Petri Nets.
"""

import ast
import operator
import re

from cpncheck.net import CPNNet as _ParsedNet


# ---------------------------------------------------------------------------
# SML -> Python
# ---------------------------------------------------------------------------

# Function to translate the SML subset emitted by the exporter into Python.
def sml_to_python(expr):
    """Translate the SML subset the exporter emits into a Python expression.

    Handles record field access (``#f0 x``), the boolean operators, and SML's
    ``~`` negation prefix.

    Args:
        expr (str): SML source as written in a guard or arc inscription.

    Returns:
        str: An equivalent Python expression over a dict named ``x``.
    """
    out = expr.strip()
    # CPN Tools writes guards wrapped in brackets: [ ... ]. Left in place they
    # would parse as a Python list literal, which is always truthy.
    if out.startswith("[") and out.endswith("]"):
        out = out[1:-1].strip()
    out = re.sub(r"#(f\d+)\s+x", r"x['\1']", out)
    out = out.replace("andalso", " and ").replace("orelse", " or ")
    # ~1.5 / ~2.0E~3 -> -1.5 / -2.0E-3 (SML negation, also inside exponents)
    out = re.sub(r"~(?=[\d.])", "-", out)
    out = out.replace("not (", " not (")
    out = re.sub(r"\btrue\b", "True", out)
    out = re.sub(r"\bfalse\b", "False", out)
    return out or "True"


# ---------------------------------------------------------------------------
# Restricted evaluation
# ---------------------------------------------------------------------------
#
# Guards and arc inscriptions come from a ``.cpn`` file, which may have been
# written by anyone. ``eval`` with empty ``__builtins__`` is no sandbox --
# ``().__class__.__base__.__subclasses__()`` reaches arbitrary code -- so the
# expressions are parsed with :mod:`ast` and only the constructs the exporter
# writes are interpreted: constants, names bound in the environment, item
# access (``x['f3']``), comparisons, ``and``/``or``/``not``, unary minus,
# arithmetic on numbers, list/tuple literals and calls to functions found in
# the environment. Attribute access, private or dunder names, lambdas,
# comprehensions and every other construct are rejected before anything runs.

_COMPARE = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne,
    ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
}
_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_ALLOWED = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.Subscript,
            ast.Compare, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not,
            ast.BinOp, ast.List, ast.Tuple, ast.Call,
            *_COMPARE, *_BINARY, *_UNARY)
_CONSTANTS = (bool, int, float, str)
_NUMBERS = (int, float)


# Function to check that an expression tree uses only the whitelisted nodes.
def _validate(tree):
    """Reject any construct outside the whitelist, before evaluating.

    The whole tree is checked, including branches that short-circuiting
    would never reach.

    Args:
        tree (ast.AST): The parsed expression.

    Raises:
        ValueError: Naming the first construct that is not allowed.
    """
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"disallowed construct in net expression: "
                             f"{type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError(f"disallowed name in net expression: {node.id!r}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, _CONSTANTS):
            raise ValueError(f"disallowed constant in net expression: "
                             f"{node.value!r}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError(f"disallowed call of {type(node.func).__name__} "
                                 "in net expression: only functions named in "
                                 "the environment may be called")
            if node.keywords:
                raise ValueError("disallowed keyword arguments in net expression")


# Function to interpret a validated expression tree.
def _interpret(node, env):
    """Evaluate a node of a tree that passed :func:`_validate`.

    Args:
        node (ast.AST): The node.
        env (dict): Names visible to the expression.

    Returns:
        The value of the node.

    Raises:
        ValueError: On an unbound name or an operation on the wrong types.
    """
    if isinstance(node, ast.Expression):
        return _interpret(node.body, env)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(f"unbound name in net expression: {node.id!r}")
        return env[node.id]
    if isinstance(node, ast.Subscript):
        value = _interpret(node.value, env)
        key = _interpret(node.slice, env)
        if not isinstance(value, (dict, list, tuple)):
            raise ValueError(f"cannot index a {type(value).__name__}")
        return value[key]
    if isinstance(node, ast.Compare):
        left = _interpret(node.left, env)
        for op, comp in zip(node.ops, node.comparators):
            right = _interpret(comp, env)
            if not _COMPARE[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.BoolOp):
        want = isinstance(node.op, ast.Or)
        value = None
        for sub in node.values:
            value = _interpret(sub, env)
            if bool(value) is want:
                return value
        return value
    if isinstance(node, ast.UnaryOp):
        operand = _interpret(node.operand, env)
        if isinstance(node.op, ast.Not):
            return not operand
        if not isinstance(operand, _NUMBERS):
            raise ValueError(f"unary {type(node.op).__name__} on a "
                             f"{type(operand).__name__}")
        return _UNARY[type(node.op)](operand)
    if isinstance(node, ast.BinOp):
        left = _interpret(node.left, env)
        right = _interpret(node.right, env)
        if not (isinstance(left, _NUMBERS) and isinstance(right, _NUMBERS)):
            raise ValueError(f"{type(node.op).__name__} on "
                             f"{type(left).__name__} and {type(right).__name__}")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, (ast.List, ast.Tuple)):
        items = [_interpret(e, env) for e in node.elts]
        return items if isinstance(node, ast.List) else tuple(items)
    if isinstance(node, ast.Call):
        func = env.get(node.func.id)
        if not callable(func):
            raise ValueError(f"{node.func.id!r} is not a function of the net")
        return func(*[_interpret(a, env) for a in node.args])
    raise ValueError(f"disallowed construct in net expression: "
                     f"{type(node).__name__}")


# Function to evaluate untrusted Python expression text safely.
def safe_eval(expr, env=None):
    """Evaluate an expression read from a net without running arbitrary code.

    A drop-in for ``eval(expr, {"__builtins__": {}}, env)`` restricted to the
    constructs listed above.

    Args:
        expr (str): Python expression text (for a guard, the output of
            :func:`sml_to_python`).
        env (dict, optional): Names the expression may read or call.

    Returns:
        The value of the expression.

    Raises:
        TypeError: If ``expr`` is not a string.
        ValueError: If the expression uses a construct outside the whitelist,
            reads an unbound name, or does not parse.
    """
    if not isinstance(expr, str):
        raise TypeError(f"safe_eval() needs a string, got {type(expr).__name__}")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"cannot parse net expression {expr!r}: {exc.msg}")
    _validate(tree)
    return _interpret(tree, dict(env or {}))


# Function to evaluate a translated SML expression against a sample.
def _eval(expr, x):
    """Evaluate a translated SML expression against a sample.

    Args:
        expr (str): SML expression.
        x (dict): Sample keyed by record field name (``f0``, ``f1``, ...).

    Returns:
        The Python value of the expression.

    Raises:
        ValueError: If the expression is not one the exporter writes.
    """
    return safe_eval(sml_to_python(expr), {"x": x})


# ---------------------------------------------------------------------------
# .cpn parsing, plus what these nets mean
# ---------------------------------------------------------------------------

# Class adding the tree-ensemble reading to the generic .cpn parser.
class CPNNet(_ParsedNet):
    """A parsed ``.cpn``, read as a decision tree, a forest or a boosting model.

    Parsing comes from :class:`cpncheck.net.CPNNet`; everything below is about
    the nets this package generates.
    """

    # Method to detect which model family the net encodes.
    def family(self):
        """Identify the model family from the shape of the top page.

        Returns:
            str: ``"Decision Tree"``, ``"Random Forest"`` or
            ``"Gradient Boosting Decision Trees"``.
        """
        top = self.top_page()
        subst = [tr for tr in top["transitions"].values() if tr["subpage"]]
        if not subst:
            return "Decision Tree"
        if all(tr["name"].startswith("Tree_") for tr in subst):
            return "Random Forest"
        return "Gradient Boosting Decision Trees"

    # Method to read the sample encoded in the input place's initial marking.
    def sample_from_initial_marking(self):
        """Read the sample encoded in the input place's initial marking.

        Returns:
            dict: Sample keyed by record field name, or None if absent.
        """
        for page in self.pages.values():
            for pl in page["places"].values():
                if pl["initmark"] and pl["name"] in ("Input",):
                    fields = re.findall(r"(f\d+)\s*=\s*\(?(~?[\d.eE~+-]+)\)?",
                                        pl["initmark"])
                    return {k: float(v.replace("~", "-")) for k, v in fields}
        return None

    # Method to list the guard/output pairs of every leaf transition on a page.
    def leaf_outputs(self, page):
        """Guard/output pairs of every leaf transition on a page.

        Args:
            page (dict): A page entry from `self.pages`.

        Returns:
            list: (transition name, guard string, output arc expression).
        """
        out = []
        for tid, tr in page["transitions"].items():
            if tr["subpage"]:
                continue
            outgoing = [a for a in page["arcs"]
                        if a["trans"] == tid and a["orient"] == "TtoP"]
            if not outgoing:
                continue
            out.append((tr["name"], tr["guard"] or "true", outgoing[0]["expr"]))
        return out

    # Method to evaluate one base-learner subpage by picking the enabled leaf.
    def eval_subpage(self, page_name, x):
        """Evaluate one base-learner subpage: pick the enabled leaf.

        Args:
            page_name (str): Name of the subpage.
            x (dict): Sample keyed by record field name.

        Returns:
            The leaf's output expression, or None if no leaf is enabled.

        Raises:
            AssertionError: If more than one leaf is enabled, i.e. the guards
                are not mutually exclusive for this sample.
        """
        enabled = [(n, e) for n, g, e in self.leaf_outputs(self.pages[page_name])
                   if _eval(g, x)]
        assert len(enabled) <= 1, (
            f"{page_name}: {len(enabled)} leaves enabled at once "
            f"({[n for n, _ in enabled]}) -- guards are not mutually exclusive"
        )
        return enabled[0][1] if enabled else None


# ---------------------------------------------------------------------------
# Per-family evaluation
# ---------------------------------------------------------------------------

# Function to parse an SML real list literal into Python floats.
def _parse_prob_list(expr):
    """Parse an SML ``real list`` literal into Python floats.

    Args:
        expr (str): For example ``[0.0, 1.0]``.

    Returns:
        list[float]: The parsed vector.
    """
    return [float(v.replace("~", "-"))
            for v in re.findall(r"~?[\d.]+(?:[eE]~?[\d+-]+)?", expr)]


# Function to compute an argmax with NumPy tie-breaking.
def _argmax(vec):
    """Argmax with NumPy tie-breaking (lowest index wins).

    Args:
        vec (list[float]): Values to compare.

    Returns:
        int: Index of the maximum.
    """
    best, bi = vec[0], 0
    for i, v in enumerate(vec[1:], start=1):
        if v > best:
            best, bi = v, i
    return bi


# Function to read the label table of a forest's decision.
def _rf_label_table(net):
    """Labels of the distribution columns, from the net's ``decide`` function.

    The exporter writes ``List.nth ([l0, l1, ...], argmax xs)`` when the
    labels are not 0..K-1, and a bare ``argmax xs`` otherwise.

    Args:
        net (CPNNet): The parsed net.

    Returns:
        list[int] | None: The labels, or None when columns are the labels.
    """
    text = "".join(net.cpnet.itertext()) if net.cpnet is not None else ""
    m = re.search(r"List\.nth\s*\(\[([^\]]*)\],\s*argmax xs\)", text)
    if not m:
        return None
    return [int(v.strip().replace("~", "-")) for v in m.group(1).split(",")]


# Function to compute the class the net denotes for one sample.
def evaluate(net, x, default_class=0):
    """Compute the class the net denotes for one sample.

    Args:
        net (CPNNet): The parsed net.
        x (dict): Sample keyed by record field name (``f0``, ``f1``, ...).
        default_class (int): Class used when every base learner abstains.

    Returns:
        int: The predicted class label.

    Raises:
        AssertionError: If the net's guards are inconsistent for this sample.
        ValueError: If the net does not match any supported topology, or the
            sample holds a missing value (NaN): the net's input is a record
            of REALs, which models real-valued inputs only. Every guard
            would be false on NaN, so the answer would silently differ from
            the classifier's, which routes NaN per split.
    """
    missing = [k for k, v in (x or {}).items()
               if isinstance(v, float) and v != v]
    if missing:
        raise ValueError(
            f"Sample has missing values (NaN) in {sorted(missing)[:10]}: the "
            "CPN models real-valued inputs only."
        )
    top = next((p for p in net.pages.values()
                if any(pl["name"] == "Prediction" for pl in p["places"].values())),
               None)
    if top is None:
        raise ValueError("no page with a Prediction place")

    # --- Decision Tree: leaves live on the top page itself ---------------
    if not any(t["subpage"] for t in top["transitions"].values()):
        enabled = [(n, e) for n, g, e in net.leaf_outputs(top) if _eval(g, x)]
        assert len(enabled) <= 1, (
            f"{len(enabled)} leaves enabled at once ({[n for n, _ in enabled]})"
        )
        return int(enabled[0][1]) if enabled else int(default_class)

    subst = {tid: tr for tid, tr in top["transitions"].items() if tr["subpage"]}

    # --- Random Forest: one PROB vector per tree, summed, then argmax ----
    if all(tr["name"].startswith("Tree_") for tr in subst.values()):
        total = None
        for tr in subst.values():
            vec_expr = net.eval_subpage(net.page_by_id[tr["subpage"]]["name"], x)
            vec = _parse_prob_list(vec_expr) if vec_expr else None
            if vec is None:
                continue
            total = vec if total is None else [a + b for a, b in zip(total, vec)]
        if total is None or all(v == 0.0 for v in total):
            return int(default_class)      # every tree abstained
        labels = _rf_label_table(net)
        best = _argmax(total)
        return labels[best] if labels else best

    # --- GBDT: per-class channels, additive scores, then threshold/argmax -
    scores = {}
    for tr in subst.values():
        ch = net.page_by_id[tr["subpage"]]
        init = next((a["expr"] for a in ch["arcs"]
                     if a["orient"] == "TtoP"
                     and ch["transitions"][a["trans"]]["name"] == "Init"), None)
        m = re.search(r",\s*\(?(~?[\d.eE~+-]+)\)?\s*\)$", init or "")
        score = float(m.group(1).replace("~", "-")) if m else 0.0

        for stage in sorted(t["subpage"] for t in ch["transitions"].values()
                            if t["subpage"]):
            val = net.eval_subpage(net.page_by_id[stage]["name"], x)
            if val is not None:
                score += float(val.strip("()").replace("~", "-"))
        scores[tr["name"]] = score

    # The decision transitions carry the threshold / argmax policy as guards.
    decides = [(t["name"], t["guard"]) for t in top["transitions"].values()
               if t["name"].startswith("Decide_")]
    env = {}
    if len(scores) == 1:
        env["sc"] = next(iter(scores.values()))
    else:
        for name, sc in scores.items():
            env[f"sc{name.rsplit('_', 1)[-1]}"] = sc

    fired = [n for n, g in decides
             if safe_eval(sml_to_python(g or "true"), env)]
    assert len(fired) == 1, (
        f"{len(fired)} Decide transitions enabled ({fired}) -- the decision "
        "guards are not a partition"
    )
    return int(fired[0].rsplit("_", 1)[-1])
