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


# Function to evaluate a translated SML expression against a sample.
def _eval(expr, x):
    """Evaluate a translated SML expression against a sample.

    Args:
        expr (str): SML expression.
        x (dict): Sample keyed by record field name (``f0``, ``f1``, ...).

    Returns:
        The Python value of the expression.
    """
    return eval(sml_to_python(expr), {"__builtins__": {}}, {"x": x})  # noqa: S307


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
        ValueError: If the net does not match any supported topology.
    """
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
        return _argmax(total)

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
             if eval(sml_to_python(g or "true"), {"__builtins__": {}}, env)]  # noqa: S307
    assert len(fired) == 1, (
        f"{len(fired)} Decide transitions enabled ({fired}) -- the decision "
        "guards are not a partition"
    )
    return int(fired[0].rsplit("_", 1)[-1])
