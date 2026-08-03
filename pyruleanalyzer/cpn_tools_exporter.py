"""CPN Tools (.cpn) HCPN exporter for pyRuleAnalyzer.

This module converts a :class:`RuleClassifier` rule set (either the initial,
unrefined rules or the final, post-refinement rules) into a **Hierarchical
Coloured Petri Net** serialised in the native CPN Tools 4.x XML format
(``format="6"``), so the resulting ``.cpn`` file can be opened, visualised,
simulated and verified directly in `CPN Tools <https://cpntools.org>`_.

The mapping implements the formalisation proved in the accompanying article
*"Coloured Petri Nets-Based Modeling and Validation of Gradient Boosting
Decision Trees for Secure Industrial Systems"*:

* **Theorem 1 (DT -> CPN_DT).** Every root-to-leaf path of a decision tree
  becomes a single transition whose guard is the Boolean translation of the
  path conditions. Because the split conditions are mutually exclusive and
  exhaustive, exactly one transition is enabled for any input sample. Each
  tree is emitted as its own CPN *subpage* with an input port ``P_in``
  (carrying the sample) and an output port ``P_out`` (carrying the leaf value).

* **Theorem 2 (GBDT -> HCPN_GBDT).** Each class channel is emitted as a page
  that evaluates the ``M`` boosting stages sequentially and accumulates the
  scaled leaf values ``s + eta * v_m`` starting from the initial estimator
  ``s_0``. Every stage is a *substitution transition* bound to the
  corresponding DT subpage. Sequential ordering (the counter ``kappa`` in the
  theorem) is realised structurally by the accumulator place-chain
  ``Acc0 -> Acc1 -> ... -> AccM``, which guarantees strict mutual exclusion and
  complete accumulation.

The top-level page distributes the input sample to every channel and applies
the decision policy: sigmoid threshold (``s_M >= 0``) for binary GBDT, and the
NumPy-compatible ``argmax`` tie-breaking (lowest index wins ties) for
multiclass GBDT. Decision Tree models are exported as a single CPN_DT page and
Random Forest models as a top page with one DT subpage per tree (each emitting
its leaf's class-probability vector) plus a soft-voting decision transition that
sums the vectors and takes the argmax -- reproducing scikit-learn's ``predict``.

The generator depends only on the Python standard library.
"""

from __future__ import annotations

import re
import math
import xml.etree.ElementTree as ET
from collections import defaultdict, OrderedDict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Stable page names
# ---------------------------------------------------------------------------
# Page names are fixed (not derived from the model name) so that model-checking
# queries are portable across generated models: the same ASK-CTL code works for
# Decision Tree, GBDT and Random Forest without editing identifiers. In CPN
# Tools a marking is read as ``Mark.<Page>'<place> <inst> <node>``, hence:
#
#   Top                  top-level page  -> Mark.Top'Input, Mark.Top'Prediction
#   Channel_<k>          GBDT class channel k
#   Tree_c<k>_s<m>       GBDT tree of class k, boosting stage m
#   Tree_<i>             Random Forest tree i
TOP_PAGE = "Top"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

class _IdGen:
    """Generates unique ``IDxxxxxxx`` identifiers (CPN Tools convention)."""

    def __init__(self) -> None:
        self._n = 1000000

    def __call__(self) -> str:
        self._n += 1
        return f"ID{self._n}"


def sml_real(value) -> str:
    """Format a Python float as a valid Standard ML (CPN ML) real literal.

    Standard ML writes unary minus as ``~`` and the exponent sign as ``~`` too,
    and every real literal must contain a decimal point. Scientific notation
    from Python (e.g. ``1e-05``) is rewritten to the SML form (``1.0E~5``).
    """
    v = float(value)
    if v != v or v in (float("inf"), float("-inf")):
        # Guard against NaN/inf leaking into the model.
        v = 0.0
    s = repr(v)
    neg = s.startswith("-")
    if neg:
        s = s[1:]
    if "e" in s or "E" in s:
        mant, exp = re.split("[eE]", s)
        if "." not in mant:
            mant += ".0"
        exp = exp.lstrip("+")
        if exp.startswith("-"):
            exp = "~" + exp[1:]
        s = f"{mant}E{exp}"
    elif "." not in s:
        s += ".0"
    return ("~" + s) if neg else s


def _ml_val(value) -> str:
    """A real literal safe to embed as a value inside record/tuple literals.

    CPN ML writes negative reals with a leading ``~`` (e.g. ``~1.5``). Inside a
    record value literal (``{f0=~1.5}``) or a tuple, the bare ``~`` form fails to
    parse, so negative values are wrapped in parentheses (``(~1.5)``).
    """
    s = sml_real(value)
    return f"({s})" if s.startswith("~") else s


def _sanitize(name: str, used: set, fallback: str = "N") -> str:
    """Turn an arbitrary rule/tree name into a unique CPN identifier."""
    clean = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    if not clean or not (clean[0].isalpha() or clean[0] == "_"):
        clean = f"{fallback}_{clean}"
    base = clean
    i = 1
    while clean in used:
        clean = f"{base}_{i}"
        i += 1
    used.add(clean)
    return clean


def _int_label(label) -> int:
    """Best-effort conversion of a class label to an int for CPN INT tokens."""
    try:
        return int(str(label).replace("Class", "").strip())
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# In-memory net representation
# ---------------------------------------------------------------------------

class _Place:
    __slots__ = ("id", "name", "colorset", "x", "y", "initmark", "port_type")

    def __init__(self, pid, name, colorset, x, y, initmark=None, port_type=None):
        self.id = pid
        self.name = name
        self.colorset = colorset
        self.x = x
        self.y = y
        self.initmark = initmark
        self.port_type = port_type  # 'In' | 'Out' | 'I/O' | None


class _Subst:
    __slots__ = ("subpage_id", "subpage_name", "pairs")

    def __init__(self, subpage_id, subpage_name, pairs):
        self.subpage_id = subpage_id
        self.subpage_name = subpage_name
        self.pairs = pairs  # list of (socket_place_id, port_place_id)


class _Trans:
    __slots__ = ("id", "name", "x", "y", "guard", "subst")

    def __init__(self, tid, name, x, y, guard=None, subst=None):
        self.id = tid
        self.name = name
        self.x = x
        self.y = y
        self.guard = guard
        self.subst = subst


class _Arc:
    __slots__ = ("id", "orient", "trans_id", "place_id", "expr")

    def __init__(self, aid, orient, trans_id, place_id, expr):
        self.id = aid
        self.orient = orient  # 'PtoT' (place->trans) | 'TtoP' (trans->place)
        self.trans_id = trans_id
        self.place_id = place_id
        self.expr = expr


class _Page:
    __slots__ = ("id", "name", "places", "transitions", "arcs", "_used_names")

    def __init__(self, pid, name):
        self.id = pid
        self.name = name
        self.places: List[_Place] = []
        self.transitions: List[_Trans] = []
        self.arcs: List[_Arc] = []
        self._used_names: set = set()


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

class CPNToolsExporter:
    """Builds a CPN Tools ``.cpn`` HCPN from a RuleClassifier rule set."""

    def __init__(self, classifier, rules, feature_names=None, sample=None,
                 is_final=False, model_name="pyRuleAnalyzer"):
        """Create an exporter.

        Args:
            classifier: The :class:`RuleClassifier` instance (for metadata such
                as algorithm type, GBDT init scores, class labels).
            rules: The list of :class:`Rule` objects to export (typically
                ``classifier.initial_rules`` or ``classifier.final_rules``).
            feature_names: Ordered feature names. If ``None`` they are taken
                from ``classifier._array_feature_names`` or inferred from rules.
            sample: Optional iterable / dict giving an example input sample used
                as the initial marking of the top-level input place (so the net
                can be simulated immediately). Defaults to all zeros.
            is_final: Whether ``rules`` is the refined rule set. When ``True``
                each DT subpage receives a catch-all fallback transition so the
                net cannot deadlock on samples that match no remaining leaf.
            model_name: Free-form label for the model. Kept for API
                compatibility; page names are fixed (see ``TOP_PAGE``) so that
                model-checking queries stay portable across models.
        """
        self.clf = classifier
        self.rules = list(rules)
        self.is_final = is_final
        self.model_name = model_name
        self.idgen = _IdGen()

        self.pages: List[_Page] = []
        # colorset registry: name -> (kind, payload, layout_string)
        self.colorsets: "OrderedDict[str, Tuple[str, object, str]]" = OrderedDict()
        self.vars: "OrderedDict[str, str]" = OrderedDict()   # var name -> colorset
        self.mls: List[str] = []

        # ---- Feature ordering / record fields --------------------------
        feats = list(feature_names) if feature_names else []
        if not feats:
            feats = list(getattr(classifier, "_array_feature_names", []) or [])
        # Always union in every feature referenced by the rules, so the SAMPLE
        # record can never miss a field that a guard references (this also
        # covers an incomplete caller-supplied feature_names list).
        seen_set = set(feats)
        for r in self.rules:
            for var, _, _ in getattr(r, "parsed_conditions", []) or []:
                if var not in seen_set:
                    seen_set.add(var)
                    feats.append(var)
        if not feats:
            feats = ["f0"]
        self.feature_names = feats
        self.field = {name: f"f{i}" for i, name in enumerate(feats)}

        # Number of classes (for Random Forest soft-voting probability vectors).
        self.n_classes = int(getattr(classifier, "num_classes", 0) or 0)
        if self.n_classes < 2:
            k = 0
            for r in self.rules:
                d = getattr(r, "class_distribution", None)
                if d is not None:
                    k = max(k, len(d))
                else:
                    k = max(k, _int_label(r.class_) + 1)
            self.n_classes = max(k, 2)

        # ---- Sample for the initial marking ----------------------------
        self.sample_values = self._resolve_sample(sample)

        # ---- Standard + SAMPLE color sets ------------------------------
        self._declare_standard_colorsets()
        self._declare_var("x", "SAMPLE")

    # ------------------------------------------------------------------
    # Declarations
    # ------------------------------------------------------------------

    def _declare_standard_colorsets(self):
        # INT and REAL come from the canonical "Standard declarations" block
        # emitted by _emit_standard_blocks (exactly as every CPN Tools net has
        # them). Here we declare only our own color sets.
        # SAMPLE: one REAL field per feature.
        fields = [(self.field[n], "REAL") for n in self.feature_names]
        layout = "colset SAMPLE = record " + \
                 " * ".join(f"{fn}:REAL" for fn, _ in fields) + ";"
        self.colorsets["SAMPLE"] = ("record", fields, layout)

    def _declare_product_ss(self):
        if "SS" not in self.colorsets:
            self.colorsets["SS"] = (
                "product", ["SAMPLE", "REAL"],
                "colset SS = product SAMPLE * REAL;",
            )

    def _declare_prob(self):
        # PROB = per-tree class probability vector (Random Forest soft voting).
        if "PROB" not in self.colorsets:
            self.colorsets["PROB"] = ("list", "REAL", "colset PROB = list REAL;")

    def _prob_literal(self, rule) -> str:
        """SML list literal of the leaf's normalised class distribution."""
        k = self.n_classes
        dist = getattr(rule, "class_distribution", None)
        if dist is None:
            idx = _int_label(rule.class_)
            vec = [1.0 if i == idx else 0.0 for i in range(k)]
        else:
            d = [float(x) for x in dist]
            if len(d) < k:
                d = d + [0.0] * (k - len(d))
            d = d[:k]
            total = sum(d)
            vec = [(x / total if total > 0 else 0.0) for x in d]
        return "[" + ", ".join(_ml_val(v) for v in vec) + "]"

    def _declare_var(self, name, colorset):
        self.vars[name] = colorset

    def _resolve_sample(self, sample) -> List[float]:
        vals = [0.0] * len(self.feature_names)
        if sample is None:
            return vals
        try:
            if isinstance(sample, dict):
                for i, name in enumerate(self.feature_names):
                    if name in sample:
                        vals[i] = float(sample[name])
            else:
                seq = list(sample)
                for i in range(min(len(seq), len(vals))):
                    vals[i] = float(seq[i])
        except (TypeError, ValueError):
            pass
        # Match the classifier's input precision, so the marking the net starts
        # from is the same value the classifier would compare against. Without
        # this, a sample within one float32 ULP of a threshold can take
        # different branches in the net and in the oracle.
        if getattr(self.clf, "input_dtype", None) == "float32":
            import numpy as _np
            vals = [float(_np.float32(v)) for v in vals]
        return vals

    def _sample_literal(self) -> str:
        parts = [f"{self.field[n]}={_ml_val(self.sample_values[i])}"
                 for i, n in enumerate(self.feature_names)]
        return "1`{" + ", ".join(parts) + "}"

    # ------------------------------------------------------------------
    # Guard / expression helpers
    # ------------------------------------------------------------------

    def _leaf_guard(self, rule) -> str:
        terms = []
        for var, op, val in getattr(rule, "parsed_conditions", []) or []:
            field = self.field.get(var)
            if field is None:
                # Unknown feature -> register a new field defensively.
                field = f"f{len(self.field)}"
                self.field[var] = field
            terms.append(f"#{field} x {op} {_ml_val(val)}")
        return " andalso ".join(terms) if terms else "true"

    @staticmethod
    def _negate_disjunction(guards: List[str]) -> str:
        # An unguarded leaf ("true") already covers the whole space, so the
        # fallback must never fire. Dropping it from the disjunction instead
        # would leave a fallback guard of not(rest), which is satisfied
        # wherever that leaf is the only match -- two transitions enabled at
        # once, and the subnet stops being deterministic. A rule promoted all
        # the way to the root produces exactly such a leaf.
        if any((not g) or g == "true" for g in guards):
            return "false"
        real = [g for g in guards if g]
        if not real:
            return "false"  # the leaves already cover the whole space
        joined = " orelse ".join(f"({g})" for g in real)
        return f"not ({joined})"

    # ------------------------------------------------------------------
    # Page construction primitives
    # ------------------------------------------------------------------

    def _new_page(self, name) -> _Page:
        page = _Page(self.idgen(), name)
        self.pages.append(page)
        return page

    def _add_place(self, page, name, colorset, x, y, initmark=None, port_type=None):
        p = _Place(self.idgen(), name, colorset, x, y, initmark, port_type)
        page.places.append(p)
        return p

    def _add_trans(self, page, name, x, y, guard=None, subst=None):
        t = _Trans(self.idgen(), _sanitize(name, page._used_names, "T"),
                   x, y, guard, subst)
        page.transitions.append(t)
        return t

    def _arc(self, page, orient, trans, place, expr):
        page.arcs.append(_Arc(self.idgen(), orient, trans.id, place.id, expr))

    # ------------------------------------------------------------------
    # Theorem 1: Decision-tree subpage (sample -> value)
    # ------------------------------------------------------------------

    def _build_dt_subpage(self, page_name, tree_rules, value_kind,
                          in_name="P_in", out_name="P_out"):
        """Build a DT subpage. ``value_kind`` is 'real' (GBDT contribution),
        'class' (DT integer label) or 'prob' (RF per-tree probability vector).
        The port places are named ``in_name`` / ``out_name`` so they match the
        socket names on the parent page (CPN Tools assigns ports to sockets by
        name). Returns (page, in, out)."""
        out_cs = {"real": "REAL", "prob": "PROB"}.get(value_kind, "INT")
        zero_vec = "[" + ", ".join(["0.0"] * self.n_classes) + "]"
        page = self._new_page(page_name)
        p_in = self._add_place(page, in_name, "SAMPLE", -420, 0, port_type="In")
        p_out = self._add_place(page, out_name, out_cs, 420, 0, port_type="Out")

        y = (len(tree_rules) - 1) * 100 / 2
        leaf_guards = []
        for idx, rule in enumerate(tree_rules):
            guard = self._leaf_guard(rule)
            leaf_guards.append(guard)
            t = self._add_trans(page, rule.name or f"leaf{idx}", 0, y, guard=guard)
            self._arc(page, "PtoT", t, p_in, "x")
            if value_kind == "real":
                contribution = getattr(rule, "contribution", None)
                out_expr = _ml_val(contribution if contribution is not None else 0.0)
            elif value_kind == "prob":
                out_expr = self._prob_literal(rule)
            else:
                out_expr = str(_int_label(rule.class_))
            self._arc(page, "TtoP", t, p_out, out_expr)
            y -= 100

        # Refined models may not be exhaustive -> add a catch-all so the
        # subpage always has exactly one enabled transition (no deadlock).
        if self.is_final:
            fb_guard = self._negate_disjunction(leaf_guards)
            if fb_guard != "false":
                tf = self._add_trans(page, f"{page_name}_default", 0, y, guard=fb_guard)
                self._arc(page, "PtoT", tf, p_in, "x")
                if value_kind == "real":
                    self._arc(page, "TtoP", tf, p_out, "0.0")
                elif value_kind == "prob":
                    # Non-matching tree adds a zero vector -> does not affect the
                    # argmax of the summed distribution (matches sklearn, which
                    # only averages the trees that reach a leaf).
                    self._arc(page, "TtoP", tf, p_out, zero_vec)
                else:
                    self._arc(page, "TtoP", tf, p_out, str(_int_label(
                        getattr(self.clf, "default_class", 0))))
        return page, p_in, p_out

    # ------------------------------------------------------------------
    # Theorem 2: GBDT class channel
    # ------------------------------------------------------------------

    def _build_channel(self, class_label, stage_trees, init_score,
                       in_name="P_in", score_name="P_score"):
        """Build a channel page for one GBDT class. ``stage_trees`` is the
        ordered list of per-stage rule lists. ``in_name`` / ``score_name`` name
        the channel's ports to match the parent-page sockets. Returns
        (page, in, score)."""
        self._declare_product_ss()
        self._declare_var("sc", "REAL")
        self._declare_var("lv", "REAL")

        cl = _int_label(class_label)
        page = self._new_page(f"Channel_{cl}")
        # Left-to-right layout on a single cursor; sub-page evaluation hangs
        # above/below each stage's Pend place. Generous spacing avoids overlap.
        step = 170
        cx = -900
        p_in = self._add_place(page, in_name, "SAMPLE", cx, 0, port_type="In")

        # Init: (x) -> Acc0 = (x, s0)
        cx += step
        t_init = self._add_trans(page, "Init", cx, 0)
        cx += step
        acc_prev = self._add_place(page, "Acc0", "SS", cx, 0)
        self._arc(page, "PtoT", t_init, p_in, "x")
        self._arc(page, "TtoP", t_init, acc_prev, f"(x, {_ml_val(init_score)})")

        for m, tree_rules in enumerate(stage_trees, start=1):
            # Socket / port share a unique name so CPN Tools links them by name.
            in_nm, out_nm = f"TreeIn_c{cl}s{m}", f"TreeOut_c{cl}s{m}"
            sub_page, sub_in, sub_out = self._build_dt_subpage(
                f"Tree_c{cl}_s{m}", tree_rules, "real",
                in_name=in_nm, out_name=out_nm)

            cx += step
            t_feed = self._add_trans(page, f"Feed_{m}", cx, 0)
            cx += step
            pend = self._add_place(page, f"Pend_{m}", "SS", cx, 0)
            dt_in = self._add_place(page, in_nm, "SAMPLE", cx, 210)
            dt_out = self._add_place(page, out_nm, "REAL", cx, -210)
            cx += step
            t_col = self._add_trans(page, f"Collect_{m}", cx, 0)
            cx += step
            acc_m = self._add_place(page, f"Acc{m}", "SS", cx, 0)

            # Substitution transition -> DT subpage. The socket places are
            # arc-connected to it: in-socket feeds the subpage's In port, the
            # out-socket receives its Out port.
            subst = _Subst(sub_page.id, sub_page.name,
                           [(dt_in.id, sub_in.id), (dt_out.id, sub_out.id)])
            t_stage = self._add_trans(page, f"Stage_{m}", pend.x, 380, subst=subst)
            self._arc(page, "PtoT", t_stage, dt_in, "x")
            self._arc(page, "TtoP", t_stage, dt_out, "lv")

            # Feed: Acc_{m-1} -> (DTin_m sample, Pend_m state)
            self._arc(page, "PtoT", t_feed, acc_prev, "(x, sc)")
            self._arc(page, "TtoP", t_feed, dt_in, "x")
            self._arc(page, "TtoP", t_feed, pend, "(x, sc)")

            # Collect: (Pend_m, DTout_m) -> Acc_m = (x, sc + lv)
            self._arc(page, "PtoT", t_col, pend, "(x, sc)")
            self._arc(page, "PtoT", t_col, dt_out, "lv")
            self._arc(page, "TtoP", t_col, acc_m, "(x, sc + lv)")

            acc_prev = acc_m

        # Finalize: Acc_M -> P_score = sc
        cx += step
        t_fin = self._add_trans(page, "Finalize", cx, 0)
        cx += step
        p_score = self._add_place(page, score_name, "REAL", cx, 0, port_type="Out")
        self._arc(page, "PtoT", t_fin, acc_prev, "(x, sc)")
        self._arc(page, "TtoP", t_fin, p_score, "sc")
        return page, p_in, p_score

    # ------------------------------------------------------------------
    # Top-level pages per algorithm
    # ------------------------------------------------------------------

    def _group_gbdt(self):
        """Return (classes, {class_label: [stage1_rules, ...]}, init_scores)."""
        classes = list(getattr(self.clf, "_gbdt_classes", None) or [])
        init_scores = dict(getattr(self.clf, "_gbdt_init_scores", None) or {})

        # Group rules by tree id (e.g. 'GBDT1T2'), preserving discovery order.
        tree_map: "OrderedDict[str, list]" = OrderedDict()
        for r in self.rules:
            tid = r.name.split("_")[0] if "_" in r.name else r.name
            tree_map.setdefault(tid, []).append(r)

        # tree id 'GBDT<class>T<stage>' -> (class, stage). Init trees are T0.
        per_class: Dict[str, Dict[int, list]] = defaultdict(dict)
        for tid, rs in tree_map.items():
            cls = str(rs[0].class_group)
            m = re.search(r"T(\d+)$", tid)
            stage = int(m.group(1)) if m else 0
            if stage == 0:
                continue  # init rule: handled via init_scores
            per_class[cls][stage] = rs

        if not classes:
            classes = sorted(per_class.keys(), key=_int_label)

        ordered = {}
        for cls in classes:
            stages = per_class.get(str(cls), {})
            ordered[cls] = [stages[s] for s in sorted(stages.keys())]
        return classes, ordered, init_scores

    def _build_gbdt(self):
        classes, stages_by_class, init_scores = self._group_gbdt()
        is_binary = bool(getattr(self.clf, "_gbdt_is_binary", False)) or len(classes) == 2

        top = self._new_page(TOP_PAGE)
        p_input = self._add_place(top, "Input", "SAMPLE", -600, 0,
                                  initmark=self._sample_literal())
        p_pred = self._add_place(top, "Prediction", "INT", 700, 0)
        t_disp = self._add_trans(top, "Distribute", -420, 0)
        self._arc(top, "PtoT", t_disp, p_input, "x")

        if is_binary:
            # Positive class = classes[1]; only that channel has trees.
            pos = classes[1] if len(classes) >= 2 else classes[0]
            neg = classes[0]
            ch_page, ch_in, ch_score = self._build_channel(
                pos, stages_by_class.get(pos, []), init_scores.get(str(pos), 0.0),
                in_name="ChIn", score_name="ChScore")

            sock_in = self._add_place(top, "ChIn", "SAMPLE", -240, 0)
            sock_score = self._add_place(top, "ChScore", "REAL", 200, 0)
            subst = _Subst(ch_page.id, ch_page.name,
                           [(sock_in.id, ch_in.id), (sock_score.id, ch_score.id)])
            self._declare_var("sc", "REAL")
            t_ch = self._add_trans(top, "Channel", -40, 0, subst=subst)
            self._arc(top, "PtoT", t_ch, sock_in, "x")
            self._arc(top, "TtoP", t_ch, sock_score, "sc")
            self._arc(top, "TtoP", t_disp, sock_in, "x")

            t_pos = self._add_trans(top, "Decide_1", 460, 120, guard="sc >= 0.0")
            self._arc(top, "PtoT", t_pos, sock_score, "sc")
            self._arc(top, "TtoP", t_pos, p_pred, str(_int_label(pos)))
            t_neg = self._add_trans(top, "Decide_0", 460, -120, guard="sc < 0.0")
            self._arc(top, "PtoT", t_neg, sock_score, "sc")
            self._arc(top, "TtoP", t_neg, p_pred, str(_int_label(neg)))
        else:
            score_socks = []
            yy = (len(classes) - 1) * 160 / 2
            for i, cls in enumerate(classes):
                ch_page, ch_in, ch_score = self._build_channel(
                    cls, stages_by_class.get(cls, []), init_scores.get(str(cls), 0.0),
                    in_name=f"ChIn_{i}", score_name=f"ChScore_{i}")
                sock_in = self._add_place(top, f"ChIn_{i}", "SAMPLE", -240, yy)
                sock_score = self._add_place(top, f"ChScore_{i}", "REAL", 220, yy)
                subst = _Subst(ch_page.id, ch_page.name,
                               [(sock_in.id, ch_in.id), (sock_score.id, ch_score.id)])
                self._declare_var(f"sc{i}", "REAL")
                t_ch = self._add_trans(top, f"Channel_{i}", -20, yy, subst=subst)
                self._arc(top, "PtoT", t_ch, sock_in, "x")
                self._arc(top, "TtoP", t_ch, sock_score, f"sc{i}")
                self._arc(top, "TtoP", t_disp, sock_in, "x")
                score_socks.append((i, cls, sock_score))
                yy -= 160

            # Decision: argmax with NumPy tie-breaking (lowest index wins ties).
            yy = (len(classes) - 1) * 120 / 2
            for i, cls, _sock in score_socks:
                terms = []
                for j, _c2, _s2 in score_socks:
                    if j == i:
                        continue
                    op = ">" if j < i else ">="
                    terms.append(f"sc{i} {op} sc{j}")
                guard = " andalso ".join(terms) if terms else "true"
                t_dec = self._add_trans(top, f"Decide_{_int_label(cls)}", 480, yy,
                                        guard=guard)
                for j, _c2, sock_j in score_socks:
                    self._arc(top, "PtoT", t_dec, sock_j, f"sc{j}")
                self._arc(top, "TtoP", t_dec, p_pred, str(_int_label(cls)))
                yy -= 120
        return top

    def _build_decision_tree(self):
        page, _pin, _pout = self._build_dt_subpage_root(
            TOP_PAGE, self.rules, "class")
        return page

    def _build_dt_subpage_root(self, page_name, tree_rules, value_kind):
        """A standalone single-page DT model (Theorem 1), with an input place
        carrying the sample initial marking and an INT output place."""
        page = self._new_page(page_name)
        p_in = self._add_place(page, "Input", "SAMPLE", -300, 0,
                               initmark=self._sample_literal())
        p_out = self._add_place(page, "Prediction", "INT", 300, 0)
        y = (len(tree_rules) - 1) * 100 / 2
        leaf_guards = []
        for idx, rule in enumerate(tree_rules):
            guard = self._leaf_guard(rule)
            leaf_guards.append(guard)
            t = self._add_trans(page, rule.name or f"leaf{idx}", 0, y, guard=guard)
            self._arc(page, "PtoT", t, p_in, "x")
            self._arc(page, "TtoP", t, p_out, str(_int_label(rule.class_)))
            y -= 100
        if self.is_final:
            fb_guard = self._negate_disjunction(leaf_guards)
            if fb_guard != "false":
                tf = self._add_trans(page, f"{page_name}_default", 0, y, guard=fb_guard)
                self._arc(page, "PtoT", tf, p_in, "x")
                self._arc(page, "TtoP", tf, p_out,
                          str(_int_label(getattr(self.clf, "default_class", 0))))
        return page, p_in, p_out

    def _build_random_forest(self):
        # Soft voting, faithful to scikit-learn: each tree emits its leaf's
        # normalised class-probability vector; the decision sums them and takes
        # argmax (NumPy tie-break: lowest index). argmax of the sum equals argmax
        # of the average, so this reproduces sklearn's predict exactly.
        self._declare_prob()
        self._declare_ml_softvote()
        tree_map: "OrderedDict[str, list]" = OrderedDict()
        for r in self.rules:
            tid = r.name.split("_")[0] if "_" in r.name else "Tree0"
            tree_map.setdefault(tid, []).append(r)

        top = self._new_page(TOP_PAGE)
        p_input = self._add_place(top, "Input", "SAMPLE", -600, 0,
                                  initmark=self._sample_literal())
        p_pred = self._add_place(top, "Prediction", "INT", 700, 0)
        t_disp = self._add_trans(top, "Distribute", -420, 0)
        self._arc(top, "PtoT", t_disp, p_input, "x")

        vote_socks = []
        yy = (len(tree_map) - 1) * 130 / 2
        for i, (tid, rs) in enumerate(tree_map.items()):
            sock_nm = f"TProb_{i}"
            sub_page, sub_in, sub_out = self._build_dt_subpage(
                f"Tree_{i}", rs, "prob",
                in_name=f"TIn_{i}", out_name=sock_nm)
            sock_in = self._add_place(top, f"TIn_{i}", "SAMPLE", -240, yy)
            sock_vote = self._add_place(top, sock_nm, "PROB", 220, yy)
            subst = _Subst(sub_page.id, sub_page.name,
                           [(sock_in.id, sub_in.id), (sock_vote.id, sub_out.id)])
            self._declare_var(f"vp{i}", "PROB")
            t_tree = self._add_trans(top, f"Tree_{i}", -20, yy, subst=subst)
            self._arc(top, "PtoT", t_tree, sock_in, "x")
            self._arc(top, "TtoP", t_tree, sock_vote, f"vp{i}")
            self._arc(top, "TtoP", t_disp, sock_in, "x")
            vote_socks.append((i, sock_vote))
            yy -= 130

        t_dec = self._add_trans(top, "Vote", 480, 0)
        for i, sock in vote_socks:
            self._arc(top, "PtoT", t_dec, sock, f"vp{i}")
        prob_list = "[" + ", ".join(f"vp{i}" for i, _ in vote_socks) + "]"
        self._arc(top, "TtoP", t_dec, p_pred, f"decide(vsum({prob_list}))")
        return top

    def _declare_ml_softvote(self):
        # Element-wise vector add, sum of a list of probability vectors, and
        # argmax with lowest-index tie-breaking (NumPy-compatible).
        self.mls.append(
            "fun vadd (xs, ys) = ListPair.map (fn (a:real, b:real) => a + b) (xs, ys);"
        )
        self.mls.append(
            "fun vsum ([] : real list list) = []\n"
            "  | vsum (h::t) = List.foldl vadd h t;"
        )
        self.mls.append(
            "fun argmax (nil : real list) = 0\n"
            "  | argmax (h::t) =\n"
            "      let fun go (_, bi, _, []) = bi\n"
            "            | go (i, bi, bv, (x:real)::xs) =\n"
            "                if x > bv then go (i+1, i, x, xs)\n"
            "                          else go (i+1, bi, bv, xs)\n"
            "      in go (1, 0, h, t) end;"
        )
        # Every tree abstained (refined subnets emit a zero vector when the
        # sample matches no remaining leaf). A plain argmax would return class 0
        # with no evidence behind it, so fall back to the declared default
        # class, matching what the Python engines do.
        default_cls = _int_label(getattr(self.clf, "default_class", 0))
        self.mls.append(
            "fun allzero (xs : real list) =\n"
            "      List.all (fn x => x <= 0.0 andalso x >= 0.0) xs;"
        )
        self.mls.append(
            "fun decide (xs : real list) =\n"
            f"      if allzero xs then {default_cls} else argmax xs;"
        )

    # ------------------------------------------------------------------
    # Build dispatcher
    # ------------------------------------------------------------------

    def build(self) -> str:
        algo = getattr(self.clf, "algorithm_type", "Decision Tree")
        if algo == "Gradient Boosting Decision Trees":
            self._build_gbdt()
        elif algo == "Random Forest":
            self._build_random_forest()
        else:
            self._build_decision_tree()
        return self._serialize()

    # ------------------------------------------------------------------
    # XML serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _obj_attrs(parent, x=0.0, y=0.0, fill_pattern="", thick="1",
                   line_type="Solid"):
        ET.SubElement(parent, "posattr", {"x": f"{x:.6f}", "y": f"{y:.6f}"})
        ET.SubElement(parent, "fillattr",
                      {"colour": "White", "pattern": fill_pattern, "filled": "false"})
        ET.SubElement(parent, "lineattr",
                      {"colour": "Black", "thick": thick, "type": line_type})
        ET.SubElement(parent, "textattr", {"colour": "Black", "bold": "false"})

    def _label(self, parent, tag, text, x, y):
        el = ET.SubElement(parent, tag, {"id": self.idgen()})
        self._obj_attrs(el, x, y, fill_pattern="Solid", thick="0")
        t = ET.SubElement(el, "text", {"tool": "CPN Tools", "version": "4.0.1"})
        t.text = text
        return el

    def _serialize(self) -> str:
        root = ET.Element("workspaceElements")
        ET.SubElement(root, "generator",
                      {"tool": "CPN Tools", "version": "4.0.1", "format": "6"})
        cpnet = ET.SubElement(root, "cpnet")

        # ---- globbox -------------------------------------------------
        globbox = ET.SubElement(cpnet, "globbox")
        self._emit_standard_blocks(globbox)
        block = ET.SubElement(globbox, "block", {"id": self.idgen()})
        ET.SubElement(block, "id").text = "Declarations"
        for name, (kind, payload, layout) in self.colorsets.items():
            self._emit_color(block, name, kind, payload, layout)
        # variables grouped by colorset
        by_cs: "OrderedDict[str, list]" = OrderedDict()
        for vname, cs in self.vars.items():
            by_cs.setdefault(cs, []).append(vname)
        for cs, names in by_cs.items():
            self._emit_var(block, cs, names)
        for ml_code in self.mls:
            ml = ET.SubElement(block, "ml", {"id": self.idgen()})
            ml.text = ml_code

        # ---- pages ---------------------------------------------------
        for page in self.pages:
            self._emit_page(cpnet, page)

        # ---- instances (hierarchy) ----------------------------------
        self._emit_instances(cpnet)

        # ---- trailing required elements -----------------------------
        ET.SubElement(cpnet, "options")
        ET.SubElement(cpnet, "binders")
        ET.SubElement(cpnet, "monitorblock", {"name": "Monitors"})
        ET.SubElement(cpnet, "IndexNode", {"expanded": "false"})

        if hasattr(ET, "indent"):       # pretty-print (Python 3.9+)
            ET.indent(root, space="  ")
        body = ET.tostring(root, encoding="unicode")
        header = (
            '<?xml version="1.0" encoding="iso-8859-1"?>\n'
            '<!DOCTYPE workspaceElements PUBLIC "-//CPN//DTD CPNXML 1.0//EN" '
            '"http://cpntools.org/DTD/6/cpn.dtd">\n'
        )
        return header + body + "\n"

    def _emit_standard_blocks(self, globbox):
        """Emit the canonical 'Standard priorities' and 'Standard declarations'
        blocks present in every CPN Tools net (matches the New net template)."""
        prio = ET.SubElement(globbox, "block", {"id": self.idgen()})
        ET.SubElement(prio, "id").text = "Standard priorities"
        for val in ("val P_HIGH = 100;", "val P_NORMAL = 1000;",
                    "val P_LOW = 10000;"):
            ml = ET.SubElement(prio, "ml", {"id": self.idgen()})
            ml.text = val
            ET.SubElement(ml, "layout").text = val

        std = ET.SubElement(globbox, "block", {"id": self.idgen()})
        ET.SubElement(std, "id").text = "Standard declarations"
        for name, tag in (("UNIT", "unit"), ("BOOL", "bool"), ("INT", "int"),
                          ("REAL", "real"), ("STRING", "string")):
            color = ET.SubElement(std, "color", {"id": self.idgen()})
            ET.SubElement(color, "id").text = name
            ET.SubElement(color, tag)
            ET.SubElement(color, "layout").text = f"colset {name} = {tag};"

    def _emit_color(self, block, name, kind, payload, layout):
        color = ET.SubElement(block, "color", {"id": self.idgen()})
        ET.SubElement(color, "id").text = name
        if kind == "record":
            rec = ET.SubElement(color, "record")
            for fname, ftype in payload:
                rf = ET.SubElement(rec, "recordfield")
                ET.SubElement(rf, "id").text = fname
                ET.SubElement(rf, "id").text = ftype
        elif kind == "product":
            prod = ET.SubElement(color, "product")
            for comp in payload:
                ET.SubElement(prod, "id").text = comp
        elif kind == "list":
            lst = ET.SubElement(color, "list")
            ET.SubElement(lst, "id").text = payload
        else:
            ET.SubElement(color, kind)
        ET.SubElement(color, "layout").text = layout

    def _emit_var(self, block, colorset, names):
        var = ET.SubElement(block, "var", {"id": self.idgen()})
        t = ET.SubElement(var, "type")
        ET.SubElement(t, "id").text = colorset
        for n in names:
            ET.SubElement(var, "id").text = n
        ET.SubElement(var, "layout").text = \
            f"var {', '.join(names)} : {colorset};"

    def _emit_page(self, cpnet, page: _Page):
        pg = ET.SubElement(cpnet, "page", {"id": page.id})
        ET.SubElement(pg, "pageattr", {"name": page.name})
        for p in page.places:
            self._emit_place(pg, p)
        for t in page.transitions:
            self._emit_trans(pg, t)
        for a in page.arcs:
            self._emit_arc(pg, a)

    def _emit_place(self, pg, p: _Place):
        el = ET.SubElement(pg, "place", {"id": p.id})
        self._obj_attrs(el, p.x, p.y)
        ET.SubElement(el, "text").text = p.name
        ET.SubElement(el, "ellipse", {"w": "60.000000", "h": "40.000000"})
        # colour set type
        tp = ET.SubElement(el, "type", {"id": self.idgen()})
        self._obj_attrs(tp, p.x + 40, p.y - 26, fill_pattern="Solid", thick="0")
        ET.SubElement(tp, "text", {"tool": "CPN Tools", "version": "4.0.1"}).text = \
            p.colorset
        if p.port_type is not None:
            port = ET.SubElement(el, "port",
                                 {"id": self.idgen(), "type": p.port_type})
            self._obj_attrs(port, p.x - 40, p.y - 26, fill_pattern="Solid", thick="0")
        if p.initmark:
            self._label(el, "initmark", p.initmark, p.x + 40, p.y + 24)

    def _emit_trans(self, pg, t: _Trans):
        el = ET.SubElement(pg, "trans", {"id": t.id, "explicit": "false"})
        self._obj_attrs(el, t.x, t.y)
        ET.SubElement(el, "text").text = t.name
        ET.SubElement(el, "box", {"w": "60.000000", "h": "40.000000"})
        if t.subst is not None:
            # CPN Tools encodes each assignment as (portId,socketId) -- the
            # subpage port place first, then the socket place on this page.
            portsock = "".join(f"({p},{s})" for s, p in t.subst.pairs)
            sub = ET.SubElement(el, "subst",
                                {"subpage": t.subst.subpage_id, "portsock": portsock})
            info = ET.SubElement(sub, "subpageinfo",
                                 {"id": self.idgen(), "name": t.subst.subpage_name})
            self._obj_attrs(info, t.x, t.y - 30, fill_pattern="Solid", thick="0")
        ET.SubElement(el, "binding", {"x": f"{t.x + 7.2:.6f}", "y": f"{t.y - 3.0:.6f}"})
        # Real CPN Tools transitions always carry the full set of inscription
        # regions. Substitution transitions in particular are only processed
        # correctly (port/socket assignment) when these elements are present.
        self._label(el, "cond", f"[{t.guard}]" if t.guard else "", t.x - 40, t.y + 28)
        self._label(el, "time", "", t.x + 30, t.y + 28)
        self._label(el, "code", "", t.x + 28, t.y - 43)
        self._label(el, "channel", "", t.x - 20, t.y - 43)
        self._label(el, "priority", "", t.x - 50, t.y - 43)

    def _emit_arc(self, pg, a: _Arc):
        el = ET.SubElement(pg, "arc",
                           {"id": a.id, "orientation": a.orient, "order": "1"})
        self._obj_attrs(el, 0, 0)
        ET.SubElement(el, "arrowattr",
                      {"headsize": "1.200000", "currentcyckle": "2"})
        ET.SubElement(el, "transend", {"idref": a.trans_id})
        ET.SubElement(el, "placeend", {"idref": a.place_id})
        if a.expr:
            self._label(el, "annot", a.expr, 0, 0)

    def _emit_instances(self, cpnet):
        referenced = set()
        children: Dict[str, list] = {}
        for page in self.pages:
            kids = []
            for t in page.transitions:
                if t.subst is not None:
                    referenced.add(t.subst.subpage_id)
                    kids.append((t.id, t.subst.subpage_id))
            children[page.id] = kids
        roots = [p for p in self.pages if p.id not in referenced]

        instances = ET.SubElement(cpnet, "instances")

        def emit(parent, page_id, trans_id=None):
            # Top-level instances carry page="..."; instances nested under a
            # substitution transition carry ONLY trans="..." (CPN Tools derives
            # the subpage from the transition). Emitting page= on a nested
            # instance leaves the ports "unassigned" in CPN Tools.
            if trans_id is None:
                attrs = {"id": self.idgen(), "page": page_id}
            else:
                attrs = {"id": self.idgen(), "trans": trans_id}
            inst = ET.SubElement(parent, "instance", attrs)
            for (st_id, sub_id) in children.get(page_id, []):
                emit(inst, sub_id, st_id)

        for root_page in roots:
            emit(instances, root_page.id)


# ---------------------------------------------------------------------------
# Public convenience function
# ---------------------------------------------------------------------------

def export_cpn_tools(classifier, filepath, rules=None, feature_names=None,
                     sample=None, use_final=None, model_name="pyRuleAnalyzer"):
    """Generate a CPN Tools ``.cpn`` HCPN file from a RuleClassifier.

    Args:
        classifier: The :class:`RuleClassifier` instance.
        filepath: Destination ``.cpn`` path.
        rules: Explicit list of rules to export. If ``None`` the rule set is
            chosen from ``use_final``.
        feature_names: Ordered feature names (defaults to classifier metadata).
        sample: Example input sample for the initial marking.
        use_final: If ``True`` export ``final_rules``; if ``False`` export
            ``initial_rules``; if ``None`` use final rules when available.
        model_name: Free-form label for the model. Page names are fixed
            (``Top``, ``Channel_k``, ``Tree_c<k>_s<m>``, ``Tree_<i>``) so that
            ASK-CTL queries are identical across generated models.

    Returns:
        The ``filepath`` written.
    """
    if rules is None:
        has_final = bool(getattr(classifier, "final_rules", None))
        if use_final is None:
            use_final = has_final
        rules = classifier.final_rules if use_final else classifier.initial_rules
    is_final = bool(use_final) if use_final is not None else \
        bool(getattr(classifier, "final_rules", None))

    if not filepath.lower().endswith(".cpn"):
        filepath = filepath + ".cpn"

    exporter = CPNToolsExporter(
        classifier, rules, feature_names=feature_names, sample=sample,
        is_final=is_final, model_name=model_name,
    )
    xml = exporter.build()
    with open(filepath, "w", encoding="iso-8859-1", errors="replace") as f:
        f.write(xml)
    return filepath
