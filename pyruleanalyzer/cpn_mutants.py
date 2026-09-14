"""Mutants of generated nets: every property must be able to fail.

A model checker that answers "true" on correct nets has shown nothing until it
is also seen answering "false" on incorrect ones. This module derives, from a
correct generated ``.cpn``, one mutant per property, each a small edit that
makes that property false for the input stored in the initial marking:

========================  ===========  ==============================================
Mutant                    Must fail    Edit
========================  ===========  ==============================================
``cycle``                 A1           a live leaf gives its input token back instead
                                       of producing its output
``deadlock``              A2, A3       the transition producing the prediction loses
                                       its output arc
``dead_branch``           A4, A8       a dormant leaf is always enabled and produces
                                       nothing: a second, predictionless end
``double_output``         A5, A7       the prediction is deposited twice
``bad_label``             A6           the prediction becomes a label outside L
``overlap``               B1, B4       a dormant leaf becomes enabled next to the
                                       live one
``hidden_overlap``        B4           two dormant leaves get the same guard; the
                                       sample does not reach them, so B1 still holds
``no_leaf``               B2           the live leaf's guard becomes false
``tree_double``           B3           a tree emits its result twice
``skip_stage``            C1           a boosting stage reads the accumulator of the
                                       stage before the previous one
``early_score``           C2           Finalize reads the second-to-last accumulator
``no_decision``           C3a          the live Decide guard becomes false
``two_decisions``         C3b          a dormant Decide guard becomes true
``partial_vote``          D1           Vote stops waiting for the first tree
``no_vote``               D2a          Vote's guard becomes false
``wrong_value``           PC           the net predicts another valid class; every
                                       CTL property still holds
========================  ===========  ==============================================

Edits only touch guards, inscriptions and arcs (copies of existing arcs), so
the mutants are still files CPN Tools opens, and the same mutants are fed to
both engines in the cross-check.
"""

import copy
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

from .cpn_semantics import CPNNet
from .cpn_statespace import ColouredNet, build_state_space

__all__ = ["CPNDocument", "Mutant", "MUTANTS", "make_mutants"]


# Class wrapping a .cpn file for in-place edits that keep it loadable.
class CPNDocument:
    """A ``.cpn`` file opened for editing.

    The XML prologue (declaration and CPN Tools DOCTYPE) is kept verbatim, and
    new elements get fresh identifiers.
    """

    # Method to open a .cpn file.
    def __init__(self, path: str):
        """Parse the file.

        Args:
            path (str): The ``.cpn`` to edit.
        """
        with open(path, "rb") as fh:
            raw = fh.read().decode("iso-8859-1")
        start = raw.index("<workspaceElements")
        self.prologue = raw[:start]
        self.root = ET.fromstring(raw[start:].encode("iso-8859-1"))
        self.pages = {p.find("pageattr").get("name"): p for p in self.root.iter("page")}
        ids = [int(m.group(1)) for el in self.root.iter()
               for m in [re.fullmatch(r"ID(\d+)", el.get("id") or "")] if m]
        self._next_id = max(ids, default=1000000) + 1

    # Method to allocate an unused element identifier.
    def new_id(self) -> str:
        """A fresh ``ID<n>`` identifier.

        Returns:
            str: The identifier.
        """
        self._next_id += 1
        return f"ID{self._next_id}"

    # Method to find a transition element by page and name.
    def trans(self, page: str, name: str):
        """The ``<trans>`` element named ``name`` on ``page``.

        Args:
            page (str): Page name.
            name (str): Transition name.

        Returns:
            Element: The transition.
        """
        for t in self.pages[page].findall("trans"):
            if (t.findtext("text") or "").strip() == name:
                return t
        raise KeyError(f"{page}/{name}")

    # Method to find a place element by page and name.
    def place(self, page: str, name: str):
        """The ``<place>`` element named ``name`` on ``page``.

        Args:
            page (str): Page name.
            name (str): Place name.

        Returns:
            Element: The place.
        """
        for p in self.pages[page].findall("place"):
            if (p.findtext("text") or "").strip() == name:
                return p
        raise KeyError(f"{page}/{name}")

    # Method to list the arcs of a transition.
    def arcs(self, page: str, trans_name: str, orientation: Optional[str] = None):
        """Arcs attached to a transition.

        Args:
            page (str): Page name.
            trans_name (str): Transition name.
            orientation (str, optional): ``"PtoT"`` or ``"TtoP"``.

        Returns:
            list: The ``<arc>`` elements.
        """
        tid = self.trans(page, trans_name).get("id")
        return [a for a in self.pages[page].findall("arc")
                if a.find("transend").get("idref") == tid
                and (orientation is None or a.get("orientation") == orientation)]

    # Method to replace a transition guard.
    def set_guard(self, page: str, trans_name: str, text: str):
        """Replace a transition's guard.

        Args:
            page (str): Page name.
            trans_name (str): Transition name.
            text (str): New guard, e.g. ``"[false]"``.
        """
        self.trans(page, trans_name).find("cond/text").text = text

    # Method to read a transition guard.
    def guard(self, page: str, trans_name: str) -> str:
        """A transition's guard text.

        Args:
            page (str): Page name.
            trans_name (str): Transition name.

        Returns:
            str: The guard.
        """
        return self.trans(page, trans_name).findtext("cond/text") or ""

    # Method to replace the inscription of an arc.
    @staticmethod
    def set_inscription(arc, text: str):
        """Replace an arc's inscription.

        Args:
            arc (Element): The arc.
            text (str): New inscription.
        """
        arc.find("annot/text").text = text

    # Method to remove an arc.
    def remove_arc(self, page: str, arc):
        """Delete an arc from a page.

        Args:
            page (str): Page name.
            arc (Element): The arc to remove.
        """
        self.pages[page].remove(arc)

    # Method to add an arc as a copy of an existing one.
    def add_arc(self, page: str, template, orientation: str, trans_el, place_el,
                text: str):
        """Add an arc, cloned from ``template`` so every attribute CPN Tools
        expects is present.

        Args:
            page (str): Page name.
            template (Element): Arc to clone.
            orientation (str): ``"PtoT"`` or ``"TtoP"``.
            trans_el (Element): Transition end.
            place_el (Element): Place end.
            text (str): Inscription.

        Returns:
            Element: The new arc.
        """
        arc = copy.deepcopy(template)
        for el in arc.iter():
            if el.get("id"):
                el.set("id", self.new_id())
        arc.set("orientation", orientation)
        arc.find("transend").set("idref", trans_el.get("id"))
        arc.find("placeend").set("idref", place_el.get("id"))
        arc.find("annot/text").text = text
        self.pages[page].append(arc)
        return arc

    # Method to change the place an arc is attached to.
    def retarget(self, arc, place_el):
        """Attach an arc to another place of the same page.

        Args:
            arc (Element): The arc.
            place_el (Element): New place end.
        """
        arc.find("placeend").set("idref", place_el.get("id"))

    # Method to name the place an arc is attached to.
    def place_of(self, page: str, arc) -> str:
        """Name of the place at the end of an arc.

        Args:
            page (str): Page name.
            arc (Element): The arc.

        Returns:
            str: The place name.
        """
        pid = arc.find("placeend").get("idref")
        for p in self.pages[page].findall("place"):
            if p.get("id") == pid:
                return (p.findtext("text") or "").strip()
        raise KeyError(pid)

    # Method to write the edited document.
    def save(self, path: str) -> str:
        """Write the document, keeping the original prologue.

        Args:
            path (str): Destination.

        Returns:
            str: The path written.
        """
        body = ET.tostring(self.root, encoding="unicode")
        with open(path, "wb") as fh:
            fh.write((self.prologue + body + "\n").encode("iso-8859-1",
                                                          "xmlcharrefreplace"))
        return path


# Class describing one mutation operator.
class Mutant:
    """A mutation operator.

    Attributes:
        name (str): Identifier used in file names and reports.
        targets (tuple): Properties the mutant must make fail.
        families (tuple): Families it applies to.
        description (str): What the edit does.
        build (callable): ``build(doc, info) -> bool``; returns False when the
            net has nothing to mutate this way.
    """

    __slots__ = ("name", "targets", "families", "description", "build")

    # Method to create a mutation operator.
    def __init__(self, name, targets, families, description, build):
        """Create the operator.

        Args:
            name (str): Identifier.
            targets (tuple): Properties that must fail.
            families (tuple): Applicable families.
            description (str): The edit.
            build (callable): The edit itself.
        """
        self.name = name
        self.targets = targets
        self.families = families
        self.description = description
        self.build = build


# Function to find which transitions occur for the net's own input.
def _analyse(path: str) -> Dict[str, object]:
    """Locate the live and dormant transitions the operators edit.

    Args:
        path (str): A correct generated net.

    Returns:
        dict: Pages, live leaves, dormant leaves, the prediction producer,
        channels and vote information.
    """
    net = CPNNet(path)
    cnet = ColouredNet(net)
    ss = build_state_space(cnet, max_nodes=500_000)
    occurring = {ti for out in ss.succ for ti, _k, _w in out}
    trans = cnet.transitions
    top = cnet.top["name"]
    family = cnet.family
    trees: Dict[str, Dict[str, List[str]]] = {}
    for ti, t in enumerate(trans):
        is_leaf = t["page"].startswith("Tree") if family != "Decision Tree" \
            else t["page"] == top
        if not is_leaf:
            continue
        info = trees.setdefault(t["page"], {"live": [], "dormant": []})
        info["live" if ti in occurring else "dormant"].append(t["name"])
    pred_idx = next(i for i, p in enumerate(cnet.places)
                    if p["name"] == "Prediction" and p["page"] == top)
    producers = [trans[ti] for ti in occurring
                 if any(p == pred_idx for p, _ in trans[ti]["out"])]
    decides = [t["name"] for t in trans if t["page"] == top
               and t["name"].startswith("Decide")]
    live_decides = [trans[ti]["name"] for ti in occurring
                    if trans[ti]["name"].startswith("Decide")]
    channels = sorted({t["page"] for t in trans if t["name"] == "Finalize"})
    labels = sorted({tok for d in ss.dead for tok in ss.markings[d][pred_idx]})
    domain = set()
    for t in trans:
        for p, terms in t["out"]:
            if p != pred_idx:
                continue
            for _count, code in terms:
                try:
                    value = eval(code, dict(cnet.env))  # noqa: S307
                except Exception:
                    continue
                if isinstance(value, int):
                    domain.add(value)
    return {"family": family, "top": top, "trees": trees, "domain": sorted(domain),
            "producer": producers[0]["name"] if producers else None,
            "producer_page": producers[0]["page"] if producers else None,
            "decides": decides, "live_decides": live_decides,
            "channels": channels, "prediction": labels}


# Function to pick the first tree page with a live leaf.
def _live_tree(info) -> Tuple[str, str]:
    """The first tree page and its live leaf.

    Args:
        info (dict): Output of :func:`_analyse`.

    Returns:
        tuple: ``(page, leaf name)``.
    """
    for page, leaves in info["trees"].items():
        if leaves["live"]:
            return page, leaves["live"][0]
    raise LookupError("no live leaf")


# Function to pick a dormant, non-catch-all leaf of a tree page.
def _dormant(info, page: str, n: int = 1) -> Optional[List[str]]:
    """Dormant leaves of a page, excluding the catch-all.

    Args:
        info (dict): Output of :func:`_analyse`.
        page (str): Tree page.
        n (int): How many are needed.

    Returns:
        list|None: ``n`` leaf names, or None if the page has fewer.
    """
    names = [t for t in info["trees"][page]["dormant"] if not t.endswith("_default")]
    return names[:n] if len(names) >= n else None


# Function to find a tree page with a live leaf and enough dormant ones.
def _page_with_dormant(info, n: int = 1) -> Optional[Tuple[str, List[str]]]:
    """A tree page whose live leaf has ``n`` dormant, non-catch-all siblings.

    Args:
        info (dict): Output of :func:`_analyse`.
        n (int): Dormant leaves needed.

    Returns:
        tuple|None: ``(page, dormant leaf names)``.
    """
    for page, leaves in info["trees"].items():
        if not leaves["live"]:
            continue
        dormant = _dormant(info, page, n)
        if dormant:
            return page, dormant
    return None


# Function to make a live leaf give its token back (A1).
def _cycle(doc: CPNDocument, info) -> bool:
    page, leaf = _live_tree(info)
    inp = doc.arcs(page, leaf, "PtoT")[0]
    for arc in doc.arcs(page, leaf, "TtoP"):
        doc.remove_arc(page, arc)
    place = doc.place(page, doc.place_of(page, inp))
    doc.add_arc(page, inp, "TtoP", doc.trans(page, leaf), place,
                inp.find("annot/text").text)
    return True


# Function to drop the output arc of the prediction producer (A2, A3).
def _deadlock(doc: CPNDocument, info) -> bool:
    page, name = info["producer_page"], info["producer"]
    for arc in doc.arcs(page, name, "TtoP"):
        if doc.place_of(page, arc) == "Prediction":
            doc.remove_arc(page, arc)
    return True


# Function to add an always-enabled dead-end leaf (A4, A8).
def _dead_branch(doc: CPNDocument, info) -> bool:
    found = _page_with_dormant(info)
    if not found:
        return False
    page, dormant = found
    doc.set_guard(page, dormant[0], "[true]")
    for arc in doc.arcs(page, dormant[0], "TtoP"):
        doc.remove_arc(page, arc)
    return True


# Function to deposit the prediction twice (A5, A7).
def _double_output(doc: CPNDocument, info) -> bool:
    page, name = info["producer_page"], info["producer"]
    for arc in doc.arcs(page, name, "TtoP"):
        if doc.place_of(page, arc) == "Prediction":
            e = arc.find("annot/text").text.strip()
            doc.set_inscription(arc, f"1`{e} ++ 1`{e}")
    return True


# Function to deposit a label outside the domain (A6).
def _bad_label(doc: CPNDocument, info) -> bool:
    page, name = info["producer_page"], info["producer"]
    for arc in doc.arcs(page, name, "TtoP"):
        if doc.place_of(page, arc) == "Prediction":
            doc.set_inscription(arc, "7")
    return True


# Function to enable a dormant leaf next to the live one (B1, B4).
def _overlap(doc: CPNDocument, info) -> bool:
    found = _page_with_dormant(info)
    if not found:
        return False
    page, dormant = found
    doc.set_guard(page, dormant[0], "[true]")
    return True


# Function to give two dormant leaves the same guard (B4 only).
def _hidden_overlap(doc: CPNDocument, info) -> bool:
    for page in info["trees"]:
        dormant = _dormant(info, page, 2)
        if dormant:
            doc.set_guard(page, dormant[0], doc.guard(page, dormant[1]))
            return True
    return False


# Function to disable the live leaf of a tree (B2).
def _no_leaf(doc: CPNDocument, info) -> bool:
    page, leaf = _live_tree(info)
    doc.set_guard(page, leaf, "[false]")
    return True


# Function to make a tree emit its result twice (B3).
def _tree_double(doc: CPNDocument, info) -> bool:
    page, leaf = _live_tree(info)
    for arc in doc.arcs(page, leaf, "TtoP"):
        e = arc.find("annot/text").text.strip()
        doc.set_inscription(arc, f"1`{e} ++ 1`{e}")
    return True


# Function to make a stage skip its predecessor (C1).
def _skip_stage(doc: CPNDocument, info) -> bool:
    for page in reversed(info["channels"]):
        try:
            feed2 = doc.arcs(page, "Feed_2", "PtoT")
        except KeyError:
            continue
        for arc in feed2:
            if doc.place_of(page, arc) == "Acc1":
                doc.retarget(arc, doc.place(page, "Acc0"))
                return True
    return False


# Function to finalise a channel before its last stage (C2).
def _early_score(doc: CPNDocument, info) -> bool:
    for page in reversed(info["channels"]):
        accs = sorted(int(m.group(1)) for p in doc.pages[page].findall("place")
                      for m in [re.fullmatch(r"Acc(\d+)",
                                             (p.findtext("text") or "").strip())] if m)
        if len(accs) < 2:
            continue
        for arc in doc.arcs(page, "Finalize", "PtoT"):
            doc.retarget(arc, doc.place(page, f"Acc{accs[-2]}"))
            return True
    return False


# Function to disable the live decision (C3a).
def _no_decision(doc: CPNDocument, info) -> bool:
    if not info["live_decides"]:
        return False
    doc.set_guard(info["top"], info["live_decides"][0], "[false]")
    return True


# Function to enable a second decision (C3b).
def _two_decisions(doc: CPNDocument, info) -> bool:
    dormant = [d for d in info["decides"] if d not in info["live_decides"]]
    if not dormant:
        return False
    doc.set_guard(info["top"], dormant[0], "[true]")
    return True


# Function to make Vote stop waiting for the first tree (D1).
def _partial_vote(doc: CPNDocument, info) -> bool:
    top = info["top"]
    arcs = doc.arcs(top, "Vote", "PtoT")
    first = next((a for a in arcs if a.find("annot/text").text.strip() == "vp0"), None)
    if first is None or len(arcs) < 2:
        return False
    doc.remove_arc(top, first)
    for arc in doc.arcs(top, "Vote", "TtoP"):
        text = arc.find("annot/text").text
        doc.set_inscription(arc, re.sub(r"\[\s*vp0\s*,\s*", "[", text))
    return True


# Function to disable Vote (D2a).
def _no_vote(doc: CPNDocument, info) -> bool:
    doc.set_guard(info["top"], "Vote", "[false]")
    return True


# Function to make the net predict another valid class (PC).
def _wrong_value(doc: CPNDocument, info) -> bool:
    page, name = info["producer_page"], info["producer"]
    current = info["prediction"][0] if info["prediction"] else 0
    other = 1 if current == 0 else 0
    for arc in doc.arcs(page, name, "TtoP"):
        if doc.place_of(page, arc) == "Prediction":
            doc.set_inscription(arc, str(other))
    return True


_ALL = ("Decision Tree", "Random Forest", "Gradient Boosting Decision Trees")
_GBDT = ("Gradient Boosting Decision Trees",)
_RF = ("Random Forest",)

#: Every mutation operator, in report order.
MUTANTS: List[Mutant] = [
    Mutant("cycle", ("A1",), _ALL, "live leaf returns its input token", _cycle),
    Mutant("deadlock", ("A2", "A3"), _ALL,
           "prediction producer loses its output arc", _deadlock),
    Mutant("dead_branch", ("A4", "A8"), _ALL,
           "dormant leaf always enabled, produces nothing", _dead_branch),
    Mutant("double_output", ("A5", "A7"), _ALL,
           "prediction deposited twice", _double_output),
    Mutant("bad_label", ("A6",), _ALL, "prediction replaced by label 7", _bad_label),
    Mutant("overlap", ("B1", "B4"), _ALL,
           "dormant leaf enabled next to the live one", _overlap),
    Mutant("hidden_overlap", ("B4",), _ALL,
           "two dormant leaves share a guard", _hidden_overlap),
    Mutant("no_leaf", ("B2",), _ALL, "live leaf guard set to false", _no_leaf),
    Mutant("tree_double", ("B3",), _ALL, "tree emits its result twice",
           _tree_double),
    Mutant("skip_stage", ("C1",), _GBDT, "Feed_2 reads Acc0", _skip_stage),
    Mutant("early_score", ("C2",), _GBDT, "Finalize reads Acc_{M-1}", _early_score),
    Mutant("no_decision", ("C3a",), _GBDT, "live Decide guard set to false",
           _no_decision),
    Mutant("two_decisions", ("C3b",), _GBDT, "dormant Decide guard set to true",
           _two_decisions),
    Mutant("partial_vote", ("D1",), _RF, "Vote no longer consumes tree 0's vote",
           _partial_vote),
    Mutant("no_vote", ("D2a",), _RF, "Vote guard set to false", _no_vote),
    Mutant("wrong_value", ("PC",), _ALL, "net predicts another valid class",
           _wrong_value),
]


# Function to write every applicable mutant of a net.
def make_mutants(path: str, out_dir: str,
                 names: Optional[List[str]] = None) -> List[Dict[str, object]]:
    """Derive the mutants of a correct generated net.

    Args:
        path (str): The correct ``.cpn``.
        out_dir (str): Where the mutants are written.
        names (list, optional): Restrict to these operators.

    Returns:
        list: ``{"name", "targets", "description", "path", "labels"}`` per
        mutant written; ``labels`` is the original net's label domain, which
        the checker must be given for A6.
    """
    info = _analyse(path)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    written = []
    for op in MUTANTS:
        if names and op.name not in names:
            continue
        if info["family"] not in op.families:
            continue
        doc = CPNDocument(path)
        if not op.build(doc, info):
            continue
        dst = os.path.join(out_dir, f"{stem}__{op.name}.cpn")
        doc.save(dst)
        written.append({"name": op.name, "targets": op.targets,
                        "description": op.description, "path": dst,
                        "labels": info["domain"] or None})
    return written
