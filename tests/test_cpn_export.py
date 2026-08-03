"""Tests for the CPN Tools (.cpn) HCPN exporter.

For every supported model type (Decision Tree, Random Forest, binary GBDT and
multiclass GBDT) and for both the initial and the refined (final) rule sets,
these tests check that the generated ``.cpn`` file is:

* well-formed XML carrying the CPN Tools DOCTYPE;
* DTD-valid against the official ``cpn.dtd`` (only when ``lxml`` is available);
* internally consistent as a hierarchy -- every substitution transition's
  ``portsock`` references a socket place that is arc-connected to it on the
  parent page and a matching-colour port place on the referenced subpage;
* free of dangling ID references and duplicate IDs;
* fully covered by the ``<instances>`` page-instance tree.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyruleanalyzer import PyRuleAnalyzer
from pyruleanalyzer.cpn_tools_exporter import (
    export_cpn_tools, sml_real, _ml_val, CPNToolsExporter,
)
from cpn_semantics import CPNNet, evaluate

# Official CPN DTD, shipped alongside the tests for offline validation. Falls
# back to the (git-ignored) formalization copy if present.
DTD_PATH = os.path.join(os.path.dirname(__file__), "cpn.dtd")
if not os.path.exists(DTD_PATH):
    DTD_PATH = os.path.join(os.path.dirname(__file__), "..", "formalization", "cpn.dtd")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_csv(tmpdir, name, n_classes=2, n_features=4, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(240, n_features)
    if n_classes == 2:
        y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    else:
        y = (X[:, 0] * n_classes).astype(int).clip(0, n_classes - 1)
    cols = [f"feat_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    tr = os.path.join(tmpdir, f"{name}_train.csv")
    te = os.path.join(tmpdir, f"{name}_test.csv")
    df.iloc[:170].to_csv(tr, index=False)
    df.iloc[170:].to_csv(te, index=False)
    return tr, te, cols


def _analyzer(tmpdir, model, params, n_classes=2):
    tr, te, cols = _make_csv(tmpdir, model[:4], n_classes=n_classes,
                             seed=abs(hash(model)) % 50)
    a = PyRuleAnalyzer.create(train_path=tr, test_path=te, model=model,
                              params=params, refine=False, save_models=False)
    a.execute_rule_refinement(test_path=te, remove_below_n_classifications=1,
                              save_final_model=False, save_report=False)
    sample = pd.read_csv(te).drop(columns=["target"]).iloc[0].to_dict()
    return a, cols, sample


# ---------------------------------------------------------------------------
# Structural validation (stdlib only)
# ---------------------------------------------------------------------------

def _validate_structure(path):
    tree = ET.parse(path)             # raises on malformed XML
    root = tree.getroot()
    assert root.tag == "workspaceElements"
    assert root.find("generator") is not None
    cpnet = root.find("cpnet")
    assert cpnet is not None

    with open(path, encoding="iso-8859-1") as f:
        head = f.read(400)
    assert "<!DOCTYPE workspaceElements" in head
    assert "cpntools.org/DTD/6/cpn.dtd" in head

    # ---- collect IDs and references --------------------------------
    all_ids, dup = set(), set()
    for el in root.iter():
        i = el.get("id")
        if i:
            if i in all_ids:
                dup.add(i)
            all_ids.add(i)
    assert not dup, f"duplicate IDs: {sorted(dup)[:10]}"

    refs = set()
    for el in root.iter():
        for attr in ("idref", "page", "trans", "subpage"):
            v = el.get(attr)
            if v:
                refs.add(v)
    for el in root.iter("subst"):
        for item in (el.get("portsock") or "").split(")"):
            item = item.strip().lstrip("(")
            if item:
                refs.update(item.split(","))
    dangling = refs - all_ids
    assert not dangling, f"dangling references: {sorted(dangling)[:10]}"
    return root


def _validate_hierarchy(root):
    place_cs, place_isport, page_of_place = {}, {}, {}
    for page in root.iter("page"):
        for pl in page.findall("place"):
            plid = pl.get("id")
            t = pl.find("type/text")
            place_cs[plid] = t.text if t is not None else None
            place_isport[plid] = pl.find("port") is not None
            page_of_place[plid] = page.get("id")

    trans_places, trans_page = {}, {}
    for page in root.iter("page"):
        for tr in page.findall("trans"):
            trans_places.setdefault(tr.get("id"), set())
            trans_page[tr.get("id")] = page.get("id")
        for arc in page.findall("arc"):
            tid = arc.find("transend").get("idref")
            plid = arc.find("placeend").get("idref")
            trans_places.setdefault(tid, set()).add(plid)

    referenced_pages, n_subst = set(), 0
    for page in root.iter("page"):
        for tr in page.findall("trans"):
            subst = tr.find("subst")
            if subst is None:
                continue
            n_subst += 1
            subpage = subst.get("subpage")
            referenced_pages.add(subpage)
            portsock = subst.get("portsock") or ""
            assert portsock, "substitution transition with empty portsock"
            for item in portsock.split(")"):
                item = item.strip().lstrip("(")
                if not item:
                    continue
                # portsock pairs are (portId, socketId)
                port, sock = item.split(",")
                assert sock in trans_places.get(tr.get("id"), set()), \
                    f"socket {sock} not arc-connected to subst {tr.get('id')}"
                assert page_of_place.get(port) == subpage, \
                    f"port {port} not on subpage {subpage}"
                assert place_isport.get(port), f"{port} is not a port place"
                assert place_cs.get(sock) == place_cs.get(port), \
                    "socket/port colour set mismatch"

    # The instance tree has exactly one node per page. Top-level instances
    # carry page="..."; nested instances (under a subst transition) carry only
    # trans="...". Verify that shape.
    n_inst = len(list(root.iter("instance")))
    n_pages = len(list(root.iter("page")))
    assert n_inst == n_pages, f"instance count {n_inst} != page count {n_pages}"
    for inst in root.iter("instance"):
        assert bool(inst.get("page")) ^ bool(inst.get("trans")), \
            "instance must have exactly one of page= / trans="
    return n_subst


def _validate_dtd(path):
    lxml = pytest.importorskip("lxml")
    from lxml import etree
    if not os.path.exists(DTD_PATH):
        pytest.skip("cpn.dtd not available")
    parser = etree.XMLParser(load_dtd=False, no_network=True, resolve_entities=False)
    tree = etree.parse(path, parser)
    dtd = etree.DTD(DTD_PATH)
    ok = dtd.validate(tree)
    assert ok, "DTD validation failed:\n" + "\n".join(
        str(e) for e in dtd.error_log.filter_from_errors()[:20])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

CASES = [
    ("Decision Tree", {"max_depth": 4, "random_state": 1}, 2),
    ("Random Forest", {"n_estimators": 4, "max_depth": 3, "random_state": 1}, 2),
    ("Gradient Boosting Decision Trees",
     {"n_estimators": 5, "max_depth": 2, "random_state": 1}, 2),
    ("Gradient Boosting Decision Trees",
     {"n_estimators": 4, "max_depth": 2, "random_state": 1}, 3),
]


@pytest.mark.parametrize("model,params,n_classes", CASES)
@pytest.mark.parametrize("which", ["initial", "final"])
def test_cpn_export_valid(tmp_path, model, params, n_classes, which):
    a, cols, sample = _analyzer(str(tmp_path), model, params, n_classes=n_classes)
    rules = (a.classifier.initial_rules if which == "initial"
             else a.classifier.final_rules)
    assert rules, "expected a non-empty rule set"
    path = export_cpn_tools(
        a.classifier, os.path.join(str(tmp_path), f"m_{which}.cpn"),
        rules=rules, feature_names=cols, sample=sample,
        use_final=(which == "final"), model_name="m")
    root = _validate_structure(path)
    _validate_hierarchy(root)
    _validate_dtd(path)


@pytest.mark.parametrize("model,params,n_classes", CASES)
@pytest.mark.parametrize("which", ["initial", "final"])
def test_cpn_prediction_consistency(tmp_path, model, params, n_classes, which):
    """The net must compute the same class as the classifier it came from.

    Structural validation says the file is a well-formed CPN; this checks what
    the net actually denotes, by reading the guards and arc expressions back
    out of the exported ``.cpn`` and evaluating them on real samples.
    """
    a, cols, sample = _analyzer(str(tmp_path), model, params, n_classes=n_classes)
    clf = a.classifier
    rules = clf.initial_rules if which == "initial" else clf.final_rules
    assert rules, "expected a non-empty rule set"

    path = export_cpn_tools(
        clf, os.path.join(str(tmp_path), f"sem_{which}.cpn"),
        rules=rules, feature_names=cols, sample=sample,
        use_final=(which == "final"), model_name="m")

    net = CPNNet(path)
    field = {name: f"f{i}" for i, name in enumerate(cols)}
    default_class = clf._default_class_int()

    # Compile the same rule set the net was built from, so both sides answer
    # for the same model.
    clf.compile_tree_arrays(rules, feature_names=cols)
    X = _samples_for(tmp_path, model, cols)

    mismatches = []
    for row in X:
        x = {field[c]: float(v) for c, v in zip(cols, row)}
        net_pred = evaluate(net, x, default_class=default_class)
        clf_pred = int(clf.predict_batch(row.reshape(1, -1), feature_names=cols)[0])
        if net_pred != clf_pred:
            mismatches.append((x, net_pred, clf_pred))

    assert not mismatches, (
        f"{len(mismatches)}/{len(X)} samples disagree between the .cpn and the "
        f"classifier; first: net={mismatches[0][1]} clf={mismatches[0][2]}"
    )


def _samples_for(tmp_path, model, cols):
    """Load the test split used to build the analyzer.

    Args:
        tmp_path: Temporary directory holding the generated CSVs.
        model (str): Model name (its prefix names the CSV files).
        cols (list): Feature column names.

    Returns:
        np.ndarray: Feature matrix of the test split.
    """
    te = os.path.join(str(tmp_path), f"{model[:4]}_test.csv")
    return pd.read_csv(te)[cols].to_numpy(dtype=float)


def test_export_hcpn_both(tmp_path):
    a, cols, sample = _analyzer(
        str(tmp_path), "Gradient Boosting Decision Trees",
        {"n_estimators": 4, "max_depth": 2, "random_state": 1})
    res = a.export_hcpn(os.path.join(str(tmp_path), "gbdt"),
                        which="both", sample=sample, feature_names=cols)
    assert set(res) == {"initial", "final"}
    for path in res.values():
        root = _validate_structure(path)
        _validate_hierarchy(root)


@pytest.mark.parametrize("model,params,n_classes", CASES)
def test_page_names_are_stable(tmp_path, model, params, n_classes):
    """Page names must not depend on the model name: ASK-CTL queries such as
    Mark.Top'Prediction have to work unchanged for DT, RF and GBDT."""
    a, cols, sample = _analyzer(str(tmp_path), model, params, n_classes=n_classes)
    path = export_cpn_tools(
        a.classifier, os.path.join(str(tmp_path), "n.cpn"),
        rules=a.classifier.initial_rules, feature_names=cols, sample=sample,
        use_final=False, model_name="whatever_label")
    root = ET.parse(path).getroot()
    names = [p.find("pageattr").get("name") for p in root.iter("page")]
    assert "Top" in names, f"no 'Top' page in {names}"
    assert not any("whatever_label" in n for n in names), \
        f"page names leak the model name: {names}"
    # The top page always exposes Input and Prediction under those exact names.
    for page in root.iter("page"):
        if page.find("pageattr").get("name") != "Top":
            continue
        places = {pl.find("text").text for pl in page.findall("place")}
        assert {"Input", "Prediction"} <= places, places


@pytest.mark.parametrize("model,params,n_classes", CASES)
def test_refinement_keeps_leaves_disjoint(tmp_path, model, params, n_classes):
    """Refinement must not leave overlapping guards inside a tree.

    Sibling promotion drops a rule's last condition. Ensemble trees often split
    on the same feature at the same threshold, so a sibling lookup that ignores
    the tree id matches a leaf of *another* tree and promotes it, generalizing
    it over ground its own tree still covers. The resulting net is
    nondeterministic: two leaf transitions enabled at once.
    """
    a, _cols, _sample = _analyzer(str(tmp_path), model, params, n_classes=n_classes)
    clf = a.classifier
    assert clf.find_overlapping_rules(rules=clf.initial_rules)["n_overlaps"] == 0
    ov = clf.find_overlapping_rules(rules=clf.final_rules)
    assert ov["n_overlaps"] == 0, (
        f"refined rules overlap in trees {ov['trees_affected']}: "
        f"{ov['overlaps'][:3]}")


def test_rf_between_tree_merge_is_refused(tmp_path):
    """Merging rules across trees is unsound under soft voting.

    Each tree casts one vote; collapsing N rules from N trees into one removes
    N-1 voters and can flip the argmax. The RF analyzer must warn and skip
    instead of silently producing a model that diverges from scikit-learn.
    """
    from pyruleanalyzer import RFAnalyzer

    tr, te, cols = _make_csv(str(tmp_path), "rfbt", n_classes=3, seed=5)
    a = PyRuleAnalyzer.create(train_path=tr, test_path=te, model="Random Forest",
                              params={"n_estimators": 6, "max_depth": 3,
                                      "random_state": 5},
                              refine=False, save_models=False)
    clf = a.classifier
    before = len(clf.initial_rules)
    with pytest.warns(RuntimeWarning, match="unsound for Random Forest"):
        RFAnalyzer(clf).execute_rule_refinement(
            file_path=te, remove_below_n_classifications=-1,
            refine_between_trees=True, save_final_model=False, save_report=False)
    # The stage was skipped, so no cross-tree collapse happened.
    assert len(clf.final_rules) == before


def test_sml_real_literals():
    assert sml_real(0.5) == "0.5"
    assert sml_real(3) == "3.0"
    assert sml_real(-2.5) == "~2.5"
    # scientific notation must be rewritten to SML form
    assert "E" in sml_real(1e-12) and "e" not in sml_real(1e-12)
    assert sml_real(-1e-12).startswith("~")
    assert sml_real(float("nan")) == "0.0"


def test_negative_values_in_marking_are_parenthesized():
    # CPN ML record-literal values must wrap negative reals in parens, e.g.
    # {f0=(~1.5)}, otherwise the initial marking fails to parse in CPN Tools.
    assert _ml_val(-1.5) == "(~1.5)"
    assert _ml_val(0.5) == "0.5"

    class _Clf:
        algorithm_type = "Decision Tree"
        _array_feature_names = ["a", "b"]
        default_class = 0
        final_rules = []
    exp = CPNToolsExporter(_Clf(), [], feature_names=["a", "b"],
                           sample={"a": -1.5, "b": 0.5})
    lit = exp._sample_literal()
    assert "f0=(~1.5)" in lit and "f1=0.5" in lit and "=~" not in lit
