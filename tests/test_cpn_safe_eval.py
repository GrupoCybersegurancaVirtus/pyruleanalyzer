"""Expressions read from a ``.cpn`` must never run arbitrary code.

Guards, arc inscriptions and decision guards come from a file that anyone may
have written. They used to go through ``eval`` with empty ``__builtins__``,
which is not a sandbox. These tests check that the restricted evaluator
rejects attribute access, dunder names and the other constructs the exporter
never writes before anything runs, that it still computes exactly what the
classifier does on the nets the exporter writes, and that the mutant driver
keeps working on top of it.

The exporter part checks that a leaf without a class distribution puts its
one-hot weight in the column of its label, not at the label's numeric value.
"""

import os
import re
import sys
import types
from xml.sax.saxutils import escape

import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pyruleanalyzer.cpn_mutants import make_mutants
from pyruleanalyzer.cpn_semantics import (CPNNet, _parse_prob_list, evaluate,
                                          safe_eval, sml_to_python)
from pyruleanalyzer.cpn_tools_exporter import CPNToolsExporter, export_cpn_tools
from test_cpn_export import CASES, _analyzer, _samples_for

NETS = os.path.join(os.path.dirname(__file__), "cpn_nets")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Toucher:
    """Writes a marker file when its method is reached.

    Reaching it needs attribute access, which the evaluator must refuse.
    """

    def __init__(self, path):
        self.path = path

    def touch(self):
        with open(self.path, "w") as f:
            f.write("ran")
        return True


def _replace_guard(src, dst, payload, contains=None):
    """Copy a ``.cpn``, replacing the first matching guard with ``payload``.

    Args:
        src (str): The net to copy.
        dst (str): Where the edited net is written.
        payload (str): The new guard text (unescaped).
        contains (str, optional): Only replace a guard containing this text
            (as it appears in the XML).

    Returns:
        str: ``dst``.
    """
    with open(src, encoding="iso-8859-1") as f:
        text = f.read()
    pattern = re.compile(r"(<cond\b[^>]*>(?:(?!</cond>).)*?<text\b[^>]*>)"
                         r"([^<]+)(</text>)", re.S)
    done = []

    def swap(m):
        if done or (contains and contains not in m.group(2)):
            return m.group(0)
        done.append(True)
        return m.group(1) + escape(payload) + m.group(3)

    text = pattern.sub(swap, text)
    assert done, "no guard to replace"
    with open(dst, "w", encoding="iso-8859-1") as f:
        f.write(text)
    return dst


def _dt_net(tmp_path):
    a, cols, sample = _analyzer(str(tmp_path), "Decision Tree",
                                {"max_depth": 3, "random_state": 1})
    path = export_cpn_tools(a.classifier, os.path.join(str(tmp_path), "dt.cpn"),
                            rules=a.classifier.initial_rules, feature_names=cols,
                            sample=sample, use_final=False, model_name="m")
    return path, cols


# ---------------------------------------------------------------------------
# The evaluator itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expr,construct", [
    ("().__class__.__base__.__subclasses__()", "Attribute"),
    ("x.keys()", "Attribute"),
    ("__import__('os')", "__import__"),
    ("__builtins__", "__builtins__"),
    ("_private", "_private"),
    ("(lambda: 1)()", "Lambda"),
    ("[c for c in x]", "ListComp"),
    ("{k: 1 for k in x}", "DictComp"),
    ("(c for c in x)", "GeneratorExp"),
    ("2 ** 3", "Pow"),
    ("x['f0'] if True else 0", "IfExp"),
    ("{'a': 1}", "Dict"),
    ("x['f0':]", "Slice"),
])
def test_disallowed_constructs_are_named(expr, construct):
    with pytest.raises(ValueError, match=re.escape(construct)):
        safe_eval(expr, {"x": {"f0": 1.0}})


def test_rejection_happens_before_anything_runs(tmp_path):
    marker = tmp_path / "ran"
    env = {"w": _Toucher(str(marker)), "x": {"f0": 1.0}}
    for expr in ("w.touch()", "x['f0'] > 0 or w.touch()",
                 "False and w.touch()", "[w.touch()][0]"):
        with pytest.raises(ValueError, match="Attribute"):
            safe_eval(expr, env)
    assert not marker.exists()


def test_calls_are_limited_to_env_functions():
    assert safe_eval("f(2, 3)", {"f": lambda a, b: a + b}) == 5
    with pytest.raises(ValueError):
        safe_eval("g(1)", {"f": abs})
    with pytest.raises(ValueError, match="keyword"):
        safe_eval("f(1, k=2)", {"f": abs})
    with pytest.raises(ValueError):
        safe_eval("f()", {"f": 3})


def test_non_numeric_arithmetic_is_refused():
    with pytest.raises(ValueError):
        safe_eval("[0] * 3")
    with pytest.raises(ValueError):
        safe_eval("'a' + 'b'")


def test_non_string_input_is_a_type_error():
    with pytest.raises(TypeError):
        safe_eval(lambda env: 1)


@pytest.mark.parametrize("sml,x,expected", [
    ("[#f0 x <= 0.5 andalso #f1 x > ~1.5]", {"f0": 0.5, "f1": 0.0}, True),
    ("#f0 x <= 0.5 andalso #f1 x > (~1.5)", {"f0": 0.6, "f1": 0.0}, False),
    ("not ((#f0 x <= 0.5) orelse (#f0 x > 0.5 andalso #f1 x <= 2.0E~3))",
     {"f0": 0.7, "f1": 1.0}, True),
    ("true", {}, True),
    ("[false]", {}, False),
])
def test_exporter_guards_still_evaluate(sml, x, expected):
    assert bool(safe_eval(sml_to_python(sml), {"x": x})) is expected


def test_arc_literals_still_evaluate():
    assert safe_eval("[0.25, 0.75]") == [0.25, 0.75]
    assert safe_eval("(-0.5)") == -0.5
    assert safe_eval("sc0 > sc1 and sc0 >= sc2",
                     {"sc0": 1.0, "sc1": 0.5, "sc2": 1.0}) is True


# ---------------------------------------------------------------------------
# Crafted nets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", [
    "().__class__.__base__.__subclasses__()",
    "__import__('os')",
    "x.__class__",
])
def test_crafted_leaf_guard_is_rejected(tmp_path, payload):
    path, _cols = _dt_net(tmp_path)
    bad = _replace_guard(path, os.path.join(str(tmp_path), "bad.cpn"), payload)
    net = CPNNet(bad)
    x = net.sample_from_initial_marking()
    with pytest.raises(ValueError):
        evaluate(net, x)


def test_crafted_decide_guard_is_rejected(tmp_path):
    bad = _replace_guard(os.path.join(NETS, "gbdt_binary.cpn"),
                         os.path.join(str(tmp_path), "bad.cpn"),
                         "sc >= 0.0 orelse ().__class__", contains="sc &gt;=")
    net = CPNNet(bad)
    x = net.sample_from_initial_marking()
    with pytest.raises(ValueError, match="Attribute"):
        evaluate(net, x)


def test_model_checker_does_not_run_crafted_guards(tmp_path):
    """cpncheck compiles guards with its own CPN ML interpreter; a guard it
    cannot read must fail as an error, never as code."""
    from pyruleanalyzer.model_checker import check_cpn

    marker = tmp_path / "ran"
    path, _cols = _dt_net(tmp_path)
    bad = _replace_guard(path, os.path.join(str(tmp_path), "bad.cpn"),
                         f"__import__('os').remove({str(marker)!r})")
    marker.write_text("keep")
    try:
        check_cpn(bad, class_labels=[0, 1])
    except Exception:
        pass
    assert marker.exists()


# ---------------------------------------------------------------------------
# Generated nets still compute what the classifier does
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,params,n_classes", CASES)
def test_generated_nets_match_classifier(tmp_path, model, params, n_classes):
    a, cols, sample = _analyzer(str(tmp_path), model, params, n_classes=n_classes)
    clf = a.classifier
    rules = clf.final_rules
    path = export_cpn_tools(clf, os.path.join(str(tmp_path), "n.cpn"),
                            rules=rules, feature_names=cols, sample=sample,
                            use_final=True, model_name="m")
    net = CPNNet(path)
    clf.compile_tree_arrays(rules, feature_names=cols)
    default_class = clf._default_class_int()
    for row in _samples_for(tmp_path, model, cols):
        x = {f"f{i}": float(v) for i, v in enumerate(row)}
        expected = int(clf.predict_batch(row.reshape(1, -1), feature_names=cols)[0])
        assert evaluate(net, x, default_class=default_class) == expected


@pytest.mark.parametrize("name", ["gbdt_binary.cpn", "rf_multiclass.cpn"])
def test_mutants_still_build(tmp_path, name):
    made = make_mutants(os.path.join(NETS, name), str(tmp_path))
    assert made
    for m in made:
        assert os.path.exists(m["path"])
        CPNNet(m["path"])


def test_mutants_of_a_decision_tree_still_build(tmp_path):
    path, _cols = _dt_net(tmp_path)
    made = make_mutants(path, os.path.join(str(tmp_path), "mut"))
    assert made and all(os.path.exists(m["path"]) for m in made)


# ---------------------------------------------------------------------------
# Exporter: label -> distribution column
# ---------------------------------------------------------------------------

def _stub(labels, num_classes=None, class_labels=None):
    clf = types.SimpleNamespace(
        algorithm_type="Random Forest", _array_feature_names=["a"],
        default_class=labels[0], final_rules=[],
        _rf_labels=lambda n: list(labels)[:n])
    if num_classes is not None:
        clf.num_classes = num_classes
    if class_labels is not None:
        clf.class_labels = class_labels
    return clf


def _rule(label):
    return types.SimpleNamespace(name="r", class_=f"Class{label}",
                                 parsed_conditions=[("a", "<=", 0.5)],
                                 class_distribution=None)


def test_prob_literal_uses_label_column():
    exp = CPNToolsExporter(_stub([10, 20, 30], num_classes=3), [_rule(20)],
                           feature_names=["a"])
    assert _parse_prob_list((exp._prob_literal(_rule(20)))) == [0.0, 1.0, 0.0]
    assert _parse_prob_list((exp._prob_literal(_rule(10)))) == [1.0, 0.0, 0.0]


def test_prob_literal_falls_back_to_label_value():
    exp = CPNToolsExporter(_stub([0, 1, 2], num_classes=3), [_rule(1)],
                           feature_names=["a"])
    assert _parse_prob_list((exp._prob_literal(_rule(1)))) == [0.0, 1.0, 0.0]
    # A label the classifier does not know keeps the old reading.
    assert _parse_prob_list((exp._prob_literal(_rule(5)))) == [0.0] * 3


def test_n_classes_fallback_uses_label_columns():
    exp = CPNToolsExporter(_stub([10, 20, 30], class_labels=[10, 20, 30]),
                           [_rule(30), _rule(10)], feature_names=["a"])
    assert exp.n_classes == 3
