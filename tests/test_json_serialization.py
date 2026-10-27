"""Tests for the JSON model format (``save_json`` / ``load_json``).

The JSON format is the safe alternative to pickle: loading it only reads data.
Unlike the ``.bin`` export it keeps the ``Rule`` objects, so a loaded model
must behave exactly like the saved one -- same predictions from the initial
and the refined rules, bit-identical thresholds, refinable again, exportable --
and malformed or hostile files must be refused with ``ValueError``.
"""

import contextlib
import copy
import io
import json
import os
import pickle
import struct
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyruleanalyzer import PyRuleAnalyzer
from pyruleanalyzer.rule_classifier import Rule, RuleClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEATS = [f"f{i}" for i in range(5)]

MODELS = {
    "dt": lambda: DecisionTreeClassifier(max_depth=5, random_state=0),
    "rf": lambda: RandomForestClassifier(n_estimators=8, max_depth=4, random_state=0),
    "gbdt": lambda: GradientBoostingClassifier(n_estimators=8, max_depth=3, random_state=0),
}

# (model, labels): multiclass with 0..K-1, 1..3 and -1..1; binary GBDT too.
CASES = [
    (m, labels)
    for m in ("dt", "rf", "gbdt")
    for labels in ((0, 1, 2), (1, 2, 3), (-1, 0, 1))
] + [("gbdt", (0, 1)), ("gbdt", (-1, 1)), ("dt", (-1, 1)), ("rf", (0, 1))]

IDS = [f"{m}-{'_'.join(map(str, labels))}" for m, labels in CASES]


def _quiet(fn, *args, **kwargs):
    """Call fn with its progress prints silenced."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _data(labels, n=300, seed=3):
    """A dataset with named columns whose classes are `labels`."""
    X, y = make_classification(n_samples=n, n_features=5, n_informative=4,
                               n_redundant=0, n_classes=len(labels),
                               random_state=seed)
    return pd.DataFrame(X, columns=FEATS), np.asarray(labels)[y]


def _fitted(model, labels):
    """An analyzer extracted from a fitted estimator, plus its data."""
    X, y = _data(labels)
    est = MODELS[model]().fit(X, y)
    return PyRuleAnalyzer.from_sklearn(est), X, y


def _refine(analyzer, X, y):
    """Refine with a low-usage threshold, so rules are actually removed."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _quiet(analyzer.execute_rule_refinement, X=X, y=y,
               remove_below_n_classifications=2)


def _roundtrip(analyzer, tmp_path, name="model.json"):
    """Save `analyzer` with save_json and load it back."""
    path = str(tmp_path / name)
    analyzer.save_json(path)
    return PyRuleAnalyzer.load_json(path)


def _bits(x):
    """The IEEE-754 bit pattern of a float."""
    return struct.pack("<d", float(x))


def _assert_rules_identical(a, b):
    """Two rule lists hold the same rules, floats compared bit for bit."""
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        assert ra.name == rb.name
        assert ra.class_ == rb.class_ and type(ra.class_) is type(rb.class_)
        assert ra.conditions == rb.conditions
        assert ra.usage_count == rb.usage_count
        assert ra.error_count == rb.error_count
        assert ra.class_group == rb.class_group
        assert len(ra.parsed_conditions) == len(rb.parsed_conditions)
        for (va, oa, ta), (vb, ob, tb) in zip(ra.parsed_conditions, rb.parsed_conditions):
            assert (va, oa) == (vb, ob)
            assert _bits(ta) == _bits(tb)
        for key in ("leaf_value", "learning_rate", "contribution"):
            x, z = getattr(ra, key), getattr(rb, key)
            assert (x is None) == (z is None)
            if x is not None:
                assert _bits(x) == _bits(z)
        if ra.class_distribution is None:
            assert rb.class_distribution is None
        else:
            assert [_bits(v) for v in ra.class_distribution] == \
                [_bits(v) for v in rb.class_distribution]


def _assert_same_model(a, b, X):
    """Two analyzers hold the same model and predict the same."""
    ca, cb = a.classifier, b.classifier
    assert ca.algorithm_type == cb.algorithm_type
    _assert_rules_identical(ca.initial_rules, cb.initial_rules)
    _assert_rules_identical(ca.final_rules, cb.final_rules)
    assert ca.class_labels == cb.class_labels
    assert ca.num_classes == cb.num_classes
    assert ca.default_class == cb.default_class
    assert ca._gbdt_is_binary == cb._gbdt_is_binary
    assert ca._gbdt_classes == cb._gbdt_classes
    if ca._gbdt_init_scores is None:
        assert cb._gbdt_init_scores is None
    else:
        assert {k: _bits(v) for k, v in ca._gbdt_init_scores.items()} == \
            {k: _bits(v) for k, v in cb._gbdt_init_scores.items()}
    assert ca._array_feature_names == cb._array_feature_names
    assert ca.input_dtype == cb.input_dtype
    assert ca.model_parameters == cb.model_parameters
    assert a.feature_names == b.feature_names
    assert a.class_names == b.class_names
    for refined in (False, True):
        np.testing.assert_array_equal(a.predict(X, use_refined=refined),
                                      b.predict(X, use_refined=refined))
        # predict_proba may fail on some label sets (a separate issue); the
        # loaded model must then behave the same way.
        try:
            expected = a.predict_proba(X, use_refined=refined)
        except Exception as exc:  # noqa: BLE001
            with pytest.raises(type(exc)):
                b.predict_proba(X, use_refined=refined)
        else:
            np.testing.assert_array_equal(expected, b.predict_proba(X, use_refined=refined))
    # The single-sample engine (native function / iterative) too.
    for _, row in X.head(25).iterrows():
        sample = row.to_dict()
        for final in (False, True):
            assert ca.classify(sample, final=final)[0] == cb.classify(sample, final=final)[0]


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,labels", CASES, ids=IDS)
def test_roundtrip_before_refinement(model, labels, tmp_path):
    analyzer, X, _ = _fitted(model, labels)
    loaded = _roundtrip(analyzer, tmp_path)
    _assert_same_model(analyzer, loaded, X)
    assert sorted(loaded.classes_) == sorted(labels)


@pytest.mark.parametrize("model,labels", CASES, ids=IDS)
def test_roundtrip_after_refinement(model, labels, tmp_path):
    analyzer, X, y = _fitted(model, labels)
    _refine(analyzer, X, y)
    assert analyzer.classifier.final_rules
    loaded = _roundtrip(analyzer, tmp_path)
    _assert_same_model(analyzer, loaded, X)
    # Refined rules are the same objects as their initial counterparts, as
    # after unpickling.
    ids = {id(r) for r in loaded.classifier.initial_rules}
    orig_ids = {id(r) for r in analyzer.classifier.initial_rules}
    assert sum(id(r) in ids for r in loaded.classifier.final_rules) == \
        sum(id(r) in orig_ids for r in analyzer.classifier.final_rules)


@pytest.mark.parametrize("model,labels", CASES, ids=IDS)
def test_refine_again_after_loading(model, labels, tmp_path):
    analyzer, X, y = _fitted(model, labels)
    _refine(analyzer, X, y)
    loaded = _roundtrip(analyzer, tmp_path)
    twin = pickle.loads(pickle.dumps(analyzer))
    _refine(twin, X, y)
    _refine(loaded, X, y)
    _assert_same_model(twin, loaded, X)


@pytest.mark.parametrize("model,labels", CASES, ids=IDS)
def test_exports_after_loading(model, labels, tmp_path):
    analyzer, X, y = _fitted(model, labels)
    _refine(analyzer, X, y)
    loaded = _roundtrip(analyzer, tmp_path)

    py_path = str(tmp_path / "model.py")
    _quiet(loaded.to_python, py_path)
    with open(py_path, encoding="utf-8") as f:
        compile(f.read(), py_path, "exec")

    bin_path = str(tmp_path / "model.bin")
    _quiet(loaded.to_binary, bin_path)
    from_bin = PyRuleAnalyzer.load_binary(bin_path)
    np.testing.assert_array_equal(from_bin.predict(X.values), loaded.predict(X))

    paths = _quiet(loaded.export_hcpn, str(tmp_path / "net"), which="both",
                   sample=X.iloc[0])
    assert set(paths) == {"initial", "final"}
    assert all(os.path.getsize(p) > 0 for p in paths.values())


def test_thresholds_bit_identical_in_text(tmp_path):
    analyzer, X, _ = _fitted("rf", (0, 1, 2))
    path = str(tmp_path / "m.json")
    analyzer.save_json(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    thresholds = [c[2] for r in data["classifier"]["rules"] for c in r["parsed_conditions"]]
    originals = [t for r in analyzer.classifier.initial_rules for _, _, t in r.parsed_conditions]
    assert sorted(map(_bits, thresholds)) == sorted(map(_bits, originals))
    # And through repr, which is what json writes.
    for t in originals:
        assert _bits(json.loads(json.dumps(float(t)))) == _bits(t)


def test_every_rule_slot_is_written():
    analyzer, _, _ = _fitted("gbdt", (0, 1, 2))
    rule = analyzer.classifier.initial_rules[1]
    assert set(rule.to_dict()) == set(Rule.__slots__)
    copy_ = Rule.from_dict(json.loads(json.dumps(rule.to_dict())))
    _assert_rules_identical([rule], [copy_])


def test_save_and_load_pick_format_by_extension(tmp_path):
    analyzer, X, y = _fitted("dt", (1, 2, 3))
    _refine(analyzer, X, y)
    json_path = str(tmp_path / "m.json")
    pkl_path = str(tmp_path / "m.pkl")
    analyzer.save(json_path)
    analyzer.save(pkl_path)
    with open(json_path, encoding="utf-8") as f:
        assert json.load(f)["format"] == "pyruleanalyzer.PyRuleAnalyzer"
    with open(pkl_path, "rb") as f:
        assert f.read(1) == b"\x80"  # a pickle, for backward compatibility
    _assert_same_model(analyzer, PyRuleAnalyzer.load(json_path), X)
    _assert_same_model(analyzer, PyRuleAnalyzer.load(pkl_path), X)


def test_rule_classifier_to_json_from_json(tmp_path):
    analyzer, X, y = _fitted("gbdt", (-1, 1))
    _refine(analyzer, X, y)
    path = str(tmp_path / "clf.json")
    analyzer.classifier.to_json(path, indent=1)
    clf = RuleClassifier.from_json(path)
    _assert_rules_identical(analyzer.classifier.final_rules, clf.final_rules)
    np.testing.assert_array_equal(
        clf.predict_batch(X.values, feature_names=FEATS, use_final=True),
        analyzer.classifier.predict_batch(X.values, feature_names=FEATS, use_final=True))
    # The default (refined) set is compiled on loading, as a pickle keeps it.
    np.testing.assert_array_equal(
        clf.predict_batch(X.values, feature_names=FEATS),
        analyzer.classifier.predict_batch(X.values, feature_names=FEATS, use_final=True))
    # load_json wraps a bare classifier file.
    wrapped = PyRuleAnalyzer.load_json(path)
    np.testing.assert_array_equal(wrapped.predict(X), analyzer.predict(X))


def test_model_parameters_keep_their_types(tmp_path):
    X, y = _data((0, 1, 2))
    est = DecisionTreeClassifier(max_depth=4, class_weight={0: 1.0, 1: 2.0, 2: 1.5},
                                 min_impurity_decrease=0.0, random_state=0).fit(X, y)
    analyzer = PyRuleAnalyzer.from_sklearn(est)
    analyzer.classifier.model_parameters["max_leaf_nodes"] = np.int64(40)
    analyzer.classifier.model_parameters["ccp_alpha"] = float("inf")
    loaded = _roundtrip(analyzer, tmp_path)
    params = loaded.get_params()
    assert params["class_weight"] == {0: 1.0, 1: 2.0, 2: 1.5}
    assert params["max_leaf_nodes"] == 40 and type(params["max_leaf_nodes"]) is int
    assert params["ccp_alpha"] == float("inf")


def test_unpicklable_hyperparameter_is_refused(tmp_path):
    analyzer, _, _ = _fitted("dt", (0, 1, 2))
    analyzer.classifier.model_parameters["random_state"] = np.random.RandomState(0)
    with pytest.raises(ValueError, match="random_state"):
        analyzer.save_json(str(tmp_path / "m.json"))


def test_untrained_analyzer_roundtrip(tmp_path):
    analyzer = PyRuleAnalyzer.new_model("Random Forest", {"n_estimators": 5})
    loaded = _roundtrip(analyzer, tmp_path)
    assert loaded.get_params() == analyzer.get_params()
    X, y = _data((0, 1, 2))
    loaded.fit(X, y)
    assert len(loaded.predict(X)) == len(X)


def test_binary_loaded_classifier_is_refused(tmp_path):
    analyzer, _, _ = _fitted("dt", (0, 1, 2))
    _quiet(analyzer.to_binary, str(tmp_path / "m.bin"))
    from_bin = PyRuleAnalyzer.load_binary(str(tmp_path / "m.bin"))
    with pytest.raises(ValueError, match="binary"):
        from_bin.save_json(str(tmp_path / "m.json"))


# ---------------------------------------------------------------------------
# Malformed and hostile files
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def saved():
    """A valid saved analyzer (refined GBDT), as a decoded JSON object."""
    import tempfile
    analyzer, X, y = _fitted("gbdt", (0, 1, 2))
    _refine(analyzer, X, y)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.json")
        analyzer.save_json(path)
        with open(path, encoding="utf-8") as f:
            return json.load(f)


def _load_obj(obj, tmp_path):
    """Write `obj` as JSON and load it as an analyzer."""
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(obj), encoding="utf-8")
    return PyRuleAnalyzer.load_json(str(path))


def _load_text(text, tmp_path):
    """Write raw text and load it as an analyzer."""
    path = tmp_path / "bad.json"
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")
    return PyRuleAnalyzer.load_json(str(path))


def test_valid_fixture_loads(saved, tmp_path):
    assert _load_obj(saved, tmp_path).classifier.final_rules


def _mutate(saved, fn):
    """A deep copy of `saved` changed by `fn`."""
    obj = copy.deepcopy(saved)
    fn(obj)
    return obj


BAD_MUTATIONS = {
    "version_2": lambda o: o.update(version=2),
    "version_str": lambda o: o.update(version="1"),
    "version_bool": lambda o: o.update(version=True),
    "classifier_version_3": lambda o: o["classifier"].update(version=3),
    "format_wrong": lambda o: o.update(format="pickle"),
    "classifier_format_wrong": lambda o: o["classifier"].update(format="x"),
    "missing_feature_names": lambda o: o.pop("feature_names"),
    "missing_classifier": lambda o: o.pop("classifier"),
    "missing_rules": lambda o: o["classifier"].pop("rules"),
    "missing_class_labels": lambda o: o["classifier"].pop("class_labels"),
    "missing_rule_name": lambda o: o["classifier"]["rules"][0].pop("name"),
    "missing_rule_threshold_list": lambda o: o["classifier"]["rules"][3].pop("parsed_conditions"),
    "extra_top_key": lambda o: o.update(extra=1),
    "extra_classifier_key": lambda o: o["classifier"].update(__class__="os.system"),
    "extra_rule_key": lambda o: o["classifier"]["rules"][0].update(__reduce__=["os.system", ["x"]]),
    "classifier_not_object": lambda o: o.update(classifier=[1, 2]),
    "rules_not_list": lambda o: o["classifier"].update(rules={"a": 1}),
    "rule_not_object": lambda o: o["classifier"]["rules"].__setitem__(0, "rule"),
    "threshold_list": lambda o: o["classifier"]["rules"][3]["parsed_conditions"][0].__setitem__(2, [1]),
    "threshold_bool": lambda o: o["classifier"]["rules"][3]["parsed_conditions"][0].__setitem__(2, True),
    "condition_short": lambda o: o["classifier"]["rules"][3]["parsed_conditions"].__setitem__(0, ["f0", "<="]),
    "operator_bad": lambda o: o["classifier"]["rules"][3]["parsed_conditions"][0].__setitem__(1, "in"),
    "usage_negative": lambda o: o["classifier"]["rules"][0].update(usage_count=-1),
    "usage_float": lambda o: o["classifier"]["rules"][0].update(usage_count=1.5),
    "usage_huge": lambda o: o["classifier"]["rules"][0].update(usage_count=10 ** 40),
    "leaf_value_str": lambda o: o["classifier"]["rules"][0].update(leaf_value="0.1"),
    "class_list": lambda o: o["classifier"]["rules"][0].update(class_=[0]),
    "class_float": lambda o: o["classifier"]["rules"][0].update(class_=1.5),
    "distribution_bad": lambda o: o["classifier"]["rules"][0].update(class_distribution=[None]),
    "conditions_not_strings": lambda o: o["classifier"]["rules"][3].update(conditions=[1, 2]),
    "algorithm_unknown": lambda o: o["classifier"].update(algorithm_type="os.system"),
    "rule_index_out_of_range": lambda o: o["classifier"]["final_rules"].append(10 ** 6),
    "rule_index_negative": lambda o: o["classifier"]["initial_rules"].append(-1),
    "rule_index_str": lambda o: o["classifier"]["initial_rules"].append("0"),
    "num_classes_str": lambda o: o["classifier"].update(num_classes="3"),
    "is_binary_int": lambda o: o["classifier"].update(gbdt_is_binary=0),
    "init_scores_dict": lambda o: o["classifier"].update(gbdt_init_scores={"0": 1.0}),
    "init_scores_bad_pair": lambda o: o["classifier"].update(gbdt_init_scores=[["0", "x"]]),
    "input_dtype_bad": lambda o: o["classifier"].update(input_dtype="object"),
    "params_list": lambda o: o["classifier"].update(model_parameters=[1]),
    "params_tag_unknown": lambda o: o["classifier"]["model_parameters"].update(
        x={"__class__": "os.system"}),
    "params_bad_float_tag": lambda o: o["classifier"]["model_parameters"].update(
        x={"__float__": "__import__('os')"}),
    "params_deep": lambda o: o["classifier"]["model_parameters"].update(
        x=json.loads("[" * 200 + "]" * 200)),
    "feature_names_nested": lambda o: o.update(feature_names=[["f0"]]),
    "class_names_null": lambda o: o.update(class_names=None),
}


@pytest.mark.parametrize("name", sorted(BAD_MUTATIONS))
def test_malformed_model_raises_value_error(name, saved, tmp_path):
    with pytest.raises(ValueError):
        _load_obj(_mutate(saved, BAD_MUTATIONS[name]), tmp_path)


@pytest.mark.parametrize("text", [
    "",
    "not json",
    "[]",
    "null",
    '"pyruleanalyzer"',
    "{",
    "[" * 100000 + "]" * 100000,
    '{"format": "pyruleanalyzer.PyRuleAnalyzer", "version": NaN}',
    '{"format": "pyruleanalyzer.PyRuleAnalyzer", "version": 1e999}',
    '{"format": "pyruleanalyzer.PyRuleAnalyzer", "version": ' + "9" * 5000 + "}",
    b"\xff\xfe\x00garbage",
], ids=["empty", "text", "list", "null", "string", "truncated", "deep",
        "nan", "inf", "huge_int", "not_utf8"])
def test_invalid_json_text_raises_value_error(text, tmp_path):
    with pytest.raises(ValueError):
        _load_text(text, tmp_path)


@pytest.mark.parametrize("literal", ["1e999", "-1e999", "NaN", "Infinity"])
def test_huge_threshold_raises_value_error(literal, saved, tmp_path):
    text = json.dumps(saved)
    rule = saved["classifier"]["rules"][3]
    value = rule["parsed_conditions"][0][2]
    old = json.dumps(rule["parsed_conditions"][0])
    new = json.dumps(rule["parsed_conditions"][0][:2])[:-1] + ", " + literal + "]"
    assert old in text, value
    with pytest.raises(ValueError):
        _load_text(text.replace(old, new, 1), tmp_path)


def test_code_like_strings_are_only_data(saved, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    payload = "__import__('os').system('echo pwned > PWNED')"
    marker = tmp_path / "PWNED"

    def hostile(o):
        """Put the payload wherever a string is allowed."""
        clf = o["classifier"]
        clf["rules"][0]["name"] = payload
        clf["model_parameters"]["criterion"] = payload
        clf["model_parameters"]["os.system('echo pwned')"] = [payload, {"k": payload}]
        clf["rules"][3]["conditions"][0] = payload
        o["class_names"] = o["class_names"] + [payload]

    loaded = _load_obj(_mutate(saved, hostile), tmp_path)
    assert not marker.exists()
    assert loaded.classifier.initial_rules[0].name == payload
    assert loaded.classifier.initial_rules[3].conditions[0] == payload
    assert loaded.classifier.model_parameters["criterion"] == payload
    assert loaded.classifier.model_parameters["os.system('echo pwned')"] == \
        [payload, {"k": payload}]
    assert loaded.class_names[-1] == payload
    assert not marker.exists()


def test_dunder_keys_round_trip_as_data(tmp_path):
    """A hyperparameter dict keyed like a tag is written as __items__ and read back."""
    analyzer, _, _ = _fitted("dt", (0, 1, 2))
    odd = {"__import__('os')": 1, "__float__": "inf", 3: None}
    analyzer.classifier.model_parameters["extra"] = odd
    loaded = _roundtrip(analyzer, tmp_path)
    assert loaded.classifier.model_parameters["extra"] == odd
