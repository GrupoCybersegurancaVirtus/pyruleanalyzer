"""Saving and loading the classifier.

Provides ``SerializationMixin``: loading a pickled classifier (``load``) and
the JSON model format -- ``to_dict`` / ``from_dict`` and ``to_json`` /
``from_json`` with their strict encoders and decoders of plain values.

`RuleClassifier` is defined after its mixins, so the methods that name it import
it locally.
"""

import pickle
import numpy as np
from ._rules import Condition, Rule


class SerializationMixin:
    """Pickle loading and the JSON model format."""

    # Method to load the model
    @staticmethod
    def load(path):
        """
        Loads a saved RuleClassifier model from a pickle (.pkl) file.

        Args:
            path (str): Path to the .pkl file.

        Returns:
            RuleClassifier: The loaded classifier instance.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    # Identification of the JSON model format (see to_json)
    _JSON_FORMAT = 'pyruleanalyzer.RuleClassifier'
    # Version 2: missing-value routing of each condition, label table,
    # estimator class. Version 1 files are still read.
    _JSON_VERSION = 2
    _JSON_ESTIMATORS = {
        'Decision Tree': ('DecisionTreeClassifier', 'ExtraTreeClassifier'),
        'Random Forest': ('RandomForestClassifier', 'ExtraTreesClassifier'),
        'Gradient Boosting Decision Trees': ('GradientBoostingClassifier',),
    }
    _JSON_MAX_DEPTH = 64
    _JSON_ALGORITHMS = ('Decision Tree', 'Random Forest', 'Gradient Boosting Decision Trees')

    # Method to encode a plain value (model parameters) for JSON
    @staticmethod
    def _json_encode_plain(value, where='value', depth=0):
        """Encode a plain Python value (None, bool, number, str, list, dict) for JSON.

        Used for `model_parameters`. Non-finite floats become
        ``{"__float__": "inf"}`` and dicts whose keys are not plain strings
        (``class_weight={0: 1.0}``) become ``{"__items__": [[key, value], ...]}``,
        so they read back with their types. Anything else (an object, a
        function) is refused rather than written in a form that could only be
        read back by executing code.

        Args:
            value: The value to encode.
            where (str): Location of the value, for the error message.
            depth (int): Nesting depth (internal).

        Returns:
            A value made of JSON types only.

        Raises:
            ValueError: If the value is not plain data.
        """
        from .rule_classifier import RuleClassifier
        if depth > RuleClassifier._JSON_MAX_DEPTH:
            raise ValueError(f'{where}: nested too deeply to be saved as JSON.')
        if value is None or isinstance(value, (bool, str)):
            return value
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            value = float(value)
            if value != value:
                return {'__float__': 'nan'}
            if value in (float('inf'), float('-inf')):
                return {'__float__': 'inf' if value > 0 else '-inf'}
            return value
        if isinstance(value, (list, tuple)):
            return [RuleClassifier._json_encode_plain(v, f'{where}[{i}]', depth + 1)
                    for i, v in enumerate(value)]
        if isinstance(value, dict):
            if all(isinstance(k, str) and not k.startswith('__') for k in value):
                return {k: RuleClassifier._json_encode_plain(v, f'{where}[{k!r}]', depth + 1)
                        for k, v in value.items()}
            return {'__items__': [
                [RuleClassifier._json_encode_plain(k, f'{where} key', depth + 1),
                 RuleClassifier._json_encode_plain(v, f'{where}[{k!r}]', depth + 1)]
                for k, v in value.items()]}
        raise ValueError(
            f'{where}: a {type(value).__name__} cannot be saved as JSON (only '
            'None, bool, numbers, strings, lists and dicts can).'
        )

    # Method to decode a plain value written by _json_encode_plain
    @staticmethod
    def _json_decode_plain(value, where='value', depth=0):
        """Decode a value written by `_json_encode_plain`.

        Args:
            value: The decoded JSON value.
            where (str): Location of the value, for the error message.
            depth (int): Nesting depth (internal).

        Returns:
            The plain Python value.

        Raises:
            ValueError: If the value is malformed.
        """
        from .rule_classifier import RuleClassifier
        if depth > RuleClassifier._JSON_MAX_DEPTH:
            raise ValueError(f'{where}: nested too deeply.')
        if value is None or isinstance(value, (bool, str, int)):
            return value
        if isinstance(value, float):
            if value != value or value in (float('inf'), float('-inf')):
                raise ValueError(f'{where}: number out of range.')
            return value
        if isinstance(value, list):
            return [RuleClassifier._json_decode_plain(v, f'{where}[{i}]', depth + 1)
                    for i, v in enumerate(value)]
        if isinstance(value, dict):
            if set(value) == {'__float__'}:
                special = {'nan': float('nan'), 'inf': float('inf'), '-inf': float('-inf')}
                if not isinstance(value['__float__'], str) or value['__float__'] not in special:
                    raise ValueError(f'{where}: invalid special float.')
                return special[value['__float__']]
            if set(value) == {'__items__'}:
                items = value['__items__']
                if not isinstance(items, list):
                    raise ValueError(f'{where}: invalid __items__.')
                out = {}
                for item in items:
                    if not isinstance(item, list) or len(item) != 2:
                        raise ValueError(f'{where}: invalid __items__ entry.')
                    key = RuleClassifier._json_decode_plain(item[0], f'{where} key', depth + 1)
                    if isinstance(key, (list, dict)):
                        raise ValueError(f'{where}: invalid dict key.')
                    out[key] = RuleClassifier._json_decode_plain(item[1], where, depth + 1)
                return out
            if any(k.startswith('__') for k in value):
                raise ValueError(f'{where}: unknown tagged value {sorted(value)[:5]}.')
            return {k: RuleClassifier._json_decode_plain(v, f'{where}[{k!r}]', depth + 1)
                    for k, v in value.items()}
        raise ValueError(f'{where}: unexpected {type(value).__name__}.')

    # Method to write a class label as a JSON value
    @staticmethod
    def _json_label_out(value):
        """A label (or feature name) as str, or int for integer (numpy) labels."""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return int(value)
        return str(value)

    # Method to parse JSON text without evaluating anything
    @staticmethod
    def _json_loads(text):
        """Parse JSON text into plain values, refusing NaN/Infinity.

        `json` only builds dicts, lists, strings, numbers, booleans and None;
        nothing in the text is imported or executed.

        Args:
            text (str | bytes): The JSON text.

        Returns:
            The decoded value.

        Raises:
            ValueError: If the text is not valid JSON.
        """
        import json

        def reject(constant):
            """Refuse the non-standard constants NaN, Infinity and -Infinity."""
            raise ValueError(f'Invalid JSON: non-standard constant {constant}.')

        try:
            return json.loads(text, parse_constant=reject)
        except RecursionError:
            raise ValueError('Invalid JSON: nested too deeply.') from None

    # Method to read a JSON file without evaluating anything
    @staticmethod
    def _json_read_file(path):
        """Read and parse a JSON model file (see `_json_loads`).

        Args:
            path (str): Path to the file.

        Returns:
            The decoded value.

        Raises:
            ValueError: If the file is not valid UTF-8 JSON.
        """
        from .rule_classifier import RuleClassifier
        with open(path, 'rb') as f:
            data = f.read()
        try:
            text = data.decode('utf-8-sig')
        except UnicodeDecodeError as exc:
            raise ValueError(f'Invalid JSON model file: {exc}') from None
        return RuleClassifier._json_loads(text)

    # Method to serialise the classifier as plain JSON types
    def to_dict(self) -> dict:
        """The classifier as a dict of plain JSON types (the JSON model format).

        Rules are stored once in ``"rules"``; ``"initial_rules"``,
        ``"final_rules"`` and ``"specific_rules"`` list indices into it, so a
        rule shared by the initial and refined sets stays one object after
        loading, as it does with pickle. Compiled caches (native function,
        tree arrays) are not stored; they are rebuilt on loading.

        Returns:
            dict: The model; see `to_json`.

        Raises:
            ValueError: If the classifier has no rules (one loaded with
                `load_binary`) or a value cannot be represented in JSON.
        """
        from .rule_classifier import RuleClassifier
        if not self.initial_rules and not self.final_rules and getattr(self, '_arrays_compiled', False):
            raise ValueError(
                'This classifier has no rules (it was loaded from a binary '
                'model); the JSON format stores rules. Save it with to_binary().'
            )
        pool, index = [], {}

        def refs(rules):
            """Indices of `rules` in the pool, adding the new ones."""
            out = []
            for rule in rules or []:
                if id(rule) not in index:
                    index[id(rule)] = len(pool)
                    pool.append(rule)
                out.append(index[id(rule)])
            return out

        initial = refs(self.initial_rules)
        final = refs(self.final_rules)
        specific = refs(getattr(self, 'specific_rules', None))
        label = RuleClassifier._json_label_out
        init_scores = getattr(self, '_gbdt_init_scores', None)
        gbdt_classes = getattr(self, '_gbdt_classes', None)
        params = getattr(self, 'model_parameters', None)
        table = getattr(self, 'label_table', None)
        estimator = getattr(self, 'sklearn_estimator', None)
        # Version 2 adds the missing-value routing of the conditions, the
        # label table and the estimator class; a model without any of them
        # is written as version 1, exactly as before.
        v2 = (table is not None or estimator is not None
              or any(isinstance(c, Condition)
                     for r in pool for c in (r.parsed_conditions or [])))
        out = {
            'format': RuleClassifier._JSON_FORMAT,
            'version': RuleClassifier._JSON_VERSION if v2 else 1,
            'algorithm_type': self.algorithm_type,
            'rules': [rule.to_dict() for rule in pool],
            'initial_rules': initial,
            'final_rules': final,
            'specific_rules': specific,
            'class_labels': [label(c) for c in self.class_labels],
            'num_classes': int(self.num_classes),
            'default_class': label(self.default_class),
            'gbdt_init_scores': None if init_scores is None else
                [[label(k), float(v)] for k, v in init_scores.items()],
            'gbdt_is_binary': bool(getattr(self, '_gbdt_is_binary', False)),
            'gbdt_classes': None if gbdt_classes is None else [label(c) for c in gbdt_classes],
            'array_feature_names': [label(f) for f in getattr(self, '_array_feature_names', None) or []],
            'input_dtype': getattr(self, 'input_dtype', None),
            'fallback_activations': int(getattr(self, 'fallback_activations', 0) or 0),
            'model_parameters': None if params is None else
                RuleClassifier._json_encode_plain(params, 'model_parameters'),
        }
        if v2:
            out['label_table'] = None if table is None else [
                RuleClassifier._json_table_label(v) for v in table]
            out['sklearn_estimator'] = estimator
        return out

    # Method to write one entry of the label table as JSON
    @staticmethod
    def _json_table_label(value):
        """A label of the code -> label table as a JSON value.

        Args:
            value: The label (str, int, float or bool).

        Returns:
            The label as str, int, float or bool.

        Raises:
            ValueError: If the label cannot be written (not finite, other type).
        """
        if isinstance(value, (bool, str, int)):
            return value
        if isinstance(value, float) and value == value and abs(value) != float('inf'):
            return value
        raise ValueError(f'label_table: {value!r:.60} cannot be saved as JSON.')

    # Method to rebuild a classifier from to_dict output
    @classmethod
    def from_dict(cls, data) -> 'RuleClassifier':
        """Rebuild a classifier from `to_dict` output, validating every field.

        The result is in the state unpickling leaves a classifier in: the
        native function is recompiled for the refined (else initial) rules and
        the tree arrays are compiled on the first batch prediction.

        Args:
            data (dict): Output of `to_dict` (decoded from JSON).

        Returns:
            RuleClassifier: The classifier.

        Raises:
            ValueError: If the format, version or any field is invalid.
        """
        if not isinstance(data, dict):
            raise ValueError('Invalid JSON model: expected an object at the top level.')
        if data.get('format') != cls._JSON_FORMAT:
            raise ValueError(
                f'Invalid JSON model: format {data.get("format")!r:.60} is not '
                f'{cls._JSON_FORMAT!r}.'
            )
        version = data.get('version')
        if isinstance(version, bool) or version not in (1, cls._JSON_VERSION):
            raise ValueError(
                f'Unsupported JSON model version {version!r:.40} (this version of '
                f'pyruleanalyzer reads versions 1 to {cls._JSON_VERSION}).'
            )
        keys = ('format', 'version', 'algorithm_type', 'rules', 'initial_rules',
                'final_rules', 'specific_rules', 'class_labels', 'num_classes',
                'default_class', 'gbdt_init_scores', 'gbdt_is_binary', 'gbdt_classes',
                'array_feature_names', 'input_dtype', 'fallback_activations',
                'model_parameters')
        if version >= 2:
            keys += ('label_table', 'sklearn_estimator')
        missing = [k for k in keys if k not in data]
        if missing:
            raise ValueError(f'Invalid JSON model: missing field(s) {missing}.')
        unknown = [k for k in data if k not in keys]
        if unknown:
            raise ValueError(f'Invalid JSON model: unknown field(s) {sorted(unknown)[:10]}.')

        algorithm_type = data['algorithm_type']
        if algorithm_type not in cls._JSON_ALGORITHMS:
            raise ValueError(f'Invalid JSON model: unknown algorithm_type {algorithm_type!r:.60}.')
        if not isinstance(data['rules'], list):
            raise ValueError('Invalid JSON model: rules must be a list.')
        pool = [Rule.from_dict(r, f'rules[{i}]') for i, r in enumerate(data['rules'])]

        def rule_list(key):
            """The rules a list of pool indices refers to."""
            refs = data[key]
            if not isinstance(refs, list):
                raise ValueError(f'Invalid JSON model: {key} must be a list of indices.')
            out = []
            for i in refs:
                if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(pool):
                    raise ValueError(f'Invalid JSON model: {key} has an invalid rule index {i!r:.40}.')
                out.append(pool[i])
            return out

        def label_list(key, optional=False):
            """A list of labels (strings or integers)."""
            value = data[key]
            if value is None and optional:
                return None
            if not isinstance(value, list):
                raise ValueError(f'Invalid JSON model: {key} must be a list.')
            return [Rule._json_label(v, key) for v in value]

        init_scores = data['gbdt_init_scores']
        if init_scores is not None:
            if not isinstance(init_scores, list):
                raise ValueError('Invalid JSON model: gbdt_init_scores must be a list of pairs.')
            pairs = {}
            for pair in init_scores:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError('Invalid JSON model: gbdt_init_scores must be a list of pairs.')
                pairs[Rule._json_label(pair[0], 'gbdt_init_scores')] = \
                    Rule._json_number(pair[1], 'gbdt_init_scores')
            init_scores = pairs
        if not isinstance(data['gbdt_is_binary'], bool):
            raise ValueError('Invalid JSON model: gbdt_is_binary must be true or false.')
        if data['input_dtype'] not in (None, 'float32'):
            raise ValueError(f'Invalid JSON model: unsupported input_dtype {data["input_dtype"]!r:.40}.')
        params = data['model_parameters']
        if params is not None:
            if not isinstance(params, dict):
                raise ValueError('Invalid JSON model: model_parameters must be an object.')
            params = cls._json_decode_plain(params, 'model_parameters')
            if not isinstance(params, dict) or not all(isinstance(k, str) for k in params):
                raise ValueError('Invalid JSON model: model_parameters must map names to values.')

        table = data.get('label_table')
        if table is not None:
            if not isinstance(table, list) or not table:
                raise ValueError('Invalid JSON model: label_table must be a non-empty list or null.')
            for v in table:
                if not isinstance(v, (str, int, float, bool)) or (
                        isinstance(v, float) and (v != v or abs(v) == float('inf'))):
                    raise ValueError('Invalid JSON model: label_table holds a value '
                                     'that is not a string or a finite number.')
            if len(set(table)) != len(table):
                raise ValueError('Invalid JSON model: label_table repeats a label.')
        estimator = data.get('sklearn_estimator')
        if estimator is not None and estimator not in cls._JSON_ESTIMATORS[algorithm_type]:
            raise ValueError(f'Invalid JSON model: sklearn_estimator {estimator!r:.60} '
                             f'is not one of {cls._JSON_ESTIMATORS[algorithm_type]}.')

        obj = cls.__new__(cls)
        obj.algorithm_type = algorithm_type
        obj.initial_rules = rule_list('initial_rules')
        obj.final_rules = rule_list('final_rules')
        obj.specific_rules = rule_list('specific_rules')
        obj.duplicated_rules = []
        obj.class_labels = label_list('class_labels')
        obj.num_classes = Rule._json_count(data['num_classes'], 'num_classes')
        obj.default_class = Rule._json_label(data['default_class'], 'default_class')
        obj._gbdt_init_scores = init_scores
        obj._gbdt_is_binary = data['gbdt_is_binary']
        obj._gbdt_classes = label_list('gbdt_classes', optional=True)
        obj._array_feature_names = label_list('array_feature_names')
        obj.input_dtype = data['input_dtype']
        obj.fallback_activations = Rule._json_count(data['fallback_activations'],
                                                    'fallback_activations')
        if params is not None:
            obj.model_parameters = params
        obj.label_table = table
        obj.sklearn_estimator = estimator
        if table is not None and len(table) != obj.num_classes:
            raise ValueError('Invalid JSON model: label_table must have one '
                             'label per class (num_classes).')
        # Compiled caches: rebuilt below / on first use, as after unpickling.
        obj._tree_arrays = []
        obj._tree_class_groups = None
        obj._tree_is_init = None
        obj._arrays_compiled = False
        obj._arrays_source = None
        obj._arrays_source_len = None
        obj._array_states = []
        obj.native_fn = None
        obj._native_source = None
        rules_to_compile = obj.final_rules if obj.final_rules else obj.initial_rules
        if rules_to_compile:
            obj.update_native_model(rules_to_compile)
            # A pickle keeps the arrays compiled last; compile the default
            # set so predict_batch() without use_final works the same way.
            if obj._array_feature_names:
                try:
                    obj._set_array_state(obj._array_state_for_rules(
                        rules_to_compile, list(obj._array_feature_names)))
                except (ValueError, KeyError, IndexError, RuntimeError):
                    pass  # compiled on first use, which reports the problem
        return obj

    # Method to save the classifier as JSON
    def to_json(self, path, indent=None):
        """Save the classifier to a JSON file (the safe alternative to pickle).

        Loading a JSON model (`from_json`) only reads data: unlike a pickle,
        a JSON file from an untrusted source cannot execute code. Unlike the
        `.bin` format it keeps the `Rule` objects, so a loaded model can still
        be refined, edited, model-checked and exported.

        The file is an object with ``"format": "pyruleanalyzer.RuleClassifier"``
        and ``"version": 2`` (1 when the model records no missing-value
        routing, label table or estimator class, as files written before
        those existed); see `to_dict` for the fields. Version 2 adds
        ``"label_table"`` (original labels by code, or null),
        ``"sklearn_estimator"`` (e.g. ``"ExtraTreesClassifier"``, or null)
        and a fourth element to each parsed condition: whether a missing
        value satisfies it. Thresholds are
        written with ``repr`` and read back bit for bit.

        Args:
            path (str): Path to the .json file.
            indent (int, optional): Indentation passed to `json.dump`.

        Raises:
            ValueError: If the classifier cannot be represented (see `to_dict`).
        """
        import json
        text = json.dumps(self.to_dict(), allow_nan=False, indent=indent)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)

    # Method to load a classifier saved with to_json
    @classmethod
    def from_json(cls, path) -> 'RuleClassifier':
        """Load a classifier saved with `to_json`.

        Safe on untrusted files: the file is parsed as JSON and validated;
        nothing in it is imported or executed.

        Args:
            path (str): Path to the .json file.

        Returns:
            RuleClassifier: The classifier.

        Raises:
            ValueError: If the file is not a valid JSON model.
        """
        return cls.from_dict(cls._json_read_file(path))
