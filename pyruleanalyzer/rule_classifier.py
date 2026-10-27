import os
import re
import sys
import copy
import time
import pickle
import struct
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from pyruleanalyzer._accel import (
    traverse_tree_batch as _accel_traverse,
    traverse_tree_batch_multi as _accel_traverse_multi,
    HAS_C_EXTENSION as _HAS_C_EXT,
)


# Function to order tree identifiers by estimator index.
def _tree_sort_key(tid):
    """Sort key that orders tree ids by their numbers, not as text.

    Tree ids carry the estimator index ('DT10', 'GBDT1T12'). Sorted as text,
    'DT10' precedes 'DT2', so the trees' outputs would be added in another
    order than scikit-learn adds its estimators; floating-point addition is
    not associative, so the sums could differ in the last bit and flip a
    decision that sits on a tie (RF) or on score 0 (binary GBDT).

    Args:
        tid (str): Tree identifier.

    Returns:
        list: Alternating text and integer parts of `tid`.
    """
    return [int(p) if p.isdigit() else p for p in re.split(r'(\d+)', str(tid))]


# Class to represent a parsed condition whose missing-value routing is known
class Condition(tuple):
    """A parsed condition ``(feature, operator, threshold)`` that a missing
    value (NaN) does NOT satisfy.

    scikit-learn (>= 1.3) routes a NaN at every split to the side recorded in
    ``tree_.missing_go_to_left``, so of the two conditions a split creates
    (``f <= t`` and ``f > t``) exactly one is satisfied by NaN. Conditions
    read from a fitted tree are `Condition` (NaN goes the other way) or
    `MissingCondition` (NaN goes this way). A plain tuple -- a rule parsed
    from text, an old pickle or JSON file -- has no known routing, and
    the engines refuse NaN inputs for a rule set that contains one.

    It is still a 3-tuple: ``var, op, value = cond``, slicing, hashing and
    equality behave as before (equality ignores the routing; compare
    `condition_signature` where it matters).
    """

    __slots__ = ()

    # NaN does not satisfy this condition.
    missing = False

    # Method to execute __repr__
    def __repr__(self):
        """Show the condition like the tuple it is."""
        return tuple.__repr__(self)


# Class to represent a parsed condition that a missing value satisfies
class MissingCondition(Condition):
    """A parsed condition that a missing value (NaN) satisfies: the split
    sends NaN to this side (see `Condition`)."""

    __slots__ = ()

    # NaN satisfies this condition.
    missing = True


# Function to build a condition with its missing-value routing
def make_condition(var, op, value, missing=None):
    """A parsed condition carrying its missing-value routing.

    Args:
        var (str): Feature name.
        op (str): Operator.
        value (float): Threshold.
        missing (bool | None): Whether NaN satisfies the condition; None
            for unknown (a plain tuple).

    Returns:
        tuple: A `MissingCondition`, a `Condition` or a plain tuple.
    """
    if missing is None:
        return (var, op, value)
    return (MissingCondition if missing else Condition)((var, op, value))


# Function to read the missing-value routing of a condition
def condition_missing(cond):
    """Whether NaN satisfies `cond`: True, False, or None when unknown.

    Args:
        cond (tuple): A parsed condition.

    Returns:
        bool | None: The routing.
    """
    if isinstance(cond, Condition):
        return cond.missing
    return None


# Function to compare conditions including their missing-value routing
def condition_signature(cond):
    """The condition with its routing, for comparisons where NaN matters.

    Args:
        cond (tuple): A parsed condition.

    Returns:
        tuple: ``(var, op, value, missing)``.
    """
    return (cond[0], cond[1], cond[2], condition_missing(cond))


# Function to format a float as a Python literal
def _py_float(value) -> str:
    """`value` as a Python expression that evaluates to the same float.

    ``repr(float('inf'))`` is ``inf``, which is not a Python literal; the
    split that sends only the missing values to the right has threshold +inf.

    Args:
        value (float): The number.

    Returns:
        str: The literal.
    """
    value = float(value)
    if value != value:
        return "float('nan')"
    if value in (float('inf'), float('-inf')):
        return "float('inf')" if value > 0 else "float('-inf')"
    return repr(value)


# Function to generate the Python test of one split
def _py_split_test(feat, threshold, missing_left) -> str:
    """The Python test that sends a sample to the left child of a split.

    ``x <= t`` is False for NaN, so NaN goes right; ``not (x > t)`` is the
    same test for every number but True for NaN, so NaN goes left. One
    lookup either way.

    Args:
        feat (str): Feature name.
        threshold (float): Split threshold.
        missing_left (bool | None): Whether NaN goes left.

    Returns:
        str: The condition, e.g. ``sample['f0'] <=0.5`` (the spacing the
        generated code always had).
    """
    if missing_left:
        return f"not (sample[{feat!r}] > {_py_float(threshold)})"
    return f"sample[{feat!r}] <={_py_float(threshold)}"


# Class to represent a rule
class Rule:
    """
    Represents a complete ruleset (path from root to leaf) in a decision tree.
    
    This class is not designed to be operated directly, but by instead using the main 
    :ref:`RuleClassifier<rule_classifier>` class.

    Attributes:
        name (str): Name of the rule (e.g., "DT1_Rule36_Class0").
        class_ (str): Class that the rule assigns to matching instances.
        conditions (List[str]): List of condition strings (like "v2 > 0.5").
        usage_count (int): Number of times the rule matched during classification.
        error_count (int): Number of times the rule matched but the prediction was wrong.
        parsed_conditions (List[Tuple[str, str, float]]): Cached parsed conditions for faster access.
            Conditions read from a fitted tree are `Condition` /
            `MissingCondition` tuples, which also record whether a missing
            value (NaN) satisfies them.
    """

    __slots__ = [
        'name', 'class_', 'conditions', 'usage_count', 'error_count',
        'parsed_conditions',
        # RF soft-voting: per-class probability distribution at this leaf
        'class_distribution',
        # GBDT-specific fields (None for DT/RF rules)
        'leaf_value', 'learning_rate', 'contribution', 'class_group',
    ]

    # Method to execute __init__
    def __init__(self, name, class_, conditions, leaf_value=None,
                 learning_rate=None, class_group=None, class_distribution=None):
        """
        Initializes a new Rule instance representing a decision path.

        Args:
            name (str): Unique identifier for the rule.
            class_ (str): Target class label as a string.
            conditions (List[str]): List of conditions required to trigger this rule.
            leaf_value (float, optional): Raw residual value at the leaf (GBDT only).
            learning_rate (float, optional): Learning rate for this tree (GBDT only).
            class_group (str, optional): The class group this tree contributes to (GBDT only).
            class_distribution (List[float], optional): Per-class probability distribution
                at the leaf node (RF soft-voting). Index i corresponds to class i.
        """
        self.name = name
        self.class_ = class_
        self.conditions = conditions
        self.usage_count = 0
        self.error_count = 0
        self.parsed_conditions = []  # Populated by parse_conditions_static
        # RF soft-voting distribution
        self.class_distribution = class_distribution
        # GBDT-specific
        self.leaf_value = leaf_value
        self.learning_rate = learning_rate
        self.class_group = class_group
        if leaf_value is not None and learning_rate is not None:
            self.contribution = learning_rate * leaf_value
        elif leaf_value is not None:
            self.contribution = leaf_value  # Init score (no LR scaling)
        else:
            self.contribution = None

    # Method to execute __repr__
    def __repr__(self):
        """
        __repr__ method.
        Returns:
            Description of return value.
        """
        return f"Rule(name={self.name}, class={self.class_}, conditions={self.conditions})"

    # Operators a parsed condition may use (see parse_conditions_static)
    _JSON_OPERATORS = ('<=', '>', '<', '>=', '==')

    # Method to check a JSON value is a finite number
    @staticmethod
    def _json_number(value, where, optional=False):
        """A JSON number as a finite float.

        Args:
            value: The decoded JSON value.
            where (str): Location of the value, for the error message.
            optional (bool): Whether None is accepted.

        Returns:
            float | None: The value as a float (None only when `optional`).

        Raises:
            ValueError: If the value is not a finite number.
        """
        if value is None and optional:
            return None
        # bool is an int subclass: True is not a threshold.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{where}: expected a number, got {type(value).__name__}.')
        try:
            result = float(value)
        except OverflowError:
            raise ValueError(f'{where}: number out of range.') from None
        if result != result or result in (float('inf'), float('-inf')):
            raise ValueError(f'{where}: expected a finite number, got {value!r}.')
        return result

    # Method to check a JSON value is a non-negative count
    @staticmethod
    def _json_count(value, where):
        """A JSON integer as a non-negative count below 2**63.

        Args:
            value: The decoded JSON value.
            where (str): Location of the value, for the error message.

        Returns:
            int: The count.

        Raises:
            ValueError: If the value is not such an integer.
        """
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2 ** 63:
            raise ValueError(f'{where}: expected a non-negative integer, got {value!r:.40}.')
        return value

    # Method to check a JSON value is a class label
    @staticmethod
    def _json_label(value, where, optional=False):
        """A class label (str or int) read from JSON.

        Args:
            value: The decoded JSON value.
            where (str): Location of the value, for the error message.
            optional (bool): Whether None is accepted.

        Returns:
            str | int | None: The label.

        Raises:
            ValueError: If the value is not a string or an integer.
        """
        if value is None and optional:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool) and abs(value) < 2 ** 63:
            return value
        raise ValueError(f'{where}: expected a class label (string or integer), got {value!r:.40}.')

    # Method to check a JSON value is a list of strings
    @staticmethod
    def _json_str_list(value, where):
        """A JSON list of strings.

        Args:
            value: The decoded JSON value.
            where (str): Location of the value, for the error message.

        Returns:
            List[str]: A new list with the strings.

        Raises:
            ValueError: If the value is not a list of strings.
        """
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError(f'{where}: expected a list of strings.')
        return list(value)

    # Method to serialise the rule as plain JSON types
    def to_dict(self) -> dict:
        """The rule as a dict of plain JSON types (see `RuleClassifier.to_json`).

        Every slot is written. Thresholds and leaf values are Python floats,
        which `json` writes with `repr`, so they read back bit for bit.

        Returns:
            dict: One key per slot of the rule.

        Raises:
            ValueError: If a slot holds a value JSON cannot represent.
        """
        def num(v):
            """Python float of a (numpy) number, None kept."""
            return None if v is None else float(v)

        def label(v):
            """A label as str, or int for integer (numpy) labels."""
            if v is None or isinstance(v, str):
                return v
            if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                return int(v)
            return str(v)

        parsed = []
        for cond in self.parsed_conditions or []:
            var, op, value = cond
            if not isinstance(value, str):
                value = float(value)
                if value in (float('inf'), float('-inf')):
                    # The split that isolates the missing values has
                    # threshold +inf, which JSON has no number for.
                    value = {'__float__': 'inf' if value > 0 else '-inf'}
            entry = [str(var), str(op), value]
            # Version 2: whether a missing value satisfies the condition.
            missing = condition_missing(cond)
            if missing is not None:
                entry.append(bool(missing))
            parsed.append(entry)
        dist = self.class_distribution
        out = {
            'name': str(self.name),
            'class_': label(self.class_),
            'conditions': [str(c) for c in (self.conditions or [])],
            'parsed_conditions': parsed,
            'usage_count': int(self.usage_count or 0),
            'error_count': int(self.error_count or 0),
            'class_distribution': None if dist is None else [float(v) for v in dist],
            'leaf_value': num(self.leaf_value),
            'learning_rate': num(self.learning_rate),
            'contribution': num(self.contribution),
            'class_group': label(self.class_group),
        }
        # Slots added after this format was written: plain values only.
        for slot in self.__slots__:
            if slot not in out:
                out[slot] = RuleClassifier._json_encode_plain(
                    getattr(self, slot, None), f'rule {self.name!r}.{slot}')
        return out

    # Method to rebuild a rule from `to_dict` output
    @classmethod
    def from_dict(cls, data, where='rule') -> 'Rule':
        """Rebuild a rule from `to_dict` output, validating every field.

        Only plain values are read; nothing in `data` is evaluated.

        Args:
            data (dict): Output of `to_dict` (decoded from JSON).
            where (str): Location of the rule, for error messages.

        Returns:
            Rule: The rule.

        Raises:
            ValueError: If a field is missing, unknown or of the wrong type.
        """
        if not isinstance(data, dict):
            raise ValueError(f'{where}: expected an object, got {type(data).__name__}.')
        known = ('name', 'class_', 'conditions', 'parsed_conditions', 'usage_count',
                 'error_count', 'class_distribution', 'leaf_value', 'learning_rate',
                 'contribution', 'class_group')
        missing = [k for k in known if k not in data]
        if missing:
            raise ValueError(f'{where}: missing field(s) {missing}.')
        unknown = [k for k in data if k not in cls.__slots__]
        if unknown:
            raise ValueError(f'{where}: unknown field(s) {sorted(map(str, unknown))[:10]}.')

        rule = cls.__new__(cls)
        if not isinstance(data['name'], str):
            raise ValueError(f'{where}.name: expected a string.')
        rule.name = data['name']
        where = f'{where} {rule.name!r:.60}'
        rule.class_ = cls._json_label(data['class_'], f'{where}.class_')
        rule.conditions = cls._json_str_list(data['conditions'], f'{where}.conditions')

        parsed = data['parsed_conditions']
        if not isinstance(parsed, list):
            raise ValueError(f'{where}.parsed_conditions: expected a list.')
        rule.parsed_conditions = []
        for i, cond in enumerate(parsed):
            at = f'{where}.parsed_conditions[{i}]'
            if (not isinstance(cond, list) or len(cond) not in (3, 4)
                    or not isinstance(cond[0], str) or cond[1] not in cls._JSON_OPERATORS
                    or (len(cond) == 4 and not isinstance(cond[3], bool))):
                raise ValueError(
                    f'{at}: expected [feature, operator, threshold] or '
                    '[feature, operator, threshold, missing].')
            raw = cond[2]
            if isinstance(raw, str):
                value = raw
            elif isinstance(raw, dict):
                # +/-inf only (a split isolating the missing values).
                if (set(raw) != {'__float__'}
                        or raw['__float__'] not in ('inf', '-inf')):
                    raise ValueError(f'{at}: invalid threshold.')
                value = float(raw['__float__'])
            else:
                value = cls._json_number(raw, at)
            rule.parsed_conditions.append(make_condition(
                cond[0], cond[1], value, cond[3] if len(cond) == 4 else None))

        rule.usage_count = cls._json_count(data['usage_count'], f'{where}.usage_count')
        rule.error_count = cls._json_count(data['error_count'], f'{where}.error_count')

        dist = data['class_distribution']
        if dist is not None:
            if not isinstance(dist, list):
                raise ValueError(f'{where}.class_distribution: expected a list or null.')
            dist = [cls._json_number(v, f'{where}.class_distribution') for v in dist]
        rule.class_distribution = dist
        for key in ('leaf_value', 'learning_rate', 'contribution'):
            setattr(rule, key, cls._json_number(data[key], f'{where}.{key}', optional=True))
        rule.class_group = cls._json_label(data['class_group'], f'{where}.class_group',
                                           optional=True)
        for slot in cls.__slots__:
            if slot not in known:
                setattr(rule, slot, RuleClassifier._json_decode_plain(
                    data.get(slot), f'{where}.{slot}'))
        return rule

# Helper function for memory estimation (used by export_to_arduino_ino)
def _compute_memory_for_rule_classifier(rf_model) -> dict:
    """Compute memory estimates for a RuleClassifier instance.

    Args:
        rf_model: A RuleClassifier with compiled tree arrays.

    Returns:
        Dict with flash and RAM byte estimates.
    """
    avr_max_flash = 143360   # Arduino Uno: ~140KB available
    avr_max_sram = 2048      # Arduino Uno: 2KB SRAM
    esp32_max_flash = 4 * 1024 * 1024  # ESP32: 4MB Flash
    esp32_max_sram = 524288              # ESP32: 512KB SRAM

    estimated_flash_bytes = 0
    total_ram_bytes = 0

    if not hasattr(rf_model, '_tree_arrays') or not rf_model._tree_arrays:
        return {
            'estimated_flash_bytes': 0,
            'total_ram_bytes': 0,
            'avr_max_flash': avr_max_flash,
            'avr_max_sram': avr_max_sram,
            'esp32_max_flash': esp32_max_flash,
            'esp32_max_sram': esp32_max_sram,
        }

    for tree in rf_model._tree_arrays:
        if 'init_score' in tree:
            continue  # GBDT init entry — no tree structure
        n_nodes = len(tree['feature_idx'])
        estimated_flash_bytes += n_nodes * 36 + 512  # ~36 bytes per node + header

    total_ram_bytes = estimated_flash_bytes * 0.1  # ~10% overhead

    return {
        'estimated_flash_bytes': estimated_flash_bytes,
        'total_ram_bytes': total_ram_bytes,
        'avr_max_flash': avr_max_flash,
        'avr_max_sram': avr_max_sram,
        'esp32_max_flash': esp32_max_flash,
        'esp32_max_sram': esp32_max_sram,
    }


# Class to handle the rule classification process
from .exporters import RuleExporterMixin
class RuleClassifier(RuleExporterMixin):
    # Represents a rule-based classifier built from decision paths in tree models.
    
    # Method to execute __init__
    def __init__(self, rules, algorithm_type='Decision Tree'):
        """
        Represents a rule-based classifier built from decision paths in tree models.
        
        This class supports rule extraction, classification, refinement, and 
        analysis of decision logic. It automatically compiles rules into a native 
        Python function for high-performance inference.

        Args:
            rules (Union[List[Rule], List[List[Rule]], Dict, str]): The extracted rules.
            algorithm_type (str): Type of model ('Decision Tree', 'Random Forest', or 'Gradient Boosting Decision Trees').
        """
        # Parse raw rules into Rule objects
        self.initial_rules = self.parse_rules(rules, algorithm_type)
        self.algorithm_type = algorithm_type
        
        # Initialize lists
        self.final_rules = []
        self.duplicated_rules = []
        self.specific_rules = []
        
        # GBDT-specific model metadata (None for DT/RF)
        self._gbdt_init_scores = None
        self._gbdt_is_binary = False
        self._gbdt_classes = None
        
        # --- OPTIMIZED CLASS & LABEL DETECTION ---
        labels = []
        counts = Counter()
        
        for rule in self.initial_rules:
            # Clean label (remove 'Class' prefix and spaces)
            raw_label = str(rule.class_).replace('Class', '').strip()
            try:
                label = int(raw_label)
            except ValueError:
                label = raw_label
            
            labels.append(label)
            counts[label] += 1

        # Sort labels safely handling mixed types
        self.class_labels = sorted(set(labels), key=lambda x: (isinstance(x, str), x))
        self.num_classes = len(self.class_labels)

        # Determine default class (Majority Voting)
        if counts:
            self.default_class = counts.most_common(1)[0][0]
        else:
            self.default_class = self.class_labels[0] if self.class_labels else 0

        # --- NATIVE OPTIMIZATION ---
        # Compile the initial rules into an in-memory Python function immediately
        # For GBDT, the caller (new_classifier) sets metadata first, then calls update_native_model
        self.native_fn = None
        if algorithm_type != 'Gradient Boosting Decision Trees':
            self.update_native_model(self.initial_rules)

        # --- ARRAY-BASED VECTORIZED PREDICTION ---
        # Initialize array attributes (compiled lazily or by explicit call)
        self._tree_arrays: list = []
        self._tree_class_groups: Optional[list] = None
        self._tree_is_init: Optional[list] = None
        self._array_feature_names: list = []
        self._arrays_compiled: bool = False
        # Rule list the current arrays were compiled from, and the other
        # compiled sets (see compile_tree_arrays / _array_state_for).
        self._arrays_source: Optional[list] = None
        self._arrays_source_len: Optional[int] = None
        self._array_states: list = []

        # --- UNCOVERED-REGION (FALLBACK) ACCOUNTING ---
        # Refinement can leave regions of the feature space with no matching
        # rule.  Every engine resolves them to `default_class`; this counter
        # records how often that happened so the loss of coverage is reported
        # instead of silently absorbed into the metrics.
        self.fallback_activations: int = 0

        # --- INPUT PRECISION ---
        # scikit-learn casts X to float32 before comparing against the
        # (float64) thresholds.  Set to 'float32' to reproduce that
        # quantization exactly; None keeps full float64 comparison.
        self.input_dtype: Optional[str] = None

        # --- LABEL TABLE ---
        # Rules, engines and exports work on integer labels. Labels that are
        # not integers ('cat', 0.5) are stored as their codes 0..K-1 (the
        # position in the estimator's classes_) and `label_table[code]` is
        # the original label; None when the labels are integers themselves.
        self.label_table: Optional[list] = None

        # --- ESTIMATOR CLASS ---
        # Name of the scikit-learn class the rules were read from, when it is
        # not the default one of `algorithm_type` (ExtraTreesClassifier for a
        # 'Random Forest'), so a refit trains the same kind of model.
        self.sklearn_estimator: Optional[str] = None

    # Method to execute _default_class_int
    def _default_class_int(self) -> int:
        """The default class as an int, for array/vectorized code paths.

        Returns:
            int: `default_class` stripped of any 'Class' prefix, or 0 when it
            cannot be interpreted as an integer.
        """
        try:
            return int(str(self.default_class).replace('Class', '').strip())
        except (ValueError, AttributeError, TypeError):
            return 0

    # Method to map the columns of a forest distribution to class labels
    def _rf_labels(self, n: int) -> List[int]:
        """The class label each column of an `n`-class distribution stands for.

        Leaf distributions are indexed like the estimator's ``classes_``; the
        argmax of their sum is a column index, which is the label only when
        the labels happen to be 0..K-1.

        Args:
            n (int): Length of the distribution.

        Returns:
            List[int]: The label of each column (the index itself when a label
            cannot be read as an integer).
        """
        out = []
        for i in range(n):
            try:
                out.append(int(str(self._label_at(i, n)).replace('Class', '').strip()))
            except (ValueError, TypeError):
                out.append(i)
        return out

    # Method to build the code -> label table of non-integer labels
    @staticmethod
    def _label_table_for(classes) -> Optional[list]:
        """The original labels of `classes` when they are not integers.

        Every engine and export works on integer labels (the C, binary and
        Arduino targets have no other type). Integer labels (also integral
        numbers such as ``1.0``, but not strings) are used as they are. Any
        other set of labels -- strings, non-integral floats -- is encoded:
        a label is stored as its position in `classes` (0..K-1) and the
        table returned here maps the code back.

        Args:
            classes: The estimator's ``classes_`` (in that order).

        Returns:
            list | None: The labels as plain Python values (str, int, float,
            bool), indexed by code; None when the labels are integers.

        Raises:
            ValueError: If a label is neither a string nor a number, or the
                labels are not distinct.
        """
        values = list(classes)
        integral = True
        for c in values:
            if isinstance(c, (str, bytes, np.str_, np.bytes_)):
                integral = False
                break
            try:
                value = float(c)
            except (TypeError, ValueError):
                integral = False
                break
            if not value.is_integer():
                integral = False
                break
        if integral:
            return None
        table = []
        for c in values:
            if isinstance(c, (bool, np.bool_)):
                table.append(bool(c))
            elif isinstance(c, (str, np.str_)):
                table.append(str(c))
            elif isinstance(c, (int, np.integer)):
                table.append(int(c))
            elif isinstance(c, (float, np.floating)):
                table.append(float(c))
            else:
                raise ValueError(
                    f'Unsupported class label {c!r:.60} ({type(c).__name__}): '
                    'labels must be strings or numbers.')
        if len(set(table)) != len(table):
            raise ValueError('Class labels must be distinct.')
        return table

    # Method to read an estimator's classes as the labels the rules store
    @staticmethod
    def _label_names(classes) -> List[str]:
        """The labels of `classes` as the rules store them, e.g. ``'-1'``.

        Integer labels are stored as themselves (``1.0`` becomes ``'1'``:
        ``str(1.0)`` would be ``'1.0'``, which no engine can turn back into
        the label). Other labels are stored as their codes ``'0'..'K-1'``;
        see `_label_table_for` for the table that maps them back.

        Args:
            classes: The estimator's ``classes_``.

        Returns:
            List[str]: One label per class, in the same order.
        """
        if RuleClassifier._label_table_for(classes) is not None:
            return [str(i) for i in range(len(list(classes)))]
        return [str(int(float(c))) for c in classes]

    # Method to translate original labels into the codes the engines use
    def _encode_labels(self, values):
        """Original labels -> the integer labels the engines answer.

        The identity when the model has no label table.

        Args:
            values: An array-like of original labels.

        Returns:
            np.ndarray | list: The codes (unchanged input without a table).

        Raises:
            ValueError: If a value is not one of the model's labels.
        """
        table = getattr(self, 'label_table', None)
        if table is None:
            return values
        code_of = {self._label_key(v): i for i, v in enumerate(table)}
        out = []
        for v in (values.tolist() if hasattr(values, 'tolist') else list(values)):
            key = self._label_key(v)
            if key not in code_of:
                raise ValueError(f'Unknown class label {v!r:.60}: the model '
                                 f'knows {table!r:.200}.')
            out.append(code_of[key])
        return np.asarray(out, dtype=np.int64)

    # Method to translate engine codes back into the original labels
    def _decode_labels(self, codes):
        """Integer labels answered by an engine -> the original labels.

        The identity when the model has no label table.

        Args:
            codes: An integer or an array of integers.

        Returns:
            The original label(s): a scalar for a scalar, an array otherwise.
        """
        table = getattr(self, 'label_table', None)
        if table is None:
            return codes
        if np.ndim(codes) == 0:
            return table[int(codes)]
        return np.asarray(table)[np.asarray(codes, dtype=np.int64)]

    # Method to give the names to show for each integer label
    def _display_names(self) -> list:
        """The label shown for each class column in reports.

        Returns:
            list: `label_table` when there is one, else `class_labels`.
        """
        table = getattr(self, 'label_table', None)
        return list(table) if table is not None else self.class_labels

    # Method to make a label hashable for lookups
    @staticmethod
    def _label_key(value):
        """A dict key under which numpy and Python spellings of a label meet.

        Args:
            value: A label.

        Returns:
            The key (np.str_ -> str, numpy numbers -> Python numbers).
        """
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode('utf-8', 'replace')
        return value

    # Method to escape text placed inside a generated C comment
    @staticmethod
    def _c_comment(text) -> str:
        """`text` made safe for a C comment: one line, no comment terminator.

        Feature names come from the data (CSV headers); a newline would end a
        ``//`` comment and a ``*/`` a block comment, turning the rest of the
        name into code.

        Args:
            text: The text to embed.

        Returns:
            str: The text on one line, with ``*/`` broken up.
        """
        return ' '.join(str(text).splitlines()).replace('*/', '* /')

    # Method to execute _assert_no_missing_values
    @staticmethod
    def _assert_no_missing_values(X) -> None:
        """Reject inputs containing NaN.

        Used for rule sets whose missing-value routing is unknown (see
        `_check_missing_values`). A scikit-learn tree routes a missing value
        to a side chosen per node (``missing_go_to_left``); rules without
        that information only test ``feature <= threshold``, every
        comparison against NaN is False, and the sample would silently
        match no rule.

        Args:
            X (np.ndarray): Input data.

        Raises:
            ValueError: If `X` contains NaN.
        """
        arr = np.asarray(X, dtype=np.float64)
        if arr.size and np.isnan(arr).any():
            n = int(np.isnan(arr).any(axis=-1).sum()) if arr.ndim > 1 else 1
            raise ValueError(
                f'Input contains NaN in {n} sample(s). This rule set does not '
                'record the missing-value routing of the original tree model '
                '(it was parsed from text, edited, or saved by an older '
                'version), so the result would silently differ from it. '
                'Impute the missing values, or extract the rules again from '
                'the fitted estimator.'
            )

    # Method to check whether a rule set records its missing-value routing
    def _rules_route_missing(self, rules: list) -> bool:
        """Whether every condition of `rules` knows where NaN goes.

        True for rules read from a fitted tree (scikit-learn >= 1.3), and for
        what refinement makes of them; False as soon as one condition is a
        plain tuple (parsed from text, edited, old pickle/JSON). Cached per
        rule list, like `_used_features`.

        Args:
            rules (List[Rule]): A rule set.

        Returns:
            bool: Whether NaN inputs can be answered faithfully.
        """
        cache = getattr(self, '_route_missing_cache', None)
        if cache is not None and cache[0] is rules and cache[1] == len(rules):
            return cache[2]
        known = all(isinstance(c, Condition)
                    for r in rules for c in (r.parsed_conditions or []))
        self._route_missing_cache = (rules, len(rules), known)
        return known

    # Method to check whether compiled arrays record the missing-value routing
    @staticmethod
    def _trees_route_missing(trees: list) -> bool:
        """Whether every compiled tree carries its ``missing_left`` array.

        Args:
            trees (List[dict]): Compiled tree arrays (GBDT init entries are
                skipped).

        Returns:
            bool: Whether NaN inputs can be routed.
        """
        return all('missing_left' in t for t in trees if 'init_score' not in t)

    # Method to refuse NaN inputs a model cannot answer faithfully
    def _check_missing_values(self, X, routes_missing: bool) -> bool:
        """Accept NaN in `X` only where the original model accepts it.

        * Gradient Boosting: scikit-learn's GradientBoostingClassifier raises
          on NaN (its trees route missing values, the ensemble does not
          accept them), so NaN is refused, as there.
        * Decision Tree / Random Forest (also ExtraTrees): NaN follows the
          per-split routing recorded in the rules (`Condition`), exactly as
          the fitted trees do; refused when that routing is unknown.

        Args:
            X (np.ndarray): Input data.
            routes_missing (bool): Whether the rule set / arrays record the
                routing.

        Returns:
            bool: Whether `X` contains NaN.

        Raises:
            ValueError: If `X` contains NaN the model cannot answer.
        """
        arr = np.asarray(X, dtype=np.float64)
        if not (arr.size and np.isnan(arr).any()):
            return False
        if self.algorithm_type == 'Gradient Boosting Decision Trees':
            n = int(np.isnan(arr).any(axis=-1).sum()) if arr.ndim > 1 else 1
            raise ValueError(
                f'Input contains NaN in {n} sample(s). scikit-learn\'s '
                'GradientBoostingClassifier does not accept missing values, '
                'so neither does a model extracted from it. Impute them, or '
                'use a Decision Tree / Random Forest / ExtraTrees model.'
            )
        if not routes_missing:
            self._assert_no_missing_values(arr)
        return True

    # Method to list the features a rule set tests
    def _used_features(self, rules: list) -> List[str]:
        """The features the model answering with `rules` reads.

        The union of the features tested by `rules`; a classifier loaded with
        `load_binary` has no rules, so its compiled column names are used.
        Cached per rule list (by identity and length), since `classify` asks
        for it on every sample.

        Args:
            rules (List[Rule]): The rule set a prediction uses.

        Returns:
            List[str]: Feature names, sorted.
        """
        if not rules:
            return list(getattr(self, '_array_feature_names', None) or [])
        cache = getattr(self, '_used_features_cache', None)
        if cache is not None and cache[0] is rules and cache[1] == len(rules):
            return cache[2]
        used = sorted({var for rule in rules
                       for var, _, _ in getattr(rule, 'parsed_conditions', None) or []})
        self._used_features_cache = (rules, len(rules), used)
        return used

    # Method to reject a sample that lacks a feature the model uses
    def _assert_features_present(self, data, rules: list) -> None:
        """Reject a single sample missing a feature the model tests.

        The engines used to disagree on a missing feature: the compiled arrays
        and the native function read it as 0, the iterative engine treated it
        as "rule does not match". Neither reproduces the original model, so
        it is an error.

        Args:
            data (Dict[str, float]): The instance to classify.
            rules (List[Rule]): The rule set the prediction uses.

        Raises:
            KeyError: Naming the missing feature(s).
        """
        missing = [f for f in self._used_features(rules) if f not in data]
        if missing:
            shown = ', '.join(repr(f) for f in missing[:10])
            more = f' (and {len(missing) - 10} more)' if len(missing) > 10 else ''
            raise KeyError(
                f'Sample is missing {len(missing)} feature(s) used by the '
                f'model: {shown}{more}.'
            )

    # Method to execute _quantize_input
    def _quantize_input(self, X):
        """Apply the configured input precision to a numpy array.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: `X` round-tripped through float32 when `input_dtype` is
            'float32', otherwise `X` unchanged.
        """
        if self.input_dtype == 'float32':
            return np.asarray(X, dtype=np.float32).astype(np.float64)
        return X

    # Method to execute reset_fallback_counter
    def reset_fallback_counter(self) -> None:
        """Zero the uncovered-region counter (see `fallback_activations`)."""
        self.fallback_activations = 0

    # Methods to support pickling and unpickling of the RuleClassifier
    def __getstate__(self):
        """Returns the state of the RuleClassifier for pickling.

        Returns:
            Dict: The state dictionary containing the attributes of the RuleClassifier.
        """
        state = self.__dict__.copy()
        # Remove the unpicklable function
        if 'native_fn' in state:
            del state['native_fn']
        state.pop('_native_source', None)
        # Compiled rule sets other than the current one are a cache: rebuilt
        # on demand, not worth the size of the pickle.
        state['_array_states'] = []
        state.pop('_used_features_cache', None)
        state.pop('_route_missing_cache', None)
        # Also remove custom_rule_removal if it's a lambda or local function
        if 'custom_rule_removal' in state:
             # Reset to default logic string or None to be safe, 
             # or just assume the user sets it again if needed.
             pass
        return state

    # Method used when unpickling
    def __setstate__(self, state):
        """
        Restores the RuleClassifier state after unpickling.

        Args:
            state (Dict): The state dictionary from __getstate__.
        """
        self.__dict__.update(state)
        self.native_fn = None
        
        # Ensure array attributes exist (backward compat with old pickles)
        if not hasattr(self, '_arrays_compiled'):
            self._tree_arrays = []
            self._tree_class_groups = None
            self._tree_is_init = None
            self._array_feature_names = []
            self._arrays_compiled = False

        # Backward compat: attributes introduced after some pickles were written
        if not hasattr(self, 'fallback_activations'):
            self.fallback_activations = 0
        if not hasattr(self, 'input_dtype'):
            self.input_dtype = None
        if not hasattr(self, '_array_states'):
            self._array_states = []
        if not hasattr(self, 'label_table'):
            self.label_table = None
        if not hasattr(self, 'sklearn_estimator'):
            self.sklearn_estimator = None
        self._native_source = None

        # Auto-recompile the native model upon loading (classify only uses it
        # for this rule set; see _native_source)
        rules_to_compile = self.final_rules if self.final_rules else self.initial_rules
        if rules_to_compile:
            self.update_native_model(rules_to_compile)
            
    # Method to parse the rules from string based on the algorithm type
    def parse_rules(self, rules, algorithm_type):
        """
        Parses the input rules into a list of Rule objects.

        Args:
            rules: The raw rules input (list, dict, or string).
            algorithm_type (str): The algorithm type.

        Returns:
            List[Rule]: A list of parsed Rule objects.
        """
        parsed_rules = []

        # Case 1: Input is already a list of Rule objects
        if isinstance(rules, list):
            # Flatten list of lists (Random Forest) if necessary
            if len(rules) > 0 and isinstance(rules[0], list):
                for sublist in rules:
                    parsed_rules.extend(sublist)
            else:
                parsed_rules = rules
            
            # Ensure parsed_conditions are populated
            for rule in parsed_rules:
                if not rule.parsed_conditions:
                    rule.parsed_conditions = self.parse_conditions_static(rule.conditions)
            return parsed_rules

        # Case 2: Input is a legacy Dictionary
        if isinstance(rules, dict):
            for class_name, rule_list in rules.items():
                parsed_rules.extend(rule_list) # Assuming values are lists of Rule objects
            return parsed_rules

        # Case 3: Input is a String (loaded from text file)
        if isinstance(rules, str):
            for line in rules.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Expected format: "RuleName: ['cond1', 'cond2']"
                if ':' in line:
                    parts = line.split(':', 1)
                    rule_name = parts[0].strip()
                    conditions_str = parts[1].strip()
                    
                    # Extract class from rule name (e.g., ..._Class0)
                    if "_Class" in rule_name:
                        class_label = rule_name.split("_Class")[-1]
                    else:
                        class_label = "Unknown"

                    # Convert string representation of list back to list
                    try:
                        # Safe eval for list strings like "['a', 'b']"
                        import ast
                        conditions = ast.literal_eval(conditions_str)
                    except Exception:
                        conditions = []

                    rule = Rule(rule_name, class_label, conditions)
                    rule.parsed_conditions = self.parse_conditions_static(conditions)
                    parsed_rules.append(rule)
            return parsed_rules
        return []
    
    # Method to parse conditions from string to tuple (variable, operator, value)
    @staticmethod
    def parse_conditions_static(conditions):
        """
        Parses condition strings into structured tuples (variable, operator, value).

        Args:
            conditions (List[str]): List of condition strings (e.g., "v1 <= 0.5").

        Returns:
            List[Tuple[str, str, float]]: A list of parsed conditions.
        """
        parsed_conditions = []
        for condition in conditions:
            parts = condition.split()
            if len(parts) == 3:
                var, op, value = parts
                try:
                    # Convert value to float directly
                    parsed_conditions.append((var, op, float(value)))
                except ValueError:
                    # Keep as string if not a number
                    parsed_conditions.append((var, op, value))
        return parsed_conditions

    # Helper method to compile a list of rules into a fast lookup function
    def _compile_tree_lookup(self, rules_to_compile):
        """
        Compiles a list of rules into a function that returns the index of the matched rule.
        Used for optimizing Random Forest analysis.
        """
        # Map rule logic to rule index
        # We need to ensure we don't change the order of rules passed in
        
        # Build tree structure. Raises if the rules are not the leaves of one
        # tree: a lookup through a wrong tree would credit the usage of one
        # rule to another, and the low-usage refinement acts on those counts.
        tree_dict = self._tree_from_rules(rules_to_compile)
        index_of = {id(r): i for i, r in enumerate(rules_to_compile)}
        for node in tree_dict.values():
            rule = node.pop('rule', None)
            # Store the index of the rule at the leaf (-1: uncovered)
            node['v'] = index_of[id(rule)] if rule is not None else -1

        # Build code
        def build_code(node_id, indent):
            """
            build_code method for RuleClassifier.
            Args:
                node_id: Description of node_id.
                indent: Description of indent.
            Returns:
                Description of return value.
            """
            node = tree_dict[node_id]
            tab = "    " * indent
            
            if node['f'] == -2: 
                # Return the index of the rule
                return f"{tab}return {node['v']}\n"
            
            feat_name = node['f']
            # Use repr(float(t)) for precision. Index the sample directly: a
            # missing feature raises KeyError instead of being read as 0. NaN
            # goes where the split sends missing values.
            code = f"{tab}if {_py_split_test(feat_name, node['t'], node.get('m'))}:\n"
            code += build_code(node['l'], indent + 1)
            code += f"{tab}else:\n"
            code += build_code(node['r'], indent + 1)
            return code

        func_code = "def lookup(sample):\n"
        func_code += build_code(0, 1)

        context = {}
        try:
            exec(func_code, {}, context)
            return context['lookup']
        except Exception:
            return None

    # Method to rebuild one tree for the native function
    def _native_tree(self, rules):
        """Rebuild one tree for `update_native_model`, or None if impossible.

        The native function is only an accelerator: when the rules are not the
        leaves of a tree (see `_tree_from_rules`) it is not built, and
        `classify` falls back to the iterative engine instead of routing
        samples through a tree that does not compute the rules.

        Args:
            rules (List[Rule]): The rules of one tree.

        Returns:
            Dict[int, dict] | None: The nodes, or None when the rules are not
            a tree (a warning says why).
        """
        try:
            return self._tree_from_rules(rules)
        except ValueError as exc:
            import warnings
            warnings.warn(f'Native model not compiled: {exc}', RuntimeWarning,
                          stacklevel=3)
            self.native_fn = None
            return None

    # Method to compile rules into a native Python function
    def update_native_model(self, rules_to_compile):
        """
        Compiles a list of Rules into an optimized in-memory Python function.
        
        This method rebuilds the decision tree structure from linear rules and 
        uses `exec()` to create a `predict(sample)` function that runs at 
        native Python speed, bypassing slow Rule object iteration.

        Args:
            rules_to_compile (List[Rule]): The rules to be compiled.
        """
        # `classify` may only answer from native_fn for the rule set it was
        # compiled from; remember which one that is.
        self.native_fn = None
        self._native_source = rules_to_compile

        # --- GBDT Compilation (Additive Scoring) ---
        if self.algorithm_type == 'Gradient Boosting Decision Trees':
            import math as _math

            # Group rules by tree identifier (prefix before first '_')
            tree_rules_map = defaultdict(list)
            for rule in rules_to_compile:
                if '_' in rule.name:
                    tid = rule.name.split('_')[0]
                else:
                    tid = rule.name
                tree_rules_map[tid].append(rule)

            # Separate init rules from tree rules, group by class_group
            init_scores = {}

            # First pass: extract init scores
            for tid, rules in tree_rules_map.items():
                if not rules:
                    continue
                class_group = rules[0].class_group
                if len(rules) == 1 and not rules[0].parsed_conditions and rules[0].contribution is not None:
                    init_scores[class_group] = rules[0].contribution

            # Now build the complete function code
            func_code = 'import math\n'

            # Identifiers come from positions, never from rule names or class
            # labels, which may hold anything (a '-' in "-1" already breaks
            # them, and a crafted name would be executed).
            fn_of = {tid: f'predict_{i}' for i, tid in enumerate(tree_rules_map)}

            # Re-iterate to build tree functions in order
            for tid, rules in tree_rules_map.items():
                if not rules:
                    continue
                # Skip init rules
                if len(rules) == 1 and not rules[0].parsed_conditions and rules[0].contribution is not None:
                    continue

                func_name = fn_of[tid]
                tree_dict = self._native_tree(rules)
                if tree_dict is None:
                    return
                for node in tree_dict.values():
                    rule = node.pop('rule', None)
                    # Uncovered region: the stage abstains (adds 0.0).
                    node['v'] = (rule.contribution if rule is not None
                                 and rule.contribution is not None else 0.0)

                # Method to _build_code.
                def _build_code(node_id, indent, td=tree_dict):
                    """
                    _build_code method for RuleClassifier.
                    Args:
                        node_id: Description of node_id.
                        indent: Description of indent.
                        td: Description of td.
                    Returns:
                        Description of return value.
                    """
                    node = td[node_id]
                    tab = '    ' * indent
                    if node['f'] == -2:
                        return f'{tab}return {repr(float(node["v"]))}\n'
                    feat = node['f']
                    # GradientBoostingClassifier refuses NaN: no routing emitted.
                    code = f"{tab}if {_py_split_test(feat, node['t'], None)}:\n"
                    code += _build_code(node['l'], indent + 1, td)
                    code += f'{tab}else:\n'
                    code += _build_code(node['r'], indent + 1, td)
                    return code

                func_code += f'def {func_name}(sample):\n'
                func_code += _build_code(0, 1)
                func_code += '\n'

            # Main prediction function
            is_binary = self._gbdt_is_binary
            classes = self._gbdt_classes or []

            func_code += 'def native_predict(sample):\n'

            if is_binary and len(classes) >= 2:
                pos_class = classes[1]
                score_init = init_scores.get(pos_class, 0.0)
                func_code += f'    score = {repr(score_init)}\n'
                for tid, rules in tree_rules_map.items():
                    if not rules:
                        continue
                    if len(rules) == 1 and not rules[0].parsed_conditions:
                        continue
                    if rules[0].class_group != pos_class:
                        continue
                    func_code += f'    score += {fn_of[tid]}(sample)\n'
                # Decide on the raw score, as scikit-learn does
                # (raw_predictions >= 0). sigmoid(score) >= 0.5 is not the
                # same test in floating point: sigmoid rounds to exactly 0.5
                # for negative scores of magnitude below ~1e-16 and would pick
                # the positive class.
                func_code += '    if score >= 0.0:\n'
                func_code += f'        return {int(classes[1])}, None, None\n'
                func_code += '    else:\n'
                func_code += f'        return {int(classes[0])}, None, None\n'
            else:
                # Multiclass: one score per class, named by its position
                col = {cl: i for i, cl in enumerate(classes)}
                for class_label in classes:
                    score_init = init_scores.get(class_label, 0.0)
                    func_code += f'    score_{col[class_label]} = {repr(score_init)}\n'
                for tid, rules in tree_rules_map.items():
                    if not rules:
                        continue
                    if len(rules) == 1 and not rules[0].parsed_conditions:
                        continue
                    if rules[0].class_group not in col:
                        continue
                    func_code += (f'    score_{col[rules[0].class_group]} += '
                                  f'{fn_of[tid]}(sample)\n')

                # Argmax
                score_vars = [f'score_{col[cl]}' for cl in classes]
                class_ints = [int(cl) for cl in classes]
                func_code += f'    scores = [{", ".join(score_vars)}]\n'
                func_code += f'    classes = {class_ints}\n'
                func_code += '    best_idx = scores.index(max(scores))\n'
                func_code += '    return classes[best_idx], None, None\n'

            context = {'math': _math}
            try:
                exec(func_code, context)
                self.native_fn = context['native_predict']
            except Exception as e:
                print(f'Error compiling GBDT native model: {e}')
                self.native_fn = None
            return

        # --- Random Forest Compilation (Soft Voting) ---
        if self.algorithm_type == 'Random Forest':
            tree_rules_map = defaultdict(list)
            for rule in rules_to_compile:
                if "_" in rule.name:
                    tid = rule.name.split('_')[0]
                else:
                    tid = "Tree0" 
                tree_rules_map[tid].append(rule)
            
            # Determine number of classes from distributions
            n_classes = 0
            for rule in rules_to_compile:
                if rule.class_distribution is not None:
                    n_classes = len(rule.class_distribution)
                    break
            
            tree_func_names = []
            func_code = ""

            for t_pos, (tid, rules) in enumerate(tree_rules_map.items()):
                # Named by position: rule names are not valid identifiers
                func_name = f"predict_{t_pos}"
                tree_func_names.append(func_name)

                # Build tree structure — store probability distribution at leaves
                tree_dict = self._native_tree(rules)
                if tree_dict is None:
                    return

                for curr, node in tree_dict.items():
                    rule = node.pop('rule', None)
                    node['v'] = None
                    if rule is None:
                        continue  # uncovered region: the tree abstains

                    # Store normalized probability distribution at leaf
                    # (raw counts -> probabilities at compile time for fast runtime)
                    if rule.class_distribution is not None:
                        raw = rule.class_distribution
                        total_c = sum(raw)
                        if total_c > 0:
                            tree_dict[curr]['v'] = [c / total_c for c in raw]
                        else:
                            tree_dict[curr]['v'] = list(raw)
                    else:
                        try:
                            clean_class = int(str(rule.class_).replace('Class', '').strip())
                        except Exception:
                            clean_class = rule.class_
                        tree_dict[curr]['v'] = clean_class

                # Method to build_rf_code.
                def build_rf_code(node_id, indent):
                    """
                    build_rf_code method for RuleClassifier.
                    Args:
                        node_id: Description of node_id.
                        indent: Description of indent.
                    Returns:
                        Description of return value.
                    """
                    node = tree_dict[node_id]
                    tab = "    " * indent
                    if node['f'] == -2:
                        val = node['v']
                        if val is None:
                            return f"{tab}return None\n" 
                        return f"{tab}return {repr(val)}\n"
                    
                    feat = node['f']
                    # Use repr(float(t)) to ensure no numpy precision issues in literals
                    code = f"{tab}if {_py_split_test(feat, node['t'], node.get('m'))}:\n"
                    code += build_rf_code(node['l'], indent + 1)
                    code += f"{tab}else:\n"
                    code += build_rf_code(node['r'], indent + 1)
                    return code

                func_code += f"def {func_name}(sample):\n"
                func_code += build_rf_code(0, 1)
                func_code += "\n"

            # Main Voting Function — soft probability averaging
            try:
                default_val = int(str(self.default_class).replace('Class', '').strip())
            except Exception:
                default_val = self.default_class
            
            def_str = repr(default_val)

            func_code += "def native_predict(sample):\n"
            func_code += "    tree_results = []\n"
            for func in tree_func_names:
                func_code += f"    v = {func}(sample)\n"
                func_code += "    if v is not None: tree_results.append(v)\n"
            func_code += f"    if not tree_results: return {def_str}, [], None\n"

            if n_classes > 0:
                # The argmax is a column of the distribution; answer its label
                func_code += f"    labels = {self._rf_labels(n_classes)!r}\n"
                # Soft voting path: each tree returns a probability list
                func_code += "    # Check if results are probability lists (soft voting)\n"
                func_code += "    if isinstance(tree_results[0], list):\n"
                func_code += f"        n_classes = {n_classes}\n"
                func_code += "        avg = [0.0] * n_classes\n"
                func_code += "        for proba in tree_results:\n"
                func_code += "            for j in range(n_classes):\n"
                func_code += "                avg[j] += proba[j]\n"
                func_code += "        n = len(tree_results)\n"
                func_code += "        avg = [p / n for p in avg]\n"
                func_code += "        best = avg.index(max(avg))\n"
                func_code += "        return labels[best], None, avg\n"
                func_code += "    else:\n"
                func_code += "        # Fallback: hard voting (distributions unavailable)\n"
                func_code += "        pred = Counter(tree_results).most_common(1)[0][0]\n"
                func_code += "        return pred, tree_results, None\n"
            else:
                # No distributions at all — hard voting fallback
                func_code += "    pred = Counter(tree_results).most_common(1)[0][0]\n"
                func_code += "    return pred, tree_results, None\n"

        # --- Decision Tree Compilation (Single Tree) ---
        else:
            tree_dict = self._native_tree(rules_to_compile)
            if tree_dict is None:
                return
            for node in tree_dict.values():
                rule = node.pop('rule', None)
                if rule is None:
                    # Uncovered region -> default_class. None, not -1: -1 is
                    # a legitimate label (y in {-1, 1}).
                    node['v'] = None
                    continue
                try:
                    node['v'] = int(str(rule.class_).replace('Class', '').strip())
                except Exception:
                    node['v'] = rule.class_

            # Method to build_dt_code.
            def build_dt_code(node_id, indent):
                """
                build_dt_code method for RuleClassifier.
                Args:
                    node_id: Description of node_id.
                    indent: Description of indent.
                Returns:
                    Description of return value.
                """
                node = tree_dict[node_id]
                tab = "    " * indent
                if node['f'] == -2:
                    val = node['v']
                    if val is None:
                        try:
                            val = int(str(self.default_class).replace('Class', '').strip())
                        except Exception:
                            val = self.default_class
                    return f"{tab}return {val!r}, None, None\n"
                
                feat = node['f']
                # Use repr(float(t)) for precision
                code = f"{tab}if {_py_split_test(feat, node['t'], node.get('m'))}:\n"
                code += build_dt_code(node['l'], indent + 1)
                code += f"{tab}else:\n"
                code += build_dt_code(node['r'], indent + 1)
                return code

            func_code = "def native_predict(sample):\n"
            func_code += build_dt_code(0, 1)

        # Compile logic into the local namespace
        context = {'Counter': Counter}
        try:
            exec(func_code, context)
            self.native_fn = context.get('native_predict')
            if self.native_fn is None:
                print("Error: 'native_predict' function not found in compiled code.")
        except Exception as e:
            print(f"Error compiling native model: {e}")
            self.native_fn = None

    # Method to execute the classification process
    def classify(self, data, final=False):
        """
        Classifies a single data instance using extracted rules.

        This method delegates the classification logic to the compiled tree
        arrays, to the native function when it was compiled from the requested
        rule set, or to iterative evaluation. All three answer with the rule
        set `final` selects.

        Args:
            data (Dict[str, float]): The instance to classify.
            final (bool): If True, uses `final_rules` (post-analysis; the
                initial rules when refinement has not run). If False, uses
                `initial_rules`.

        Returns:
            Tuple[int, List[int]|None, List[float]|None]: 
                - Predicted class label (int).
                - List of votes (Random Forest only).
                - Class probabilities (Random Forest only).
        """
        # Every path below must answer with the rule set `final` selects:
        # final_rules, or initial_rules when there are none yet.
        rules = self._rules_for(final)

        # Checked up front: the fast paths below swallow exceptions, so this
        # must not be left to predict_batch or it would be silently masked.
        # NaN is answered (by every path) only when the rules record where
        # each split sends it.
        for _v in data.values():
            if _v != _v:   # NaN, whatever the numeric type
                self._check_missing_values(
                    np.array([np.nan]),
                    self._rules_route_missing(rules) if rules
                    else self._trees_route_missing(self._tree_arrays))
                break

        # A missing feature is an error on every engine (also checked before
        # the fast paths, which would swallow it or read it as 0).
        self._assert_features_present(data, rules)

        # --- FAST PATH 1: Array-based single-sample prediction ---
        # predict_batch selects (and, the first time, compiles) the arrays of
        # the requested rule set, so this path cannot answer with the other.
        if self._arrays_compiled:
            try:
                row = np.array(
                    [[data.get(f, 0.0) for f in self._array_feature_names]],
                    dtype=np.float64,
                )
                # predict_batch applies the configured input precision itself.
                pred = self.predict_batch(
                    row, feature_names=list(self._array_feature_names),
                    use_final=final)[0]
                clean_class = int(pred)
                return clean_class, None, None
            except Exception:
                pass  # Fallback to slower paths

        if self.input_dtype == 'float32':
            data = {
                k: (float(np.float32(v)) if isinstance(v, (int, float)) else v)
                for k, v in data.items()
            }

        # --- FAST PATH 2: Native function (exec-compiled) ---
        # Only valid for the rule set it was compiled from (update_native_model
        # is also called on the refined rules, and on unpickling).
        if (self.native_fn is not None
                and getattr(self, '_native_source', None) is rules):
            try:
                # native_fn returns (prediction, votes, proba)
                return self.native_fn(data) # pyright: ignore[reportCallIssue]
            except Exception:
                # Fallback to standard execution if something goes wrong (e.g., missing keys)
                pass

        # --- SLOW PATH: Iterative Execution ---

        predicted_class = self.default_class
        votes = None
        proba = None

        if self.algorithm_type == 'Random Forest':
            n_dist = next((len(r.class_distribution) for r in rules
                           if r.class_distribution is not None), 0)
            predicted_class, votes, proba, _ = self.classify_rf(
                data, rules, labels=self._rf_labels(n_dist))
            if predicted_class is None:
                # No tree had a matching rule: uncovered region in every tree.
                predicted_class = self.default_class
                self.fallback_activations += 1

        elif self.algorithm_type == 'Gradient Boosting Decision Trees':
            predicted_class, _, _ = self.classify_gbdt(
                data, rules, self._gbdt_init_scores,
                self._gbdt_is_binary, self._gbdt_classes,
            )

        elif self.algorithm_type == 'Decision Tree':
            matched_rule = self.classify_dt(data, rules)
            if matched_rule:
                predicted_class = matched_rule.class_
            else:
                # Uncovered region: `predicted_class` stays at default_class.
                self.fallback_activations += 1

        # Ensure the returned class is always an integer for consistency with metrics
        try:
            clean_class = int(str(predicted_class).replace('Class', '').strip())
        except (ValueError, AttributeError):
            # Fallback if conversion fails, try to map from class_labels or return raw
            clean_class = predicted_class
            
        return clean_class, votes, proba

    # Method to classify data using Decision Tree rules
    @staticmethod
    def classify_dt(data, rules):
        """
        Classifies a single data instance using extracted rules from a decision tree.
        Iterates through the list until the first rule is satisfied.

        Args:
            data (Dict[str, float]): Instance data.
            rules (List[Rule]): List of rule instances.

        Returns:
            Rule|None: The first Rule object that matches, or None.
        """
        for rule in rules:
            rule_satisfied = True

            # Iterate over pre-parsed conditions for performance
            for cond in rule.parsed_conditions:
                var, op, value = cond
                # Use data.get() to avoid KeyErrors, defaulting to None implies failure
                instance_value = data.get(var)

                if instance_value is None:
                    rule_satisfied = False
                    break

                if instance_value != instance_value:
                    # NaN: satisfied only on the side its split sends
                    # missing values to (unknown routing: not satisfied).
                    if getattr(cond, 'missing', False):
                        continue
                    rule_satisfied = False
                    break

                # Note: 'value' is already a float from parse_conditions_static
                # 'instance_value' should be float from process_data

                if op == '<=':
                    if not (instance_value <= value): 
                        rule_satisfied = False
                elif op == '>':
                    if not (instance_value > value): 
                        rule_satisfied = False
                elif op == '<':
                     if not (instance_value < value): 
                        rule_satisfied = False
                elif op == '>=':
                     if not (instance_value >= value): 
                        rule_satisfied = False
                
                if not rule_satisfied:
                    break
            
            if rule_satisfied:
                return rule
                
        return None

    # Method to classify data using Random Forest rules    
    @staticmethod
    def classify_rf(data, rules, labels=None):
        """
        Classifies a single data instance using extracted rules from a Random Forest.
        Uses soft probability averaging (like sklearn) when class distributions are
        available, otherwise falls back to hard majority voting.

        Args:
            data (Dict[str, float]): Instance data.
            rules (List[Rule]): List of rule instances.
            labels (List[int], optional): The label of each distribution
                column. Without it, soft voting answers the column index.

        Returns:
            Tuple: (Predicted Label, Votes List, Proba List, Matched Rules List)
        """
        if not rules: 
            return None, [], [], []
        
        # 1. Organize rules by Tree (DT1, DT2...)
        tree_rules = defaultdict(list)
        for rule in rules:
            tree_identifier = rule.name.split('_')[0]
            tree_rules[tree_identifier].append(rule)

        votes = []
        all_matched_rules = []
        tree_probas = []  # Per-tree probability distributions for soft voting
        has_distributions = True  # Track if all matched rules have distributions

        # 2. Evaluate each tree
        for _, rules_in_tree in tree_rules.items():
            matched_rule = RuleClassifier.classify_dt(data, rules_in_tree)
            
            if matched_rule:
                # Extract clean label for votes list
                try:
                    label = int(str(matched_rule.class_).replace('Class', '').strip())
                except ValueError:
                    label = matched_rule.class_
                
                votes.append(label)
                all_matched_rules.append(matched_rule)
                
                # Collect class distribution for soft voting (normalize raw counts to probabilities)
                if matched_rule.class_distribution is not None:
                    dist = matched_rule.class_distribution
                    total_counts = sum(dist)
                    if total_counts > 0:
                        tree_probas.append([c / total_counts for c in dist])
                    else:
                        tree_probas.append(dist)
                else:
                    has_distributions = False

        # 3. Aggregate Results
        if not votes:
            return None, [], [], []

        # --- Soft Voting (probability averaging, matches sklearn behavior) ---
        if has_distributions and tree_probas:
            n_classes = len(tree_probas[0])
            # Average probabilities across trees
            avg_proba = [0.0] * n_classes
            for proba in tree_probas:
                for i in range(n_classes):
                    avg_proba[i] += proba[i]
            n_trees = len(tree_probas)
            avg_proba = [p / n_trees for p in avg_proba]
            
            # Predicted class = argmax of averaged probabilities
            best_idx = avg_proba.index(max(avg_proba))
            predicted_class = (labels[best_idx]
                               if labels is not None and len(labels) == n_classes
                               else best_idx)
            probas_out = avg_proba
        else:
            # --- Fallback: Hard Voting ---
            counts = Counter(votes)
            total_votes = len(votes)
            
            unique_classes = sorted(list(set(r.class_ for r in rules)))
            proba_map = {k: v / total_votes for k, v in counts.items()}
            for cls in unique_classes:
                if cls not in proba_map:
                    proba_map[cls] = 0.0
            
            predicted_class = counts.most_common(1)[0][0]
            probas_out = [proba_map[cls] for cls in unique_classes]

        return predicted_class, votes, probas_out, all_matched_rules

    # Method to classify data using GBDT rules (additive scoring)
    @staticmethod
    def classify_gbdt(data, rules, init_scores, is_binary, classes):
        """
        Classifies a single data instance using GBDT additive scoring.

        For each class group, the method sums the init score plus the
        contribution of the first matching rule in each tree. Binary
        classification takes the positive class when the score is >= 0 (as
        scikit-learn does); multiclass uses argmax.

        Args:
            data (Dict[str, float]): Instance data.
            rules (List[Rule]): All GBDT Rule objects (init + tree rules).
            init_scores (Dict[str, float]): Init scores per class group.
            is_binary (bool): Whether this is binary classification.
            classes (List[str]): List of class label strings.

        Returns:
            Tuple[int, List[Rule], None]:
                - Predicted class label (int).
                - List of matched rules (one per tree per class group).
                - None (kept for API consistency).
        """
        # Group rules by tree identifier (prefix before first '_')
        # e.g., 'GBDT1T0' for init, 'GBDT1T1' for tree 1 of class 1
        tree_rules_map = defaultdict(list)
        for rule in rules:
            if '_' in rule.name:
                tree_id = rule.name.split('_')[0]
            else:
                tree_id = rule.name
            tree_rules_map[tree_id].append(rule)

        # Compute scores per class group
        scores = {}
        matched_rules = []

        if is_binary:
            # Binary: only positive class (classes[1]) has trees
            pos_class = classes[1]
            score = init_scores.get(pos_class, 0.0)

            for tree_id, tree_rules in tree_rules_map.items():
                # Skip init rules (no conditions)
                if tree_rules and tree_rules[0].class_group != pos_class:
                    continue

                for rule in tree_rules:
                    # Init rules have no conditions — skip scoring (already in init_score)
                    if not rule.parsed_conditions and rule.contribution is not None:
                        # This is the init rule — already accounted for
                        matched_rules.append(rule)
                        continue

                    # Check if rule matches
                    matched = True
                    for cond in rule.parsed_conditions:
                        var, op, value = cond
                        instance_value = data.get(var)
                        if instance_value is None:
                            matched = False
                            break
                        if instance_value != instance_value:
                            # NaN follows the routing of its split
                            if getattr(cond, 'missing', False):
                                continue
                            matched = False
                            break
                        if op == '<=':
                            if not (instance_value <= value):
                                matched = False
                        elif op == '>':
                            if not (instance_value > value):
                                matched = False
                        elif op == '<':
                            if not (instance_value < value):
                                matched = False
                        elif op == '>=':
                            if not (instance_value >= value):
                                matched = False
                        if not matched:
                            break

                    if matched:
                        if rule.contribution is not None:
                            score += rule.contribution
                        matched_rules.append(rule)
                        break  # Only one match per tree

            # Decide on the raw score, as scikit-learn does (raw >= 0); see
            # update_native_model for why not sigmoid(score) >= 0.5.
            predicted_class = int(classes[1]) if score >= 0.0 else int(classes[0])

        else:
            # Multiclass: compute score for each class group
            for class_label in classes:
                scores[class_label] = init_scores.get(class_label, 0.0)

            for tree_id, tree_rules in tree_rules_map.items():
                if not tree_rules:
                    continue
                class_group = tree_rules[0].class_group

                for rule in tree_rules:
                    if not rule.parsed_conditions and rule.contribution is not None:
                        matched_rules.append(rule)
                        continue

                    matched = True
                    for cond in rule.parsed_conditions:
                        var, op, value = cond
                        instance_value = data.get(var)
                        if instance_value is None:
                            matched = False
                            break
                        if instance_value != instance_value:
                            # NaN follows the routing of its split
                            if getattr(cond, 'missing', False):
                                continue
                            matched = False
                            break
                        if op == '<=':
                            if not (instance_value <= value):
                                matched = False
                        elif op == '>':
                            if not (instance_value > value):
                                matched = False
                        elif op == '<':
                            if not (instance_value < value):
                                matched = False
                        elif op == '>=':
                            if not (instance_value >= value):
                                matched = False
                        if not matched:
                            break

                    if matched:
                        if rule.contribution is not None and class_group in scores:
                            scores[class_group] += rule.contribution
                        matched_rules.append(rule)
                        break

            # Argmax
            best_class = max(scores, key=lambda c: scores[c])
            try:
                predicted_class = int(best_class)
            except (ValueError, TypeError):
                predicted_class = best_class

        return predicted_class, matched_rules, None

    # =========================================================================
    # ARRAY-BASED VECTORIZED PREDICTION (High-Performance Path)
    # =========================================================================
    #
    # The methods below convert rules into flat numpy arrays (mirroring sklearn's
    # internal tree representation) and use vectorized numpy operations for
    # batch prediction.  This eliminates:
    #   - Linear scans over Rule objects
    #   - Per-sample Python dict lookups
    #   - Per-call tree grouping with string splitting
    #   - Pure-Python voting loops
    #
    # Array layout per tree (parallel arrays, one element per node):
    #   feature_idx[i]    int32   feature column index (-2 = leaf)
    #   threshold[i]      float64 split threshold
    #   children_left[i]  int32   left child node index (-1 = none)
    #   children_right[i] int32   right child node index (-1 = none)
    #   missing_left[i]   uint8   1 if a NaN goes left at this split (only
    #                             when the rules record the routing)
    #   value[i]          varies  leaf payload:
    #       DT:   int32   class label
    #       RF:   float64 array of shape (n_classes,) — raw sample counts
    #       GBDT: float64 contribution (learning_rate * leaf_value)
    # =========================================================================

    # Method to rebuild the tree a rule set was read from
    @staticmethod
    def _tree_from_rules(rules: list, validate: bool = True) -> Dict[int, dict]:
        """
        Rebuild the binary tree that one tree's rules were read from.

        Every compiled engine (tree arrays, binary, C header, native function)
        routes a sample down this tree instead of testing the rules one by one,
        so it only computes what the rules say if the rules really are the
        leaves of one tree. Four ways they can fail to be, each of which would
        otherwise make a compiled engine answer differently from the iterative
        engine and from the CPN, without any error:

        - two rules want different splits at a node they share;
        - a rule stops at a node other rules go past (its region contains
          theirs), or goes past a node where another rule stops;
        - two rules stop at the same node (identical regions);
        - two rules disagree on where a shared split sends missing values.

        The second and third are overlaps: the iterative engine takes the first match,
        the tree keeps one of them, and the CPN becomes nondeterministic.

        Args:
            rules (List[Rule]): The rules of one tree.
            validate (bool): Raise on any of the conditions above.

        Returns:
            Dict[int, dict]: Nodes keyed by id; internal nodes carry 'f', 't',
            'l', 'r' and 'm' (whether NaN goes left: True/False, None when
            the conditions do not record it); a leaf has ``f == -2`` and
            'rule' set to the rule that ends there, or None for a region no
            rule covers.

        Raises:
            ValueError: If `validate` is set and the rules are not the leaves
                of one tree.
        """
        nodes: Dict[int, dict] = {
            0: {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'rule': None}
        }
        next_id = 1

        def _name(r):
            return getattr(r, 'name', '?')

        for rule in rules:
            curr = 0
            for cond in rule.parsed_conditions:
                var, op, val = cond
                goes_left = op in ('<=', '<')
                # Where the split sends NaN: True (left), False (right) or
                # None (unknown: the condition is a plain tuple).
                miss = condition_missing(cond)
                m = None if miss is None else (miss if goes_left else not miss)
                node = nodes[curr]
                if node['f'] == -2:
                    if validate and node['rule'] is not None:
                        raise ValueError(
                            'Rule set is not a tree and cannot be compiled: rule '
                            f'{_name(rule)!r} continues past the leaf of rule '
                            f'{_name(node["rule"])!r}, whose region therefore '
                            'contains it (overlapping rules).'
                        )
                    node.update({'f': var, 't': val, 'm': m,
                                 'l': next_id, 'r': next_id + 1})
                    for child in (next_id, next_id + 1):
                        nodes[child] = {'l': -1, 'r': -1, 'f': -2, 't': -2.0,
                                        'rule': None}
                    next_id += 2
                elif validate and (node['f'] != var
                                   or float(node['t']) != float(val)):
                    raise ValueError(
                        'Rule set is not prefix-closed and cannot be compiled '
                        'into tree arrays: rule '
                        f'{_name(rule)!r} requires the split '
                        f'({var} {op} {val}) at a node already split on '
                        f'({node["f"]} @ {node["t"]}). '
                        'Rules sharing a prefix of conditions must agree on the '
                        'split at every shared node. This usually means a custom '
                        'rule-removal function produced a rule set that is no '
                        'longer a tree.'
                    )
                elif node.get('m') != m:
                    if validate and node.get('m') is not None and m is not None:
                        raise ValueError(
                            'Rule set is not a tree and cannot be compiled: rule '
                            f'{_name(rule)!r} sends missing values (NaN) to the '
                            f'other side of the split ({var} {op} {val}) than '
                            'the rules sharing it.'
                        )
                    node['m'] = None   # unknown routing
                curr = node['l'] if goes_left else node['r']

            node = nodes[curr]
            if validate and node['f'] != -2:
                raise ValueError(
                    'Rule set is not a tree and cannot be compiled: rule '
                    f'{_name(rule)!r} stops at an internal node, so its region '
                    'contains the regions of other rules (overlapping rules).'
                )
            if validate and node['rule'] is not None:
                raise ValueError(
                    'Rule set is not a tree and cannot be compiled: rules '
                    f'{_name(node["rule"])!r} and {_name(rule)!r} cover the same '
                    'region.'
                )
            node['rule'] = rule
        return nodes

    # Method to execute _build_single_tree_arrays
    @staticmethod
    def _build_single_tree_arrays(
        rules: list,
        feature_name_to_idx: Dict[str, int],
        algorithm_type: str,
        n_classes: int = 0,
        default_class: int = 0,
        validate_structure: bool = True,
        class_labels: Optional[list] = None,
    ) -> dict:
        """
        Convert a list of rules (all from one tree) into flat numpy arrays.

        The reconstruction walks each rule from the root (see
        `_tree_from_rules`), so it requires the rules to be the leaves of one
        tree: rules sharing a prefix of conditions must agree on the split at
        every shared node, and no rule's region may contain another's. This
        holds for rules extracted from a tree and is preserved by the built-in
        refinement operations, but a custom removal function (see
        `set_custom_rule_removal`) can break it.  When that happens rules land
        in the wrong leaf, silently; `validate_structure` turns that silent
        corruption into an error.

        Args:
            rules (List[Rule]): Rules for a single tree.
            feature_name_to_idx (Dict[str, int]): Mapping feature_name -> column index.
            algorithm_type (str): 'Decision Tree', 'Random Forest', or
                'Gradient Boosting Decision Trees'.
            n_classes (int): Number of classes (needed for RF leaf distributions).
            default_class (int): Class assigned to uncovered regions (DT only).
            validate_structure (bool): Raise if the rules are not the leaves
                of one tree.
            class_labels (List[int], optional): The label of each RF
                distribution column (see `_rf_labels`). A rule without a
                distribution is one-hot encoded in its label's column.

        Returns:
            Dict with keys: 'feature_idx', 'threshold', 'children_left',
            'children_right', 'value', 'max_depth', 'is_hole', and
            'missing_left' (uint8, 1 where a split sends NaN left) when the
            rules record the missing-value routing.  Each array is
            indexed by node id; 'is_hole' marks leaves that no rule reached,
            i.e. uncovered regions of the feature space.

        Raises:
            ValueError: If `validate_structure` is set and the rules are not
                the leaves of one tree (conflicting split or overlap).
        """
        # 1. Reconstruct tree structure
        tree_dict = RuleClassifier._tree_from_rules(rules, validate_structure)

        for curr, node in tree_dict.items():
            rule = node.pop('rule')
            node['v'] = None
            if rule is None:
                continue

            # Set leaf value based on algorithm type
            if algorithm_type == 'Gradient Boosting Decision Trees':
                tree_dict[curr]['v'] = float(rule.contribution) if rule.contribution is not None else 0.0
            elif algorithm_type == 'Random Forest':
                if rule.class_distribution is not None:
                    tree_dict[curr]['v'] = list(rule.class_distribution)
                else:
                    # Fallback: one-hot encode the class label, in the
                    # column that label stands for (labels 10/20/30 are
                    # columns 0/1/2, not 10/20/30).
                    try:
                        label = int(str(rule.class_).replace('Class', '').strip())
                    except (ValueError, AttributeError):
                        label = None
                    if class_labels is not None and label in class_labels:
                        cls_idx = list(class_labels).index(label)
                    elif label is not None:
                        cls_idx = label   # unknown label: read as a column
                    else:
                        cls_idx = 0
                    one_hot = [0.0] * max(n_classes, cls_idx + 1)
                    one_hot[cls_idx] = 1.0
                    tree_dict[curr]['v'] = one_hot
            else:
                # Decision Tree -- the rule's own label, read exactly as the
                # iterative engine, the native function and the CPN read it.
                # (argmax of the distribution is its *index*, which is the
                # label only when the labels happen to be 0..K-1.)
                try:
                    tree_dict[curr]['v'] = int(str(rule.class_).replace('Class', '').strip())
                except (ValueError, AttributeError):
                    dist = getattr(rule, 'class_distribution', None)
                    m = re.search(r'(\d+)$', str(rule.class_))
                    if m:
                        tree_dict[curr]['v'] = int(m.group(1))
                    elif dist is not None:
                        tree_dict[curr]['v'] = int(np.argmax(dist))
                    else:
                        tree_dict[curr]['v'] = 0

        # 2. Convert to flat numpy arrays
        n_nodes = len(tree_dict)
        feature_idx = np.full(n_nodes, -2, dtype=np.int32)
        threshold = np.zeros(n_nodes, dtype=np.float64)
        children_left = np.full(n_nodes, -1, dtype=np.int32)
        children_right = np.full(n_nodes, -1, dtype=np.int32)

        if algorithm_type == 'Random Forest':
            value = np.zeros((n_nodes, n_classes), dtype=np.float64)
        elif algorithm_type == 'Gradient Boosting Decision Trees':
            value = np.zeros(n_nodes, dtype=np.float64)
        else:
            value = np.full(n_nodes, int(default_class), dtype=np.int32)

        # Leaves that no rule reached: uncovered regions left behind by
        # refinement.  They resolve to `default_class` (DT), to an abstention
        # (zero probability vector for RF, zero contribution for GBDT), and are
        # flagged here so callers can report how often they are hit.
        is_hole = np.zeros(n_nodes, dtype=bool)

        # Where each split sends a missing value (1 = left). Kept only when
        # every split knows it; leaves are 0 (both children are the leaf).
        missing_left = np.zeros(n_nodes, dtype=np.uint8)
        routing_known = True

        for node_id, node in tree_dict.items():
            feat = node['f']
            if feat == -2:
                # Leaf node — self-referencing children so traversal
                # samples that reached a leaf simply stay in place.
                # We also set feature_idx to 0 (any valid column) so that
                # the gather X[samples, feature_idx[node_ids]] always reads
                # a valid column.  The threshold is set to -inf so the
                # comparison (feat_val <= threshold) is always False, making
                # the traversal pick children_right which points back here.
                feature_idx[node_id] = 0  # valid column (not -2)
                threshold[node_id] = -np.inf
                children_left[node_id] = node_id   # self-loop
                children_right[node_id] = node_id   # self-loop
                is_hole[node_id] = node['v'] is None
                if algorithm_type == 'Random Forest':
                    v = node['v']
                    if v is not None:
                        for ci in range(min(len(v), n_classes)):
                            value[node_id, ci] = v[ci]
                elif algorithm_type == 'Gradient Boosting Decision Trees':
                    value[node_id] = float(node['v']) if node['v'] is not None else 0.0
                else:
                    value[node_id] = (
                        int(node['v']) if node['v'] is not None
                        else int(default_class)
                    )
            else:
                feature_idx[node_id] = feature_name_to_idx.get(feat, -2)
                threshold[node_id] = float(node['t'])
                children_left[node_id] = node['l']
                children_right[node_id] = node['r']
                if node.get('m') is None:
                    routing_known = False
                else:
                    missing_left[node_id] = 1 if node['m'] else 0

        # Compute tree depth by BFS from root
        max_depth = 0
        stack = [(0, 0)]  # (node_id, depth)
        while stack:
            nid, d = stack.pop()
            if nid < 0 or nid >= n_nodes:
                continue
            node = tree_dict.get(nid)
            if node is None:
                continue
            if node['f'] == -2:
                # Leaf
                if d > max_depth:
                    max_depth = d
            else:
                stack.append((node['l'], d + 1))
                stack.append((node['r'], d + 1))

        out = {
            'feature_idx': feature_idx,
            'threshold': threshold,
            'children_left': children_left,
            'children_right': children_right,
            'value': value,
            'max_depth': max_depth,
            'is_hole': is_hole,
        }
        if routing_known:
            out['missing_left'] = missing_left
        return out

    # Method to pick the rule set a prediction should use
    def _rules_for(self, use_final: bool) -> list:
        """The rule set a prediction with `use_final` must use.

        Args:
            use_final (bool): True for the refined model, False for the one
                extracted from the trained estimator.

        Returns:
            List[Rule]: `final_rules` when requested and present, otherwise
            `initial_rules` (before refinement they are the same model).
        """
        if use_final and self.final_rules:
            return self.final_rules
        return self.initial_rules

    # Method to execute compile_tree_arrays
    def compile_tree_arrays(self, rules: Optional[list] = None, feature_names: Optional[list] = None) -> None:
        """
        Compile rules into numpy tree arrays for vectorized prediction.

        After calling this method, `predict_batch(X)` becomes available.
        This is called automatically by `update_native_model` but can also be
        called explicitly after rule removal to refresh the arrays.

        The compiled set becomes the *current* one, used by `predict_batch`
        and the binary/C/Arduino exports when no rule set is named. Each set
        compiled is also kept, so `predict_batch(use_final=...)` can switch
        between the initial and the refined model without recompiling.
        Compile again after editing a rule list in place.

        Args:
            rules (List[Rule], optional): Rules to compile. Defaults to
                final_rules if available, else initial_rules.
            feature_names (List[str], optional): Ordered feature names matching
                the columns of X that will be passed to predict_batch.
                If None, they are inferred from the rules.
        """
        if rules is None:
            rules = self.final_rules if self.final_rules else self.initial_rules

        state = self._compile_array_state(rules, feature_names)
        self._set_array_state(state)
        states = [st for st in getattr(self, '_array_states', [])
                  if st['rules'] is not rules]
        self._array_states = (states + [state])[-4:]

    # Method to compile one rule set into tree arrays
    def _compile_array_state(self, rules: list, feature_names: Optional[list] = None) -> dict:
        """
        Compile one rule set into tree arrays without making it current.

        Args:
            rules (List[Rule]): Rules to compile.
            feature_names (List[str], optional): Column order of X. If None,
                inferred from the rules (sorted).

        Returns:
            dict: 'rules' (the list compiled, kept by identity), 'n_rules',
            'feature_names', 'trees', 'class_groups', 'is_init'.
        """
        # --- Infer feature names from rules if not provided ---
        if feature_names is None:
            feat_set: set = set()
            for rule in rules:
                for var, _, _ in rule.parsed_conditions:
                    feat_set.add(var)
            feature_names = sorted(feat_set)

        feature_name_to_idx = {name: i for i, name in enumerate(feature_names)}

        n_classes = self.num_classes
        class_groups = None
        is_init = None

        # --- Group rules by tree ---
        if self.algorithm_type == 'Decision Tree':
            # Single tree — all rules belong to the same tree
            tree_arrays = [
                self._build_single_tree_arrays(
                    rules, feature_name_to_idx, self.algorithm_type, n_classes,
                    default_class=self._default_class_int(),
                )
            ]

        elif self.algorithm_type == 'Random Forest':
            tree_rules_map: Dict[str, list] = defaultdict(list)
            for rule in rules:
                tid = rule.name.split('_')[0] if '_' in rule.name else 'Tree0'
                tree_rules_map[tid].append(rule)

            tree_arrays = []
            for tid in sorted(tree_rules_map.keys(), key=_tree_sort_key):
                arr = self._build_single_tree_arrays(
                    tree_rules_map[tid], feature_name_to_idx,
                    self.algorithm_type, n_classes,
                    default_class=self._default_class_int(),
                    class_labels=self._rf_labels(n_classes),
                )
                tree_arrays.append(arr)

        elif self.algorithm_type == 'Gradient Boosting Decision Trees':
            tree_rules_map = defaultdict(list)
            for rule in rules:
                tid = rule.name.split('_')[0] if '_' in rule.name else rule.name
                tree_rules_map[tid].append(rule)

            tree_arrays = []
            class_groups = []
            is_init = []

            for tid in sorted(tree_rules_map.keys(), key=_tree_sort_key):
                trules = tree_rules_map[tid]
                if not trules:
                    continue
                # Check if this is an init rule (single rule, no conditions)
                if (len(trules) == 1 and not trules[0].parsed_conditions
                        and trules[0].contribution is not None):
                    is_init.append(True)
                    class_groups.append(trules[0].class_group)
                    # Store init score directly — no tree structure needed
                    tree_arrays.append({
                        'init_score': float(trules[0].contribution),
                    })
                else:
                    is_init.append(False)
                    class_groups.append(trules[0].class_group)
                    arr = self._build_single_tree_arrays(
                        trules, feature_name_to_idx,
                        self.algorithm_type, n_classes,
                        default_class=self._default_class_int(),
                    )
                    tree_arrays.append(arr)
        else:
            raise ValueError(f'Unsupported algorithm_type: {self.algorithm_type}')

        return {
            'rules': rules,
            'n_rules': len(rules),
            'feature_names': list(feature_names),
            'trees': tree_arrays,
            'class_groups': class_groups,
            'is_init': is_init,
        }

    # Method to make a compiled rule set the current one
    def _set_array_state(self, state: dict) -> None:
        """Install `state` (from `_compile_array_state`) as the current arrays.

        Args:
            state (dict): A compiled rule set.
        """
        self._array_feature_names = list(state['feature_names'])
        self._tree_arrays = state['trees']
        self._tree_class_groups = state['class_groups']
        self._tree_is_init = state['is_init']
        self._arrays_source = state['rules']
        self._arrays_source_len = state['n_rules']
        self._arrays_compiled = True

    # Method to get the compiled arrays of the current rule set
    def _current_array_state(self) -> dict:
        """The current arrays as a state dict (see `_compile_array_state`).

        Returns:
            dict: The arrays `predict_batch` uses when no rule set is named.
        """
        return {
            'rules': getattr(self, '_arrays_source', None),
            'n_rules': getattr(self, '_arrays_source_len', None),
            'feature_names': list(self._array_feature_names),
            'trees': self._tree_arrays,
            'class_groups': self._tree_class_groups,
            'is_init': self._tree_is_init,
        }

    # Method to get the compiled arrays of the rule set a prediction needs
    def _array_state_for(self, use_final: Optional[bool],
                         feature_names: Optional[list] = None) -> dict:
        """The compiled arrays of the rule set selected by `use_final`.

        Args:
            use_final (bool | None): None for the current arrays (whatever
                `compile_tree_arrays` compiled last); True/False for the
                refined/initial rule set, compiled on first use.
            feature_names (List[str], optional): Column order of X, used only
                when the set must be compiled and nothing was compiled before.

        Returns:
            dict: A compiled rule set (see `_compile_array_state`).

        Raises:
            RuntimeError: If `use_final` is None and nothing was compiled.
        """
        compiled = getattr(self, '_arrays_compiled', False)
        rules = None
        if use_final is not None and (self.initial_rules or self.final_rules):
            rules = self._rules_for(use_final)
        if rules is None:
            # No rule set named, or no rules at all (a model loaded with
            # load_binary holds only its arrays): use the current arrays.
            if not compiled:
                raise RuntimeError(
                    'Tree arrays not compiled. Call compile_tree_arrays() first.'
                )
            return self._current_array_state()
        names = (list(self._array_feature_names)
                 if getattr(self, '_array_feature_names', None) else feature_names)
        return self._array_state_for_rules(rules, names)

    # Method to get (compiling if needed) the arrays of one rule list
    def _array_state_for_rules(self, rules: list,
                               feature_names: Optional[list] = None) -> dict:
        """The compiled arrays of `rules`, compiled and kept on first use.

        Args:
            rules (List[Rule]): The rule list (matched by identity).
            feature_names (List[str], optional): Column order when compiling.

        Returns:
            dict: A compiled rule set (see `_compile_array_state`).
        """
        if (getattr(self, '_arrays_compiled', False)
                and getattr(self, '_arrays_source', None) is rules
                and getattr(self, '_arrays_source_len', None) == len(rules)):
            return self._current_array_state()
        states = getattr(self, '_array_states', [])
        for st in states:
            if st['rules'] is rules and st['n_rules'] == len(rules):
                return st

        state = self._compile_array_state(rules, feature_names)
        states = [st for st in states if st['rules'] is not rules]
        self._array_states = (states + [state])[-4:]
        return state

    # Method to execute _traverse_tree_batch
    @staticmethod
    def _traverse_tree_batch(
        X: np.ndarray,
        feature_idx: np.ndarray,
        threshold: np.ndarray,
        children_left: np.ndarray,
        children_right: np.ndarray,
        max_depth: int = 50,
    ) -> np.ndarray:
        """
        Traverse a single tree for all samples simultaneously using numpy indexing.

        This is the core vectorized operation:  O(depth) steps, each step processes
        ALL N samples in parallel via numpy fancy indexing -- no masking, no branching.

        Leaf nodes are set up as self-loops (children point to themselves) with
        threshold = -inf, so samples that have already reached a leaf simply stay
        in place each iteration.

        Args:
            X (np.ndarray): Input data, shape (n_samples, n_features).
            feature_idx (np.ndarray): Per-node feature index, shape (n_nodes,).
                Leaf nodes have a valid column index (not -2) since we read
                from X unconditionally.
            threshold (np.ndarray): Per-node threshold, shape (n_nodes,).
                Leaf nodes have -inf so (val <= -inf) is always False.
            children_left (np.ndarray): Per-node left child, shape (n_nodes,).
                Leaf nodes point to themselves.
            children_right (np.ndarray): Per-node right child, shape (n_nodes,).
                Leaf nodes point to themselves.
            max_depth (int): Maximum depth of the tree. The traversal loop runs
                exactly max_depth iterations.

        Returns:
            np.ndarray: Array of leaf node indices, shape (n_samples,).
        """
        n_samples = X.shape[0]
        node_ids = np.zeros(n_samples, dtype=np.int32)  # Start at root (node 0)
        sample_idx = np.arange(n_samples)  # Precompute once

        for _ in range(max_depth):
            # Gather the split feature column and threshold for every sample
            feat = feature_idx[node_ids]          # shape (n_samples,)

            # Gather the feature value from X using fancy indexing
            # For leaf nodes: feat is a valid column (0), thresh is -inf,
            # so go_left is False, and children_right[leaf] = leaf (no-op).
            go_left = X[sample_idx, feat] <= threshold[node_ids]

            node_ids = np.where(go_left, children_left[node_ids],
                                children_right[node_ids])

        return node_ids

    # Method to execute predict_batch
    def predict_batch(
        self,
        X: np.ndarray,
        feature_names: Optional[list] = None,
        use_final: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Vectorized batch prediction over a numpy array.

        This is the high-performance prediction method.  It uses the pre-compiled
        tree arrays (from compile_tree_arrays) and numpy vectorized traversal.

        Args:
            X (np.ndarray): Input data, shape (n_samples, n_features).
                Columns must be ordered to match the feature_names used during
                compile_tree_arrays (or the feature_names argument here).
            feature_names (list, optional): Feature names corresponding to columns
                of X.  If provided and different from compiled order, X columns
                are reordered accordingly.
            use_final (bool, optional): True predicts with final_rules (the
                refined model; initial_rules if there are none), False with
                initial_rules, each compiled on first use and kept. None (the
                default) uses whatever `compile_tree_arrays` compiled last.

        Returns:
            np.ndarray: Predicted class labels, shape (n_samples,), dtype int.
        """
        st = self._array_state_for(use_final, feature_names)
        trees, groups, is_init = st['trees'], st['class_groups'], st['is_init']

        # Reorder columns if feature_names differ from compiled order
        if feature_names is not None and list(feature_names) != st['feature_names']:
            col_order = [list(feature_names).index(f) for f in st['feature_names']]
            X = X[:, col_order]

        self._check_missing_values(X, self._trees_route_missing(trees))
        X = self._quantize_input(X)
        n_samples = X.shape[0]

        # --- Decision Tree ---
        if self.algorithm_type == 'Decision Tree':
            tree = trees[0]
            leaf_ids = _accel_traverse(
                X, tree['feature_idx'], tree['threshold'],
                tree['children_left'], tree['children_right'],
                tree['max_depth'], tree.get('missing_left'),
            )
            # Uncovered regions already carry `default_class` in `value`; count
            # them so the loss of coverage is reported, not silently absorbed.
            holes = tree.get('is_hole')
            if holes is not None:
                self.fallback_activations += int(holes[leaf_ids].sum())
            predictions = tree['value'][leaf_ids]
            return predictions

        # --- Random Forest (Soft Voting) ---
        elif self.algorithm_type == 'Random Forest':
            n_classes = self.num_classes
            n_trees = len(trees)

            # Build list of tree tuples for multi-tree C traversal
            tree_tuples = [
                (t['feature_idx'], t['threshold'],
                 t['children_left'], t['children_right'], t['max_depth'],
                 t.get('missing_left'))
                for t in trees
            ]
            # all_leaf_ids: shape (n_trees, n_samples), int32
            all_leaf_ids = _accel_traverse_multi(X, tree_tuples)

            # Accumulate probability contributions from each tree
            prob_sum = np.zeros((n_samples, n_classes), dtype=np.float64)
            for t_idx in range(n_trees):
                leaf_ids = all_leaf_ids[t_idx]
                # Get raw counts at leaves
                raw_counts = trees[t_idx]['value'][leaf_ids]
                # Normalize to probabilities per sample
                totals = raw_counts.sum(axis=1, keepdims=True)
                totals = np.where(totals == 0, 1.0, totals)
                probas = raw_counts / totals
                prob_sum += probas

            # Average and argmax
            avg_proba = prob_sum / n_trees
            # argmax is a column of the distribution; answer its label
            labels = np.asarray(self._rf_labels(n_classes), dtype=np.int32)
            predictions = labels[np.argmax(avg_proba, axis=1)]

            # Every tree abstained (refinement left this sample uncovered in
            # all of them): the argmax would silently return class 0 with no
            # evidence behind it.  Resolve to `default_class` and count it.
            abstained = prob_sum.sum(axis=1) == 0.0
            n_abstained = int(abstained.sum())
            if n_abstained:
                predictions[abstained] = np.int32(self._default_class_int())
                self.fallback_activations += n_abstained
            return predictions

        # --- GBDT (Additive Scoring) ---
        elif self.algorithm_type == 'Gradient Boosting Decision Trees':
            classes = self._gbdt_classes or []
            is_binary = self._gbdt_is_binary
            assert groups is not None
            assert is_init is not None

            # Separate init trees from real trees and batch-traverse real trees
            real_tree_indices = []
            tree_tuples = []
            for i, tree in enumerate(trees):
                if not is_init[i]:
                    real_tree_indices.append(i)
                    tree_tuples.append((
                        tree['feature_idx'], tree['threshold'],
                        tree['children_left'], tree['children_right'],
                        tree['max_depth'], tree.get('missing_left'),
                    ))

            # Batch traverse all real trees at once
            if tree_tuples:
                all_leaf_ids = _accel_traverse_multi(X, tree_tuples)
            else:
                all_leaf_ids = np.empty((0, n_samples), dtype=np.int32)

            # A stage whose region is uncovered contributes 0.0 (abstention).
            # Track, per sample, whether any stage abstained.
            touched_hole = np.zeros(n_samples, dtype=bool)

            if is_binary and len(classes) >= 2:
                # Binary: accumulate score for positive class
                pos_class = classes[1]
                scores = np.zeros(n_samples, dtype=np.float64)

                # Add init scores
                for i, tree in enumerate(trees):
                    if is_init[i]:
                        if groups[i] == pos_class:
                            scores += tree['init_score']

                # Add real tree contributions
                for batch_idx, orig_idx in enumerate(real_tree_indices):
                    if groups[orig_idx] != pos_class:
                        continue
                    leaf_ids = all_leaf_ids[batch_idx]
                    scores += trees[orig_idx]['value'][leaf_ids]
                    holes = trees[orig_idx].get('is_hole')
                    if holes is not None:
                        touched_hole |= holes[leaf_ids]

                # Decide on the raw score, as scikit-learn does
                # (raw_predictions >= 0); see update_native_model.
                predictions = np.where(
                    scores >= 0.0, int(classes[1]), int(classes[0])
                ).astype(np.int32)
                self.fallback_activations += int(touched_hole.sum())
                return predictions

            else:
                # Multiclass: accumulate scores per class
                class_to_col = {cl: idx for idx, cl in enumerate(classes)}
                n_cls = len(classes)
                scores = np.zeros((n_samples, n_cls), dtype=np.float64)

                # Add init scores
                for i, tree in enumerate(trees):
                    if is_init[i]:
                        cg = groups[i]
                        col = class_to_col.get(cg)
                        if col is not None:
                            scores[:, col] += tree['init_score']

                # Add real tree contributions
                for batch_idx, orig_idx in enumerate(real_tree_indices):
                    cg = groups[orig_idx]
                    col = class_to_col.get(cg)
                    if col is None:
                        continue
                    leaf_ids = all_leaf_ids[batch_idx]
                    scores[:, col] += trees[orig_idx]['value'][leaf_ids]
                    holes = trees[orig_idx].get('is_hole')
                    if holes is not None:
                        touched_hole |= holes[leaf_ids]

                # Argmax -> class label
                best_col = np.argmax(scores, axis=1)
                class_arr = np.array([int(cl) for cl in classes], dtype=np.int32)
                predictions = class_arr[best_col]
                self.fallback_activations += int(touched_hole.sum())
                return predictions

        raise ValueError(f'Unsupported algorithm_type: {self.algorithm_type}')

    # Method to execute predict_batch_proba
    def predict_batch_proba(
        self,
        X: np.ndarray,
        feature_names: Optional[list] = None,
        use_final: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Vectorized batch probability prediction.

        Args:
            X (np.ndarray): Input data, shape (n_samples, n_features).
            feature_names (list, optional): Feature names for column reordering.
            use_final (bool, optional): Rule set to use, as in `predict_batch`.

        Returns:
            np.ndarray: Predicted probabilities, shape (n_samples, n_classes).
                For DT, this is a one-hot array.
                For RF, this is the averaged probability across trees.
                For GBDT binary, shape (n_samples, 2) with sigmoid probabilities.
                For GBDT multiclass, shape (n_samples, n_classes) with softmax probabilities.
        """
        st = self._array_state_for(use_final, feature_names)
        trees, groups, is_init = st['trees'], st['class_groups'], st['is_init']

        if feature_names is not None and list(feature_names) != st['feature_names']:
            col_order = [list(feature_names).index(f) for f in st['feature_names']]
            X = X[:, col_order]

        # Same input handling as predict_batch, so the probabilities describe
        # the leaves predict_batch routes the samples to.
        self._check_missing_values(X, self._trees_route_missing(trees))
        X = self._quantize_input(X)
        n_samples = X.shape[0]

        # --- Decision Tree ---
        if self.algorithm_type == 'Decision Tree':
            tree = trees[0]
            leaf_ids = _accel_traverse(
                X, tree['feature_idx'], tree['threshold'],
                tree['children_left'], tree['children_right'],
                tree['max_depth'], tree.get('missing_left'),
            )
            preds = tree['value'][leaf_ids]
            # One-hot encode, in the column of each predicted *label*
            # (classes_ order): labels 1..3 are columns 0..2, and labels
            # 10/20/30 are not column indices at all.
            n_classes = self.num_classes
            proba = np.zeros((n_samples, n_classes), dtype=np.float64)
            col_of = {lab: j for j, lab in enumerate(self._rf_labels(n_classes))}
            uniq, inverse = np.unique(preds, return_inverse=True)
            cols = np.array([col_of.get(int(u), -1) for u in uniq],
                            dtype=np.int64)[inverse.reshape(-1)]
            rows = np.flatnonzero(cols >= 0)
            proba[rows, cols[rows]] = 1.0
            return proba

        # --- Random Forest ---
        elif self.algorithm_type == 'Random Forest':
            n_classes = self.num_classes
            prob_sum = np.zeros((n_samples, n_classes), dtype=np.float64)
            n_trees = len(trees)

            # Batch traverse all trees at once
            tree_tuples = [
                (t['feature_idx'], t['threshold'],
                 t['children_left'], t['children_right'], t['max_depth'],
                 t.get('missing_left'))
                for t in trees
            ]
            all_leaf_ids = _accel_traverse_multi(X, tree_tuples)

            for t_idx in range(n_trees):
                leaf_ids = all_leaf_ids[t_idx]
                raw_counts = trees[t_idx]['value'][leaf_ids]
                totals = raw_counts.sum(axis=1, keepdims=True)
                totals = np.where(totals == 0, 1.0, totals)
                probas = raw_counts / totals
                prob_sum += probas
            return prob_sum / n_trees

        # --- GBDT ---
        elif self.algorithm_type == 'Gradient Boosting Decision Trees':
            classes = self._gbdt_classes or []
            is_binary = self._gbdt_is_binary
            assert groups is not None
            assert is_init is not None
            class_to_col = {cl: idx for idx, cl in enumerate(classes)}
            n_cls = len(classes)
            scores = np.zeros((n_samples, n_cls), dtype=np.float64)

            # Separate init vs real trees, batch traverse real ones
            real_tree_indices = []
            tree_tuples_gbdt = []
            for i, tree in enumerate(trees):
                if is_init[i]:
                    cg = groups[i]
                    col = class_to_col.get(cg)
                    if col is not None:
                        scores[:, col] += tree['init_score']
                else:
                    real_tree_indices.append(i)
                    tree_tuples_gbdt.append((
                        tree['feature_idx'], tree['threshold'],
                        tree['children_left'], tree['children_right'],
                        tree['max_depth'], tree.get('missing_left'),
                    ))

            if tree_tuples_gbdt:
                all_leaf_ids = _accel_traverse_multi(X, tree_tuples_gbdt)
                for batch_idx, orig_idx in enumerate(real_tree_indices):
                    cg = groups[orig_idx]
                    col = class_to_col.get(cg)
                    if col is None:
                        continue
                    leaf_ids = all_leaf_ids[batch_idx]
                    scores[:, col] += trees[orig_idx]['value'][leaf_ids]

            if is_binary and n_cls == 2:
                # Sigmoid on positive class score
                prob_pos = 1.0 / (1.0 + np.exp(-scores[:, 1]))
                proba = np.column_stack([1.0 - prob_pos, prob_pos])
                return proba
            else:
                # Softmax
                exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
                proba = exp_scores / exp_scores.sum(axis=1, keepdims=True)
                return proba

        raise ValueError(f'Unsupported algorithm_type: {self.algorithm_type}')

    # =========================================================================
    # BINARY / C HEADER EXPORT
    # =========================================================================

    # Method to serialise the code -> label table for the binary format
    @staticmethod
    def _pack_label_table(table: list) -> bytes:
        """The label table as the bytes of the binary format (version 4).

        Args:
            table (list): Original labels (str, float, int, bool), by code.

        Returns:
            bytes: ``n_labels`` (uint16) then one typed entry per label.

        Raises:
            ValueError: If a label cannot be written.
        """
        out = [struct.pack('<H', len(table))]
        for lab in table:
            if isinstance(lab, bool):
                out.append(struct.pack('<BB', 3, 1 if lab else 0))
            elif isinstance(lab, str):
                encoded = lab.encode('utf-8')
                if len(encoded) > 0xFFFF:
                    raise ValueError('A class label is too long for the binary format.')
                out.append(struct.pack('<BH', 0, len(encoded)) + encoded)
            elif isinstance(lab, int):
                out.append(struct.pack('<Bq', 2, lab))
            elif isinstance(lab, float):
                out.append(struct.pack('<Bd', 1, lab))
            else:
                raise ValueError(f'Cannot write class label {lab!r:.60} to a binary model.')
        return b''.join(out)

    # Method to execute export_to_binary
    def export_to_binary(self, filepath: str = 'model.bin') -> None:
        """
        Export the compiled tree arrays to a compact binary file.

        File format (all little-endian):
            Header:
                4 bytes  magic: b'PYRA'
                1 byte   version: 1, or 2 when the input precision is set,
                         3 with column labels, 4 with missing-value routing
                         or a label table (see the flags)
                1 byte   algorithm_type: 0=DT, 1=RF, 2=GBDT
                2 bytes  n_features (uint16)
                2 bytes  n_classes (uint16)
                2 bytes  n_trees (uint16)
                4 bytes  default_class (int32)
            Version 2 and later:
                1 byte   flags: bit 0 = compare inputs rounded to float32
                         (``input_dtype='float32'``, as scikit-learn does).
                         Version 4 also: bit 1 = per-node missing-value
                         routing stored with each tree; bit 2 = column
                         labels follow; bit 3 = label table follows.
            Version 3 (a Random Forest whose labels are not 0..K-1), or
            version 4 with flag bit 2:
                n_classes * 4 bytes  label of each distribution column (int32)
            Version 4 with flag bit 3 (labels that are not integers; the
            model answers their codes 0..K-1, the table names them):
                2 bytes  n_labels (uint16)
                For each label, in code order:
                    1 byte  type: 0 = text, 1 = float64, 2 = int64, 3 = bool
                    text:   2 bytes length (uint16) + utf-8 bytes
                    float64 / int64: 8 bytes; bool: 1 byte
            For GBDT additionally:
                1 byte   is_binary (bool)
                2 bytes  n_gbdt_classes (uint16)
                For each gbdt_class:
                    4 bytes  class_label (int32)
                For each gbdt_class:
                    8 bytes  init_score (float64)
            Feature names:
                For each feature:
                    2 bytes  name_len (uint16)
                    N bytes  name (utf-8)
            For each tree:
                4 bytes  n_nodes (int32)
                For GBDT:
                    1 byte   is_init (bool)
                    4 bytes  class_group (int32, only if not init)
                    If is_init:
                        8 bytes  init_score (float64)
                        continue to next tree
                feature_idx:   n_nodes * 4 bytes (int32)
                threshold:     n_nodes * 8 bytes (float64)
                children_left: n_nodes * 4 bytes (int32)
                children_right:n_nodes * 4 bytes (int32)
                missing_left:  n_nodes * 1 byte (uint8, 1 = a NaN goes
                               left; version 4 with flag bit 1 only)
                value:
                    DT:   n_nodes * 4 bytes (int32)
                    RF:   n_nodes * n_classes * 8 bytes (float64)
                    GBDT: n_nodes * 8 bytes (float64)
                max_depth:     4 bytes (int32)

        Models that record no missing-value routing and whose labels are
        integers are written exactly as before (versions 1 to 3).

        Args:
            filepath (str): Output file path.
        """
        import struct

        if not hasattr(self, '_arrays_compiled') or not self._arrays_compiled:
            raise RuntimeError(
                'Tree arrays not compiled. Call compile_tree_arrays() first.'
            )

        algo_map = {'Decision Tree': 0, 'Random Forest': 1,
                     'Gradient Boosting Decision Trees': 2}
        algo_byte = algo_map[self.algorithm_type]

        try:
            default_cls = int(str(self.default_class).replace('Class', '').strip())
        except (ValueError, AttributeError):
            default_cls = 0

        # Count real trees (skip GBDT init entries for tree count)
        n_trees = len(self._tree_arrays)

        with open(filepath, 'wb') as f:
            # --- Header ---
            # Version 1 had no room for the input precision; a model that
            # rounds its inputs to float32 is written as version 2, so a
            # reloaded model routes samples exactly like the one exported.
            f32 = getattr(self, 'input_dtype', None) == 'float32'
            # Version 3 adds the label of each column of a forest's
            # distributions, written only when they are not 0..K-1.
            rf_labels = (self._rf_labels(self.num_classes)
                         if self.algorithm_type == 'Random Forest' else None)
            has_labels = (rf_labels is not None
                          and rf_labels != list(range(self.num_classes)))
            # Version 4: the missing-value routing of every split, the code
            # -> label table, and the column labels of a Decision Tree too
            # (predict_proba of a reloaded tree needs them).
            routes_missing = self._c_routes_missing()
            table = getattr(self, 'label_table', None)
            col_labels = self._rf_labels(self.num_classes)
            dt_labels = (self.algorithm_type == 'Decision Tree'
                         and col_labels != list(range(self.num_classes)))
            if routes_missing or table is not None or dt_labels:
                version = 4
                has_labels = has_labels or dt_labels
                rf_labels = col_labels
            else:
                version = 3 if has_labels else (2 if f32 else 1)
            flags = ((1 if f32 else 0)
                     | (2 if version == 4 and routes_missing else 0)
                     | (4 if version == 4 and has_labels else 0)
                     | (8 if version == 4 and table is not None else 0))
            f.write(b'PYRA')                                      # magic
            f.write(struct.pack('<B', version))                    # version
            f.write(struct.pack('<B', algo_byte))                  # algorithm_type
            f.write(struct.pack('<H', len(self._array_feature_names)))  # n_features
            f.write(struct.pack('<H', self.num_classes))           # n_classes
            f.write(struct.pack('<H', n_trees))                    # n_trees
            f.write(struct.pack('<i', default_cls))                # default_class
            if version >= 2:
                f.write(struct.pack('<B', flags))                  # flags
            if has_labels:
                for lab in rf_labels:
                    f.write(struct.pack('<i', lab))                # class labels
            if flags & 8:
                f.write(self._pack_label_table(table))             # label table

            # --- GBDT metadata ---
            if self.algorithm_type == 'Gradient Boosting Decision Trees':
                classes = self._gbdt_classes or []
                f.write(struct.pack('<B', 1 if self._gbdt_is_binary else 0))
                f.write(struct.pack('<H', len(classes)))
                for cl in classes:
                    f.write(struct.pack('<i', int(cl)))
                init_scores = self._gbdt_init_scores or {}
                for cl in classes:
                    f.write(struct.pack('<d', init_scores.get(cl, 0.0)))

            # --- Feature names ---
            for name in self._array_feature_names:
                encoded = name.encode('utf-8')
                f.write(struct.pack('<H', len(encoded)))
                f.write(encoded)

            # --- Trees ---
            tree_class_groups = self._tree_class_groups
            for i, tree in enumerate(self._tree_arrays):
                if 'init_score' in tree:
                    # GBDT init entry
                    f.write(struct.pack('<i', 0))       # n_nodes = 0 (sentinel)
                    f.write(struct.pack('<B', 1))        # is_init = True
                    f.write(struct.pack('<d', tree['init_score']))
                    assert tree_class_groups is not None
                    f.write(struct.pack('<i', int(tree_class_groups[i])))
                    continue

                n_nodes = len(tree['feature_idx'])
                f.write(struct.pack('<i', n_nodes))

                if self.algorithm_type == 'Gradient Boosting Decision Trees':
                    f.write(struct.pack('<B', 0))        # is_init = False
                    assert tree_class_groups is not None
                    f.write(struct.pack('<i', int(tree_class_groups[i])))

                f.write(tree['feature_idx'].tobytes())
                f.write(tree['threshold'].tobytes())
                f.write(tree['children_left'].tobytes())
                f.write(tree['children_right'].tobytes())
                if flags & 2:
                    f.write(np.ascontiguousarray(
                        tree['missing_left'], dtype=np.uint8).tobytes())
                f.write(tree['value'].tobytes())
                f.write(struct.pack('<i', tree.get('max_depth', 50)))

        print(f'[EXPORT] Binary model saved to {filepath} '
              f'({os.path.getsize(filepath)} bytes)')

    # Method to execute load_binary
    @classmethod
    def load_binary(cls, filepath: str) -> 'RuleClassifier':
        """
        Load a RuleClassifier from a binary file created by export_to_binary().

        The returned object supports predict_batch() but NOT rule-level
        operations (no Rule objects are reconstructed).

        Args:
            filepath (str): Path to the .bin file.

        Returns:
            RuleClassifier: A classifier ready for predict_batch().

        Raises:
            ValueError: If the file is not a valid model: unknown magic,
                version or algorithm, truncated or trailing data, or tree
                arrays the traversal could not walk safely (see
                `_validate_tree_arrays`). The traversal does no bounds
                checks, so the file is never trusted.
        """
        import struct

        algo_names = {0: 'Decision Tree', 1: 'Random Forest',
                      2: 'Gradient Boosting Decision Trees'}

        with open(filepath, 'rb') as f:
            data = f.read()
        pos = 0

        # Read exactly `n` bytes, raising on a short read (not struct.error)
        def _read(n):
            nonlocal pos
            if n < 0 or pos + n > len(data):
                raise ValueError(
                    f'Invalid binary model file: truncated at byte {pos} '
                    f'(needed {n} more bytes, {len(data) - pos} left).'
                )
            chunk = data[pos:pos + n]
            pos += n
            return chunk

        # Unpack one struct format from the next bytes
        def _unpack(fmt):
            return struct.unpack(fmt, _read(struct.calcsize(fmt)))[0]

        magic = _read(4) if len(data) >= 4 else data
        if magic != b'PYRA':
            raise ValueError(f'Invalid binary model file (magic={magic!r})')
        version = _unpack('<B')
        if version not in (1, 2, 3, 4):
            raise ValueError(f'Unsupported binary model version: {version}')
        algo_byte = _unpack('<B')
        if algo_byte not in algo_names:
            raise ValueError(f'Invalid binary model file: unknown algorithm '
                             f'type {algo_byte}.')
        n_features = _unpack('<H')
        n_classes = _unpack('<H')
        n_trees = _unpack('<H')
        default_cls = _unpack('<i')
        flags = _unpack('<B') if version >= 2 else 0
        known_flags = 0x0F if version >= 4 else 0x01
        if flags & ~known_flags:
            raise ValueError(f'Invalid binary model file: unknown flags {flags:#04x}.')
        class_labels = list(range(n_classes))
        if version == 3 or flags & 4:
            class_labels = [_unpack('<i') for _ in range(n_classes)]
        label_table = None
        if flags & 8:
            n_labels = _unpack('<H')
            if n_labels != n_classes:
                raise ValueError('Invalid binary model file: the label table '
                                 f'has {n_labels} labels for {n_classes} classes.')
            label_table = []
            for _ in range(n_labels):
                kind = _unpack('<B')
                if kind == 0:
                    try:
                        label_table.append(_read(_unpack('<H')).decode('utf-8'))
                    except UnicodeDecodeError as e:
                        raise ValueError(f'Invalid binary model file: label '
                                         f'is not utf-8 ({e}).') from None
                elif kind == 1:
                    label_table.append(_unpack('<d'))
                elif kind == 2:
                    label_table.append(_unpack('<q'))
                elif kind == 3:
                    label_table.append(bool(_unpack('<B')))
                else:
                    raise ValueError(f'Invalid binary model file: unknown '
                                     f'label type {kind}.')
            if len(set(label_table)) != len(label_table):
                raise ValueError('Invalid binary model file: repeated class labels.')
        has_missing = bool(flags & 2)

        algorithm_type = algo_names[algo_byte]
        if algorithm_type == 'Random Forest' and n_classes < 1:
            raise ValueError('Invalid binary model file: a Random Forest '
                             'needs at least one class.')

        # GBDT metadata
        gbdt_is_binary = False
        gbdt_classes = []
        gbdt_init_scores = {}
        if algorithm_type == 'Gradient Boosting Decision Trees':
            gbdt_is_binary = bool(_unpack('<B'))
            n_gbdt_classes = _unpack('<H')
            for _ in range(n_gbdt_classes):
                gbdt_classes.append(_unpack('<i'))
            for cl in gbdt_classes:
                gbdt_init_scores[cl] = _unpack('<d')

        # Feature names
        feature_names = []
        for _ in range(n_features):
            name_len = _unpack('<H')
            try:
                feature_names.append(_read(name_len).decode('utf-8'))
            except UnicodeDecodeError as e:
                raise ValueError(f'Invalid binary model file: feature name '
                                 f'is not utf-8 ({e}).') from None

        # Bytes per node: feature_idx, threshold, both children, value.
        value_size = {'Decision Tree': 4, 'Random Forest': n_classes * 8,
                      'Gradient Boosting Decision Trees': 8}[algorithm_type]
        node_size = 4 + 8 + 4 + 4 + value_size + (1 if has_missing else 0)

        # Trees
        tree_arrays = []
        tree_class_groups = []
        tree_is_init = []

        for t in range(n_trees):
            n_nodes = _unpack('<i')

            if algorithm_type == 'Gradient Boosting Decision Trees':
                is_init = bool(_unpack('<B'))
                if is_init:
                    init_score = _unpack('<d')
                    cg = _unpack('<i')
                    tree_arrays.append({'init_score': init_score})
                    tree_class_groups.append(cg)
                    tree_is_init.append(True)
                    continue
                else:
                    cg = _unpack('<i')
                    tree_class_groups.append(cg)
                    tree_is_init.append(False)

            # Bounded by the bytes left, so a forged count cannot make us
            # allocate (or traverse) more than the file holds.
            if n_nodes < 1 or n_nodes * node_size > len(data) - pos:
                raise ValueError(
                    f'Invalid binary model file: tree {t} has n_nodes='
                    f'{n_nodes} ({len(data) - pos} bytes left).'
                )

            feature_idx = np.frombuffer(_read(n_nodes * 4), dtype='<i4').astype(np.int32)
            threshold_arr = np.frombuffer(_read(n_nodes * 8), dtype='<f8').astype(np.float64)
            children_left = np.frombuffer(_read(n_nodes * 4), dtype='<i4').astype(np.int32)
            children_right = np.frombuffer(_read(n_nodes * 4), dtype='<i4').astype(np.int32)
            missing_left = None
            if has_missing:
                missing_left = np.frombuffer(_read(n_nodes), dtype=np.uint8).copy()

            if algorithm_type == 'Decision Tree':
                value_arr = np.frombuffer(_read(n_nodes * 4), dtype='<i4').astype(np.int32)
            elif algorithm_type == 'Random Forest':
                value_arr = np.frombuffer(
                    _read(n_nodes * n_classes * 8), dtype='<f8'
                ).astype(np.float64).reshape(n_nodes, n_classes)
            else:  # GBDT
                value_arr = np.frombuffer(_read(n_nodes * 8), dtype='<f8').astype(np.float64)

            tree = {
                'feature_idx': feature_idx,
                'threshold': threshold_arr,
                'children_left': children_left,
                'children_right': children_right,
                'value': value_arr,
                'max_depth': _unpack('<i'),
            }
            if missing_left is not None:
                tree['missing_left'] = missing_left
            cls._validate_tree_arrays(tree, n_features, t)
            tree_arrays.append(tree)

        if pos != len(data):
            raise ValueError(
                f'Invalid binary model file: {len(data) - pos} unexpected '
                'trailing bytes.'
            )

        # Build a minimal RuleClassifier instance
        # We use __new__ to bypass __init__ (no Rule objects needed)
        obj = cls.__new__(cls)
        obj.algorithm_type = algorithm_type
        obj.num_classes = n_classes
        obj.class_labels = class_labels
        obj.default_class = default_cls
        obj.initial_rules = []
        obj.final_rules = []
        obj.duplicated_rules = []
        obj.specific_rules = []
        obj.native_fn = None
        obj._gbdt_init_scores = gbdt_init_scores if gbdt_init_scores else None
        obj._gbdt_is_binary = gbdt_is_binary
        obj._gbdt_classes = gbdt_classes if gbdt_classes else None
        obj._array_feature_names = feature_names
        obj._tree_arrays = tree_arrays
        obj._tree_class_groups = tree_class_groups if tree_class_groups else None
        obj._tree_is_init = tree_is_init if tree_is_init else None
        obj._arrays_compiled = True
        # No rules: predict_batch always answers with these arrays.
        obj._arrays_source = None
        obj._arrays_source_len = None
        obj._array_states = []
        obj._native_source = None
        obj.fallback_activations = 0
        obj.input_dtype = 'float32' if flags & 1 else None
        obj.label_table = label_table
        obj.sklearn_estimator = None

        return obj

    # Method to check that tree arrays can be traversed safely
    @staticmethod
    def _validate_tree_arrays(tree: dict, n_features: int, tree_no: int = 0) -> None:
        """Check one tree's arrays against what the traversal assumes.

        The C traversal reads ``X[i, feature_idx[node]]`` and follows the
        children without bounds checks, so arrays from an untrusted source
        must hold:

        - a leaf is a self-loop (both children are the node itself) with
          ``feature_idx == 0`` and ``threshold == -inf``, as
          `_build_single_tree_arrays` writes it;
        - any other node has ``0 <= feature_idx < n_features`` and both
          children in ``[0, n_nodes)``, neither being the node itself;
        - ``0 <= max_depth <= n_nodes``;
        - the split nodes form a tree rooted at node 0: no node is reached
          twice (no cycle, no child shared by two splits), every node is
          reachable from the root, and no leaf is deeper than ``max_depth``
          (the traversal runs exactly ``max_depth`` steps, so a deeper leaf
          would leave samples on a split node).

        Args:
            tree (dict): 'feature_idx', 'threshold', 'children_left',
                'children_right', 'max_depth'.
            n_features (int): Number of input columns the model reads.
            tree_no (int): Position of the tree, for the error message.

        Raises:
            ValueError: Naming the first violated invariant.
        """
        feat = tree['feature_idx']
        thr = tree['threshold']
        left = tree['children_left']
        right = tree['children_right']
        n_nodes = len(feat)

        # Raise with the tree and the first offending node
        def _fail(what, mask):
            node = int(np.flatnonzero(mask)[0])
            raise ValueError(
                f'Invalid tree arrays (tree {tree_no}, node {node}): {what}.'
            )

        if n_nodes < 1 or not (len(thr) == len(left) == len(right) == n_nodes):
            raise ValueError(f'Invalid tree arrays (tree {tree_no}): '
                             'arrays are empty or of different lengths.')
        missing = tree.get('missing_left')
        if missing is not None:
            if len(missing) != n_nodes:
                raise ValueError(f'Invalid tree arrays (tree {tree_no}): '
                                 'missing_left has the wrong length.')
            bad = np.asarray(missing) > 1
            if bad.any():
                _fail('missing_left must be 0 or 1', bad)
        ids = np.arange(n_nodes)
        is_leaf = (left == ids) & (right == ids)
        bad = is_leaf & ((feat != 0) | ~np.isneginf(thr))
        if bad.any():
            _fail('a leaf must have feature_idx 0 and threshold -inf', bad)
        inner = ~is_leaf
        bad = inner & ((feat < 0) | (feat >= n_features))
        if bad.any():
            _fail(f'feature_idx out of range [0, {n_features})', bad)
        bad = inner & ((left < 0) | (left >= n_nodes)
                       | (right < 0) | (right >= n_nodes))
        if bad.any():
            _fail(f'child index out of range [0, {n_nodes})', bad)
        bad = inner & ((left == ids) | (right == ids))
        if bad.any():
            _fail('a split node cannot be its own child', bad)
        depth = int(tree['max_depth'])
        if not 0 <= depth <= n_nodes:
            raise ValueError(f'Invalid tree arrays (tree {tree_no}): '
                             f'max_depth={depth} not in [0, {n_nodes}].')

        # The split nodes must form a tree rooted at node 0: walk it level by
        # level, so a child pointing back to an ancestor (a cycle) or to a
        # node another split already reached shows up as a node seen twice.
        # Such arrays stay in bounds but route samples wrongly.
        seen = np.zeros(n_nodes, dtype=bool)
        seen[0] = True
        frontier = np.zeros(1, dtype=np.intp)
        level = 0
        while True:
            frontier = frontier[inner[frontier]]
            if frontier.size == 0:
                break
            level += 1
            if level > depth:
                # The traversal runs exactly max_depth steps, so a sample
                # deeper than that would stop on a split node and read its
                # (meaningless) value.
                raise ValueError(
                    f'Invalid tree arrays (tree {tree_no}): max_depth={depth} '
                    f'is smaller than the depth of the tree (> {depth}).')
            kids = np.concatenate((left[frontier], right[frontier]))
            if seen[kids].any() or np.unique(kids).size != kids.size:
                counts = np.bincount(kids, minlength=n_nodes)
                _fail('node reached twice (a cycle or a child shared by two '
                      'splits): the split nodes do not form a tree',
                      (counts > 1) | (seen & (counts > 0)))
            seen[kids] = True
            frontier = kids
        # `_build_single_tree_arrays` numbers only the nodes it creates while
        # walking from the root, so every node is reachable.
        if not seen.all():
            _fail('node not reachable from the root', ~seen)

    # Method to check whether the C exports route missing values
    def _c_routes_missing(self) -> bool:
        """Whether the C/Arduino exports carry the missing-value routing.

        Gradient Boosting never does: scikit-learn's GradientBoostingClassifier
        refuses NaN, so its exports are written as before.

        Returns:
            bool: True when every compiled tree has its ``missing_left``
            array and there is at least one tree to traverse.
        """
        if self.algorithm_type == 'Gradient Boosting Decision Trees':
            return False
        trees = self._tree_arrays
        return (self._trees_route_missing(trees)
                and any('init_score' not in t for t in trees))

    # Method to generate the C function that walks one tree
    def _c_traverse_function(self, name: str, routes_missing: bool,
                             leaf_comment: str = '') -> List[str]:
        """The C source of the tree traversal of the C/Arduino exports.

        Without missing-value routing it is the function the exports always
        had (``x <= t`` goes left, so a NaN goes right). With it, a NaN goes
        to the side the split recorded (``missing_left``), as in
        scikit-learn.

        Args:
            name (str): Function name.
            routes_missing (bool): Emit the ``missing_left`` parameter and
                the ``isnan`` test.
            leaf_comment (str): Comment appended to the leaf test.

        Returns:
            List[str]: The lines, followed by an empty line.
        """
        f32 = getattr(self, 'input_dtype', None) == 'float32'
        # scikit-learn casts X to float32 before comparing.
        value = ('(double)(float)features[feat_idx[node]]' if f32
                 else 'features[feat_idx[node]]')
        out = [
            f'static inline int32_t {name}(',
            '    const double *features,',
            '    const int32_t *feat_idx,',
            '    const double *thresh,',
            '    const int32_t *left,',
            '    const int32_t *right,',
        ]
        if routes_missing:
            out.append('    const uint8_t *missing_left,')
        out += [
            '    int max_depth',
            ') {',
            '    int32_t node = 0;',
            '    int d;',
            '    for (d = 0; d < max_depth; d++) {',
            f'        if (left[node] == node) break;{leaf_comment}',
        ]
        if routes_missing:
            out += [
                f'        double x = {value};',
                '        if (isnan(x))  /* missing value: the side the split recorded */',
                '            node = missing_left[node] ? left[node] : right[node];',
                '        else if (x <= thresh[node])',
            ]
        else:
            out.append(f'        if ({value} <= thresh[node])')
        out += [
            '            node = left[node];',
            '        else',
            '            node = right[node];',
            '    }',
            '    return node;',
            '}',
            '',
        ]
        return out

    # Method to pass each tree's missing-value routing to the traversal
    @staticmethod
    def _c_with_missing_args(source: str, routes_missing: bool) -> str:
        """Add ``treeN_missing_left`` to every traversal call of `source`.

        Args:
            source (str): Generated C source.
            routes_missing (bool): Whether the traversal takes the argument.

        Returns:
            str: The source, unchanged when `routes_missing` is False.
        """
        if not routes_missing:
            return source
        return re.sub(r'tree(\d+)_right, TREE\1_DEPTH\)',
                      r'tree\1_right, tree\1_missing_left, TREE\1_DEPTH)', source)

    # Method to document the code -> label table in generated C
    def _c_label_table_comment(self, line_comment: bool = False) -> List[str]:
        """Comment lines naming the original label of each integer code.

        The C, Arduino and binary targets answer integer codes; for labels
        that are not integers the code is the position in ``classes_``.

        Args:
            line_comment (bool): ``//`` comments (Arduino) instead of a block.

        Returns:
            List[str]: The lines (none when the labels are integers).
        """
        table = getattr(self, 'label_table', None)
        if table is None:
            return []

        def show(lab):
            """A label on one ASCII comment line."""
            text = repr(lab) if not isinstance(lab, str) else lab
            return self._c_comment(text.encode('ascii', 'backslashreplace').decode('ascii'))

        if line_comment:
            out = ['// Class labels: pyra_predict() returns the code; original labels:']
            out += [f'//   {i} = {show(lab)}' for i, lab in enumerate(table)]
        else:
            out = ['/* Class labels: predict() returns the code; original labels:']
            out += [f'   {i} = {show(lab)}' for i, lab in enumerate(table)]
            out.append('*/')
        out.append('')
        return out

    # Method to execute export_to_c_header
    def export_to_c_header(
        self,
        filepath: str = 'model.h',
        guard_name: str = 'PYRULEANALYZER_MODEL_H',
    ) -> None:
        """
        Export the compiled tree arrays as a standalone C header file.

        The generated header contains const arrays suitable for Arduino /
        embedded targets. It includes a ``predict(const float *features)``
        function that traverses the trees and returns the predicted class.

        Args:
            filepath (str): Output file path.
            guard_name (str): Include-guard macro name.
        """
        if not hasattr(self, '_arrays_compiled') or not self._arrays_compiled:
            raise RuntimeError(
                'Tree arrays not compiled. Call compile_tree_arrays() first.'
            )

        try:
            default_cls = int(str(self.default_class).replace('Class', '').strip())
        except (ValueError, AttributeError):
            default_cls = 0

        lines: List[str] = []
        a = lines.append
        routes_missing = self._c_routes_missing()

        a(f'#ifndef {guard_name}')
        a(f'#define {guard_name}')
        a('')
        a('/* Auto-generated by pyruleanalyzer — do not edit */')
        a('')
        a('#include <stdint.h>')
        if routes_missing:
            a('#include <math.h>  /* HUGE_VAL; isnan: missing values (NaN) */')
        else:
            a('#include <math.h>  /* HUGE_VAL */')
        a('')
        a(f'#define N_FEATURES {len(self._array_feature_names)}')
        a(f'#define N_CLASSES  {self.num_classes}')
        a(f'#define N_TREES    {len(self._tree_arrays)}')
        a(f'#define DEFAULT_CLASS {default_cls}')
        a('')

        # Feature name comments
        a('/* Feature order:')
        for i, name in enumerate(self._array_feature_names):
            a(f'   [{i}] {self._c_comment(name)}')
        a('*/')
        a('')
        for line in self._c_label_table_comment():
            a(line)

        # Emit per-tree arrays
        for t_idx, tree in enumerate(self._tree_arrays):
            if 'init_score' in tree:
                continue  # GBDT init handled separately

            n_nodes = len(tree['feature_idx'])
            tree_depth = tree.get('max_depth', 50)
            a(f'/* Tree {t_idx}: {n_nodes} nodes, depth {tree_depth} */')
            a(f'#define TREE{t_idx}_DEPTH {tree_depth}')

            # feature_idx
            vals = ', '.join(str(v) for v in tree['feature_idx'])
            a(f'static const int32_t tree{t_idx}_feature[{n_nodes}] = {{{vals}}};')

            # threshold — convert special float values to C-compatible literals
            def _format_c_float(v):
                """Format a float value as a C-compatible literal.

                Special values (inf, -inf, nan) are mapped to the C89
                ``HUGE_VAL`` macro of <math.h> (+infinity on IEEE targets,
                avr-libc included):
                  * +inf  → "HUGE_VAL"   (the split isolating missing values)
                  * -inf  → "-HUGE_VAL"  (leaves; never compared)
                  * nan   → "-HUGE_VAL"  (fallback; treated as -inf)

                A literal such as ``1e500`` is out of range: GCC turns it
                into infinity with a warning, MSVC refuses to compile it.
                """
                import math
                if math.isinf(v):
                    return "HUGE_VAL" if v > 0 else "-HUGE_VAL"
                if math.isnan(v):
                    return "-HUGE_VAL"  # safe fallback
                return f'{v:.17g}'

            vals = ', '.join(_format_c_float(v) for v in tree['threshold'])
            a(f'static const double tree{t_idx}_threshold[{n_nodes}] = {{{vals}}};')

            # children_left
            vals = ', '.join(str(v) for v in tree['children_left'])
            a(f'static const int32_t tree{t_idx}_left[{n_nodes}] = {{{vals}}};')

            # children_right
            vals = ', '.join(str(v) for v in tree['children_right'])
            a(f'static const int32_t tree{t_idx}_right[{n_nodes}] = {{{vals}}};')

            # missing_left: 1 where a split sends a missing value (NaN) left
            if routes_missing:
                vals = ', '.join(str(int(v)) for v in tree['missing_left'])
                a(f'static const uint8_t tree{t_idx}_missing_left[{n_nodes}] = {{{vals}}};')

            # value
            if self.algorithm_type == 'Decision Tree':
                vals = ', '.join(str(v) for v in tree['value'])
                a(f'static const int32_t tree{t_idx}_value[{n_nodes}] = {{{vals}}};')
            elif self.algorithm_type == 'Random Forest':
                flat = tree['value'].flatten()
                vals = ', '.join(f'{v:.17g}' for v in flat)
                a(f'static const double tree{t_idx}_value[{n_nodes * self.num_classes}] = {{{vals}}};')
            elif self.algorithm_type == 'Gradient Boosting Decision Trees':
                vals = ', '.join(f'{v:.17g}' for v in tree['value'])
                a(f'static const double tree{t_idx}_value[{n_nodes}] = {{{vals}}};')

            a('')

        # Predict function
        lines.extend(self._c_traverse_function(
            'traverse_tree', routes_missing, '  /* leaf: self-loop */'))

        # Main predict function depends on algorithm type
        if self.algorithm_type == 'Decision Tree':
            a('static inline int32_t predict(const double *features) {')
            a('    int32_t leaf = traverse_tree(features, tree0_feature, tree0_threshold, tree0_left, tree0_right, TREE0_DEPTH);')
            a('    return tree0_value[leaf];')
            a('}')
        elif self.algorithm_type == 'Random Forest':
            a('static inline int32_t predict(const double *features) {')
            a(f'    double prob_sum[N_CLASSES] = {{0}};')
            a(f'    int voted = 0;')
            for t_idx in range(len(self._tree_arrays)):
                a(f'    {{')
                a(f'        int32_t leaf = traverse_tree(features, tree{t_idx}_feature, tree{t_idx}_threshold, tree{t_idx}_left, tree{t_idx}_right, TREE{t_idx}_DEPTH);')
                a(f'        double total = 0.0;')
                a(f'        int k;')
                a(f'        for (k = 0; k < N_CLASSES; k++) total += tree{t_idx}_value[leaf * N_CLASSES + k];')
                a(f'        if (total > 0.0) {{')
                a(f'            voted = 1;')
                a(f'            for (k = 0; k < N_CLASSES; k++) prob_sum[k] += tree{t_idx}_value[leaf * N_CLASSES + k] / total;')
                a(f'        }}')
                a(f'    }}')
            # Every tree abstained (refinement left the sample uncovered in
            # all of them): default class, as predict_batch and the CPN do.
            a(f'    if (!voted) return DEFAULT_CLASS;')
            a(f'    int32_t best = 0;')
            a(f'    int k;')
            a(f'    for (k = 1; k < N_CLASSES; k++) {{')
            a(f'        if (prob_sum[k] > prob_sum[best]) best = k;')
            a(f'    }}')
            # best is a column of the distribution; answer its label
            rf_labels = ', '.join(str(v) for v in self._rf_labels(self.num_classes))
            a(f'    static const int32_t class_map[N_CLASSES] = {{ {rf_labels} }};')
            a(f'    return class_map[best];')
            a('}')
        elif self.algorithm_type == 'Gradient Boosting Decision Trees':
            classes = self._gbdt_classes or []
            init_scores = self._gbdt_init_scores or {}
            tree_class_groups_c = self._tree_class_groups
            assert tree_class_groups_c is not None

            if self._gbdt_is_binary and len(classes) >= 2:
                a('#include <math.h>')
                a('')
                a('static inline int32_t predict(const double *features) {')
                pos_class = classes[1]
                init_s = init_scores.get(pos_class, 0.0)
                a(f'    double score = {init_s:.17g};')
                for t_idx, tree in enumerate(self._tree_arrays):
                    if 'init_score' in tree:
                        continue
                    if tree_class_groups_c[t_idx] != pos_class:
                        continue
                    a(f'    score += tree{t_idx}_value[traverse_tree(features, tree{t_idx}_feature, tree{t_idx}_threshold, tree{t_idx}_left, tree{t_idx}_right, TREE{t_idx}_DEPTH)];')
                # Decide on the raw score, as scikit-learn does (raw >= 0).
                a(f'    return (score >= 0.0) ? {int(classes[1])} : {int(classes[0])};')
                a('}')
            else:
                a('#include <math.h>')
                a('')
                a('static inline int32_t predict(const double *features) {')
                a(f'    double scores[{len(classes)}] = {{')
                inits = ', '.join(f'{init_scores.get(cl, 0.0):.17g}' for cl in classes)
                a(f'        {inits}')
                a(f'    }};')
                for t_idx, tree in enumerate(self._tree_arrays):
                    if 'init_score' in tree:
                        continue
                    cg = tree_class_groups_c[t_idx]
                    col = list(classes).index(cg) if cg in classes else 0
                    a(f'    scores[{col}] += tree{t_idx}_value[traverse_tree(features, tree{t_idx}_feature, tree{t_idx}_threshold, tree{t_idx}_left, tree{t_idx}_right, TREE{t_idx}_DEPTH)];')
                a(f'    int32_t best = 0;')
                a(f'    int k;')
                a(f'    for (k = 1; k < {len(classes)}; k++) {{')
                a(f'        if (scores[k] > scores[best]) best = k;')
                a(f'    }}')
                # Map col index back to class label
                a(f'    static const int32_t class_map[{len(classes)}] = {{')
                cls_vals = ', '.join(str(int(cl)) for cl in classes)
                a(f'        {cls_vals}')
                a(f'    }};')
                a(f'    return class_map[best];')
                a('}')

        a('')
        a(f'#endif /* {guard_name} */')
        a('')

        with open(filepath, 'w') as f:
            f.write(self._c_with_missing_args('\n'.join(lines), routes_missing))

        print(f'[EXPORT] C header saved to {filepath} '
              f'({os.path.getsize(filepath)} bytes)')

    # Method to export Arduino .ino sketch
    def export_to_arduino_ino(
        self,
        filepath: str = 'model.ino',
        board_model: str = 'auto',
        serial_baud: int = 115200,
        include_sensor_placeholders: bool = True,
        include_memory_check: bool = True,
        max_flash_percent: float = 90.0,
        max_ram_percent: float = 85.0,
    ) -> dict:
        """
        Export the compiled model to a complete Arduino/ESP32 .ino sketch file.

        The generated sketch is self-contained and ready for upload via
        Arduino IDE or PlatformIO. It includes:
          - The C header with tree data (inline as #define constants)
          - The predict() function inline in the sketch
          - A setup() that prints model metadata
          - A loop() that reads features, runs prediction, and outputs results

        Parameters
        ----------
        filepath : str
            Output .ino file path. Default is 'model.ino'.
        board_model : str
            Target board to determine memory limits and architecture hints.
            Options: 'auto', 'uno', 'nano', 'mega', 'esp32', 'leonardo'.
            Default is 'auto' (detects from model size).
        serial_baud : int
            Serial baud rate for output. Default is 115200.
        include_sensor_placeholders : bool
            If True, includes a read_features() function with TODO placeholders
            that the user should replace with real sensor readings.
            Default is True.
        include_memory_check : bool
            If True, includes compile-time checks for Flash/RAM limits in
            comments at the top of the file. Also prints memory report to stdout.
            Default is True.
        max_flash_percent : float
            Maximum percentage of Flash memory to allow (for compatibility check).
            Default is 90.0.
        max_ram_percent : float
            Maximum percentage of SRAM to allow (for compatibility check).
            Default is 85.0.

        Returns
        -------
        dict
            Dictionary with keys:
              - 'ino_path': Path to the generated .ino file
              - 'memory_check': Dict with memory estimation results
              - 'metadata': Model metadata embedded in the sketch
        """
        if not hasattr(self, '_arrays_compiled') or not self._arrays_compiled:
            raise RuntimeError(
                'Tree arrays not compiled. Call compile_tree_arrays() first.'
            )

        # Memory limits per board type (Flash bytes, SRAM bytes)
        BOARD_MEMORY = {
            'uno':      (143360, 2048),     # Arduino Uno: ~140KB flash, 2KB SRAM
            'nano':     (143360, 2048),     # Arduino Nano: same as Uno
            'mega':     (1048576, 8192),    # Arduino Mega: 1MB flash, 8KB SRAM
            'leonardo': (32768, 2560),      # Leonardo: ~32KB for sketch, 2.5KB SRAM
            'esp32':    (4194304, 524288),  # ESP32: 4MB flash, 512KB SRAM
        }

        # Estimate Flash and RAM usage
        estimates = _compute_memory_for_rule_classifier(self)
        est_flash = estimates['estimated_flash_bytes']
        est_ram = estimates['total_ram_bytes']

        # Determine target board
        if board_model == 'auto':
            # Auto-detect: pick the smallest board that fits
            if est_flash <= 143360 and est_ram <= 2048:
                board_model = 'uno'
            elif est_flash <= 1048576 and est_ram <= 8192:
                board_model = 'mega'
            else:
                board_model = 'esp32'

        max_flash, max_sram = BOARD_MEMORY.get(board_model, (143360, 2048))
        
        flash_pct = (est_flash / max_flash) * 100 if max_flash > 0 else 0
        ram_pct = (est_ram / max_sram) * 100 if max_sram > 0 else 0
        
        fits_flash = est_flash <= (max_flash * max_flash_percent / 100.0)
        fits_ram = est_ram <= (max_sram * max_ram_percent / 100.0)

        # Default class
        try:
            default_cls = int(str(self.default_class).replace('Class', '').strip())
        except (ValueError, AttributeError):
            default_cls = 0

        # Build the sketch content
        lines = []
        a = lines.append

        # Header comment block with metadata and memory check
        a('// ============================================================')
        a(f'// Auto-generated Arduino/ESP32 Sketch by pyruleanalyzer')
        a(f'// Model: {self.algorithm_type}')
        a(f'// Features: {len(self._array_feature_names)} | Classes: {self.num_classes} | Trees: {len(self._tree_arrays)}')
        a(f'// Target Board: {board_model.upper()}')
        a('// ============================================================')
        a('')

        if include_memory_check:
            flash_status = 'OK' if fits_flash else 'OVER!'
            ram_status = 'OK' if fits_ram else 'OVER!'
            a('// --- Memory Check (Target: ' + board_model.upper() + ') ---')
            a(f'// Flash: {est_flash:,} / {max_flash:,} bytes ({flash_pct:.1f}%) [{flash_status}]')
            a(f'// SRAM:  {est_ram:,} / {max_sram:,} bytes ({ram_pct:.1f}%) [{ram_status}]')
            if not fits_flash or not fits_ram:
                a('// WARNING: Model may NOT fit on this board! Reduce tree depth or number of trees.')
            a('')

        # Include the inline C header content
        a('// ============================================================')
        a(f'// MODEL DATA ({len(self._tree_arrays)} trees, {len(self._array_feature_names)} features)')
        a('// ============================================================')
        
        # Generate inline header (same logic as export_to_c_header but without #ifndef guards)
        default_cls_inline = default_cls
        routes_missing = self._c_routes_missing()

        if routes_missing:
            a('#include <math.h>  // HUGE_VAL; isnan: missing values (NaN)')
        else:
            a('#include <math.h>  // HUGE_VAL')
        a(f'#define N_FEATURES {len(self._array_feature_names)}')
        a(f'#define N_CLASSES  {self.num_classes}')
        a(f'#define N_TREES    {len(self._tree_arrays)}')
        a(f'#define DEFAULT_CLASS {default_cls_inline}')
        a('')

        # Feature name comments
        a('// Feature order:')
        for i, name in enumerate(self._array_feature_names):
            a(f'//   [{i}] {self._c_comment(name)}')
        a('')
        for line in self._c_label_table_comment(line_comment=True):
            a(line)

        # Per-tree arrays (same format as C header)
        for t_idx, tree in enumerate(self._tree_arrays):
            if 'init_score' in tree:
                continue  # GBDT init handled separately

            n_nodes = len(tree['feature_idx'])
            tree_depth = tree.get('max_depth', 50)
            a(f'// Tree {t_idx}: {n_nodes} nodes, depth {tree_depth}')
            a(f'#define TREE{t_idx}_DEPTH {tree_depth}')

            # feature_idx
            vals = ', '.join(str(v) for v in tree['feature_idx'])
            a(f'static const int32_t tree{t_idx}_feature[{n_nodes}] = {{{vals}}};')

            # threshold with C-compatible float formatting
            def _format_c_float_inline(v):
                # HUGE_VAL, not 1e500, which MSVC refuses (see the C header).
                import math
                if math.isinf(v):
                    return "HUGE_VAL" if v > 0 else "-HUGE_VAL"
                if math.isnan(v):
                    return "-HUGE_VAL"
                return f'{v:.17g}'

            vals = ', '.join(_format_c_float_inline(v) for v in tree['threshold'])
            a(f'static const double tree{t_idx}_threshold[{n_nodes}] = {{{vals}}};')

            # children_left
            vals = ', '.join(str(v) for v in tree['children_left'])
            a(f'static const int32_t tree{t_idx}_left[{n_nodes}] = {{{vals}}};')

            # children_right
            vals = ', '.join(str(v) for v in tree['children_right'])
            a(f'static const int32_t tree{t_idx}_right[{n_nodes}] = {{{vals}}};')

            # missing_left: 1 where a split sends a missing value (NaN) left
            if routes_missing:
                vals = ', '.join(str(int(v)) for v in tree['missing_left'])
                a(f'static const uint8_t tree{t_idx}_missing_left[{n_nodes}] = {{{vals}}};')

            # value
            if self.algorithm_type == 'Decision Tree':
                vals = ', '.join(str(v) for v in tree['value'])
                a(f'static const int32_t tree{t_idx}_value[{n_nodes}] = {{{vals}}};')
            elif self.algorithm_type == 'Random Forest':
                flat = tree['value'].flatten()
                vals = ', '.join(f'{v:.17g}' for v in flat)
                a(f'static const double tree{t_idx}_value[{n_nodes * self.num_classes}] = {{{vals}}};')
            elif self.algorithm_type == 'Gradient Boosting Decision Trees':
                vals = ', '.join(f'{v:.17g}' for v in tree['value'])
                a(f'static const double tree{t_idx}_value[{n_nodes}] = {{{vals}}};')

            a('')

        # Inline traverse_tree function
        a('// ============================================================')
        a('// PREDICTION ENGINE (C inline — no library needed)')
        a('// ============================================================')
        a('')
        lines.extend(self._c_traverse_function('pyra_traverse_tree', routes_missing))

        # Main predict function based on algorithm type
        if self.algorithm_type == 'Decision Tree':
            a('static inline int32_t pyra_predict(const double *features) {')
            a('    int32_t leaf = pyra_traverse_tree(features, tree0_feature, tree0_threshold, tree0_left, tree0_right, TREE0_DEPTH);')
            a('    return tree0_value[leaf];')
            a('}')
        elif self.algorithm_type == 'Random Forest':
            a('#include <math.h>')
            a('')
            a('static inline int32_t pyra_predict(const double *features) {')
            a(f'    double prob_sum[N_CLASSES] = {{0}};')
            a(f'    int voted = 0;')
            for t_idx in range(len(self._tree_arrays)):
                a(f'    {{')
                a(f'        int32_t leaf = pyra_traverse_tree(features, tree{t_idx}_feature, tree{t_idx}_threshold, tree{t_idx}_left, tree{t_idx}_right, TREE{t_idx}_DEPTH);')
                a(f'        double total = 0.0;')
                a(f'        int k;')
                a(f'        for (k = 0; k < N_CLASSES; k++) total += tree{t_idx}_value[leaf * N_CLASSES + k];')
                a(f'        if (total > 0.0) {{')
                a(f'            voted = 1;')
                a(f'            for (k = 0; k < N_CLASSES; k++) prob_sum[k] += tree{t_idx}_value[leaf * N_CLASSES + k] / total;')
                a(f'        }}')
                a(f'    }}')
            # Every tree abstained (refinement left the sample uncovered in
            # all of them): default class, as predict_batch and the CPN do.
            a(f'    if (!voted) return DEFAULT_CLASS;')
            a(f'    int32_t best = 0;')
            a(f'    int k;')
            a(f'    for (k = 1; k < N_CLASSES; k++) {{')
            a(f'        if (prob_sum[k] > prob_sum[best]) best = k;')
            a(f'    }}')
            # best is a column of the distribution; answer its label
            rf_labels = ', '.join(str(v) for v in self._rf_labels(self.num_classes))
            a(f'    static const int32_t class_map[N_CLASSES] = {{ {rf_labels} }};')
            a(f'    return class_map[best];')
            a('}')
        elif self.algorithm_type == 'Gradient Boosting Decision Trees':
            classes = self._gbdt_classes or []
            init_scores = self._gbdt_init_scores or {}
            tree_class_groups_c = self._tree_class_groups
            assert tree_class_groups_c is not None

            if self._gbdt_is_binary and len(classes) >= 2:
                a('#include <math.h>')
                a('')
                a('static inline int32_t pyra_predict(const double *features) {')
                pos_class = classes[1]
                init_s = init_scores.get(pos_class, 0.0)
                a(f'    double score = {init_s:.17g};')
                for t_idx, tree in enumerate(self._tree_arrays):
                    if 'init_score' in tree:
                        continue
                    if tree_class_groups_c[t_idx] != pos_class:
                        continue
                    a(f'    score += tree{t_idx}_value[pyra_traverse_tree(features, tree{t_idx}_feature, tree{t_idx}_threshold, tree{t_idx}_left, tree{t_idx}_right, TREE{t_idx}_DEPTH)];')
                # Decide on the raw score, as scikit-learn does (raw >= 0).
                a(f'    return (score >= 0.0) ? {int(classes[1])} : {int(classes[0])};')
                a('}')
            else:
                a('#include <math.h>')
                a('')
                a('static inline int32_t pyra_predict(const double *features) {')
                a(f'    double scores[{len(classes)}] = {{')
                inits = ', '.join(f'{init_scores.get(cl, 0.0):.17g}' for cl in classes)
                a(f'        {inits}')
                a(f'    }};')
                for t_idx, tree in enumerate(self._tree_arrays):
                    if 'init_score' in tree:
                        continue
                    cg = tree_class_groups_c[t_idx]
                    col = list(classes).index(cg) if cg in classes else 0
                    a(f'    scores[{col}] += tree{t_idx}_value[pyra_traverse_tree(features, tree{t_idx}_feature, tree{t_idx}_threshold, tree{t_idx}_left, tree{t_idx}_right, TREE{t_idx}_DEPTH)];')
                a(f'    int32_t best = 0;')
                a(f'    int k;')
                a(f'    for (k = 1; k < {len(classes)}; k++) {{')
                a(f'        if (scores[k] > scores[best]) best = k;')
                a(f'    }}')
                a(f'    static const int32_t class_map[{len(classes)}] = {{')
                cls_vals = ', '.join(str(int(cl)) for cl in classes)
                a(f'        {cls_vals}')
                a(f'    }};')
                a(f'    return class_map[best];')
                a('}')

        a('')

        # Arduino sketch section
        a('// ============================================================')
        a('// ARDUINO SKETCH SECTION')
        a('// ============================================================')
        a('')
        a(f'#define SERIAL_BAUD {serial_baud}')
        a('#define SAMPLE_INTERVAL_MS 1000')
        a('')

        # Global feature variables
        a('// Feature buffer (modify read_features() to populate these)')
        # double, matching pyra_predict(const double *): on AVR double is the
        # same 4-byte type as float, and on ESP32 a float buffer read through
        # a double pointer would be garbage.
        a(f'double features[N_FEATURES];')
        a('')

        if include_sensor_placeholders:
            a('// ============================================================')
            a('// TODO: Replace with your sensor reading logic')
            a('// ============================================================')
            a('void read_features(void) {')
            for i in range(min(len(self._array_feature_names), 10)):
                feat_name = self._array_feature_names[i] if i < len(self._array_feature_names) else f'feature_{i}'
                a(f'    features[{i}] = 0.0; // TODO: read {self._c_comment(feat_name)}')
            remaining = max(0, len(self._array_feature_names) - 10)
            if remaining > 0:
                a(f'    // ... and {remaining} more features')
            a('}')
            a('')

        # Setup function
        a('void setup(void) {')
        a('    Serial.begin(SERIAL_BAUD);')
        a('    while (!Serial) delay(10);')
        a('    Serial.println(F("pyruleanalyzer model loaded"));')
        a(f'    Serial.print(F("Algorithm: ")); Serial.println(F("{self.algorithm_type}"));')
        a(f'    Serial.print(F("Features: ")); Serial.print(N_FEATURES);')
        a(f'    Serial.print(F(" | Classes: ")); Serial.print(N_CLASSES);')
        a(f'    Serial.print(F(" | Trees: ")); Serial.println(N_TREES);')
        a('}')
        a('')

        # Loop function
        a('void loop(void) {')
        a('    unsigned long start_us = micros();')
        a('')
        if include_sensor_placeholders:
            a('    read_features();  // Populate features[] from sensors')
        else:
            a('    // TODO: populate features[] from your sensors')
        a('')
        a('    int32_t result = pyra_predict(features);')
        a('')
        a('    Serial.print(F("{\\"class\\":"));')
        a('    Serial.print(result);')
        a('    Serial.print(F(", \\"trees\\":"));')
        a(f'    Serial.print(N_TREES);')
        a('    Serial.println("}");')
        a('')
        a('    unsigned long elapsed_us = micros() - start_us;')
        a('    Serial.print(F("("));')
        a('    Serial.print(elapsed_us);')
        a('    Serial.println(F(" us)"));')
        a('')
        a('    delay(SAMPLE_INTERVAL_MS);')
        a('}')

        # Write the file
        ino_dir = os.path.dirname(filepath)
        if ino_dir:
            os.makedirs(ino_dir, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self._c_with_missing_args('\n'.join(lines), routes_missing))

        # Print memory report to stdout
        avr_pct_flash = (est_flash / max_flash) * 100 if max_flash > 0 else 0
        avr_pct_ram = (est_ram / max_sram) * 100 if max_sram > 0 else 0
        
        print(f'  ✅ Arduino sketch generated: {filepath}')
        print(f'     Flash: {est_flash:,} bytes ({avr_pct_flash:.1f}% of {max_flash:,})')
        print(f'     SRAM:  {est_ram:,} bytes ({avr_pct_ram:.1f}% of {max_sram:,})')
        
        if not fits_flash or not fits_ram:
            print('     ⚠️  WARNING: Model may NOT fit on target board!')

        file_size = os.path.getsize(filepath)
        
        return {
            'ino_path': filepath,
            'memory_check': {
                'board_model': board_model,
                'flash_bytes': est_flash,
                'ram_bytes': est_ram,
                'max_flash': max_flash,
                'max_sram': max_sram,
                'flash_percent': avr_pct_flash,
                'ram_percent': avr_pct_ram,
                'fits_flash': fits_flash,
                'fits_ram': fits_ram,
            },
            'metadata': {
                'algorithm_type': self.algorithm_type,
                'n_features': len(self._array_feature_names),
                'n_classes': self.num_classes,
                'n_trees': len(self._tree_arrays),
                'default_class': default_cls,
                'file_size_bytes': file_size,
            },
        }

    # Method to detect rules whose regions overlap
    def find_overlapping_rules(self, rules=None, per_tree=True, max_rules_per_tree=20000):
        """
        Find pairs of rules whose feature-space regions intersect.

        Each rule's conditions are axis-aligned inequalities, so its region is
        a box (a product of intervals). Two rules overlap exactly when their
        boxes intersect on every feature. Rules extracted from a tree never
        overlap — the leaves partition the space — but refinement can break
        that: promoting a sibling drops a condition and may extend the rule
        over ground already covered by other rules.

        This matters because an overlap is invisible to accuracy. The iterative
        engine resolves it by taking the first matching rule, which is an
        implicit priority order that a tree does not have and that the CPN model
        does not reproduce. A rule set with overlaps is a *rule list*, not a
        decision tree.

        Args:
            rules (List[Rule], optional): Rules to check. Defaults to
                final_rules if available, else initial_rules.
            per_tree (bool): Compare only rules belonging to the same tree
                (the name prefix before the first '_'). Rules from different
                trees of an ensemble are expected to overlap and are skipped.
            max_rules_per_tree (int): Trees with more rules than this are
                skipped, with their id reported under 'skipped_trees', to keep
                the quadratic comparison bounded.

        Returns:
            Dict with keys:
                'overlaps': list of (name_i, name_j, same_class) tuples;
                'n_overlaps': total number of overlapping pairs;
                'n_conflicting': pairs that also disagree on the predicted
                    class (these change the prediction, not just the shape);
                'trees_affected': sorted list of tree ids containing overlaps;
                'skipped_trees': tree ids skipped for exceeding the size limit.
        """
        if rules is None:
            rules = self.final_rules if self.final_rules else self.initial_rules

        # A Decision Tree is a single tree even though its rule names contain
        # '_' ("Rule0_Class1"); splitting on it would put every rule in its own
        # group and compare nothing.
        group_by_tree = per_tree and self.algorithm_type != 'Decision Tree'

        groups: Dict[str, list] = defaultdict(list)
        for rule in rules:
            tid = (rule.name.split('_')[0]
                   if (group_by_tree and '_' in rule.name) else '_all')
            groups[tid].append(rule)

        overlaps = []
        skipped = []
        for tid, group in groups.items():
            if len(group) > max_rules_per_tree:
                skipped.append(tid)
                continue

            # Bounds per rule: lo/hi arrays plus inclusiveness flags.
            feats = sorted({v for r in group for v, _, _ in r.parsed_conditions})
            if not feats:
                continue
            col = {f: i for i, f in enumerate(feats)}
            n, d = len(group), len(feats)

            lo = np.full((n, d), -np.inf)
            hi = np.full((n, d), np.inf)
            lo_open = np.zeros((n, d), dtype=bool)   # lower bound is strict (>)
            hi_open = np.zeros((n, d), dtype=bool)   # upper bound is strict (<)

            for i, rule in enumerate(group):
                for var, op, val in rule.parsed_conditions:
                    c = col[var]
                    if op in ('<=', '<'):
                        if val < hi[i, c] or (val == hi[i, c] and op == '<'):
                            hi[i, c] = val
                            hi_open[i, c] = (op == '<')
                    else:
                        if val > lo[i, c] or (val == lo[i, c] and op == '>'):
                            lo[i, c] = val
                            lo_open[i, c] = (op == '>')

            for i in range(n - 1):
                j = slice(i + 1, n)
                # Intervals [lo_a, hi_a] and [lo_b, hi_b] intersect when
                # max(lo) < min(hi), or they touch at a point that both sides
                # include.
                lo_max = np.maximum(lo[i], lo[j])
                hi_min = np.minimum(hi[i], hi[j])
                touching = lo_max == hi_min
                # Inclusiveness comes from whichever bound is binding: a looser
                # bound being strict says nothing about the tighter one. Only
                # when both coincide do they both have to be inclusive.
                lo_incl = np.where(
                    lo[i] > lo[j], ~lo_open[i],
                    np.where(lo[j] > lo[i], ~lo_open[j],
                             ~(lo_open[i] | lo_open[j])))
                hi_incl = np.where(
                    hi[i] < hi[j], ~hi_open[i],
                    np.where(hi[j] < hi[i], ~hi_open[j],
                             ~(hi_open[i] | hi_open[j])))
                per_feat = (lo_max < hi_min) | (touching & lo_incl & hi_incl)
                hits = np.nonzero(per_feat.all(axis=1))[0]
                for k in hits:
                    other = group[i + 1 + int(k)]
                    overlaps.append((
                        group[i].name, other.name,
                        str(group[i].class_) == str(other.class_),
                    ))

        trees = sorted({name.split('_')[0] for name, _, _ in overlaps})
        return {
            'overlaps': overlaps,
            'n_overlaps': len(overlaps),
            'n_conflicting': sum(1 for _, _, same in overlaps if not same),
            'trees_affected': trees,
            'skipped_trees': skipped,
        }

    # Method to find similar rules between trees
    def find_duplicated_rules_between_trees(self):
        """
        Identifies semantically similar rules between different trees in the forest.

        This method compares rules across the full rule set to find groups that:
        - Use the same set of variables, values, and logical operators.
        - Belong to the same target class.
        
        Optimization: Uses cached 'parsed_conditions' and sorting to ensure 
        condition order doesn't affect detection (A and B == B and A).

        Returns:
            List[List[Rule]]: A list of groups, where each group is a list of similar rules.
        """
        rules_by_signature = defaultdict(list)
        
        # Operate on final_rules (or initial_rules depending on pipeline stage)
        target_rules = self.final_rules if self.final_rules else self.initial_rules

        for rule in target_rules:
            # A rule without conditions is a GBDT init score (or a tree that
            # is a single leaf): it is not a region shared between trees, and
            # summing it with a leaf value would mix an unscaled init score
            # with learning-rate-scaled contributions.
            if not rule.parsed_conditions:
                continue
            # Create a canonical signature for the rule's logic
            # 1. We use parsed_conditions directly (no string parsing)
            # 2. We sort the conditions so that logic order doesn't matter
            # 3. We use the class label
            
            # signature format: (class_label, tuple_of_sorted_conditions)
            # condition format: (var, op, value) - already float from parsing
            # The missing-value routing is part of the region: two trees
            # with the same thresholds may send NaN different ways.
            sorted_conditions = tuple(sorted(
                (condition_signature(c) for c in rule.parsed_conditions),
                key=lambda s: (s[0], s[1], s[2], str(s[3]))))
            signature = (rule.class_, sorted_conditions)
            
            rules_by_signature[signature].append(rule)
            
        # Return only the groups that have more than one similar rule
        return [group for group in rules_by_signature.values() if len(group) > 1]
    
    # Method to find duplicated rules within the same tree (Boundary Redundancy)
    def find_duplicated_rules(self):
        """
        Identifies nearly identical rules within the same decision tree context.

        This method searches for rule pairs that:
        - Have the same class label.
        - Share all conditions except the last one (same path parent).
        - Differ only in the final condition, which splits the same variable at
          the same threshold with complementary operators (``<=``/``>`` or
          ``<``/``>=``), e.g. v1 <= 5 vs v1 > 5.

        Such pairs are considered duplicates because they imply the split at that
        boundary was unnecessary for the final classification outcome.

        Returns:
            List[Tuple[Rule,Rule]]: A list of tuples, each representing a pair of duplicated rules.
        """
        duplicated_rules = []
        rules_by_prefix = defaultdict(list)

        # Target rules to analyze
        target_rules = self.final_rules if self.final_rules else self.initial_rules

        # Boundary redundancy is a property of a *single* tree: the two rules
        # must be the two children of one split. Ensemble trees often split on
        # the same feature at the same threshold, so the signature has to
        # include the tree id -- otherwise leaves of different trees can be
        # merged as if they were siblings, silently dropping a leaf from one of
        # them. A Decision Tree is one tree even though its rule names contain
        # '_' ("Rule0_Class1"), so it is grouped as a whole.
        group_by_tree = self.algorithm_type != 'Decision Tree'

        # 1. Group rules by tree, class and their "Prefix" (all conditions except the last)
        for rule in target_rules:
            if not rule.parsed_conditions:
                continue

            tree_id = (rule.name.split('_')[0]
                       if (group_by_tree and '_' in rule.name) else '_all')

            # Prefix signature: (tree, class, conditions[0:-1])
            # We assume order matters here because it represents a tree path
            prefix_key = (tree_id, rule.class_, tuple(rule.parsed_conditions[:-1]))
            rules_by_prefix[prefix_key].append(rule)
            
        # 2. Check for boundary redundancy within groups
        for prefix, candidates in rules_by_prefix.items():
            if len(candidates) < 2:
                continue
            
            # Compare candidates pairwise O(k^2) where k is small (usually 2 for binary trees)
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    rule1 = candidates[i]
                    rule2 = candidates[j]
                    
                    # Access last condition directly from cached tuples
                    # Format: (var, op, value)
                    cond1 = rule1.parsed_conditions[-1]
                    cond2 = rule2.parsed_conditions[-1]
                    
                    var1, op1, val1 = cond1
                    var2, op2, val2 = cond2
                    
                    # Check 1: Must refer to the same variable
                    if var1 != var2:
                        continue

                    # Check 2: Must refer to the same threshold value
                    # Using a small epsilon for float comparison safety
                    if val1 != val2 and not abs(val1 - val2) <= 1e-9:
                        continue
                    
                    # Check 3: Complementary operators, (<= vs >) or (< vs >=),
                    # so the two rules split one region exactly in two.
                    is_duplicate = {op1, op2} in ({'<=', '>'}, {'<', '>='})

                    # GBDT: siblings must also add exactly the same value to
                    # the score (v_i = v_j). The merged rule keeps rule1's
                    # value, so any tolerance here would move the score of
                    # rule2's region and could flip a decision near 0.
                    if is_duplicate and self.algorithm_type == 'Gradient Boosting Decision Trees':
                        if rule1.contribution != rule2.contribution:
                            is_duplicate = False

                    # RF soft voting: siblings must have proportionally identical
                    # probability distributions.  Merging siblings with different
                    # distributions changes the per-tree probability, which can
                    # flip the overall soft vote even though the hard vote
                    # (majority class) stays the same.
                    if is_duplicate and self.algorithm_type == 'Random Forest':
                        d1 = rule1.class_distribution
                        d2 = rule2.class_distribution
                        if d1 is not None and d2 is not None:
                            t1 = sum(d1)
                            t2 = sum(d2)
                            if t1 > 0 and t2 > 0:
                                p1 = [c / t1 for c in d1]
                                p2 = [c / t2 for c in d2]
                                if any(abs(a - b) > 1e-9 for a, b in zip(p1, p2)):
                                    is_duplicate = False

                    if is_duplicate:
                        duplicated_rules.append((rule1, rule2))
                            
        return duplicated_rules
    
    # Method to set a custom rule removal function
    def set_custom_rule_removal(self, custom_function):
        """
        Allows the user to override the rule removal logic by employing their own implementation.

        This enables the injection of external logic to handle rule pruning or duplicate detection
        according to specific domain needs.

        Args:
            custom_function (Callable[[List[Rule]], Tuple[List[Rule], List[Tuple[Rule, Rule]]]]): 
                A callback function that takes a list of Rule instances as an argument and 
                returns a tuple containing:
                1. A new list of rules after processing (filtered).
                2. A list of pairs of rules identified as duplicates/removed.
        """
        # Python allows dynamic assignment of instance methods
        self.custom_rule_removal = custom_function

    # Method to remove rules based on custom logic (Default placeholder)
    def custom_rule_removal(self, rules):
        """
        Placeholder for custom rule removal logic. 
        
        By default, this method performs no operations and returns the rule set unchanged.
        It is intended to be overwritten via `set_custom_rule_removal`.

        Args:
            rules (List[Rule]): The input list of Rule instances.

        Returns:
            Tuple[List[Rule], List[Tuple[Rule, Rule]]]: 
                - The original list of rules (unchanged).
                - An empty list representing no duplicates found.
        """
        return rules, []

    # Method to adjust and remove duplicated rules
    def adjust_and_remove_rules(self, method="refine_duplicates"):
        """
        Adjusts and removes duplicated rules from the rule set based on the specified method.

        With ``"refine_duplicates"`` it finds sibling leaves whose split is redundant
        (:meth:`find_duplicated_rules`) and merges each pair into their parent,
        one round at a time. With ``"custom"`` it delegates to the function set
        via :meth:`set_custom_rule_removal`.

        Args:
            method (str): ``"refine_duplicates"`` (default) or ``"custom"``.

        Returns:
            Tuple[List[Rule], List[Tuple[Rule, Rule]]]: 
                - A new list of rules (merged + remaining).
                - A list of identified duplicated pairs (used for convergence checks).
        """
        if method == "custom":
            return self.custom_rule_removal(self.initial_rules)
        
        if method != "refine_duplicates":
            raise ValueError(f"Invalid method: {method}. Use 'refine_duplicates' or 'custom'.")

        # Determine source rules: Use final_rules if populated (iteration n), else initial (iteration 0)
        source_rules = self.final_rules if self.final_rules else self.initial_rules

        # 1. Boundary redundancy within the same tree: siblings that can be
        # merged into their parent
        sibling_pairs = self.find_duplicated_rules()

        rules_to_remove = set()
        new_generalized_rules = []

        if sibling_pairs:
            print(f"Merging {len(sibling_pairs)} pairs of duplicated rules...")

        for rule1, rule2 in sibling_pairs:
            rules_to_remove.add(rule1)
            rules_to_remove.add(rule2)

            # Create a new generalized rule (Parent Logic): the last condition,
            # which differentiated the two siblings, is dropped.
            new_generalized_rules.append(self._merge_sibling_pair(rule1, rule2))

        # 2. Rebuild the Rule Set
        # Keep rules that were NOT marked for removal + the newly created generalized rules
        final_list = new_generalized_rules + [r for r in source_rules if r not in rules_to_remove]
        
        return final_list, sibling_pairs
    
    # Method to name the class at an index of a class distribution
    def _label_at(self, idx: int, n: int) -> str:
        """The class label at index `idx` of an `n`-class distribution.

        Distributions are indexed like the estimator's ``classes_``, which are
        the labels only when those are 0..K-1; `class_labels` holds the real
        ones when it covers every class.

        Args:
            idx (int): Index into the distribution.
            n (int): Length of the distribution.

        Returns:
            str: The label, as the rules store it.
        """
        labels = getattr(self, 'class_labels', None)
        if labels is not None and len(labels) == n:
            return str(labels[idx])
        return str(idx)

    # Method to merge two sibling leaves into their parent
    def _merge_sibling_pair(self, rule1, rule2):
        """The rule of the parent of two sibling leaves with the same output.

        `find_duplicated_rules` only pairs siblings whose outputs agree (same
        class; for GBDT the same leaf value, for RF proportional
        distributions), so the parent rule predicts exactly what both did.

        Args:
            rule1 (Rule): One leaf.
            rule2 (Rule): Its sibling.

        Returns:
            Rule: The rule of the parent node. Its usage and error counts are
            the sums of the children's: its region is their union, and the
            low-usage stage decides on these counts.
        """
        common_conditions = rule1.conditions[:-1]
        name = f"{rule1.name}_&_{rule2.name}"
        if self.algorithm_type == 'Gradient Boosting Decision Trees':
            merged = Rule(name, rule1.class_, common_conditions,
                          leaf_value=rule1.leaf_value,
                          learning_rate=rule1.learning_rate,
                          class_group=rule1.class_group)
            merged.contribution = rule1.contribution  # == rule2.contribution
        else:
            d1 = rule1.class_distribution
            d2 = rule2.class_distribution
            if d1 is not None and d2 is not None:
                dist = [a + b for a, b in zip(d1, d2)]
            elif d1 is not None or d2 is not None:
                dist = list(d1 if d1 is not None else d2)
            else:
                dist = None
            # Both leaves predict rule1.class_; so does their parent.
            merged = Rule(name, rule1.class_, common_conditions,
                          class_distribution=dist)
        merged.parsed_conditions = list(rule1.parsed_conditions[:-1])
        merged.usage_count = rule1.usage_count + rule2.usage_count
        merged.error_count = rule1.error_count + rule2.error_count
        return merged

    # Method to remove boundary redundancy until none is left
    def merge_boundary_redundancy(self):
        """
        Merge sibling leaves with the same output, repeatedly, up to the root.

        This is the boundary-redundancy stage of the refinement: two sibling
        leaves that produce the same output are replaced by their parent, and
        since that may make the parent redundant with its own sibling, the
        stage repeats until no pair is left. The model's predictions do not
        change. Operates on `final_rules` (seeded from `initial_rules`).

        Returns:
            List[Tuple[Rule, Rule]]: Every pair merged, over all rounds.
        """
        if not self.final_rules:
            self.final_rules = list(self.initial_rules)
        merged_pairs = []
        while True:
            pairs = self.find_duplicated_rules()
            if not pairs:
                return merged_pairs
            remove = set()
            new_rules = []
            for rule1, rule2 in pairs:
                if id(rule1) in remove or id(rule2) in remove:
                    continue
                remove.update((id(rule1), id(rule2)))
                new_rules.append(self._merge_sibling_pair(rule1, rule2))
                merged_pairs.append((rule1, rule2))
            self.final_rules = [r for r in self.final_rules
                                if id(r) not in remove] + new_rules

    # Method to find the sibling of a rule in the tree
    def _find_sibling(self, rule, rules):
        """
        Finds the sibling rule of a given rule in the decision tree.

        Two rules are siblings if they share the same prefix (all conditions
        except the last) and their last conditions are complementary (same
        variable and threshold but opposite operators: <= vs >, or < vs >=).

        Args:
            rule (Rule): The rule whose sibling to find.
            rules (List[Rule]): The full list of rules to search.

        Returns:
            Rule|None: The sibling rule if found, otherwise None.
        """
        if not rule.parsed_conditions:
            return None

        # Build the prefix key (all conditions except the last)
        rule_prefix = tuple(rule.parsed_conditions[:-1])
        last_var, last_op, last_val = rule.parsed_conditions[-1]

        # Define complementary operators
        complement = {'<=': '>', '>': '<=', '<': '>=', '>=': '<'}
        expected_op = complement.get(last_op)
        if expected_op is None:
            return None

        for candidate in rules:
            if candidate is rule:
                continue
            if not candidate.parsed_conditions:
                continue

            # Check prefix match
            candidate_prefix = tuple(candidate.parsed_conditions[:-1])
            if candidate_prefix != rule_prefix:
                continue

            # Check last condition: same variable, same threshold, complementary op
            c_var, c_op, c_val = candidate.parsed_conditions[-1]
            if (c_var == last_var and c_op == expected_op
                    and (c_val == last_val or abs(c_val - last_val) < 1e-9)):
                return candidate

        return None

    # Method to promote sibling when removing specific rules
    def _promote_siblings(self, rules_to_remove, all_rules, accept=None,
                          fill_holes=None):
        """
        Removes rules and hands each removed region to what is left of its tree.

        Removing a leaf removes the split above it: the parent node is replaced
        by the leaf's other child, which now answers for the whole parent
        region. Concretely, every rule on the other side of that split drops
        the split's condition (the condition at the parent's depth):

        * **Sibling leaf** (promotion): the sibling drops its last condition
          and becomes the parent. Its class distribution absorbs the removed
          rule's, and its class is the majority of the sum.
        * **Sibling sub-branch** (grafting): every live leaf of the sub-branch
          drops the parent's condition, so the sub-branch is raised into the
          parent's place and its own splits route the removed region. Leaf
          outputs are unchanged (the removed rule's training distribution
          cannot be attributed to individual leaves without data).

        Either way the rules of each tree remain a partition: no region is
        left uncovered (no `default_class` fallback) and no two rules overlap.
        A rule that is the only rule of its tree (no conditions) is never
        removed, since that would leave the tree without an answer anywhere.

        Usage counts move with promotion (the sibling covers the union of the
        two regions). After grafting, the removed rule's samples spread over
        the raised leaves, which cannot be known without data: the analyzers
        recount usage and errors on the refinement set afterwards
        (`_refinement_evaluator(...).recount()`).

        **Gradient Boosting** (``fill_holes=False``, its default): a removed
        leaf whose sibling is a sub-branch is removed without grafting, and the
        tree adds 0.0 to the score on its region. In an additive model a tree
        that does not answer simply contributes nothing (it is not a fallback
        to `default_class`; the other trees and the init score still decide),
        whereas grafting would apply leaf values fitted to the residuals of a
        different region.

        **Critical**: Rules are processed from deepest to shallowest, so deeper
        removals and promotions happen first and may create the sibling a
        shallower removal needs.

        **Performance**: sibling leaves are found through a hash index keyed by
        (tree, prefix, variable, operator, threshold); only the sub-branch case
        scans the rules of one tree.

        Example:
            Given tree:
                          [x <= 5]
                         /        \\
                    [y <= 3]      D (usage: 200)
                    /      \\
                A (uso: 0)  [z <= 7]
                            /      \\
                         B (uso:0)  C (uso: 1)

            Rules to remove: A (depth 2), B (depth 3)

            Processing deepest first:
            1. Remove B (depth 3) -> promote sibling C: becomes "x<=5 AND y>3 -> C"
            2. Remove A (depth 2) -> now promoted C is A's sibling -> promote C: becomes "x<=5 -> C"

            Removing only A instead grafts the [z <= 7] sub-branch: B becomes
            "x<=5 AND z<=7" and C "x<=5 AND z>7", and A's region is routed by
            the z split instead of falling back to default_class.

        Args:
            rules_to_remove (List[Rule]): Rules identified for removal.
            all_rules (List[Rule]): The complete set of current rules.
            accept (Callable[[Rule, List[Rule], List[Rule]], bool], optional):
                Called before each removal with the rule to remove, the rules
                whose regions the removal changes (as they are now) and those
                rules as they would become; the removal happens only if it
                returns True. Used for error-aware refinement (see
                `_refinement_evaluator`). Default: every removal happens.
            fill_holes (bool, optional): Graft sub-branches (True) or leave the
                region uncovered (False). Default: True, except for Gradient
                Boosting.

        Returns:
            List[Rule]: The adjusted rule set (copies; the input rules are not
            modified).
        """
        complement = {'<=': '>', '>': '<=', '<': '>=', '>=': '<'}
        if fill_holes is None:
            fill_holes = self.algorithm_type != 'Gradient Boosting Decision Trees'

        # Siblings only exist *within* one tree. Ensemble trees routinely split
        # on the same feature at the same threshold, so a key that ignores the
        # tree would match a leaf of one tree as the sibling of a leaf of
        # another and promote it -- generalizing a rule over ground still
        # covered by its own tree's leaves (overlapping guards). Group by tree
        # id, using the same convention as find_overlapping_rules: a Decision
        # Tree is a single tree even though its rule names contain '_'.
        group_by_tree = self.algorithm_type != 'Decision Tree'

        # Method to _tree_id.
        def _tree_id(r):
            """Tree the rule belongs to (prefix before the first '_')."""
            if group_by_tree and '_' in r.name:
                return r.name.split('_')[0]
            return '_all'

        # Method to _make_key.
        def _make_key(r):
            """Index key: (tree_id, prefix, var, op, threshold_rounded)."""
            if not r.parsed_conditions:
                return None
            prefix = tuple(r.parsed_conditions[:-1])
            var, op, val = r.parsed_conditions[-1]
            return (_tree_id(r), prefix, var, op, round(val, 9))

        # Promotion rewrites a rule's conditions, class and name. `final_rules`
        # and `initial_rules` hold the *same* Rule objects, so mutating them in
        # place would silently rewrite the unrefined model too: its rules would
        # gain the generalized regions, start overlapping each other, and stop
        # reproducing the original tree. Clone first, so refinement only ever
        # touches its own copies.
        def _clone(r):
            c = copy.copy(r)
            c.conditions = list(r.conditions)
            c.parsed_conditions = list(r.parsed_conditions)
            dist = getattr(r, 'class_distribution', None)
            if dist is not None:
                c.class_distribution = list(dist)
            return c

        clones = {id(r): _clone(r) for r in all_rules}
        all_rules = [clones[id(r)] for r in all_rules]
        rules_to_remove = [clones[id(r)] for r in rules_to_remove if id(r) in clones]

        # --- Build sibling index: key -> list of rules with that key ---
        sibling_index: dict = {}
        rules_of_tree: dict = defaultdict(list)
        for r in all_rules:
            key = _make_key(r)
            if key is not None:
                sibling_index.setdefault(key, []).append(r)
            rules_of_tree[_tree_id(r)].append(r)

        # Method to _unindex.
        def _unindex(r):
            """Drop `r` from the sibling index (before its conditions change)."""
            key = _make_key(r)
            if key is not None and key in sibling_index:
                lst = [x for x in sibling_index[key] if x is not r]
                if lst:
                    sibling_index[key] = lst
                else:
                    del sibling_index[key]

        # Method to _reindex.
        def _reindex(r):
            """Insert `r` in the sibling index under its current conditions."""
            key = _make_key(r)
            if key is not None:
                sibling_index.setdefault(key, []).append(r)

        # Work on a mutable copy
        working_rules = list(all_rules)
        promoted_count = 0
        grafted_count = 0
        kept_count = 0
        rejected_count = 0

        # Sort rules to remove by depth (deepest first)
        sorted_to_remove = sorted(
            rules_to_remove,
            key=lambda r: len(r.parsed_conditions),
            reverse=True
        )

        # Track which rules have been removed by identity
        removed_ids: set = set()

        for rule in sorted_to_remove:
            if id(rule) in removed_ids:
                continue

            if not rule.parsed_conditions:
                # The rule is its whole tree: nothing could take over.
                kept_count += 1
                continue

            tid = _tree_id(rule)
            prefix = tuple(rule.parsed_conditions[:-1])
            depth = len(prefix)
            last_var, last_op, last_val = rule.parsed_conditions[-1]
            expected_op = complement.get(last_op)

            # --- O(1) sibling lookup via index ---
            sibling = None
            branch = []
            if expected_op is not None:
                # The sibling has the complementary operator, in the same tree
                sib_key = (tid, prefix, last_var, expected_op, round(last_val, 9))
                for c in sibling_index.get(sib_key, []):
                    if c is not rule and id(c) not in removed_ids:
                        sibling = c
                        break
                if sibling is None and fill_holes:
                    # The other child is a sub-branch: its live leaves extend
                    # the prefix with the complementary condition.
                    branch = [
                        r for r in rules_of_tree[tid]
                        if r is not rule and id(r) not in removed_ids
                        and len(r.parsed_conditions) > depth + 1
                        and tuple(r.parsed_conditions[:depth]) == prefix
                        and r.parsed_conditions[depth][0] == last_var
                        and r.parsed_conditions[depth][1] == expected_op
                        and (r.parsed_conditions[depth][2] == last_val
                             or abs(r.parsed_conditions[depth][2] - last_val) < 1e-9)
                    ]

            if sibling is None and not branch:
                if fill_holes:
                    # No other side to hand the region to (the rules are not
                    # the leaves of a tree here): keep the rule rather than
                    # open a hole.
                    kept_count += 1
                    continue
                # Gradient Boosting: the tree abstains (adds 0.0) on the
                # removed region.
                if accept is not None and not accept(rule, [rule], []):
                    rejected_count += 1
                    continue
                _unindex(rule)
                removed_ids.add(id(rule))
                continue

            # Proposal: the rules on the other side of the split, without the
            # split's condition.
            if sibling is not None:
                new = _clone(sibling)
                new.conditions = sibling.conditions[:-1]
                new.parsed_conditions = sibling.parsed_conditions[:-1]
                # Combine class_distribution: the promoted sibling now covers
                # the region of the removed rule as well, so add their counts
                if (getattr(rule, 'class_distribution', None) is not None
                        and getattr(sibling, 'class_distribution', None) is not None):
                    new.class_distribution = [
                        a + b for a, b in zip(sibling.class_distribution, rule.class_distribution)
                    ]
                    # Update class_ to match the new majority class (a label,
                    # not the index into the distribution)
                    dist = new.class_distribution
                    new.class_ = self._label_at(dist.index(max(dist)), len(dist))
                if "_promoted" not in new.name:
                    new.name = f"{new.name}_promoted"
                plan = [(sibling, new)]
            else:
                plan = []
                for r in branch:
                    new = _clone(r)
                    new.conditions = r.conditions[:depth] + r.conditions[depth + 1:]
                    new.parsed_conditions = (r.parsed_conditions[:depth]
                                             + r.parsed_conditions[depth + 1:])
                    if "_grafted" not in new.name:
                        new.name = f"{new.name}_grafted"
                    plan.append((r, new))

            if accept is not None and not accept(
                    rule, [rule] + [old for old, _ in plan], [new for _, new in plan]):
                rejected_count += 1
                continue

            for old, new in plan:
                _unindex(old)
                old.conditions = new.conditions
                old.parsed_conditions = new.parsed_conditions
                old.class_distribution = new.class_distribution
                old.class_ = new.class_
                old.name = new.name
                _reindex(old)

            if sibling is not None:
                sibling.usage_count += rule.usage_count
                sibling.error_count += rule.error_count
                promoted_count += 1
            else:
                grafted_count += 1

            # Remove the rule from the index; mark removed (deferred filtering)
            _unindex(rule)
            removed_ids.add(id(rule))

        # Single-pass filtering instead of per-iteration list rebuild
        working_rules = [r for r in working_rules if id(r) not in removed_ids]

        if promoted_count > 0:
            print(f"Promoted {promoted_count} sibling rule(s) after specific rule removal.")
        if grafted_count > 0:
            print(f"Grafted {grafted_count} sibling sub-branch(es) into the "
                  "place of their parent after specific rule removal.")
        if kept_count > 0:
            print(f"Kept {kept_count} low-usage rule(s) that no other rule of "
                  "their tree could take over.")
        if rejected_count > 0:
            print(f"Kept {rejected_count} low-usage rule(s) whose removal "
                  "would add errors on the refinement set.")
        if promoted_count or grafted_count:
            self._warn_promotions_that_overlap(working_rules, tree_of=_tree_id)

        return working_rules

    # Method to run the low-usage stage on the refinement set
    def _remove_low_usage(self, rules_to_remove, X, y, feature_names,
                          max_error_increase=None):
        """Remove `rules_to_remove` from `final_rules`, then recount usage.

        The low-usage stage of the analyzers. Each removal hands the rule's
        region to the rest of its tree (`_promote_siblings`); with
        `max_error_increase` set, a removal happens only while the
        refinement-set errors stay within that many samples of the errors
        before the stage (0: reduced-error pruning, errors never increase).
        Usage and error counts of the resulting rules are then recounted on
        the refinement set, since grafting spreads a removed region over
        several leaves.

        Args:
            rules_to_remove (List[Rule]): Candidates (rules of `final_rules`).
            X (np.ndarray): Refinement samples (from `_prepare_test_data`).
            y (array-like): True labels.
            feature_names (List[str]): Column names of `X`.
            max_error_increase (int, optional): Error budget; None (default)
                removes every candidate (usage-only criterion).

        Returns:
            List[Rule]: The refined rule set (also stored in `final_rules`).
        """
        accept = None
        if max_error_increase is not None:
            if max_error_increase < 0:
                raise ValueError("max_error_increase must be >= 0 or None.")
            evaluator = self._refinement_evaluator(
                X, y, feature_names, self.final_rules, max_error_increase)
            accept = evaluator.accept
        self.final_rules = self._promote_siblings(
            rules_to_remove, self.final_rules, accept=accept)
        if max_error_increase is not None:
            print(f"Refinement-set errors: {evaluator.base_errors} -> "
                  f"{evaluator.errors()} (budget +{max_error_increase}).")
        self._refinement_evaluator(
            X, y, feature_names, self.final_rules).recount(self.final_rules)
        return self.final_rules

    # Method to build an evaluator of rule-set changes on the refinement set
    def _refinement_evaluator(self, X, y, feature_names, rules, max_error_increase=0):
        """Predictions and errors of `rules` on a refinement set, kept up to date.

        Backs error-aware refinement (reduced-error pruning on the refinement
        set) and the recount of usage/error statistics after refinement. The
        semantics are those of the compiled engines: DT, the label of the
        matching rule (`default_class` when none matches); RF, the argmax of
        the sum of the matching leaves' normalised distributions
        (`default_class` when every tree abstains); GBDT, init score plus the
        matching contributions (binary: score >= 0).

        Each change touches one tree, and only on the samples that the changed
        rules cover, so a candidate is evaluated on those samples alone.

        Args:
            X (np.ndarray): Refinement samples, already prepared like the
                analyzers do (`_prepare_test_data`, which applies the input
                precision).
            y (array-like): True labels.
            feature_names (List[str]): Column names of `X`.
            rules (List[Rule]): Current rule set.
            max_error_increase (int): How many more refinement samples the
                accepted changes may misclassify, in total, than `rules` do.

        Returns:
            types.SimpleNamespace: ``accept(rule, old_rules, new_rules)``, the
            callback for `_promote_siblings` (commits the change when the
            error budget allows it); ``recount(rules)``, which sets
            `usage_count`/`error_count` of `rules` from the current state (call
            on the rules the state describes); ``errors()``, the current
            number of misclassified samples; ``base_errors``.
        """
        import types

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n = len(y)
        col = {f: i for i, f in enumerate(feature_names)}
        ops = {'<=': np.less_equal, '>': np.greater, '<': np.less, '>=': np.greater_equal}
        algo = self.algorithm_type
        is_gbdt = algo == 'Gradient Boosting Decision Trees'
        default = self._default_class_int()

        # Method to _lab.
        def _lab(v):
            """A rule label as the analyzers compare it with y."""
            try:
                return int(str(v).replace('Class', '').strip())
            except (ValueError, AttributeError):
                return v

        y = np.array([_lab(v) for v in y])

        # Method to _tree_id.
        def _tree_id(r):
            """Tree of a rule (a Decision Tree is one tree)."""
            if algo == 'Decision Tree':
                return '_all'
            return r.name.split('_')[0] if '_' in r.name else r.name

        # Method to _is_init.
        def _is_init(r):
            """GBDT init score: not a leaf (scored via `_gbdt_init_scores`)."""
            return is_gbdt and not r.parsed_conditions

        # Method to _mask.
        def _mask(r, rows):
            """Which of `rows` satisfy every condition of `r` (a NaN
            satisfies the side its split sends missing values to)."""
            m = np.ones(len(rows), dtype=bool)
            for cond in r.parsed_conditions:
                var, op, thr = cond
                values = X[rows, col[var]]
                hit = ops[op](values, thr)
                if getattr(cond, 'missing', False):
                    hit |= np.isnan(values)
                m &= hit
            return m

        if algo == 'Random Forest':
            k = next((len(r.class_distribution) for r in rules
                      if r.class_distribution is not None), self.num_classes or 0)
            rf_labels = np.asarray(self._rf_labels(k))
        elif is_gbdt:
            classes = list(self._gbdt_classes or [])
            k = len(classes)
            gcol = {c: i for i, c in enumerate(classes)}
            init = np.array([self._gbdt_init_scores.get(c, 0.0) for c in classes],
                            dtype=np.float64)
        else:
            k = 1

        # Method to _output.
        def _output(tree_rules, rows):
            """One tree's output on `rows`: (len(rows), k), zero = abstains."""
            out = np.zeros((len(rows), k), dtype=np.float64)
            label = np.full(len(rows), default, dtype=object) if algo == 'Decision Tree' else None
            for r in tree_rules:
                m = _mask(r, rows)
                if not m.any():
                    continue
                if algo == 'Decision Tree':
                    label[m] = _lab(r.class_)
                elif algo == 'Random Forest':
                    d = np.asarray(r.class_distribution if r.class_distribution is not None
                                   else [0.0] * k, dtype=np.float64)
                    if r.class_distribution is None:
                        # No distribution: a vote for the rule's class.
                        hit = np.flatnonzero(rf_labels == _lab(r.class_))
                        if len(hit):
                            d[hit[0]] = 1.0
                    s = d.sum()
                    out[m] = d / s if s > 0 else d
                else:
                    out[m, gcol[r.class_group]] = r.contribution or 0.0
            return label if algo == 'Decision Tree' else out

        # Method to _predict.
        def _predict(agg):
            """Labels from the aggregated outputs of every tree."""
            if algo == 'Decision Tree':
                return agg
            if algo == 'Random Forest':
                pred = rf_labels[np.argmax(agg, axis=1)].astype(object)
                pred[agg.sum(axis=1) == 0.0] = default
                return pred
            score = agg + init
            if self._gbdt_is_binary:
                return np.where(score[:, 1] >= 0, _lab(classes[1]),
                                _lab(classes[0])).astype(object)
            return np.array([_lab(classes[i]) for i in np.argmax(score, axis=1)],
                            dtype=object)

        all_rows = np.arange(n)
        trees: dict = defaultdict(list)
        for r in rules:
            if not _is_init(r):
                trees[_tree_id(r)].append(r)
        outs = {t: _output(rs, all_rows) for t, rs in trees.items()}
        if algo == 'Decision Tree':
            agg = outs.get('_all', np.full(n, default, dtype=object))
        else:
            agg = np.zeros((n, k), dtype=np.float64)
            for o in outs.values():
                agg += o
        state = {'agg': agg, 'pred': _predict(agg)}
        state['errors'] = int(np.sum(state['pred'] != y))
        base_errors = state['errors']
        limit = base_errors + max_error_increase

        # Method to accept.
        def accept(rule, old_rules, new_rules):
            """Commit the change if the error budget allows it."""
            covered = np.zeros(n, dtype=bool)
            for r in old_rules:
                covered[_mask(r, all_rows)] = True
            rows = np.flatnonzero(covered)
            if len(rows) == 0:
                return True
            tid = _tree_id(rule)
            new_out = _output(new_rules, rows)
            if algo == 'Decision Tree':
                new_agg = new_out
            else:
                new_agg = state['agg'][rows] - outs[tid][rows] + new_out
            new_pred = _predict(new_agg)
            delta = (int(np.sum(new_pred != y[rows]))
                     - int(np.sum(state['pred'][rows] != y[rows])))
            if state['errors'] + delta > limit:
                return False
            outs[tid][rows] = new_out
            state['agg'][rows] = new_agg
            state['pred'][rows] = new_pred
            state['errors'] += delta
            return True

        # Method to recount.
        def recount(target_rules):
            """Usage = samples matched; errors as the analyzers count them."""
            for r in target_rules:
                if _is_init(r):
                    continue
                m = _mask(r, all_rows)
                r.usage_count = int(m.sum())
                if is_gbdt:
                    r.error_count = int(np.sum(state['pred'][m] != y[m]))
                else:
                    r.error_count = int(np.sum(y[m] != _lab(r.class_)))

        return types.SimpleNamespace(
            accept=accept, recount=recount,
            errors=lambda: state['errors'], base_errors=base_errors)

    # Method to report promotions that swallowed still-live rules
    @staticmethod
    def _warn_promotions_that_overlap(rules, tree_of=None) -> int:
        """
        Report promoted rules whose region now contains other live rules.

        Promotion drops a rule's last condition so it covers the region of the
        removed sibling. When that sibling was a whole sub-branch with leaves
        still in the rule set, the promoted rule spreads over them and the rule
        set stops being a partition: two rules match the same sample. Accuracy
        does not reveal this (the iterative engine just takes the first match),
        but the CPN model becomes nondeterministic and the set is no longer a
        decision tree.

        Detection is a prefix test: rules from a tree are prefix-structured, so
        a rule lies inside a promoted rule's region exactly when its conditions
        extend the promoted rule's conditions. Sorting makes those a contiguous
        range, giving O(n log n) instead of a quadratic comparison.

        Only rules of the same tree can overlap in this sense: the trees of an
        ensemble all cover the whole space and often split on the same
        thresholds, so comparing across trees would flag every promotion whose
        conditions happen to prefix a path of another tree.

        Args:
            rules (List[Rule]): The rule set after promotion.
            tree_of (Callable[[Rule], str], optional): Tree a rule belongs to.
                Defaults to one tree for all rules.

        Returns:
            int: How many promoted rules swallowed at least one live rule.
        """
        import bisect

        promoted = [r for r in rules
                    if r.name.endswith('_promoted') or r.name.endswith('_grafted')]
        if not promoted:
            return 0

        if tree_of is None:
            def tree_of(_r):
                return '_all'
        keys_by_tree: dict = defaultdict(list)
        for r in rules:
            keys_by_tree[tree_of(r)].append(tuple(r.parsed_conditions))
        for keys in keys_by_tree.values():
            keys.sort()
        affected = 0
        for r in promoted:
            keys = keys_by_tree[tree_of(r)]
            p = tuple(r.parsed_conditions)
            i = bisect.bisect_right(keys, p)
            if i < len(keys) and keys[i][:len(p)] == p:
                affected += 1

        if affected:
            # Plain ASCII: this runs with stdout redirected to files whose
            # encoding is the system codepage (cp1252 on Windows).
            print(
                f'  WARNING: {affected} promoted rule(s) now overlap rules '
                'still in the set: the refined rule set is no longer a '
                'partition. Predictions become order-dependent and the CPN '
                'model becomes nondeterministic. Inspect with '
                'find_overlapping_rules().'
            )
        return affected

    # Exports the rule set to a standalone native Python classifier file
    def export_to_native_python(self, feature_names=None, filename="files/fast_classifier.py",
                               rules=None):
        """
        Generates a standalone Python file with the decision logic.
        
        For Decision Trees, it exports a single nested if/else function.
        For Random Forests, it exports multiple functions (one per tree) and a voting aggregator.

        Args:
            feature_names (List[str], optional): Kept for compatibility.
            filename (str): Output filename.
            rules (List[Rule], optional): Rules to export. Defaults to
                final_rules if available, else initial_rules.
        """
        print(f"[EXPORT] Generating native classifier: {filename}")

        rules_to_export = rules if rules is not None else (
            self.final_rules if self.final_rules else self.initial_rules)
        
        with open(filename, "w") as f:
            
            # --- GBDT Export Strategy (Additive Scoring) ---
            if self.algorithm_type == 'Gradient Boosting Decision Trees':
                f.write("import math\n\n")

                tree_rules_map = defaultdict(list)
                for rule in rules_to_export:
                    if '_' in rule.name:
                        tid = rule.name.split('_')[0]
                    else:
                        tid = rule.name
                    tree_rules_map[tid].append(rule)

                # Identifiers come from positions, never from rule names or
                # class labels (see update_native_model).
                fn_of = {tid: f'predict_{i}' for i, tid in enumerate(tree_rules_map)}

                # Collect init scores and write tree functions
                init_scores = {}
                for tid, rules in tree_rules_map.items():
                    if not rules:
                        continue
                    if len(rules) == 1 and not rules[0].parsed_conditions and rules[0].contribution is not None:
                        init_scores[rules[0].class_group] = rules[0].contribution
                        continue

                    func_name = fn_of[tid]
                    tree_dict = self._tree_from_rules(rules)
                    for node in tree_dict.values():
                        rule = node.pop('rule', None)
                        # Uncovered region: the stage abstains (adds 0.0).
                        node['v'] = (rule.contribution if rule is not None
                                     and rule.contribution is not None else 0.0)

                    # Method to _build_export_code.
                    def _build_export_code(node_id, indent, td=tree_dict):
                        """
                        _build_export_code method for RuleClassifier.
                        Args:
                            node_id: Description of node_id.
                            indent: Description of indent.
                            td: Description of td.
                        Returns:
                            Description of return value.
                        """
                        node = td[node_id]
                        tab = '    ' * indent
                        if node['f'] == -2:
                            return f'{tab}return {repr(float(node["v"]))}\n'
                        feat = node['f']
                        # GradientBoostingClassifier refuses NaN: no routing emitted.
                        code = f"{tab}if {_py_split_test(feat, node['t'], None)}:\n"
                        code += _build_export_code(node['l'], indent + 1, td)
                        code += f'{tab}else:\n'
                        code += _build_export_code(node['r'], indent + 1, td)
                        return code

                    f.write(f'def {func_name}(sample):\n')
                    f.write(_build_export_code(0, 1))
                    f.write('\n')

                # Main predict function
                is_binary = self._gbdt_is_binary
                classes = self._gbdt_classes or []

                f.write('def predict(sample):\n')

                if is_binary and len(classes) >= 2:
                    pos_class = classes[1]
                    score_init = init_scores.get(pos_class, 0.0)
                    f.write(f'    score = {repr(score_init)}\n')
                    for tid, rules in tree_rules_map.items():
                        if not rules:
                            continue
                        if len(rules) == 1 and not rules[0].parsed_conditions:
                            continue
                        if rules[0].class_group != pos_class:
                            continue
                        f.write(f'    score += {fn_of[tid]}(sample)\n')
                    # Decide on the raw score, as scikit-learn does (raw >= 0).
                    f.write(f'    return {int(classes[1])} if score >= 0.0 else {int(classes[0])}\n')
                else:
                    col = {cl: i for i, cl in enumerate(classes)}
                    for class_label in classes:
                        score_init = init_scores.get(class_label, 0.0)
                        f.write(f'    score_{col[class_label]} = {repr(score_init)}\n')
                    for tid, rules in tree_rules_map.items():
                        if not rules:
                            continue
                        if len(rules) == 1 and not rules[0].parsed_conditions:
                            continue
                        if rules[0].class_group not in col:
                            continue
                        f.write(f'    score_{col[rules[0].class_group]} += '
                                f'{fn_of[tid]}(sample)\n')
                    score_vars = [f'score_{col[cl]}' for cl in classes]
                    class_ints = [int(cl) for cl in classes]
                    f.write(f'    scores = [{", ".join(score_vars)}]\n')
                    f.write(f'    classes = {class_ints}\n')
                    f.write('    return classes[scores.index(max(scores))]\n')

            # --- Random Forest Export Strategy (Soft Voting) ---
            elif self.algorithm_type == 'Random Forest':
                tree_rules_map = defaultdict(list)
                for rule in rules_to_export:
                    if "_" in rule.name:
                        tid = rule.name.split('_')[0]
                    else:
                        tid = "Tree0" 
                    tree_rules_map[tid].append(rule)
                
                # Determine number of classes from distributions
                n_classes = 0
                for rule in rules_to_export:
                    if rule.class_distribution is not None:
                        n_classes = len(rule.class_distribution)
                        break
                
                # Generate a function for each tree
                tree_func_names = []
                for t_pos, (tid, rules) in enumerate(tree_rules_map.items()):
                    # Named by position: rule names are not valid identifiers
                    func_name = f"predict_{t_pos}"
                    tree_func_names.append(func_name)
                    
                    # Build tree structure — store probability distributions at leaves
                    tree_dict = self._tree_from_rules(rules)

                    for curr, node in tree_dict.items():
                        rule = node.pop('rule', None)
                        node['v'] = None
                        if rule is None:
                            continue  # uncovered region: the tree abstains

                        if rule.class_distribution is not None:
                            raw = rule.class_distribution
                            total_c = sum(raw)
                            if total_c > 0:
                                tree_dict[curr]['v'] = [c / total_c for c in raw]
                            else:
                                tree_dict[curr]['v'] = list(raw)
                        else:
                            try:
                                clean_class = int(str(rule.class_).replace('Class', '').strip())
                            except Exception:
                                clean_class = rule.class_
                            tree_dict[curr]['v'] = clean_class

                    # Build code string for this tree
                    def build_rf_code(node_id, indent):
                        """
                        build_rf_code method for RuleClassifier.
                        Args:
                            node_id: Description of node_id.
                            indent: Description of indent.
                        Returns:
                            Description of return value.
                        """
                        node = tree_dict[node_id]
                        tab = "    " * indent
                        if node['f'] == -2:
                            val = node['v']
                            if val is None:
                                return f"{tab}return None\n" 
                            return f"{tab}return {repr(val)}\n"
                        
                        feat = node['f']
                        # Use repr(float(t)) for precision
                        code = f"{tab}if {_py_split_test(feat, node['t'], node.get('m'))}:\n"
                        code += build_rf_code(node['l'], indent + 1)
                        code += f"{tab}else:\n"
                        code += build_rf_code(node['r'], indent + 1)
                        return code

                    f.write(f"def {func_name}(sample):\n")
                    f.write(build_rf_code(0, 1))
                    f.write("\n")

                # Generate Main Prediction Function — soft voting
                try:
                    default_val = int(str(self.default_class).replace('Class', '').strip())
                except Exception:
                    default_val = self.default_class
                
                def_str = repr(default_val)

                f.write("def predict(sample):\n")
                f.write("    tree_results = []\n")
                for func in tree_func_names:
                    f.write(f"    v = {func}(sample)\n")
                    f.write("    if v is not None: tree_results.append(v)\n")
                f.write(f"    if not tree_results: return {def_str}\n")

                if n_classes > 0:
                    # The argmax is a column of the distribution; answer its label
                    f.write(f"    labels = {self._rf_labels(n_classes)!r}\n")
                    f.write("    # Soft voting: average probability distributions\n")
                    f.write("    if isinstance(tree_results[0], list):\n")
                    f.write(f"        n_classes = {n_classes}\n")
                    f.write("        avg = [0.0] * n_classes\n")
                    f.write("        for proba in tree_results:\n")
                    f.write("            for j in range(n_classes):\n")
                    f.write("                avg[j] += proba[j]\n")
                    f.write("        n = len(tree_results)\n")
                    f.write("        avg = [p / n for p in avg]\n")
                    f.write("        return labels[avg.index(max(avg))]\n")
                    f.write("    else:\n")
                    f.write("        from collections import Counter\n")
                    f.write("        return Counter(tree_results).most_common(1)[0][0]\n")
                else:
                    f.write("    from collections import Counter\n")
                    f.write("    return Counter(tree_results).most_common(1)[0][0]\n")

            # --- Decision Tree Export Strategy (Single Tree) ---
            else:
                tree_dict = self._tree_from_rules(rules_to_export)
                for node in tree_dict.values():
                    rule = node.pop('rule', None)
                    if rule is None:
                        # Uncovered region -> default_class. None, not -1: -1
                        # is a legitimate label (y in {-1, 1}).
                        node['v'] = None
                        continue
                    try:
                        node['v'] = int(str(rule.class_).replace('Class', '').strip())
                    except Exception:
                        node['v'] = rule.class_

                # Method to build_dt_code.
                def build_dt_code(node_id, indent):
                    """
                    build_dt_code method for RuleClassifier.
                    Args:
                        node_id: Description of node_id.
                        indent: Description of indent.
                    Returns:
                        Description of return value.
                    """
                    node = tree_dict[node_id]
                    tab = "    " * indent
                    if node['f'] == -2:
                        val = node['v']
                        if val is None:
                            try:
                                val = int(str(self.default_class).replace('Class', '').strip())
                            except Exception:
                                val = self.default_class
                        return f"{tab}return {val!r}\n"
                    
                    feat = node['f']
                    # Use repr(float(t)) for precision
                    code = f"{tab}if {_py_split_test(feat, node['t'], node.get('m'))}:\n"
                    code += build_dt_code(node['l'], indent + 1)
                    code += f"{tab}else:\n"
                    code += build_dt_code(node['r'], indent + 1)
                    return code

                f.write("def predict(sample):\n")
                f.write(build_dt_code(0, 1))

        f32 = getattr(self, 'input_dtype', None) == 'float32'
        table = getattr(self, 'label_table', None)
        if f32 or table is not None:
            # scikit-learn casts X to float32 before comparing it with the
            # thresholds; the standalone file must too (without numpy). And
            # the trees answer label codes; a table names the original labels.
            with open(filename) as f:  # same encoding it was written with
                code = f.read()
            assert code.count("def predict(sample):\n") == 1
            code = code.replace("def predict(sample):\n", "def _predict(sample):\n")
            call = "_predict(sample)"
            if f32:
                code += (
                    "\n\nimport struct as _struct\n\n\n"
                    "def _f32(v):\n"
                    "    \"\"\"Round v to float32, as scikit-learn does before comparing.\"\"\"\n"
                    "    try:\n"
                    "        return _struct.unpack('f', _struct.pack('f', v))[0]\n"
                    "    except (OverflowError, TypeError, _struct.error):\n"
                    "        return v\n"
                )
                call = "_predict({k: _f32(v) for k, v in sample.items()})"
            if table is not None:
                # ascii(): the file is written in the platform encoding.
                code += (
                    "\n\n# Original class label of each code the trees answer.\n"
                    f"_LABELS = {ascii(list(table))}\n"
                )
                call = f"_LABELS[{call}]"
            code += f"\n\ndef predict(sample):\n    return {call}\n"
            with open(filename, "w") as f:
                f.write(code)

        print(f"[EXPORT] File '{filename}' generated.")
        
    # Method to export to multiple formats at once
    def export_all(self, base_name="model", feature_names=None, export_python=True, export_binary=True, export_c=False):
        """
        Exports the classifier to multiple formats in one call.
        
        This convenience method allows selective export to different formats:
        - Python (.py): Standalone classifier without dependencies
        - Binary (.bin): Compact format for fast loading
        - C Header (.h): For embedded systems integration
        
        Args:
            base_name (str): Base filename without extension. Default is 'model'.
            feature_names (List[str], optional): Feature names for export. Default is None.
            export_python (bool): Whether to export to Python format. Default is True.
            export_binary (bool): Whether to export to binary format. Default is True.
            export_c (bool): Whether to export to C header format. Default is False.
            
        Example:
            # Export only to binary (no Python or C)
            classifier.export_all("my_model", export_python=False, export_c=False)
            
            # Export to Python and C only
            classifier.export_all("my_model", export_binary=False, export_c=True)
        """
        print("\n" + "=" * 60)
        print("EXPORTING CLASSIFIER")
        print("=" * 60)
        
        # Determine full base path
        if '/' not in base_name and '\\' not in base_name:
            base_path = f"files/{base_name}"
        else:
            base_path = base_name
        
        # Export to Python
        if export_python:
            py_filename = f"{base_path}.py"
            self.export_to_native_python(feature_names=feature_names, filename=py_filename)
            print(f"  [OK] Python export: {py_filename}")
        
        # Export to Binary
        if export_binary:
            bin_filename = f"{base_path}.bin"
            self.export_to_binary(filepath=bin_filename)
            print(f"  [OK] Binary export: {bin_filename}")
        
        # Export to C Header
        if export_c:
            c_filename = f"{base_path}.h"
            self.export_to_c_header(filepath=c_filename)
            print(f"  [OK] C Header export: {c_filename}")
        
        print("=" * 60 + "\n")

    # Helper to save the final model state
    def _save_final_model(self, path=None):
        """Helper to save the current state of the classifier.
        
        Args:
            path (str, optional): Path to save the model. Defaults to 'files/final_model.pkl'.
        """
        if path is None:
            path = 'files/final_model.pkl'
        try:
            with open(path, 'wb') as model_file:
                pickle.dump(self, model_file)
            print(f"Final classifier saved to '{path}'.")
        except Exception as e:
            print(f"Error saving model: {e}")

    # Helper to write the analysis report
    def _write_report(self, filename, file_path, correct, total, prune_thresh):
        """Helper to write the consolidated text report.
        
        Args:
            filename (str): Path to the output report file.
            file_path (str): Path to the dataset used.
            correct (int): Number of correct predictions.
            total (int): Total number of samples.
            prune_thresh (int): Pruning threshold used.
        """
        with open(filename, 'w') as f:
            f.write(f"Results for dataset: {file_path}\n")
            f.write(f"Accuracy: {correct/total:.5f}\n")
            f.write(f"Total Final Rules: {len(self.final_rules)}\n")
            
            if prune_thresh > -1:
                f.write(f"Rules Pruned (<= {prune_thresh}): {len(self.specific_rules)}\n")
            
            f.write("\n--- Top Active Rules ---\n")
            # Sort by usage
            active_rules = sorted(
                [r for r in self.final_rules if r.usage_count > 0],
                key=lambda r: r.usage_count, 
                reverse=True
            )
            
            for rule in active_rules:
                err_rate = (rule.error_count / rule.usage_count) * 100
                f.write(f"{rule.name}: Used {rule.usage_count}, Errors {rule.error_count} ({err_rate:.2f}%)\n")

    # Method to calculate structural complexity
    @staticmethod
    def calculate_structural_complexity(rules: List[Rule], n_features_total: int) -> Dict[str, Any]:
        """
        Computes the Structural Complexity Score (SCS) and other interpretability metrics for a rule set.

        This method calculates a comprehensive set of metrics, including:
        - A novel 'Structural Complexity Score' based on depth and attribute usage.
        - Traditional metrics like rule counts and depth statistics.

        The Structural Complexity Score (SCS) measures the overall cognitive load of the model
        as an absolute volume index:

        SCS = sum( d_i * (U_i / |F|) )

        Where:
        - d_i: depth of rule i
        - U_i: number of unique features used in rule i
        - |F|: total number of available features (n_features_total)

        Unlike proportion-based metrics, this additive nature ensures that any rule refinement
        or pruning results in an immediate and mathematically evident reduction in the score.
        A score closer to zero indicates a highly concise and interpretable model.

        Args:
            rules (List[Rule]): A list of Rule objects to analyze.
            n_features_total (int): The total number of available features in the dataset.

        Returns:
            Dict[str, Any]: A dictionary containing detailed complexity metrics.
        """
        if not rules or n_features_total <= 0:
            return {
                "total_rules": 0,
                "features_used": 0,
                "total_features": n_features_total,
                "max_depth": 0,
                "mean_rule_depth": 0.0,
                "complexity_score": 0.0,
            }

        total_rules = len(rules)
        rule_depths = []
        all_features_used = set()
        
        structural_complexity_score = 0.0

        for rule in rules:
            # 1. Depth Calculation (d_i)
            depth = len(rule.conditions)
            rule_depths.append(depth)

            # 2. Attribute Usage (U_i) using cached parsed_conditions
            # item format: (var, op, value)
            unique_features_in_rule = {item[0] for item in rule.parsed_conditions}
            all_features_used.update(unique_features_in_rule)
            u_i = len(unique_features_in_rule)

            # 3. Additive SCS Calculation
            structural_complexity_score += depth * (u_i / n_features_total)

        # --- Metrics Calculation ---
        
        # Depth Statistics
        max_depth = max(rule_depths) if rule_depths else 0
        mean_rule_depth = sum(rule_depths) / total_rules if total_rules > 0 else 0.0
        n_features_used = len(all_features_used)

        result = {
            "total_rules": total_rules,
            "features_used": n_features_used,
            "total_features": n_features_total,
            "max_depth": max_depth,
            "mean_rule_depth": float(mean_rule_depth),
            "complexity_score": float(structural_complexity_score),
        }

        return result

    # Method to display classification metrics
    @staticmethod
    def display_metrics(y_true, y_pred, correct, total, file=None, class_names=None):
        """
        Computes and displays classification performance metrics.

        Calculates accuracy, precision, recall, F1 score, specificity, and displays
        the confusion matrix (using class names if available).

        Args:
            y_true (List[int]): True class labels.
            y_pred (List[int]): Predicted class labels.
            correct (int): Count of correct predictions.
            total (int): Total count of predictions.
            file (Optional[TextIO]): Output file object.
            class_names (Optional[List[str]]): List of class names for display.
        """
        # Handle None predictions (fallback to -1)
        y_pred_safe = [p if p is not None else -1 for p in y_pred]

        # 1. Calculate Standard Metrics
        accuracy = correct / total if total > 0 else 0.0
        precision = precision_score(y_true, y_pred_safe, average='macro', zero_division=0.0)
        recall = recall_score(y_true, y_pred_safe, average='macro', zero_division=0.0)
        f1 = f1_score(y_true, y_pred_safe, average='macro', zero_division=0.0)

        # 2. Confusion Matrix & Specificity
        # Get active labels (excluding placeholder -1)
        unique_labels = set(y_true) | set(y for y in y_pred_safe if y != -1)
        labels = sorted(list(unique_labels))
        
        cm = confusion_matrix(y_true, y_pred_safe, labels=labels)

        # Compute Specificity per class: TN / (TN + FP)
        specificities = []
        for i in range(len(labels)):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fn - fp
            
            denominator = tn + fp
            spec = tn / denominator if denominator > 0 else 0.0
            specificities.append(spec)
        
        specificity = np.mean(specificities) if specificities else 0.0

        # 3. Prepare Output Strings (Consolidated for Print and File)
        header = f'\nCorrect: {correct}, Errors: {total - correct}, Total: {total}'
        metrics_str = (
            f'Accuracy: {accuracy:.5f}\n'
            f'Precision (macro): {precision:.5f}\n'
            f'Recall (macro): {recall:.5f}\n'
            f'F1 Score (macro): {f1:.5f}\n'
            f'Specificity (macro): {specificity:.5f}'
        )
        
        # 4. Handle Confusion Matrix Display
        cm_output = "\nConfusion Matrix with Labels:\n"
        
        # Try to map integers to class names for the DataFrame
        display_labels = labels
        mapped_successfully = False
        
        if class_names is not None:
            try:
                # Map only if labels are valid indices
                mapped_names = [str(class_names[int(i)]) for i in labels if 0 <= int(i) < len(class_names)]
                if len(mapped_names) == len(labels):
                    display_labels = mapped_names
                    mapped_successfully = True
            except (ValueError, IndexError):
                pass # Fallback to integer labels
        
        if mapped_successfully:
            # Pandas DF string representation
            cm_df = pd.DataFrame(cm, index=display_labels, columns=display_labels)
            cm_body = str(cm_df)
        else:
            # Raw matrix representation
            cm_body = f"Labels: {display_labels}\n{cm}"

        full_report = f"{header}\n{metrics_str}\n{cm_output}{cm_body}\n"

        # 5. Output
        print(full_report)
        if file:
            file.write(full_report)

    # Method to compare initial and final results
    def compare_initial_final_results(self, file_path: str = None, X = None, y = None):
        """
        Delegates to DTAnalyzer, RFAnalyzer, or GBDTAnalyzer.  If ``execute_rule_refinement``
        was called, uses the cached analyzer. Otherwise creates one.

        Args:
            file_path: Path to the CSV test file.
            X: Dataframe or array for test data.
            y: True labels.
        """
        from .dt_analyzer import DTAnalyzer
        from .rf_analyzer import RFAnalyzer
        from .gbdt_analyzer import GBDTAnalyzer

        # Reuse the analyzer created during execute_rule_refinement when possible
        if hasattr(self, '_analyzer') and self._analyzer is not None:
            self._analyzer.compare_initial_final_results(file_path, X, y)
        elif self.algorithm_type == 'Random Forest':
            RFAnalyzer(self).compare_initial_final_results(file_path, X, y)
        elif self.algorithm_type == 'Decision Tree':
            DTAnalyzer(self).compare_initial_final_results(file_path, X, y)
        elif self.algorithm_type == 'Gradient Boosting Decision Trees':
            GBDTAnalyzer(self).compare_initial_final_results(file_path, X, y)
        else:
            raise ValueError(f"Unsupported algorithm type: {self.algorithm_type}")

    # Method to compare initial and final results for Decision Tree
    def compare_initial_final_results_dt(self, df_test=None, target_column_name=None, file_path=None):
        """Delegates to DTAnalyzer. Kept for backward compatibility.

        Args:
            df_test: Ignored (kept for signature compatibility).
            target_column_name: Ignored (kept for signature compatibility).
            file_path (str, optional): Path to the CSV test file.
        """
        from .dt_analyzer import DTAnalyzer
        # If called via the old dispatcher with df_test, we still need a file_path.
        # The new DTAnalyzer.compare_initial_final_results loads data from the path.
        if file_path is None:
            raise ValueError(
                "compare_initial_final_results_dt now requires 'file_path'. "
                "Use compare_initial_final_results(file_path) instead."
            )
        DTAnalyzer(self).compare_initial_final_results(file_path)

    # Method to compare initial and final results for Random Forest
    def compare_initial_final_results_rf(self, df_test=None, target_column_name=None, file_path=None):
        """Delegates to RFAnalyzer. Kept for backward compatibility.

        Args:
            df_test: Ignored (kept for signature compatibility).
            target_column_name: Ignored (kept for signature compatibility).
            file_path (str, optional): Path to the CSV test file.
        """
        from .rf_analyzer import RFAnalyzer
        if file_path is None:
            raise ValueError(
                "compare_initial_final_results_rf now requires 'file_path'. "
                "Use compare_initial_final_results(file_path) instead."
            )
        RFAnalyzer(self).compare_initial_final_results(file_path)

    # Method to edit rules manually
    def edit_rules(self):
        """
        Starts an interactive prompt in the terminal to allow manual editing of rules.
        
        The user can list, select, and modify the conditions or the class of a rule.
        CRITICAL: Updates the native compiled model immediately upon saving changes.
        """
        print("\n" + "*"*80)
        print("MANUAL RULE EDITING MODE")
        print("*"*80 + "\n")
        
        # Always work on the list of final rules
        target_rules = self.final_rules if self.final_rules else self.initial_rules
        
        if not target_rules:
            print("No rules available to edit. Please run the analysis first.")
            return

        while True:
            print(f"\n--- Current Rules ({len(target_rules)} rules) ---")
            # Show a subset if too many rules, or user pagination logic could go here
            # For now, we list them all but standard output buffering handles it
            for i, rule in enumerate(target_rules):
                # Display first 3 conditions only for brevity in list view
                cond_preview = rule.conditions[:3] + ["..."] if len(rule.conditions) > 3 else rule.conditions
                print(f"  [{i+1}] {rule.name}: Class={rule.class_}, Conditions={cond_preview}")

            print("\nEnter the NUMBER or NAME of the rule you want to edit (or 'exit' to finish):")
            user_input = input("> ").strip()

            if user_input.lower() == 'exit':
                break

            selected_rule = None
            
            # 1. Try to interpret as index number
            try:
                rule_index = int(user_input) - 1
                if 0 <= rule_index < len(target_rules):
                    selected_rule = target_rules[rule_index]
                else:
                    print(f"ERROR: Number '{user_input}' is out of range.")
                    continue
            except ValueError:
                # 2. Try to match by rule name (case-insensitive)
                for rule in target_rules:
                    if rule.name.lower() == user_input.lower():
                        selected_rule = rule
                        break
                
                if not selected_rule:
                    print(f"ERROR: No rule found with name '{user_input}'.")
                    continue

            # --- Editing Loop for Selected Rule ---
            while True:
                print(f"\n--- Editing Rule: {selected_rule.name} ---")
                print(f"  Current Class: {selected_rule.class_}")
                print("  Current Conditions:")
                for i, cond in enumerate(selected_rule.conditions):
                    print(f"    [{i}] {cond}")
                
                print("\nEdit Options:")
                print("  [a]dd condition")
                print("  [r]emove condition")
                print("  [c]lass (change prediction)")
                print("  [s]ave and return to main menu")
                
                action = input("Choose an action > ").strip().lower()

                if action == 'a':
                    print("Format: 'variable operator value' (e.g., 'dur <= 0.05')")
                    new_cond_str = input("Enter new condition: ").strip()
                    
                    validation = self._validate_and_parse_condition(new_cond_str)
                    if validation:
                        formatted_cond, parsed_tuple = validation
                        # Update both text representation and parsed cache
                        selected_rule.conditions.append(formatted_cond)
                        selected_rule.parsed_conditions.append(parsed_tuple)
                        print(f"Condition '{formatted_cond}' added.")
                    else:
                        print("ERROR: Invalid format or value. Ensure value is numeric.")
                
                elif action == 'r':
                    if not selected_rule.conditions:
                        print("Rule has no conditions to remove.")
                        continue
                    try:
                        idx = int(input("Enter index to remove: "))
                        if 0 <= idx < len(selected_rule.conditions):
                            removed = selected_rule.conditions.pop(idx)
                            # Keep parsed_conditions in sync
                            if len(selected_rule.parsed_conditions) > idx:
                                selected_rule.parsed_conditions.pop(idx)
                            print(f"Condition '{removed}' removed.")
                        else:
                            print("ERROR: Index out of range.")
                    except ValueError:
                        print("ERROR: Invalid index.")

                elif action == 'c':
                    new_class = input("Enter new class (e.g., '0' or 'Attack'): ").strip()
                    # Try to keep type consistency
                    try:
                        # If existing labels are ints, try to cast input to int
                        if self.class_labels and isinstance(self.class_labels[0], int):
                            int(new_class) # check if possible
                    except ValueError:
                        pass # Keep as string if conversion fails
                    
                    selected_rule.class_ = new_class
                    print(f"Rule class changed to '{new_class}'.")

                elif action == 's':
                    # Mark as edited
                    if "_edited" not in selected_rule.name:
                        selected_rule.name += "_edited"
                        
                    # Re-validate structure just in case. Conditions kept
                    # from the tree keep their missing-value routing; an
                    # added one has none (NaN is then refused).
                    known = {tuple(c): c for c in selected_rule.parsed_conditions}
                    selected_rule.parsed_conditions = [
                        known.get(tuple(c), c)
                        for c in self.parse_conditions_static(selected_rule.conditions)]
                    self._route_missing_cache = None
                    self._used_features_cache = None
                    
                    self.update_native_model(self.final_rules if self.final_rules else self.initial_rules)
                    
                    # Save to disk
                    try:
                        with open('files/edited_model.pkl', 'wb') as f_out:
                            pickle.dump(self, f_out)
                        print("Changes saved. Native model recompiled.")
                    except Exception as e:
                        print(f"Error saving model: {e}")
                        
                    break # Exit editing loop
                
                else:
                    print("Invalid option.")

        print("\n************************************** END OF EDITING MODE **************************************\n")

    # Helper to validate and parse condition strings
    def _validate_and_parse_condition(self, condition_str):
        """
        Validates user input for a condition and returns parsed data.

        Args:
            condition_str (str): Input string (e.g., "v1 <= 0.5").

        Returns:
            Tuple[str, Tuple]|None: (Formatted String, Parsed Tuple) if valid, else None.
        """
        parts = condition_str.split()
        if len(parts) != 3:
            return None
        
        var, op, val_str = parts
        
        # Validate operator
        if op not in ['<', '>', '<=', '>=', '==']:
            return None
            
        # Validate numeric value
        try:
            val_float = float(val_str)
            # Return standardized format
            return (f"{var} {op} {val_float}", (var, op, val_float))
        except ValueError:
            return None

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

    # Method to apply the input precision to test rows
    @staticmethod
    def _quantize_rows(X_test, clf):
        """Apply `clf`'s input precision to the refinement's test rows.

        The refinement counts rule usage with the iterative engine; without
        this, a sample within one float32 ULP of a threshold would be counted
        on one side while every prediction engine routes it to the other.

        Args:
            X_test (list | np.ndarray): Test rows.
            clf (RuleClassifier | None): The classifier.

        Returns:
            The rows, float32-rounded when ``clf.input_dtype == 'float32'``.
        """
        if clf is None or getattr(clf, 'input_dtype', None) != 'float32':
            return X_test
        try:
            return np.asarray(X_test, dtype=np.float32).astype(np.float64).tolist()
        except (TypeError, ValueError):
            return X_test

    # Method to prepare test data
    @staticmethod
    def _prepare_test_data(file_path=None, X=None, y=None, clf=None):
        """
        Helper method to prepare test data either from a CSV file or direct arrays/DataFrames.
        Returns:
            X_test (list or array), y_test (list or array), feature_names (list)
        """
        import pandas as pd
        import numpy as np

        if X is not None and y is not None:
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
                X_test = X.values.tolist()
            else:
                feature_names = clf._array_feature_names if clf and hasattr(clf, '_array_feature_names') else [f"Feature_{i}" for i in range(len(X[0]))]
                X_test = X.tolist() if hasattr(X, 'tolist') else list(X)
            y_test = y.tolist() if hasattr(y, 'tolist') else list(y)
            if clf is not None:
                # The rules answer label codes: compare y as codes too.
                if getattr(clf, 'label_table', None) is not None:
                    y_test = clf._encode_labels(y_test).tolist()
                # NaN only where every engine can answer it like the model.
                try:
                    arr = np.asarray(X_test, dtype=np.float64)
                except (TypeError, ValueError):
                    arr = None
                if arr is not None:
                    rules = clf.final_rules or clf.initial_rules
                    clf._check_missing_values(arr, clf._rules_route_missing(rules))
            return RuleClassifier._quantize_rows(X_test, clf), y_test, feature_names

        if file_path:
            _, _, X_test, y_test, _, _, feature_names = RuleClassifier.process_data(
                '.', file_path, is_test_only=True
            )
            return RuleClassifier._quantize_rows(X_test, clf), y_test, feature_names
            
        raise ValueError("Must provide either 'file_path' or both 'X' and 'y'.")

    # Method to execute process_data
    @staticmethod
    def process_data(train_path, test_path, is_test_only=False):
        """
        Loads and preprocesses training and testing datasets from CSV files.

        Handles header detection automatically. If no header is found, columns are named
        v1, v2, ..., class. Performs Label Encoding on categorical features to ensure
        all data is numeric for the classifier.

        Args:
            train_path (str): Path to training CSV (ignored if is_test_only=True).
            test_path (str): Path to testing CSV.
            is_test_only (bool): If True, skips loading training data and returns None for train artifacts.

        Returns:
            Tuple containing (X_train, y_train, X_test, y_test, class_names, target_column, feature_names).
        """
        
        # Helper function to consolidate loading logic
        def load_csv_robust(path):
            """
            load_csv_robust method for RuleClassifier.
            Args:
                path: Description of path.
            Returns:
                Description of return value.
            """
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            
            # 1. Detect Header
            has_header = False
            try:
                with open(path, 'r', encoding='latin-1') as f:
                    first_line = f.readline().strip()
                    # Heuristic: if line contains letters, assume it's a header
                    has_header = any(c.isalpha() for c in first_line)
            except Exception:
                pass

            header_row = 0 if has_header else None
            
            # 2. Read CSV
            df = pd.read_csv(path, header=header_row, encoding='latin-1', on_bad_lines='skip')
            
            # 3. Standardize Column Names
            if has_header:
                df.columns = [str(col).strip() for col in df.columns]
            else:
                # Assign generic names: v1, v2, ..., class
                df.columns = [f'v{i+1}' for i in range(df.shape[1]-1)] + ['class']
            
            return df

        # --- LOAD TEST DATA ---
        df_test = load_csv_robust(test_path)

        # Case 1: Test Only (Analysis Mode)
        if is_test_only:
            feature_names = df_test.columns[:-1].tolist()
            target_name = df_test.columns[-1]
            
            # Pre-processing for Test Only mode (Minimal Encoding)
            # Since we don't have the train set's encoder, we try to force numeric conversion
            for col in feature_names:
                if df_test[col].dtype == 'object':
                    try:
                        # Try simple numeric conversion first
                        df_test[col] = pd.to_numeric(df_test[col])
                    except ValueError:
                        # Fallback: simple factorization if it's a string category
                        codes, _ = pd.factorize(df_test[col])
                        df_test[col] = pd.Series(codes, index=df_test.index)

            X_test = df_test.iloc[:, :-1].values.astype(float)
            y_test = df_test.iloc[:, -1].values.astype(int) # Assumes target is interpretable as int
            
            return None, None, X_test, y_test, None, target_name, feature_names

        # --- LOAD TRAIN DATA ---
        df_train = load_csv_robust(train_path)

        # --- DATA PROCESSING (Label Encoding) ---
        # We fit on Train and transform both Train and Test to ensure consistency
        for col in df_train.columns:
            # Check if feature is categorical (object type)
            if df_train[col].dtype == 'object':
                le = LabelEncoder()
                # Convert to string to handle mixed types safely
                df_train[col] = pd.Series(le.fit_transform(df_train[col].astype(str)), index=df_train.index)
                
                # Transform Test set if column exists
                if col in df_test.columns:
                    # Handle unseen labels: map them to the first known class to avoid crashes
                    df_test[col] = df_test[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    df_test[col] = pd.Series(le.transform(df_test[col].astype(str)), index=df_test.index)

        # Extract Metadata
        feature_names = df_train.columns[:-1].tolist()
        target_name = df_train.columns[-1]
        
        # Create robust class names list
        unique_classes = sorted(df_train[target_name].unique())
        class_names = [str(c) for c in unique_classes]

        # Convert to NumPy arrays
        X_train = df_train.iloc[:, :-1].values.astype(float)
        y_train = df_train.iloc[:, -1].values.astype(int)
        
        # Ensure Test set aligns with Train set features
        # If test set is missing columns, fill with 0; if extra, ignore (basic alignment)
        X_test = df_test[feature_names].values.astype(float)
        y_test = df_test[target_name].values.astype(int)

        return X_train, y_train, X_test, y_test, class_names, target_name, feature_names

    # Method to read where each split of a fitted tree sends missing values
    @staticmethod
    def _missing_go_to_left(tree_):
        """The per-node missing-value routing of a fitted scikit-learn tree.

        scikit-learn >= 1.3 sends a NaN to the left child of a split where
        ``missing_go_to_left`` is set. A split learned from data with NaN
        records the side chosen during training; any other split sends NaN
        to the child that received more training samples. Every split has
        a side, so a NaN is never left unrouted.

        Args:
            tree_: A fitted ``sklearn.tree._tree.Tree``.

        Returns:
            np.ndarray | None: One bool per node, or None for an older
            scikit-learn without the attribute (the routing stays unknown
            and NaN inputs are refused).
        """
        flags = getattr(tree_, 'missing_go_to_left', None)
        if flags is None:
            return None
        return np.asarray(flags).astype(bool)

    # Method to extract rules from a tree model
    @staticmethod
    def get_rules(tree, feature_names, class_names):
        """
        Extracts human-readable decision rules from a scikit-learn DecisionTreeClassifier.

        This method traverses the tree structure to generate logical condition paths from root to leaf.
        
        Optimization:
        It builds both the string representation (for display) and the parsed tuple representation 
        (for calculation) simultaneously. This avoids re-parsing strings later and preserves 
        exact floating-point precision.

        Args:
            tree (DecisionTreeClassifier): A trained scikit-learn decision tree model.
            feature_names (List[str]): A list of feature names corresponding to the tree input features.
            class_names (List[str]): A list of class names corresponding to output labels.

        Returns:
            List[Rule]: A list of extracted Rule objects.
        """
        tree_ = tree.tree_ if hasattr(tree, 'tree_') else tree

        # Mapping feature indices to names
        feature_name = [
            feature_names[i] if i != -2 else "undefined!"
            for i in tree_.feature
        ]
        # Where each split sends a missing value (scikit-learn >= 1.3); the
        # conditions record it (see `Condition`).
        missing_left = RuleClassifier._missing_go_to_left(tree_)

        rules = []

        # We pass two lists down the recursion:
        # 1. conditions_str: ["v1 <= 0.5", ...] (For Human Readability/Display)
        # 2. conditions_parsed: [("v1", "<=", 0.5), ...] (For Logic/Native Execution)
        def recurse(node, conditions_str, conditions_parsed):
            """
            Recursively traverses the decision tree and extracts rules.

            Args:
                node (int): Current node index in the tree structure.
                conditions_str (List[str]): List of condition strings accumulated so far (e.g., ["v1 <= 0.5", "v2 > 0.3"]).
                conditions_parsed (List[Tuple[str, str, float]]): List of parsed conditions as tuples (variable, operator, value).
            """
            if tree_.feature[node] != -2:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                
                # CRITICAL: Use str(threshold) to preserve precision in the string representation
                # e.g., prevents 1.16e-7 from being rounded to 0.000000 in text output
                threshold_str = f"{threshold}"
                nan_left = None if missing_left is None else bool(missing_left[node])

                # Left child (<=)
                # We append the exact float 'threshold' to the parsed list, avoiding string-to-float conversion errors later
                recurse(
                    tree_.children_left[node],
                    conditions_str + [f"{name} <= {threshold_str}"],
                    conditions_parsed + [make_condition(name, '<=', threshold, nan_left)]
                )

                # Right child (>)
                recurse(
                    tree_.children_right[node],
                    conditions_str + [f"{name} > {threshold_str}"],
                    conditions_parsed + [make_condition(
                        name, '>', threshold, None if nan_left is None else not nan_left)]
                )
            else:
                # Leaf node
                value_array = tree_.value[node].flatten()
                class_idx = np.argmax(value_array)
                
                # Store weighted sample counts per class for RF soft voting.
                # sklearn's tree_.value stores normalized probabilities (sum=1),
                # so we multiply by the weighted sample count at this leaf to
                # recover the actual (weighted) sample counts.
                # Raw counts allow correct merging (element-wise addition)
                # during adjust_and_remove_rules; normalization to probabilities
                # happens at classification time in classify_rf / native model.
                weighted_n = tree_.weighted_n_node_samples[node]
                class_dist = (value_array * weighted_n).tolist()
                
                # Ensure we handle class names correctly (int or str)
                try:
                    class_label = class_names[class_idx]
                except (IndexError, TypeError):
                    class_label = str(class_idx)

                rule_name = f"Rule{len(rules)}_Class{class_label}"
                
                # Create the Rule object with class distribution
                new_rule = Rule(rule_name, str(class_label), conditions_str,
                                class_distribution=class_dist)
                
                # OPTIMIZATION: Assign the pre-parsed conditions immediately.
                # This bypasses 'parse_conditions_static' later, saving time and keeping precision.
                new_rule.parsed_conditions = conditions_parsed
                
                rules.append(new_rule)

        recurse(0, [], [])
        return rules

    # Method to extract rules from a Random Forest model
    @staticmethod
    def get_tree_rules(model, feature_names, class_names, algorithm_type='Random Forest'):
        """
        Extracts rules from a trained scikit-learn model (Decision Tree or Random Forest).

        For Decision Trees, this returns one list of rules. 
        For Random Forests, it returns a list of lists (one list of rules per estimator).

        Args:
            model (Union[DecisionTreeClassifier, RandomForestClassifier]): The trained model.
            feature_names (List[str]): List of feature names.
            class_names (List[str]): List of class names.
            algorithm_type (str): Type of model; either 'Decision Tree' or 'Random Forest'.

        Returns:
            Union[List[Rule], List[List[Rule]]]: The extracted rules.
        """
        # Remove debug print to clean up output
        # print("Feature names:", feature_names)

        rules = []
        if algorithm_type == 'Random Forest':
            # Iterate over all trees in the forest
            for i, estimator in enumerate(model.estimators_):
                tree_rules = RuleClassifier.get_rules(estimator, feature_names, class_names)
                # Add Tree Identifier to Rule Names (DT{i}_...)
                for rule in tree_rules:
                    rule.name = f"DT{i}_{rule.name}"
                rules.append(tree_rules)
        
        elif algorithm_type == 'Decision Tree':
            # Single tree
            rules.append(RuleClassifier.get_rules(model, feature_names, class_names))
        
        else:
            raise ValueError(f"Unsupported algorithm type: {algorithm_type}")
        
        return rules

    # Method to extract rules from a Gradient Boosting model
    @staticmethod
    def get_gbdt_rules(model, feature_names, class_names):
        """
        Extracts rules from a trained GradientBoostingClassifier into standard Rule objects.

        Each leaf in each boosting tree becomes a Rule with GBDT-specific metadata
        (leaf_value, learning_rate, contribution, class_group). An additional
        init rule (no conditions) is created per class group to represent the
        initial score from the prior estimator.

        Args:
            model (GradientBoostingClassifier): A fitted sklearn GBDT model.
            feature_names (List[str]): Feature names used during training.
            class_names (List[str]): Class label strings.

        Returns:
            Tuple[List[Rule], Dict[str, float], bool, List[str]]:
                - List of all Rule objects (init rules + tree rules).
                - Dict mapping class_group -> init_score.
                - Whether the model is binary classification.
                - List of class label strings.
        """
        classes = RuleClassifier._label_names(model.classes_)
        is_binary = len(classes) == 2
        learning_rate = model.get_params()['learning_rate']

        # The init score is read once, as a constant (one rule per class).
        # That holds for 'zero' and for scikit-learn's default prior
        # (a DummyClassifier); any other init estimator scores depend on x,
        # and rules built from a single constant would be silently wrong.
        from sklearn.dummy import DummyClassifier
        init = getattr(model, 'init_', 'zero')
        if not (isinstance(init, str) and init == 'zero'):
            if (not isinstance(init, DummyClassifier)
                    or getattr(init, 'strategy', 'prior') == 'stratified'):
                raise ValueError(
                    'Unsupported GradientBoostingClassifier init estimator '
                    f'{type(init).__name__}: only init=None (the default '
                    "constant prior) or init='zero' can be expressed as "
                    'rules, because any other init gives an initial score '
                    'that depends on the input.'
                )

        all_rules = []
        init_scores = {}

        # For binary: only positive class (index 1) has trees, stored in estimators_[:, 0]
        # For multiclass: each class has its own trees, stored in estimators_[:, class_idx]
        class_iter = (
            [(1, classes[1])] if is_binary
            else list(enumerate(classes))
        )

        for class_idx, class_label in class_iter:
            # Compute init score from the prior estimator (DummyClassifier)
            if isinstance(init, str):
                init_score = 0.0
            else:
                n_features = len(feature_names)
                dummy_sample = np.zeros((1, n_features))
                init_preds = np.asarray(init.predict_proba(dummy_sample))
                if is_binary:
                    # Binary GBDT: sklearn uses log-odds (logit) of
                    # the positive class as the initial raw score.
                    p = float(init_preds[0, 1])
                    p = np.clip(p, 1e-15, 1 - 1e-15)
                    init_score = float(np.log(p / (1 - p)))
                else:
                    # Multiclass GBDT: sklearn uses centered log-priors
                    # log(p_k) - mean(log(p_j) for all j).
                    log_priors = np.log(init_preds[0] + 1e-15)
                    mean_log = float(np.mean(log_priors))
                    init_score = float(log_priors[class_idx] - mean_log)

            init_scores[class_label] = init_score

            # Create init rule (no conditions, just a constant score)
            init_rule_name = f'GBDT{class_label}T0_Init_Class{class_label}'
            init_rule = Rule(
                init_rule_name, str(class_label), [],
                leaf_value=init_score, learning_rate=None,
                class_group=str(class_label),
            )
            init_rule.parsed_conditions = []
            all_rules.append(init_rule)

            # Extract rules from each estimator tree
            estimator_col = class_idx if not is_binary else 0
            for tree_idx, estimator in enumerate(model.estimators_[:, estimator_col]):
                tree_ = estimator.tree_
                missing_left = RuleClassifier._missing_go_to_left(tree_)
                tree_rules = []

                # Method to recurse.
                def recurse(node_id, conditions_str, conditions_parsed):
                    """
                    recurse method for RuleClassifier.
                    Args:
                        node_id: Description of node_id.
                        conditions_str: Description of conditions_str.
                        conditions_parsed: Description of conditions_parsed.
                    Returns:
                        Description of return value.
                    """
                    left = int(tree_.children_left[node_id])
                    right = int(tree_.children_right[node_id])

                    if left == right:  # leaf
                        leaf_value = float(tree_.value[node_id, 0, 0])
                        leaf_idx = len(tree_rules)
                        rule_name = (
                            f'GBDT{class_label}T{tree_idx + 1}'
                            f'_Rule{leaf_idx}_Class{class_label}'
                        )
                        rule = Rule(
                            rule_name, str(class_label), list(conditions_str),
                            leaf_value=leaf_value, learning_rate=learning_rate,
                            class_group=str(class_label),
                        )
                        rule.parsed_conditions = list(conditions_parsed)
                        tree_rules.append(rule)
                        return

                    feat_idx = int(tree_.feature[node_id])
                    feat_name = feature_names[feat_idx]
                    threshold = float(tree_.threshold[node_id])
                    threshold_str = f'{threshold}'
                    # Recorded although GradientBoostingClassifier refuses
                    # NaN: the regions are those of the fitted trees.
                    nan_left = (None if missing_left is None
                                else bool(missing_left[node_id]))

                    # Left child (<=)
                    recurse(
                        left,
                        conditions_str + [f'{feat_name} <= {threshold_str}'],
                        conditions_parsed + [make_condition(
                            feat_name, '<=', threshold, nan_left)],
                    )
                    # Right child (>)
                    recurse(
                        right,
                        conditions_str + [f'{feat_name} > {threshold_str}'],
                        conditions_parsed + [make_condition(
                            feat_name, '>', threshold,
                            None if nan_left is None else not nan_left)],
                    )

                recurse(0, [], [])
                all_rules.extend(tree_rules)

        return all_rules, init_scores, is_binary, classes

    # Method to generate a classifier model based on rules
    @staticmethod
    def generate_classifier_model(rules, class_names_map, algorithm_type='Random Forest', save_initial_model=True):
        """
        Instantiates a RuleClassifier from extracted rules and optionally saves it.

        Optimization: 
        This method now passes the Rule objects directly to the constructor, avoiding 
        the intermediate string serialization that caused floating-point precision loss.

        Args:
            rules (Union[List[Rule], List[List[Rule]]]): The extracted rules.
            class_names_map (Dict[str, int]): Map of class names to indices (kept for compatibility).
            algorithm_type (str): 'Random Forest' or 'Decision Tree'.
            save_initial_model (bool): Whether to save the initial model to 'initial_model.pkl'. Default is True.

        Returns:
            RuleClassifier: The initialized classifier.
        """
        # Direct instantiation using the list of Rule objects.
        # This preserves the high-precision 'parsed_conditions' created in get_rules.
        classifier = RuleClassifier(rules, algorithm_type=algorithm_type)
        
        print(f"Algorithm Type: {classifier.algorithm_type}")

        if save_initial_model:
            path = 'files/initial_model.pkl'
            try:
                with open(path, 'wb') as model_file:
                    pickle.dump(classifier, model_file)
                print(f"Classifier file saved: {path}")
            except Exception as e:
                print(f"Warning: Could not save classifier to {path}. Error: {e}")

        return classifier
    
    # Method to execute create
    @staticmethod
    def create(train, test, model="Decision Tree", parameters=None, algorithm_type=None):
        """
        Creates a new rule-based classifier from CSV data files.

        Args:
            train (str): Path to training CSV file.
            test (str): Path to test CSV file.
            model (str): Model type - 'Decision Tree', 'Random Forest', or 
                        'Gradient Boosting Decision Trees'. Default is 'Decision Tree'.
            parameters (dict, optional): Model hyperparameters. Default is None.
            algorithm_type (str, optional): Deprecated. Use 'model' instead.

        Returns:
            RuleClassifier: The created classifier.
        """
        if algorithm_type is not None:
            import warnings
            warnings.warn(
                "algorithm_type is deprecated, use 'model' instead",
                DeprecationWarning, stacklevel=2
            )
            model = algorithm_type
        
        if parameters is None:
            parameters = {"random_state": 42}
        
        return RuleClassifier.new_classifier(
            train_path=train,
            test_path=test,
            model_parameters=parameters,
            algorithm_type=model,
            save_initial_model=False,
            save_sklearn_model=False
        )
    
    # Method to execute export
    def export(self, base_name="model", formats=None, feature_names=None, use_final=None):
        """
        Exports the classifier to one or more file formats.

        Args:
            base_name (str): Base name for exported files (without extension).
            formats (list, optional): List of formats to export to.
                                    Options: "python", "binary", "c".
                                    If None, exports to python and binary only.
            feature_names (list, optional): Feature names for export.
            use_final (bool, optional): Export final_rules (True) or
                initial_rules (False). Defaults to final_rules if available,
                else initial_rules. Every format exports this same rule set.

        Returns:
            dict: Dictionary with export results for each format.
        """
        rules = (self._rules_for(use_final) if use_final is not None
                 else (self.final_rules if self.final_rules else self.initial_rules))
        if formats is None:
            formats = ["python", "binary"]
        
        formats = [f.lower() for f in formats]
        
        export_python = "python" in formats
        export_binary = "binary" in formats or "bin" in formats
        export_c = "c" in formats or "header" in formats
        export_cpn = "cpn" in formats or "cpntools" in formats or "hcpn" in formats
        
        print("\n" + "=" * 60)
        print("EXPORTING CLASSIFIER")
        print("=" * 60)
        
        if '/' not in base_name and '\\' not in base_name:
            base_path = f"files/{base_name}"
        else:
            base_path = base_name
        
        os.makedirs(os.path.dirname(base_path) or '.', exist_ok=True)

        results = {}

        if export_python:
            py_filename = f"{base_path}.py"
            self.export_to_native_python(feature_names=feature_names, filename=py_filename,
                                         rules=rules)
            results["python"] = py_filename
            print(f"  [OK] Python export: {py_filename}")

        if export_binary or export_c:
            # The binary and C exports serialize the *current* arrays, which
            # are whatever was compiled last -- possibly the other rule set.
            # Compile the exported one for the duration of the export.
            saved = self._current_array_state() if self._arrays_compiled else None
            names = feature_names or (list(self._array_feature_names)
                                      if self._array_feature_names else None)
            self._set_array_state(self._array_state_for_rules(rules, names))
            try:
                if export_binary:
                    bin_filename = f"{base_path}.bin"
                    self.export_to_binary(filepath=bin_filename)
                    results["binary"] = bin_filename
                    print(f"  [OK] Binary export: {bin_filename}")

                if export_c:
                    c_filename = f"{base_path}.h"
                    self.export_to_c_header(filepath=c_filename)
                    results["c"] = c_filename
                    print(f"  [OK] C Header export: {c_filename}")
            finally:
                if saved is not None:
                    self._set_array_state(saved)

        if export_cpn:
            cpn_filename = f"{base_path}.cpn"
            self.to_cpn_tools(filepath=cpn_filename, rules=rules,
                              feature_names=feature_names,
                              use_final=rules is not self.initial_rules)
            results["cpn"] = cpn_filename
            print(f"  [OK] CPN Tools export: {cpn_filename}")

        print("=" * 60 + "\n")
        
        return results
    
    # Method to create a new classifier
    @staticmethod
    def new_classifier(train_path, test_path, model_parameters, model_path=None, algorithm_type='Random Forest', save_initial_model=True, save_sklearn_model=True):
        """
        Orchestrates the creation of a new classifier from scratch.

        Pipeline:
        1. Load and Process Data (using robust CSV loader).
        2. Train (or Load) Scikit-Learn Model.
        3. Evaluate Scikit-Learn Model (Benchmark).
        4. Extract Rules from Tree(s).
        5. Generate RuleClassifier.

        Args:
            train_path (str): Path to training CSV.
            test_path (str): Path to testing CSV.
            model_parameters (dict): Arguments for the sklearn classifier.
            model_path (Optional[str]): Path to existing .pkl model to skip training.
            algorithm_type (str): 'Random Forest', 'Decision Tree', or 'Gradient Boosting Decision Trees'.
            save_initial_model (bool): Whether to save the initial model to 'initial_model.pkl'. Default is True.
            save_sklearn_model (bool): Whether to save the sklearn model to 'sklearn_model.pkl'. Default is True.

        Returns:
            RuleClassifier: The final rule-based model.
        """
        
        print("\n" + "*"*80)
        print("GENERATING A NEW CLASSIFIER")
        print("*"*80 + "\n")
        
        print("Processing Database...")
        # 1. Process Data
        # Returns correctly encoded data and class names
        X_train, y_train, X_test, y_test, class_names, _, feature_names = RuleClassifier.process_data(train_path, test_path)

        # Ensure class_names is always a list (process_data may return None in test-only mode)
        if class_names is None:
            class_names = [str(c) for c in sorted(np.unique(np.asarray(y_test)))]

        # 2. Train or Load Model
        if model_path:
            print(f"Loading model from: {model_path}")
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)
        else:
            if X_train is None or y_train is None:
                raise ValueError("Training data could not be loaded. Please check the train_path parameter.")
            
            print(f"Training a new Scikit-Learn {algorithm_type}...")
            if algorithm_type == 'Random Forest':
                model = RandomForestClassifier(**model_parameters)
            elif algorithm_type == 'Decision Tree':
                model = DecisionTreeClassifier(**model_parameters)
            elif algorithm_type == 'Gradient Boosting Decision Trees':
                model = GradientBoostingClassifier(**model_parameters)
            else:
                raise ValueError(f"Unsupported algorithm type: {algorithm_type}")
            
            # Fit model
            # For GBDT, fit with DataFrame so feature_names_in_ is populated
            if algorithm_type == 'Gradient Boosting Decision Trees':
                X_train_fit = pd.DataFrame(X_train, columns=feature_names)
                model.fit(X_train_fit, np.asarray(y_train))
            else:
                model.fit(X_train, np.asarray(y_train))

            # Save Sklearn model
            if save_sklearn_model:
                model_save_path = 'files/sklearn_model.pkl'
                with open(model_save_path, 'wb') as model_file: 
                    pickle.dump(model, model_file)
                print(f"Trained model saved at: {model_save_path}")

        # 3. Test Scikit-Learn Model (Benchmark)
        print("\nTesting Scikit-Learn Model (Benchmark):")
        # For GBDT, use DataFrame to avoid feature name warnings
        if algorithm_type == 'Gradient Boosting Decision Trees':
            X_test_predict = pd.DataFrame(X_test, columns=feature_names)
        else:
            X_test_predict = X_test
        y_pred = model.predict(X_test_predict)
        
        correct = np.sum(y_pred == y_test)
        total = len(y_test)
        
        # Display metrics using the extracted class names for better readability
        RuleClassifier.display_metrics(y_test, y_pred, correct, total, class_names=class_names)

        # --- GBDT: Extract standard Rule objects, same as DT/RF ---
        if algorithm_type == 'Gradient Boosting Decision Trees':
            print("\nExtracting GBDT rules into standard Rule objects...")
            rules, init_scores, is_binary, gbdt_classes = RuleClassifier.get_gbdt_rules(
                model, feature_names, class_names
            )
            print(f"Extracted {len(rules)} GBDT rules.")

            # Build RuleClassifier using standard constructor
            # (constructor defers native compilation for GBDT)
            classifier = RuleClassifier(rules, algorithm_type=algorithm_type)

            # Set GBDT-specific metadata
            classifier._gbdt_init_scores = init_scores
            classifier._gbdt_is_binary = is_binary
            classifier._gbdt_classes = gbdt_classes
            # scikit-learn compares float32(x) with the thresholds
            classifier.input_dtype = 'float32'

            # Compile native model now that metadata is set
            classifier.update_native_model(classifier.initial_rules)

            # Save initial model (after metadata is set)
            if save_initial_model:
                path = 'files/initial_model.pkl'
                try:
                    with open(path, 'wb') as model_file:
                        pickle.dump(classifier, model_file)
                    print(f"Classifier file saved: {path}")
                except Exception as e:
                    print(f"Warning: Could not save classifier to {path}. Error: {e}")

            print(f"Algorithm Type: {classifier.algorithm_type}")
            return classifier

        # 4. Extract Rules
        # We pass class_names so the rules are generated with "Class_Normal" instead of "Class_0"
        print(f"\nExtracting rules from {algorithm_type}...")
        rules = RuleClassifier.get_tree_rules(model, feature_names, class_names, algorithm_type=algorithm_type)

        # 5. Generate RuleClassifier
        # We assume mapping corresponds to the sorted unique labels from process_data
        class_names_map = {str(name): i for i, name in enumerate(class_names)}
        
        print("Initializing RuleClassifier...")
        classifier = RuleClassifier.generate_classifier_model(rules, class_names_map, algorithm_type, save_initial_model=save_initial_model)
        # scikit-learn compares float32(x) with the thresholds
        classifier.input_dtype = 'float32'

        return classifier
