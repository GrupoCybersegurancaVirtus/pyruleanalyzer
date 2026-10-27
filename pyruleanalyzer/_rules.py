"""Parsed conditions and rules.

Provides ``Rule``, one root-to-leaf path of a tree, the condition tuples
``Condition`` / ``MissingCondition`` that also record where a missing value
(NaN) goes, and the helpers the engines and code generators share
(``make_condition``, ``condition_missing``, ``condition_signature``,
``_py_float``, ``_py_split_test``, ``_tree_sort_key``).

Their public import path is still :mod:`pyruleanalyzer.rule_classifier`,
which re-exports them; it is also the module pickles record for these
classes (``__module__`` is set there). `Rule` names `RuleClassifier`
(defined there) through local imports.
"""

import re
import numpy as np


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
        from .rule_classifier import RuleClassifier
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
        from .rule_classifier import RuleClassifier
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
