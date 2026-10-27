"""Labels and input checks of the rule classifier.

Provides ``LabelsMixin``, the part of :class:`RuleClassifier` that maps the
estimator's class labels to the integer codes every engine works on (and
back), and that checks inputs before a prediction: missing values (NaN) and
whether the rules record where they go, the features the rules use, and the
float32 quantisation scikit-learn applies to its inputs.

`RuleClassifier` is defined after its mixins, so the methods that name it import
it locally.
"""

import numpy as np
from typing import List, Optional
from ._rules import Condition


class LabelsMixin:
    """Label tables, label encoding and input validation."""

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
        from .rule_classifier import RuleClassifier
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
