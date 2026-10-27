"""The rule-based classifier.

Provides ``RuleClassifier``, the classifier built from the decision paths
(rules) of a fitted decision tree, random forest or gradient boosting model:
its state, pickling, and the parsing of rule conditions. The rest of its
behaviour lives in one mixin per responsibility, combined here:

* :mod:`._labels` -- label codes and input checks (``LabelsMixin``)
* :mod:`._engine` -- iterative single-sample engine (``IterativeEngineMixin``)
* :mod:`._native_python` -- generated Python code (``NativePythonMixin``)
* :mod:`._arrays` -- compiled arrays and batch prediction (``ArrayEngineMixin``)
* :mod:`._binary_format` -- the ``.bin`` format (``BinaryFormatMixin``)
* :mod:`._c_export` -- C header and Arduino sketch (``CExportMixin``)
* :mod:`._refinement` -- rule refinement (``RefinementMixin``)
* :mod:`._serialization` -- pickle loading and JSON (``SerializationMixin``)
* :mod:`._reporting` -- reports, metrics, rule editing (``ReportingMixin``)
* :mod:`._extraction` -- data loading and rule extraction (``ExtractionMixin``)
* :mod:`.exporters` -- the ``to_*`` export front-ends (``RuleExporterMixin``)

``Rule``, ``Condition`` and the other rule helpers are defined in
:mod:`._rules` and re-exported here: ``pyruleanalyzer.rule_classifier``
remains their import path, and the module pickles refer to.
"""

from collections import Counter
from typing import Optional

# Re-exported: this module is the public (and pickled) home of these names.
from ._rules import (  # noqa: F401
    _tree_sort_key, Condition, MissingCondition, make_condition,
    condition_missing, condition_signature, _py_float, _py_split_test, Rule,
)
from ._c_export import _compute_memory_for_rule_classifier  # noqa: F401

# Pickles record a class by module and name. Rule and the conditions were
# always pickled as pyruleanalyzer.rule_classifier.<name>: keep that name,
# so old pickles load and new ones still load in older versions. Assigning
# this module's own __name__ (the object RuleClassifier.__module__ is too)
# keeps the pickles byte-identical to those written before the split.
for _cls in (Condition, MissingCondition, Rule):
    _cls.__module__ = __name__
del _cls

from ._labels import LabelsMixin
from ._engine import IterativeEngineMixin
from ._native_python import NativePythonMixin
from ._arrays import ArrayEngineMixin
from ._binary_format import BinaryFormatMixin
from ._c_export import CExportMixin
from ._refinement import RefinementMixin
from ._serialization import SerializationMixin
from ._reporting import ReportingMixin
from ._extraction import ExtractionMixin
from .exporters import RuleExporterMixin


# Class to handle the rule classification process
class RuleClassifier(
    LabelsMixin,
    IterativeEngineMixin,
    NativePythonMixin,
    ArrayEngineMixin,
    BinaryFormatMixin,
    CExportMixin,
    RefinementMixin,
    SerializationMixin,
    ReportingMixin,
    ExtractionMixin,
    RuleExporterMixin,
):
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
