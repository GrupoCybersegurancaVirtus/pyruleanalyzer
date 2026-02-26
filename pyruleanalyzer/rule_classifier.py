import os
import re
import pickle
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
    """

    __slots__ = [
        'name', 'class_', 'conditions', 'usage_count', 'error_count',
        'parsed_conditions',
        # RF soft-voting: per-class probability distribution at this leaf
        'class_distribution',
        # GBDT-specific fields (None for DT/RF rules)
        'leaf_value', 'learning_rate', 'contribution', 'class_group',
    ]

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

    def __repr__(self):
        return f"Rule(name={self.name}, class={self.class_}, conditions={self.conditions})"

# Class to handle the rule classification process
class RuleClassifier:
    # Represents a rule-based classifier built from decision paths in tree models.
    
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
        
        # Auto-recompile the native model upon loading
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
        
        # Build tree structure
        tree_dict = {0: {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': -1}}
        next_id = 1
        
        for idx, rule in enumerate(rules_to_compile):
            curr = 0
            for var, op, val in rule.parsed_conditions:
                if tree_dict[curr]['f'] == -2:
                    tree_dict[curr].update({'f': var, 't': val, 'l': next_id, 'r': next_id + 1})
                    tree_dict[next_id] = {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': -1}
                    tree_dict[next_id + 1] = {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': -1}
                    next_id += 2
                
                if op in ['<=', '<']:
                    curr = tree_dict[curr]['l']
                else:
                    curr = tree_dict[curr]['r']
            
            # Store the index of the rule at the leaf
            tree_dict[curr]['v'] = idx

        # Build code
        def build_code(node_id, indent):
            node = tree_dict[node_id]
            tab = "    " * indent
            
            if node['f'] == -2: 
                # Return the index of the rule
                return f"{tab}return {node['v']}\n"
            
            feat_name = node['f']
            # Use repr(float(t)) for precision
            code = f"{tab}if sample.get('{feat_name}', 0) <= {repr(float(node['t']))}:\n"
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

            # Re-iterate to build tree functions in order
            for tid, rules in tree_rules_map.items():
                if not rules:
                    continue
                # Skip init rules
                if len(rules) == 1 and not rules[0].parsed_conditions and rules[0].contribution is not None:
                    continue

                func_name = f'predict_{tid}'
                tree_dict = {0: {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': 0.0}}
                next_id = 1

                for rule in rules:
                    curr = 0
                    for var, op, val in rule.parsed_conditions:
                        if tree_dict[curr]['f'] == -2:
                            tree_dict[curr].update({'f': var, 't': val, 'l': next_id, 'r': next_id + 1})
                            tree_dict[next_id] = {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': 0.0}
                            tree_dict[next_id + 1] = {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': 0.0}
                            next_id += 2
                        if op in ['<=', '<']:
                            curr = tree_dict[curr]['l']
                        else:
                            curr = tree_dict[curr]['r']
                    tree_dict[curr]['v'] = rule.contribution if rule.contribution is not None else 0.0

                def _build_code(node_id, indent, td=tree_dict):
                    node = td[node_id]
                    tab = '    ' * indent
                    if node['f'] == -2:
                        return f'{tab}return {repr(float(node["v"]))}\n'
                    feat = node['f']
                    code = f"{tab}if sample.get('{feat}', 0) <= {repr(float(node['t']))}:\n"
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
                    func_name = f'predict_{tid}'
                    func_code += f'    score += {func_name}(sample)\n'
                func_code += '    prob = 1.0 / (1.0 + math.exp(-score))\n'
                func_code += '    if prob >= 0.5:\n'
                func_code += f'        return {int(classes[1])}, None, None\n'
                func_code += '    else:\n'
                func_code += f'        return {int(classes[0])}, None, None\n'
            else:
                # Multiclass
                for class_label in classes:
                    score_init = init_scores.get(class_label, 0.0)
                    func_code += f'    score_{class_label} = {repr(score_init)}\n'
                for tid, rules in tree_rules_map.items():
                    if not rules:
                        continue
                    if len(rules) == 1 and not rules[0].parsed_conditions:
                        continue
                    class_group = rules[0].class_group
                    func_name = f'predict_{tid}'
                    func_code += f'    score_{class_group} += {func_name}(sample)\n'

                # Argmax
                score_vars = [f'score_{cl}' for cl in classes]
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
            
            for tid, rules in tree_rules_map.items():
                func_name = f"predict_{tid}"
                tree_func_names.append(func_name)
                
                # Build tree structure — store probability distribution at leaves
                tree_dict = {0: {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': None}}
                next_id = 1
                
                for rule in rules:
                    curr = 0
                    for var, op, val in rule.parsed_conditions:
                        if tree_dict[curr]['f'] == -2:
                            tree_dict[curr].update({'f': var, 't': val, 'l': next_id, 'r': next_id + 1})
                            tree_dict[next_id] = {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': None}
                            tree_dict[next_id + 1] = {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': None}
                            next_id += 2
                        if op in ['<=', '<']:
                            curr = tree_dict[curr]['l']
                        else:
                            curr = tree_dict[curr]['r']
                    
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

                def build_rf_code(node_id, indent):
                    node = tree_dict[node_id]
                    tab = "    " * indent
                    if node['f'] == -2:
                        val = node['v']
                        if val is None:
                            return f"{tab}return None\n" 
                        return f"{tab}return {repr(val)}\n"
                    
                    feat = node['f']
                    # Use repr(float(t)) to ensure no numpy precision issues in literals
                    code = f"{tab}if sample.get('{feat}', 0) <= {repr(float(node['t']))}:\n"
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
            
            if isinstance(default_val, str):
                def_str = f"'{default_val}'"
            else:
                def_str = str(default_val)

            func_code += "def native_predict(sample):\n"
            func_code += "    tree_results = []\n"
            for func in tree_func_names:
                func_code += f"    v = {func}(sample)\n"
                func_code += "    if v is not None: tree_results.append(v)\n"
            func_code += f"    if not tree_results: return {def_str}, [], None\n"
            
            if n_classes > 0:
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
                func_code += "        return best, None, avg\n"
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
            tree_dict = {0: {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': -1}}
            next_id = 1
            for rule in rules_to_compile:
                curr = 0
                for var, op, val in rule.parsed_conditions:
                    if tree_dict[curr]['f'] == -2:
                        tree_dict[curr].update({'f': var, 't': val, 'l': next_id, 'r': next_id + 1})
                        tree_dict[next_id] = {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': -1}
                        tree_dict[next_id + 1] = {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': -1}
                        next_id += 2
                    if op in ['<=', '<']:
                        curr = tree_dict[curr]['l']
                    else:
                        curr = tree_dict[curr]['r']
                
                try:
                    clean_class = int(str(rule.class_).replace('Class', '').strip())
                except Exception:
                    clean_class = rule.class_
                tree_dict[curr]['v'] = clean_class

            def build_dt_code(node_id, indent):
                node = tree_dict[node_id]
                tab = "    " * indent
                if node['f'] == -2: 
                    val = node['v']
                    if val == -1: 
                        try:
                            val = int(str(self.default_class).replace('Class', '').strip())
                        except Exception:
                            val = self.default_class
                    if isinstance(val, str):
                        return f"{tab}return '{val}', None, None\n"
                    return f"{tab}return {val}, None, None\n"
                
                feat = node['f']
                # Use repr(float(t)) for precision
                code = f"{tab}if sample.get('{feat}', 0) <= {repr(float(node['t']))}:\n"
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

        This method delegates the classification logic. If 'final' is False (using initial rules)
        and the native function is compiled, it uses the high-performance in-memory function.
        Otherwise, it falls back to iterative evaluation.

        Args:
            data (Dict[str, float]): The instance to classify.
            final (bool): If True, uses `final_rules` (post-analysis).

        Returns:
            Tuple[int, List[int]|None, List[float]|None]: 
                - Predicted class label (int).
                - List of votes (Random Forest only).
                - Class probabilities (Random Forest only).
        """
        # --- FAST PATH 1: Array-based single-sample prediction ---
        # Works for BOTH initial and final rules (arrays are compiled for whichever
        # rule set is current). This fixes the final=True bypass bug.
        if self._arrays_compiled:
            try:
                row = np.array(
                    [[data.get(f, 0.0) for f in self._array_feature_names]],
                    dtype=np.float64,
                )
                pred = self.predict_batch(row)[0]
                clean_class = int(pred)
                return clean_class, None, None
            except Exception:
                pass  # Fallback to slower paths

        # --- FAST PATH 2: Native function (exec-compiled, initial rules only) ---
        if not final and self.native_fn is not None:
            try:
                # native_fn returns (prediction, votes, proba)
                return self.native_fn(data) # pyright: ignore[reportCallIssue]
            except Exception:
                # Fallback to standard execution if something goes wrong (e.g., missing keys)
                pass

        # --- SLOW PATH: Iterative Execution ---
        rules = self.final_rules if final else self.initial_rules
        
        predicted_class = self.default_class
        votes = None
        proba = None

        if self.algorithm_type == 'Random Forest':
            predicted_class, votes, proba, _ = self.classify_rf(data, rules)
        
        elif self.algorithm_type == 'Gradient Boosting Decision Trees':
            predicted_class, _, _ = self.classify_gbdt(
                data, rules, self._gbdt_init_scores,
                self._gbdt_is_binary, self._gbdt_classes,
            )

        elif self.algorithm_type == 'Decision Tree':
            matched_rule = self.classify_dt(data, rules)
            if matched_rule:
                predicted_class = matched_rule.class_

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
            for var, op, value in rule.parsed_conditions:
                # Use data.get() to avoid KeyErrors, defaulting to None implies failure
                instance_value = data.get(var)
                
                if instance_value is None:
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
    def classify_rf(data, rules):
        """
        Classifies a single data instance using extracted rules from a Random Forest.
        Uses soft probability averaging (like sklearn) when class distributions are
        available, otherwise falls back to hard majority voting.

        Args:
            data (Dict[str, float]): Instance data.
            rules (List[Rule]): List of rule instances.

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
            predicted_class = best_idx
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
        classification uses sigmoid; multiclass uses argmax.

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
        import math

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
                    for var, op, value in rule.parsed_conditions:
                        instance_value = data.get(var)
                        if instance_value is None:
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

            # Sigmoid
            prob = 1.0 / (1.0 + math.exp(-score))
            predicted_class = int(classes[1]) if prob >= 0.5 else int(classes[0])

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
                    for var, op, value in rule.parsed_conditions:
                        instance_value = data.get(var)
                        if instance_value is None:
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
    #   value[i]          varies  leaf payload:
    #       DT:   int32   class label
    #       RF:   float64 array of shape (n_classes,) — raw sample counts
    #       GBDT: float64 contribution (learning_rate * leaf_value)
    # =========================================================================

    @staticmethod
    def _build_single_tree_arrays(
        rules: list,
        feature_name_to_idx: Dict[str, int],
        algorithm_type: str,
        n_classes: int = 0,
    ) -> dict:
        """
        Convert a list of rules (all from one tree) into flat numpy arrays.

        Args:
            rules (List[Rule]): Rules for a single tree.
            feature_name_to_idx (Dict[str, int]): Mapping feature_name -> column index.
            algorithm_type (str): 'Decision Tree', 'Random Forest', or
                'Gradient Boosting Decision Trees'.
            n_classes (int): Number of classes (needed for RF leaf distributions).

        Returns:
            Dict with keys: 'feature_idx', 'threshold', 'children_left',
            'children_right', 'value'.  Each is a numpy array indexed by node id.
        """
        # 1. Reconstruct tree structure
        tree_dict: Dict[int, dict] = {
            0: {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': None}
        }
        next_id = 1

        for rule in rules:
            curr = 0
            for var, op, val in rule.parsed_conditions:
                if tree_dict[curr]['f'] == -2:
                    tree_dict[curr].update({
                        'f': var, 't': val,
                        'l': next_id, 'r': next_id + 1,
                    })
                    tree_dict[next_id] = {
                        'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': None,
                    }
                    tree_dict[next_id + 1] = {
                        'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': None,
                    }
                    next_id += 2
                if op in ('<=', '<'):
                    curr = tree_dict[curr]['l']
                else:
                    curr = tree_dict[curr]['r']

            # Set leaf value based on algorithm type
            if algorithm_type == 'Gradient Boosting Decision Trees':
                tree_dict[curr]['v'] = float(rule.contribution) if rule.contribution is not None else 0.0
            elif algorithm_type == 'Random Forest':
                if rule.class_distribution is not None:
                    tree_dict[curr]['v'] = list(rule.class_distribution)
                else:
                    # Fallback: one-hot encode the class label
                    try:
                        cls_idx = int(str(rule.class_).replace('Class', '').strip())
                    except (ValueError, AttributeError):
                        cls_idx = 0
                    one_hot = [0.0] * max(n_classes, cls_idx + 1)
                    one_hot[cls_idx] = 1.0
                    tree_dict[curr]['v'] = one_hot
            else:
                # Decision Tree — store class label as int
                # Use class_distribution (argmax) when available, otherwise
                # extract trailing digits from the class label string.
                if hasattr(rule, 'class_distribution') and rule.class_distribution is not None:
                    dist = rule.class_distribution
                    tree_dict[curr]['v'] = int(np.argmax(dist))
                else:
                    cls_str = str(rule.class_)
                    m = re.search(r'(\d+)$', cls_str)
                    if m:
                        tree_dict[curr]['v'] = int(m.group(1))
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
            value = np.full(n_nodes, -1, dtype=np.int32)

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
                if algorithm_type == 'Random Forest':
                    v = node['v']
                    if v is not None:
                        for ci in range(min(len(v), n_classes)):
                            value[node_id, ci] = v[ci]
                elif algorithm_type == 'Gradient Boosting Decision Trees':
                    value[node_id] = float(node['v']) if node['v'] is not None else 0.0
                else:
                    value[node_id] = int(node['v']) if node['v'] is not None else -1
            else:
                feature_idx[node_id] = feature_name_to_idx.get(feat, -2)
                threshold[node_id] = float(node['t'])
                children_left[node_id] = node['l']
                children_right[node_id] = node['r']

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

        return {
            'feature_idx': feature_idx,
            'threshold': threshold,
            'children_left': children_left,
            'children_right': children_right,
            'value': value,
            'max_depth': max_depth,
        }

    def compile_tree_arrays(self, rules: Optional[list] = None, feature_names: Optional[list] = None) -> None:
        """
        Compile rules into numpy tree arrays for vectorized prediction.

        After calling this method, `predict_batch(X)` becomes available.
        This is called automatically by `update_native_model` but can also be
        called explicitly after rule removal to refresh the arrays.

        Args:
            rules (List[Rule], optional): Rules to compile. Defaults to
                final_rules if available, else initial_rules.
            feature_names (List[str], optional): Ordered feature names matching
                the columns of X that will be passed to predict_batch.
                If None, they are inferred from the rules.
        """
        if rules is None:
            rules = self.final_rules if self.final_rules else self.initial_rules

        # --- Infer feature names from rules if not provided ---
        if feature_names is None:
            feat_set: set = set()
            for rule in rules:
                for var, _, _ in rule.parsed_conditions:
                    feat_set.add(var)
            feature_names = sorted(feat_set)

        self._array_feature_names = list(feature_names)
        feature_name_to_idx = {name: i for i, name in enumerate(feature_names)}

        n_classes = self.num_classes

        # --- Group rules by tree ---
        if self.algorithm_type == 'Decision Tree':
            # Single tree — all rules belong to the same tree
            tree_arrays = [
                self._build_single_tree_arrays(
                    rules, feature_name_to_idx, self.algorithm_type, n_classes
                )
            ]
            self._tree_arrays = tree_arrays
            self._tree_class_groups = None
            self._tree_is_init = None

        elif self.algorithm_type == 'Random Forest':
            tree_rules_map: Dict[str, list] = defaultdict(list)
            for rule in rules:
                tid = rule.name.split('_')[0] if '_' in rule.name else 'Tree0'
                tree_rules_map[tid].append(rule)

            tree_arrays = []
            for tid in sorted(tree_rules_map.keys()):
                arr = self._build_single_tree_arrays(
                    tree_rules_map[tid], feature_name_to_idx,
                    self.algorithm_type, n_classes,
                )
                tree_arrays.append(arr)
            self._tree_arrays = tree_arrays
            self._tree_class_groups = None
            self._tree_is_init = None

        elif self.algorithm_type == 'Gradient Boosting Decision Trees':
            tree_rules_map = defaultdict(list)
            for rule in rules:
                tid = rule.name.split('_')[0] if '_' in rule.name else rule.name
                tree_rules_map[tid].append(rule)

            tree_arrays = []
            class_groups = []
            is_init = []

            for tid in sorted(tree_rules_map.keys()):
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
                    )
                    tree_arrays.append(arr)

            self._tree_arrays = tree_arrays
            self._tree_class_groups = class_groups
            self._tree_is_init = is_init

        self._arrays_compiled = True

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

    def predict_batch(
        self,
        X: np.ndarray,
        feature_names: Optional[list] = None,
        use_final: bool = True,
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
            use_final (bool): If True, uses arrays compiled from final_rules.
                If False, uses arrays compiled from initial_rules.

        Returns:
            np.ndarray: Predicted class labels, shape (n_samples,), dtype int.
        """
        if not hasattr(self, '_arrays_compiled') or not self._arrays_compiled:
            raise RuntimeError(
                'Tree arrays not compiled. Call compile_tree_arrays() first.'
            )

        # Reorder columns if feature_names differ from compiled order
        if feature_names is not None and feature_names != self._array_feature_names:
            col_order = [feature_names.index(f) for f in self._array_feature_names]
            X = X[:, col_order]

        n_samples = X.shape[0]

        # --- Decision Tree ---
        if self.algorithm_type == 'Decision Tree':
            tree = self._tree_arrays[0]
            leaf_ids = _accel_traverse(
                X, tree['feature_idx'], tree['threshold'],
                tree['children_left'], tree['children_right'],
                tree['max_depth'],
            )
            predictions = tree['value'][leaf_ids]
            return predictions

        # --- Random Forest (Soft Voting) ---
        elif self.algorithm_type == 'Random Forest':
            n_classes = self.num_classes
            n_trees = len(self._tree_arrays)

            # Build list of tree tuples for multi-tree C traversal
            tree_tuples = [
                (t['feature_idx'], t['threshold'],
                 t['children_left'], t['children_right'], t['max_depth'])
                for t in self._tree_arrays
            ]
            # all_leaf_ids: shape (n_trees, n_samples), int32
            all_leaf_ids = _accel_traverse_multi(X, tree_tuples)

            # Accumulate probability contributions from each tree
            prob_sum = np.zeros((n_samples, n_classes), dtype=np.float64)
            for t_idx in range(n_trees):
                leaf_ids = all_leaf_ids[t_idx]
                # Get raw counts at leaves
                raw_counts = self._tree_arrays[t_idx]['value'][leaf_ids]
                # Normalize to probabilities per sample
                totals = raw_counts.sum(axis=1, keepdims=True)
                totals = np.where(totals == 0, 1.0, totals)
                probas = raw_counts / totals
                prob_sum += probas

            # Average and argmax
            avg_proba = prob_sum / n_trees
            predictions = np.argmax(avg_proba, axis=1).astype(np.int32)
            return predictions

        # --- GBDT (Additive Scoring) ---
        elif self.algorithm_type == 'Gradient Boosting Decision Trees':
            classes = self._gbdt_classes or []
            is_binary = self._gbdt_is_binary
            assert self._tree_class_groups is not None
            assert self._tree_is_init is not None

            # Separate init trees from real trees and batch-traverse real trees
            real_tree_indices = []
            tree_tuples = []
            for i, tree in enumerate(self._tree_arrays):
                if not self._tree_is_init[i]:
                    real_tree_indices.append(i)
                    tree_tuples.append((
                        tree['feature_idx'], tree['threshold'],
                        tree['children_left'], tree['children_right'],
                        tree['max_depth'],
                    ))

            # Batch traverse all real trees at once
            if tree_tuples:
                all_leaf_ids = _accel_traverse_multi(X, tree_tuples)
            else:
                all_leaf_ids = np.empty((0, n_samples), dtype=np.int32)

            if is_binary and len(classes) >= 2:
                # Binary: accumulate score for positive class
                pos_class = classes[1]
                scores = np.zeros(n_samples, dtype=np.float64)

                # Add init scores
                for i, tree in enumerate(self._tree_arrays):
                    if self._tree_is_init[i]:
                        if self._tree_class_groups[i] == pos_class:
                            scores += tree['init_score']

                # Add real tree contributions
                for batch_idx, orig_idx in enumerate(real_tree_indices):
                    if self._tree_class_groups[orig_idx] != pos_class:
                        continue
                    leaf_ids = all_leaf_ids[batch_idx]
                    scores += self._tree_arrays[orig_idx]['value'][leaf_ids]

                # Sigmoid
                prob = 1.0 / (1.0 + np.exp(-scores))
                predictions = np.where(
                    prob >= 0.5, int(classes[1]), int(classes[0])
                ).astype(np.int32)
                return predictions

            else:
                # Multiclass: accumulate scores per class
                class_to_col = {cl: idx for idx, cl in enumerate(classes)}
                n_cls = len(classes)
                scores = np.zeros((n_samples, n_cls), dtype=np.float64)

                # Add init scores
                for i, tree in enumerate(self._tree_arrays):
                    if self._tree_is_init[i]:
                        cg = self._tree_class_groups[i]
                        col = class_to_col.get(cg)
                        if col is not None:
                            scores[:, col] += tree['init_score']

                # Add real tree contributions
                for batch_idx, orig_idx in enumerate(real_tree_indices):
                    cg = self._tree_class_groups[orig_idx]
                    col = class_to_col.get(cg)
                    if col is None:
                        continue
                    leaf_ids = all_leaf_ids[batch_idx]
                    scores[:, col] += self._tree_arrays[orig_idx]['value'][leaf_ids]

                # Argmax -> class label
                best_col = np.argmax(scores, axis=1)
                class_arr = np.array([int(cl) for cl in classes], dtype=np.int32)
                predictions = class_arr[best_col]
                return predictions

        raise ValueError(f'Unsupported algorithm_type: {self.algorithm_type}')

    def predict_batch_proba(
        self,
        X: np.ndarray,
        feature_names: Optional[list] = None,
    ) -> np.ndarray:
        """
        Vectorized batch probability prediction.

        Args:
            X (np.ndarray): Input data, shape (n_samples, n_features).
            feature_names (list, optional): Feature names for column reordering.

        Returns:
            np.ndarray: Predicted probabilities, shape (n_samples, n_classes).
                For DT, this is a one-hot array.
                For RF, this is the averaged probability across trees.
                For GBDT binary, shape (n_samples, 2) with sigmoid probabilities.
                For GBDT multiclass, shape (n_samples, n_classes) with softmax probabilities.
        """
        if not hasattr(self, '_arrays_compiled') or not self._arrays_compiled:
            raise RuntimeError(
                'Tree arrays not compiled. Call compile_tree_arrays() first.'
            )

        if feature_names is not None and feature_names != self._array_feature_names:
            col_order = [feature_names.index(f) for f in self._array_feature_names]
            X = X[:, col_order]

        n_samples = X.shape[0]

        # --- Decision Tree ---
        if self.algorithm_type == 'Decision Tree':
            tree = self._tree_arrays[0]
            leaf_ids = _accel_traverse(
                X, tree['feature_idx'], tree['threshold'],
                tree['children_left'], tree['children_right'],
                tree['max_depth'],
            )
            preds = tree['value'][leaf_ids]
            # One-hot encode
            n_classes = self.num_classes
            proba = np.zeros((n_samples, n_classes), dtype=np.float64)
            for i in range(n_samples):
                proba[i, int(preds[i])] = 1.0
            return proba

        # --- Random Forest ---
        elif self.algorithm_type == 'Random Forest':
            n_classes = self.num_classes
            prob_sum = np.zeros((n_samples, n_classes), dtype=np.float64)
            n_trees = len(self._tree_arrays)

            # Batch traverse all trees at once
            tree_tuples = [
                (t['feature_idx'], t['threshold'],
                 t['children_left'], t['children_right'], t['max_depth'])
                for t in self._tree_arrays
            ]
            all_leaf_ids = _accel_traverse_multi(X, tree_tuples)

            for t_idx in range(n_trees):
                leaf_ids = all_leaf_ids[t_idx]
                raw_counts = self._tree_arrays[t_idx]['value'][leaf_ids]
                totals = raw_counts.sum(axis=1, keepdims=True)
                totals = np.where(totals == 0, 1.0, totals)
                probas = raw_counts / totals
                prob_sum += probas
            return prob_sum / n_trees

        # --- GBDT ---
        elif self.algorithm_type == 'Gradient Boosting Decision Trees':
            classes = self._gbdt_classes or []
            is_binary = self._gbdt_is_binary
            assert self._tree_class_groups is not None
            assert self._tree_is_init is not None
            class_to_col = {cl: idx for idx, cl in enumerate(classes)}
            n_cls = len(classes)
            scores = np.zeros((n_samples, n_cls), dtype=np.float64)

            # Separate init vs real trees, batch traverse real ones
            real_tree_indices = []
            tree_tuples_gbdt = []
            for i, tree in enumerate(self._tree_arrays):
                if self._tree_is_init[i]:
                    cg = self._tree_class_groups[i]
                    col = class_to_col.get(cg)
                    if col is not None:
                        scores[:, col] += tree['init_score']
                else:
                    real_tree_indices.append(i)
                    tree_tuples_gbdt.append((
                        tree['feature_idx'], tree['threshold'],
                        tree['children_left'], tree['children_right'],
                        tree['max_depth'],
                    ))

            if tree_tuples_gbdt:
                all_leaf_ids = _accel_traverse_multi(X, tree_tuples_gbdt)
                for batch_idx, orig_idx in enumerate(real_tree_indices):
                    cg = self._tree_class_groups[orig_idx]
                    col = class_to_col.get(cg)
                    if col is None:
                        continue
                    leaf_ids = all_leaf_ids[batch_idx]
                    scores[:, col] += self._tree_arrays[orig_idx]['value'][leaf_ids]

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

    def export_to_binary(self, filepath: str = 'model.bin') -> None:
        """
        Export the compiled tree arrays to a compact binary file.

        File format (all little-endian):
            Header:
                4 bytes  magic: b'PYRA'
                1 byte   version: 1
                1 byte   algorithm_type: 0=DT, 1=RF, 2=GBDT
                2 bytes  n_features (uint16)
                2 bytes  n_classes (uint16)
                2 bytes  n_trees (uint16)
                4 bytes  default_class (int32)
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
                value:
                    DT:   n_nodes * 4 bytes (int32)
                    RF:   n_nodes * n_classes * 8 bytes (float64)
                    GBDT: n_nodes * 8 bytes (float64)

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
            f.write(b'PYRA')                                      # magic
            f.write(struct.pack('<B', 1))                          # version
            f.write(struct.pack('<B', algo_byte))                  # algorithm_type
            f.write(struct.pack('<H', len(self._array_feature_names)))  # n_features
            f.write(struct.pack('<H', self.num_classes))           # n_classes
            f.write(struct.pack('<H', n_trees))                    # n_trees
            f.write(struct.pack('<i', default_cls))                # default_class

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
                f.write(tree['value'].tobytes())
                f.write(struct.pack('<i', tree.get('max_depth', 50)))

        print(f'[EXPORT] Binary model saved to {filepath} '
              f'({os.path.getsize(filepath)} bytes)')

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
        """
        import struct

        algo_names = {0: 'Decision Tree', 1: 'Random Forest',
                      2: 'Gradient Boosting Decision Trees'}

        with open(filepath, 'rb') as f:
            magic = f.read(4)
            if magic != b'PYRA':
                raise ValueError(f'Invalid binary model file (magic={magic!r})')
            version = struct.unpack('<B', f.read(1))[0]
            if version != 1:
                raise ValueError(f'Unsupported binary model version: {version}')
            algo_byte = struct.unpack('<B', f.read(1))[0]
            n_features = struct.unpack('<H', f.read(2))[0]
            n_classes = struct.unpack('<H', f.read(2))[0]
            n_trees = struct.unpack('<H', f.read(2))[0]
            default_cls = struct.unpack('<i', f.read(4))[0]

            algorithm_type = algo_names[algo_byte]

            # GBDT metadata
            gbdt_is_binary = False
            gbdt_classes = []
            gbdt_init_scores = {}
            if algorithm_type == 'Gradient Boosting Decision Trees':
                gbdt_is_binary = bool(struct.unpack('<B', f.read(1))[0])
                n_gbdt_classes = struct.unpack('<H', f.read(2))[0]
                for _ in range(n_gbdt_classes):
                    gbdt_classes.append(struct.unpack('<i', f.read(4))[0])
                for cl in gbdt_classes:
                    gbdt_init_scores[cl] = struct.unpack('<d', f.read(8))[0]

            # Feature names
            feature_names = []
            for _ in range(n_features):
                name_len = struct.unpack('<H', f.read(2))[0]
                feature_names.append(f.read(name_len).decode('utf-8'))

            # Trees
            tree_arrays = []
            tree_class_groups = []
            tree_is_init = []

            for _ in range(n_trees):
                n_nodes = struct.unpack('<i', f.read(4))[0]

                if algorithm_type == 'Gradient Boosting Decision Trees':
                    is_init = bool(struct.unpack('<B', f.read(1))[0])
                    if is_init:
                        init_score = struct.unpack('<d', f.read(8))[0]
                        cg = struct.unpack('<i', f.read(4))[0]
                        tree_arrays.append({'init_score': init_score})
                        tree_class_groups.append(cg)
                        tree_is_init.append(True)
                        continue
                    else:
                        cg = struct.unpack('<i', f.read(4))[0]
                        tree_class_groups.append(cg)
                        tree_is_init.append(False)

                feature_idx = np.frombuffer(f.read(n_nodes * 4), dtype=np.int32).copy()
                threshold_arr = np.frombuffer(f.read(n_nodes * 8), dtype=np.float64).copy()
                children_left = np.frombuffer(f.read(n_nodes * 4), dtype=np.int32).copy()
                children_right = np.frombuffer(f.read(n_nodes * 4), dtype=np.int32).copy()

                if algorithm_type == 'Decision Tree':
                    value_arr = np.frombuffer(f.read(n_nodes * 4), dtype=np.int32).copy()
                elif algorithm_type == 'Random Forest':
                    value_arr = np.frombuffer(
                        f.read(n_nodes * n_classes * 8), dtype=np.float64
                    ).reshape(n_nodes, n_classes).copy()
                else:  # GBDT
                    value_arr = np.frombuffer(f.read(n_nodes * 8), dtype=np.float64).copy()

                tree_arrays.append({
                    'feature_idx': feature_idx,
                    'threshold': threshold_arr,
                    'children_left': children_left,
                    'children_right': children_right,
                    'value': value_arr,
                    'max_depth': struct.unpack('<i', f.read(4))[0],
                })

        # Build a minimal RuleClassifier instance
        # We use __new__ to bypass __init__ (no Rule objects needed)
        obj = cls.__new__(cls)
        obj.algorithm_type = algorithm_type
        obj.num_classes = n_classes
        obj.class_labels = list(range(n_classes))
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

        return obj

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

        a(f'#ifndef {guard_name}')
        a(f'#define {guard_name}')
        a('')
        a('/* Auto-generated by pyruleanalyzer — do not edit */')
        a('')
        a('#include <stdint.h>')
        a('')
        a(f'#define N_FEATURES {len(self._array_feature_names)}')
        a(f'#define N_CLASSES  {self.num_classes}')
        a(f'#define N_TREES    {len(self._tree_arrays)}')
        a(f'#define DEFAULT_CLASS {default_cls}')
        a('')

        # Feature name comments
        a('/* Feature order:')
        for i, name in enumerate(self._array_feature_names):
            a(f'   [{i}] {name}')
        a('*/')
        a('')

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

            # threshold
            vals = ', '.join(f'{v:.17g}' for v in tree['threshold'])
            a(f'static const double tree{t_idx}_threshold[{n_nodes}] = {{{vals}}};')

            # children_left
            vals = ', '.join(str(v) for v in tree['children_left'])
            a(f'static const int32_t tree{t_idx}_left[{n_nodes}] = {{{vals}}};')

            # children_right
            vals = ', '.join(str(v) for v in tree['children_right'])
            a(f'static const int32_t tree{t_idx}_right[{n_nodes}] = {{{vals}}};')

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
        a('static inline int32_t traverse_tree(')
        a('    const double *features,')
        a('    const int32_t *feat_idx,')
        a('    const double *thresh,')
        a('    const int32_t *left,')
        a('    const int32_t *right,')
        a('    int max_depth')
        a(') {')
        a('    int32_t node = 0;')
        a('    int d;')
        a('    for (d = 0; d < max_depth; d++) {')
        a('        if (left[node] == node) break;  /* leaf: self-loop */')
        a('        if (features[feat_idx[node]] <= thresh[node])')
        a('            node = left[node];')
        a('        else')
        a('            node = right[node];')
        a('    }')
        a('    return node;')
        a('}')
        a('')

        # Main predict function depends on algorithm type
        if self.algorithm_type == 'Decision Tree':
            a('static inline int32_t predict(const double *features) {')
            a('    int32_t leaf = traverse_tree(features, tree0_feature, tree0_threshold, tree0_left, tree0_right, TREE0_DEPTH);')
            a('    return tree0_value[leaf];')
            a('}')
        elif self.algorithm_type == 'Random Forest':
            a('static inline int32_t predict(const double *features) {')
            a(f'    double prob_sum[N_CLASSES] = {{0}};')
            for t_idx in range(len(self._tree_arrays)):
                a(f'    {{')
                a(f'        int32_t leaf = traverse_tree(features, tree{t_idx}_feature, tree{t_idx}_threshold, tree{t_idx}_left, tree{t_idx}_right, TREE{t_idx}_DEPTH);')
                a(f'        double total = 0.0;')
                a(f'        int k;')
                a(f'        for (k = 0; k < N_CLASSES; k++) total += tree{t_idx}_value[leaf * N_CLASSES + k];')
                a(f'        if (total > 0.0) {{')
                a(f'            for (k = 0; k < N_CLASSES; k++) prob_sum[k] += tree{t_idx}_value[leaf * N_CLASSES + k] / total;')
                a(f'        }}')
                a(f'    }}')
            a(f'    int32_t best = 0;')
            a(f'    int k;')
            a(f'    for (k = 1; k < N_CLASSES; k++) {{')
            a(f'        if (prob_sum[k] > prob_sum[best]) best = k;')
            a(f'    }}')
            a(f'    return best;')
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
                a(f'    double prob = 1.0 / (1.0 + exp(-score));')
                a(f'    return (prob >= 0.5) ? {int(classes[1])} : {int(classes[0])};')
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
            f.write('\n'.join(lines))

        print(f'[EXPORT] C header saved to {filepath} '
              f'({os.path.getsize(filepath)} bytes)')

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
            # Create a canonical signature for the rule's logic
            # 1. We use parsed_conditions directly (no string parsing)
            # 2. We sort the conditions so that logic order doesn't matter
            # 3. We use the class label
            
            # signature format: (class_label, tuple_of_sorted_conditions)
            # condition format: (var, op, value) - already float from parsing
            sorted_conditions = tuple(sorted(rule.parsed_conditions))
            signature = (rule.class_, sorted_conditions)
            
            rules_by_signature[signature].append(rule)
            
        # Return only the groups that have more than one similar rule
        return [group for group in rules_by_signature.values() if len(group) > 1]
    
    # Method to find duplicated rules within the same tree (Boundary Redundancy)
    def find_duplicated_rules(self, type='soft'):
        """
        Identifies nearly identical rules within the same decision tree context.

        This method searches for rule pairs that:
        - Have the same class label.
        - Share all conditions except the last one (same path parent).
        - Differ only in the final condition boundary (e.g., v1 <= 5 vs v1 > 5).

        Such pairs are considered duplicates because they imply the split at that 
        boundary was unnecessary for the final classification outcome.

        Args:
            type (str): Strictness level ('soft' or 'medium').

        Returns:
            List[Tuple[Rule,Rule]]: A list of tuples, each representing a pair of duplicated rules.
        """
        duplicated_rules = []
        rules_by_prefix = defaultdict(list)
        
        # Target rules to analyze
        target_rules = self.final_rules if self.final_rules else self.initial_rules

        # 1. Group rules by class and their "Prefix" (all conditions except the last)
        for rule in target_rules:
            if not rule.parsed_conditions:
                continue
            
            # Prefix signature: (class, conditions[0:-1])
            # We assume order matters here because it represents a tree path
            prefix_key = (rule.class_, tuple(rule.parsed_conditions[:-1]))
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
                    if abs(val1 - val2) > 1e-9:
                        continue
                    
                    # Check 3: Check for complementary operators
                    op_pair = {op1, op2}
                    is_duplicate = False
                    
                    if type == 'soft':
                        # Standard complements: (<= vs >) or (< vs >=)
                        if op_pair in [{'<=', '>'}, {'<', '>='}]:
                            is_duplicate = True
                    elif type == 'medium':
                        # Broader definition allowing overlaps or loose boundaries
                        if op_pair in [{'<=', '>'}, {'<', '>='}, {'<', '>'}, {'>=', '<'}, {'<=', '<'}]:
                            is_duplicate = True

                    # GBDT: siblings must also have the same leaf_value
                    # to be considered redundant (different leaf values change
                    # the residual sum and affect predictions)
                    if is_duplicate and self.algorithm_type == 'Gradient Boosting Decision Trees':
                        if (rule1.leaf_value is not None and rule2.leaf_value is not None
                                and abs(rule1.leaf_value - rule2.leaf_value) > 1e-9):
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
    def adjust_and_remove_rules(self, method):
        """
        Adjusts and removes duplicated rules from the rule set based on the specified method.

        This method analyzes the current rule set to identify duplicates. It creates generalized 
        rules by merging sibling nodes (soft) or representative rules for inter-tree duplicates (hard).

        Args:
            method (str): Strategy for rule refinement. Must be "custom", "soft", or "hard".

        Returns:
            Tuple[List[Rule], List[Tuple[Rule, Rule]]]: 
                - A new list of rules (merged + remaining).
                - A list of identified duplicated pairs (used for convergence checks).
        """
        if method == "custom":
            return self.custom_rule_removal(self.initial_rules)
        
        if method not in ["soft", "medium", "hard", "custom"]:
            raise ValueError(f"Invalid method: {method}. Use 'soft', 'medium', 'hard' or 'custom'.")

        # Determine source rules: Use final_rules if populated (iteration n), else initial (iteration 0)
        source_rules = self.final_rules if self.final_rules else self.initial_rules

        # 1. Soft/Medium Check: Boundary Redundancy within the same tree
        # This identifies siblings that can be merged into their parent
        similar_rules_soft = self.find_duplicated_rules(type=method if method in ['soft', 'medium'] else 'soft')
        
        rules_to_remove = set()
        new_generalized_rules = []

        if similar_rules_soft:
            # Silence per-iteration print to avoid clutter if desired, or keep it.
            # user asked for "Merging X pairs..." to be consolidated, but seeing progress is also good.
            # I'll keep the granular print but the summary is what counts.
            print(f"Merging {len(similar_rules_soft)} pairs of duplicated rules...")

        for rule1, rule2 in similar_rules_soft:
            rules_to_remove.add(rule1)
            rules_to_remove.add(rule2)

            # Create a new generalized rule (Parent Logic)
            # We strip the last condition which differentiated the two siblings
            common_conditions = rule1.conditions[:-1]
            
            new_rule_name = f"{rule1.name}_&_{rule2.name}"
            if self.algorithm_type == 'Gradient Boosting Decision Trees':
                new_rule = Rule(new_rule_name, rule1.class_, common_conditions,
                                leaf_value=rule1.leaf_value,
                                learning_rate=rule1.learning_rate,
                                class_group=rule1.class_group)
            else:
                # Combine class distributions: element-wise sum of raw counts
                # This correctly represents the parent node's distribution
                combined_dist = None
                merged_class = rule1.class_
                if rule1.class_distribution is not None and rule2.class_distribution is not None:
                    combined_dist = [
                        a + b for a, b in zip(rule1.class_distribution, rule2.class_distribution)
                    ]
                    # Update class_ to reflect the majority class of the combined distribution
                    best_idx = combined_dist.index(max(combined_dist))
                    merged_class = str(best_idx)
                elif rule1.class_distribution is not None:
                    combined_dist = list(rule1.class_distribution)
                elif rule2.class_distribution is not None:
                    combined_dist = list(rule2.class_distribution)
                
                new_rule = Rule(new_rule_name, merged_class, common_conditions,
                                class_distribution=combined_dist)
            
            # CRITICAL: Parse immediately so this rule is ready for the next analysis iteration
            new_rule.parsed_conditions = self.parse_conditions_static(new_rule.conditions)
            
            new_generalized_rules.append(new_rule)

        # 2. Hard Check: Semantic Redundancy between trees
        if method == "hard":
            print("Analyzing duplicated rules between trees...")
            if self.algorithm_type == 'Random Forest':
                similar_rule_groups = self.find_duplicated_rules_between_trees()
                
                for group in similar_rule_groups:
                    # Mark all for removal
                    for rule in group:
                        rules_to_remove.add(rule)
                    
                    # Create ONE representative rule from the group
                    # We pick the first one as the template
                    representative = group[0]
                    
                    # Merge names for traceability
                    new_name = "_&_".join(sorted([r.name for r in group]))
                    
                    # Create representative rule — preserve class_distribution
                    new_rule = Rule(new_name, representative.class_, representative.conditions,
                                    class_distribution=representative.class_distribution)
                    new_rule.parsed_conditions = self.parse_conditions_static(new_rule.conditions)
                    
                    new_generalized_rules.append(new_rule)

        # 3. Rebuild the Rule Set
        # Keep rules that were NOT marked for removal + the newly created generalized rules
        final_list = new_generalized_rules + [r for r in source_rules if r not in rules_to_remove]
        
        return final_list, similar_rules_soft
    
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
            if c_var == last_var and c_op == expected_op and abs(c_val - last_val) < 1e-9:
                return candidate

        return None

    # Method to promote sibling when removing specific rules
    def _promote_siblings(self, rules_to_remove, all_rules):
        """
        Promotes sibling rules when their counterparts are removed.

        When a low-usage rule is removed, its sibling (the other child of
        the same parent node) becomes the sole representative of that region.
        The sibling's last condition (which only existed to distinguish it
        from the removed rule) is stripped, effectively promoting it to the
        parent node's level.

        **Critical**: Rules are processed from deepest to shallowest. This
        ensures that when a removed rule's sibling is not a direct leaf but
        a sub-branch (i.e., its sibling at the same depth doesn't exist as
        a single rule), the deeper removals and promotions happen first,
        potentially creating the sibling that the shallower removal needs.

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

            If processed shallowest first, step 2 would fail (no direct sibling found).

        Args:
            rules_to_remove (List[Rule]): Rules identified for removal.
            all_rules (List[Rule]): The complete set of current rules.

        Returns:
            List[Rule]: The adjusted rule set after promotions.
        """
        # Work on a mutable copy
        working_rules = list(all_rules)
        promoted_count = 0

        # Sort rules to remove by depth (deepest first)
        sorted_to_remove = sorted(
            rules_to_remove,
            key=lambda r: len(r.parsed_conditions),
            reverse=True
        )

        # Track which rules have been removed by identity
        removed_ids = set()

        for rule in sorted_to_remove:
            if id(rule) in removed_ids:
                continue

            # Find the sibling of the rule being removed
            sibling = self._find_sibling(rule, working_rules)

            if sibling is not None and id(sibling) not in removed_ids:
                # Promote the sibling: remove its last condition
                sibling.conditions = sibling.conditions[:-1]
                sibling.parsed_conditions = sibling.parsed_conditions[:-1]

                # Combine class_distribution: the promoted sibling now covers
                # the region of the removed rule as well, so add their counts
                if (hasattr(rule, 'class_distribution') and rule.class_distribution is not None
                        and hasattr(sibling, 'class_distribution') and sibling.class_distribution is not None):
                    sibling.class_distribution = [
                        a + b for a, b in zip(sibling.class_distribution, rule.class_distribution)
                    ]
                    # Update class_ to match the new majority class
                    best_idx = sibling.class_distribution.index(max(sibling.class_distribution))
                    sibling.class_ = str(best_idx)

                # Update its name to indicate promotion
                if "_promoted" not in sibling.name:
                    sibling.name = f"{sibling.name}_promoted"

                promoted_count += 1
            # If no sibling is found, the rule is simply removed.
            # Samples that would have matched it will fall through to
            # default_class (existing fallback behavior).

            # Remove the rule from working set
            removed_ids.add(id(rule))
            working_rules = [r for r in working_rules if id(r) != id(rule)]

        if promoted_count > 0:
            print(f"Promoted {promoted_count} sibling rule(s) after specific rule removal.")

        return working_rules

    # Exports the rule set to a standalone native Python classifier file
    def export_to_native_python(self, feature_names=None, filename="examples/files/fast_classifier.py"):
        """
        Generates a standalone Python file with the decision logic.
        
        For Decision Trees, it exports a single nested if/else function.
        For Random Forests, it exports multiple functions (one per tree) and a voting aggregator.

        Args:
            feature_names (List[str], optional): Kept for compatibility.
            filename (str): Output filename.
        """
        print(f"[EXPORT] Generating native classifier: {filename}")
        
        rules_to_export = self.final_rules if self.final_rules else self.initial_rules
        
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

                # Collect init scores and write tree functions
                init_scores = {}
                for tid, rules in tree_rules_map.items():
                    if not rules:
                        continue
                    if len(rules) == 1 and not rules[0].parsed_conditions and rules[0].contribution is not None:
                        init_scores[rules[0].class_group] = rules[0].contribution
                        continue

                    func_name = f'predict_{tid}'
                    tree_dict = {0: {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': 0.0}}
                    next_id = 1
                    for rule in rules:
                        curr = 0
                        for var, op, val in rule.parsed_conditions:
                            if tree_dict[curr]['f'] == -2:
                                tree_dict[curr].update({'f': var, 't': val, 'l': next_id, 'r': next_id + 1})
                                tree_dict[next_id] = {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': 0.0}
                                tree_dict[next_id + 1] = {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': 0.0}
                                next_id += 2
                            if op in ['<=', '<']:
                                curr = tree_dict[curr]['l']
                            else:
                                curr = tree_dict[curr]['r']
                        tree_dict[curr]['v'] = rule.contribution if rule.contribution is not None else 0.0

                    def _build_export_code(node_id, indent, td=tree_dict):
                        node = td[node_id]
                        tab = '    ' * indent
                        if node['f'] == -2:
                            return f'{tab}return {repr(float(node["v"]))}\n'
                        feat = node['f']
                        code = f"{tab}if sample.get('{feat}', 0) <= {repr(float(node['t']))}:\n"
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
                        f.write(f'    score += predict_{tid}(sample)\n')
                    f.write('    prob = 1.0 / (1.0 + math.exp(-score))\n')
                    f.write(f'    return {int(classes[1])} if prob >= 0.5 else {int(classes[0])}\n')
                else:
                    for class_label in classes:
                        score_init = init_scores.get(class_label, 0.0)
                        f.write(f'    score_{class_label} = {repr(score_init)}\n')
                    for tid, rules in tree_rules_map.items():
                        if not rules:
                            continue
                        if len(rules) == 1 and not rules[0].parsed_conditions:
                            continue
                        class_group = rules[0].class_group
                        f.write(f'    score_{class_group} += predict_{tid}(sample)\n')
                    score_vars = [f'score_{cl}' for cl in classes]
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
                for tid, rules in tree_rules_map.items():
                    func_name = f"predict_{tid}"
                    tree_func_names.append(func_name)
                    
                    # Build tree structure — store probability distributions at leaves
                    tree_dict = {0: {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': None}}
                    next_id = 1
                    
                    for rule in rules:
                        curr = 0
                        for var, op, val in rule.parsed_conditions:
                            if tree_dict[curr]['f'] == -2:
                                tree_dict[curr].update({'f': var, 't': val, 'l': next_id, 'r': next_id + 1})
                                tree_dict[next_id] = {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': None}
                                tree_dict[next_id + 1] = {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': None}
                                next_id += 2
                            if op in ['<=', '<']:
                                curr = tree_dict[curr]['l']
                            else:
                                curr = tree_dict[curr]['r']
                        
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
                        node = tree_dict[node_id]
                        tab = "    " * indent
                        if node['f'] == -2:
                            val = node['v']
                            if val is None:
                                return f"{tab}return None\n" 
                            return f"{tab}return {repr(val)}\n"
                        
                        feat = node['f']
                        # Use repr(float(t)) for precision
                        code = f"{tab}if sample.get('{feat}', 0) <= {repr(float(node['t']))}:\n"
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
                
                if isinstance(default_val, str):
                    def_str = f"'{default_val}'"
                else:
                    def_str = str(default_val)

                f.write("def predict(sample):\n")
                f.write("    tree_results = []\n")
                for func in tree_func_names:
                    f.write(f"    v = {func}(sample)\n")
                    f.write("    if v is not None: tree_results.append(v)\n")
                f.write(f"    if not tree_results: return {def_str}\n")
                
                if n_classes > 0:
                    f.write("    # Soft voting: average probability distributions\n")
                    f.write("    if isinstance(tree_results[0], list):\n")
                    f.write(f"        n_classes = {n_classes}\n")
                    f.write("        avg = [0.0] * n_classes\n")
                    f.write("        for proba in tree_results:\n")
                    f.write("            for j in range(n_classes):\n")
                    f.write("                avg[j] += proba[j]\n")
                    f.write("        n = len(tree_results)\n")
                    f.write("        avg = [p / n for p in avg]\n")
                    f.write("        return avg.index(max(avg))\n")
                    f.write("    else:\n")
                    f.write("        from collections import Counter\n")
                    f.write("        return Counter(tree_results).most_common(1)[0][0]\n")
                else:
                    f.write("    from collections import Counter\n")
                    f.write("    return Counter(tree_results).most_common(1)[0][0]\n")

            # --- Decision Tree Export Strategy (Single Tree) ---
            else:
                tree_dict = {0: {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': -1}}
                next_id = 1
                for rule in rules_to_export:
                    curr = 0
                    for var, op, val in rule.parsed_conditions:
                        if tree_dict[curr]['f'] == -2:
                            tree_dict[curr].update({'f': var, 't': val, 'l': next_id, 'r': next_id + 1})
                            tree_dict[next_id] = {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': -1}
                            tree_dict[next_id + 1] = {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': -1}
                            next_id += 2
                        if op in ['<=', '<']:
                            curr = tree_dict[curr]['l']
                        else:
                            curr = tree_dict[curr]['r']
                    
                    try:
                        clean_class = int(str(rule.class_).replace('Class', '').strip())
                    except Exception:
                        clean_class = rule.class_
                    tree_dict[curr]['v'] = clean_class

                def build_dt_code(node_id, indent):
                    node = tree_dict[node_id]
                    tab = "    " * indent
                    if node['f'] == -2:
                        val = node['v']
                        if val == -1:
                            try:
                                val = int(str(self.default_class).replace('Class', '').strip())
                            except Exception:
                                val = self.default_class
                        if isinstance(val, str):
                            return f"{tab}return '{val}'\n"
                        return f"{tab}return {val}\n"
                    
                    feat = node['f']
                    # Use repr(float(t)) for precision
                    code = f"{tab}if sample.get('{feat}', 0) <= {repr(float(node['t']))}:\n"
                    code += build_dt_code(node['l'], indent + 1)
                    code += f"{tab}else:\n"
                    code += build_dt_code(node['r'], indent + 1)
                    return code

                f.write("def predict(sample):\n")
                f.write(build_dt_code(0, 1))
        
        print(f"[EXPORT] File '{filename}' generated.")
        
    # Method to execute the rule analysis and identify duplicated rules
    def execute_rule_analysis(self, file_path, remove_duplicates="none", remove_below_n_classifications=-1):
        """
        Executes a full rule evaluation and pruning process on a given dataset.

        This method:
        - Applies optional duplicate rule removal (iteratively until convergence).
        - Recompiles the native Python model for speed.
        - Runs evaluation using the appropriate algorithm via DTAnalyzer/RFAnalyzer/GBDTAnalyzer.
        - Optionally removes rules used less than or equal to a given threshold.
        - Tracks and prints redundancy metrics by type.

        Args:
            file_path (str): Path to the CSV file containing data for evaluation.
            remove_duplicates (str): Strategy ("soft", "hard", "custom", "none").
            remove_below_n_classifications (int): Threshold for pruning low-usage rules.
        """
        from .dt_analyzer import DTAnalyzer
        from .rf_analyzer import RFAnalyzer
        from .gbdt_analyzer import GBDTAnalyzer

        print("\n" + "*"*80)
        print("EXECUTING RULE ANALYSIS")
        print("*"*80 + "\n")

        # Start with a copy of initial rules
        self.final_rules = list(self.initial_rules)

        # 1. Create the appropriate analyzer
        if self.algorithm_type == 'Decision Tree':
            analyzer = DTAnalyzer(self)
        elif self.algorithm_type == 'Random Forest':
            analyzer = RFAnalyzer(self)
        elif self.algorithm_type == 'Gradient Boosting Decision Trees':
            analyzer = GBDTAnalyzer(self)
        else:
            raise ValueError(f"Unsupported algorithm type: {self.algorithm_type}")

        # Store analyzer for later use (e.g., compare_initial_final_results)
        self._analyzer = analyzer

        # 2. Duplicate Removal Loop
        all_intra_pairs = []
        all_inter_groups = []

        if remove_duplicates != "none":
            iteration = 1
            total_merged = 0
            while True:
                prev_count = len(self.final_rules)

                # Track inter-tree groups separately for RF "hard" method
                inter_tree_groups = []
                if remove_duplicates == "hard" and self.algorithm_type == 'Random Forest':
                    inter_tree_groups = self.find_duplicated_rules_between_trees()
                    all_inter_groups.extend(inter_tree_groups)

                self.final_rules, self.duplicated_rules = self.adjust_and_remove_rules(remove_duplicates)
                
                merged_count = len(self.duplicated_rules)
                total_merged += merged_count
                all_intra_pairs.extend(self.duplicated_rules)
                
                # Convergence check: stop if no duplicates found or count didn't change
                if not self.duplicated_rules or len(self.final_rules) == prev_count:
                    break
                iteration += 1
            
            print(f"Total duplicated rules merged/removed: {total_merged}")

        # 3. Update redundancy counters on the analyzer
        if isinstance(analyzer, DTAnalyzer):
            analyzer.track_intra_tree_from_duplicates(all_intra_pairs)
        elif isinstance(analyzer, RFAnalyzer):
            analyzer.track_from_adjust_and_remove(
                remove_duplicates, all_intra_pairs, all_inter_groups
            )
        elif isinstance(analyzer, GBDTAnalyzer):
            analyzer.track_from_adjust_and_remove(
                remove_duplicates, all_intra_pairs
            )

        # 4. CRITICAL OPTIMIZATION: Update native model immediately
        self.update_native_model(self.final_rules)

        # 5. Delegate to the specific analyzer for evaluation + low-usage pruning
        analyzer.execute_rule_analysis(file_path, remove_below_n_classifications)

    # Method to execute the rule analysis for Decision Tree
    def execute_rule_analysis_dt(self, file_path, remove_below_n_classifications=-1):
        """Delegates to DTAnalyzer. Kept for backward compatibility.

        Args:
            file_path (str): Path to the CSV file.
            remove_below_n_classifications (int): Minimum usage count threshold.
        """
        from .dt_analyzer import DTAnalyzer
        analyzer = DTAnalyzer(self)
        analyzer.execute_rule_analysis(file_path, remove_below_n_classifications)

    # Method to execute the rule analysis for Random Forest
    def execute_rule_analysis_rf(self, file_path, remove_below_n_classifications=-1):
        """Delegates to RFAnalyzer. Kept for backward compatibility.

        Args:
            file_path (str): Path to the CSV file.
            remove_below_n_classifications (int): Minimum usage count threshold.
        """
        from .rf_analyzer import RFAnalyzer
        analyzer = RFAnalyzer(self)
        analyzer.execute_rule_analysis(file_path, remove_below_n_classifications)

    # Helper to save the final model state
    def _save_final_model(self):
        """Helper to save the current state of the classifier."""
        try:
            with open('examples/files/final_model.pkl', 'wb') as model_file:
                pickle.dump(self, model_file)
            print("Final classifier saved to 'examples/files/final_model.pkl'.")
        except Exception as e:
            print(f"Error saving model: {e}")

    # Helper to write the analysis report
    def _write_report(self, filename, file_path, correct, total, prune_thresh):
        """Helper to write the consolidated text report."""
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
        Computes a normalized complexity score and other interpretability metrics for a rule set.

        This method calculates a comprehensive set of metrics, including:
        - A novel 'Structural Complexity Score' based on depth balance and attribute usage.
        - Traditional metrics like rule counts and depth statistics.

        The primary score combines two dimensions:
        1.  Depth Balance (D_bal): Ratio of mean rule depth to max depth.
            D_bal = mean_rule_depth / max_depth
            (Values closer to 1 indicate a balanced tree structure).

        2.  Normalized Attribute Usage (A_norm): Measures feature diversity across rules.
            A_norm = sum(unique_attributes_per_rule) / (total_rules * n_features_total)
            (Higher values indicate the model leverages a wider range of features).

        The final 'complexity_score' (D_bal * A_norm) rewards models that are
        both structurally balanced and diverse in feature usage.

        Args:
            rules (List[Rule]): A list of Rule objects to analyze.
            n_features_total (int): The total number of available features in the dataset.

        Returns:
            Dict[str, Any]: A dictionary containing detailed complexity metrics.
        """
        if not rules:
            return {
                "total_rules": 0,
                "mean_rule_depth": 0.0,
                "max_depth": 0,
                "depth_balance": 0.0,
                "attribute_usage_norm": 0.0,
                "complexity_score": 0.0,
            }

        total_rules = len(rules)
        
        # Accumulators for single-pass analysis
        rule_depths = []
        all_features_used = set()
        sum_unique_attr_per_rule = 0
        
        # Helper for Random Forest analysis: maps tree_name -> list of rule depths
        tree_stats = defaultdict(list)

        for rule in rules:
            # 1. Depth Calculation
            depth = len(rule.conditions)
            rule_depths.append(depth)

            # 2. Attribute Usage (using cached parsed_conditions)
            # item format: (var, op, value)
            unique_features_in_rule = {item[0] for item in rule.parsed_conditions}
            all_features_used.update(unique_features_in_rule)
            sum_unique_attr_per_rule += len(unique_features_in_rule)

            # 3. Random Forest Grouping (on-the-fly)
            # Assumes naming convention "DTx_..."
            if '_' in rule.name:
                tree_id = rule.name.split('_')[0]
                tree_stats[tree_id].append(depth)

        # --- Metrics Calculation ---
        
        # Depth Statistics
        max_depth = max(rule_depths) if rule_depths else 0
        mean_rule_depth = sum(rule_depths) / total_rules if total_rules > 0 else 0.0
        n_features_used = len(all_features_used)

        # 1. Depth Balance (D_bal)
        depth_balance = (mean_rule_depth / max_depth) if max_depth > 0 else 0.0

        # 2. Normalized Attribute Usage (A_norm)
        denominator = total_rules * n_features_total
        attribute_usage_norm = (sum_unique_attr_per_rule / denominator) if denominator > 0 else 0.0

        # 3. Final Complexity Score
        complexity_score = depth_balance * attribute_usage_norm

        result = {
            "total_rules": total_rules,
            "features_used": n_features_used,
            "total_features": n_features_total,
            "max_depth": max_depth,
            "mean_rule_depth": float(mean_rule_depth),
            "depth_balance": float(depth_balance),
            "attribute_usage_norm": float(attribute_usage_norm),
            "complexity_score": float(complexity_score),
        }

        # --- Extra metrics for Random Forest ---
        # Calculate only if multiple trees were detected
        if len(tree_stats) > 1:
            num_trees = len(tree_stats)
            
            # tree_stats values are lists of depths for each tree
            avg_rules_per_tree = sum(len(depths) for depths in tree_stats.values()) / num_trees
            avg_max_depth_per_tree = sum(max(depths) for depths in tree_stats.values()) / num_trees
            
            # Mean depth per tree, then average of those means
            avg_mean_depth_per_tree = sum((sum(d) / len(d)) for d in tree_stats.values()) / num_trees

            result.update({
                "average_rules_per_tree": float(avg_rules_per_tree),
                "average_max_depth_per_tree": float(avg_max_depth_per_tree),
                "average_rule_depth_per_tree": float(avg_mean_depth_per_tree),
            })

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
    def compare_initial_final_results(self, file_path):
        """
        Compares the classification performance of the initial and final rule sets.

        Delegates to DTAnalyzer, RFAnalyzer, or GBDTAnalyzer.  If ``execute_rule_analysis``
        was called earlier in the same session, the cached ``_analyzer`` is
        reused; otherwise a fresh one is created.

        Args:
            file_path (str): Path to the CSV file used for evaluation.
        """
        from .dt_analyzer import DTAnalyzer
        from .rf_analyzer import RFAnalyzer
        from .gbdt_analyzer import GBDTAnalyzer

        # Reuse the analyzer created during execute_rule_analysis when possible
        if hasattr(self, '_analyzer') and self._analyzer is not None:
            self._analyzer.compare_initial_final_results(file_path)
        elif self.algorithm_type == 'Random Forest':
            RFAnalyzer(self).compare_initial_final_results(file_path)
        elif self.algorithm_type == 'Decision Tree':
            DTAnalyzer(self).compare_initial_final_results(file_path)
        elif self.algorithm_type == 'Gradient Boosting Decision Trees':
            GBDTAnalyzer(self).compare_initial_final_results(file_path)
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
                        
                    # Re-validate structure just in case
                    selected_rule.parsed_conditions = self.parse_conditions_static(selected_rule.conditions)
                    
                    self.update_native_model(self.final_rules if self.final_rules else self.initial_rules)
                    
                    # Save to disk
                    try:
                        with open('examples/files/edited_model.pkl', 'wb') as f_out:
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

    # Method to process data
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
                
                # Left child (<=)
                # We append the exact float 'threshold' to the parsed list, avoiding string-to-float conversion errors later
                recurse(
                    tree_.children_left[node], 
                    conditions_str + [f"{name} <= {threshold_str}"],
                    conditions_parsed + [(name, '<=', threshold)] 
                )
                
                # Right child (>)
                recurse(
                    tree_.children_right[node], 
                    conditions_str + [f"{name} > {threshold_str}"],
                    conditions_parsed + [(name, '>', threshold)]
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
        classes = [str(c) for c in model.classes_]
        is_binary = len(classes) == 2
        learning_rate = model.get_params()['learning_rate']

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
            init = model.init_
            if init == 'zero':
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
                tree_rules = []

                def recurse(node_id, conditions_str, conditions_parsed):
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

                    # Left child (<=)
                    recurse(
                        left,
                        conditions_str + [f'{feat_name} <= {threshold_str}'],
                        conditions_parsed + [(feat_name, '<=', threshold)],
                    )
                    # Right child (>)
                    recurse(
                        right,
                        conditions_str + [f'{feat_name} > {threshold_str}'],
                        conditions_parsed + [(feat_name, '>', threshold)],
                    )

                recurse(0, [], [])
                all_rules.extend(tree_rules)

        return all_rules, init_scores, is_binary, classes

    # Method to generate a classifier model based on rules
    @staticmethod
    def generate_classifier_model(rules, class_names_map, algorithm_type='Random Forest'):
        """
        Instantiates a RuleClassifier from extracted rules and saves it.

        Optimization: 
        This method now passes the Rule objects directly to the constructor, avoiding 
        the intermediate string serialization that caused floating-point precision loss.

        Args:
            rules (Union[List[Rule], List[List[Rule]]]): The extracted rules.
            class_names_map (Dict[str, int]): Map of class names to indices (kept for compatibility).
            algorithm_type (str): 'Random Forest' or 'Decision Tree'.

        Returns:
            RuleClassifier: The initialized classifier.
        """
        # Direct instantiation using the list of Rule objects.
        # This preserves the high-precision 'parsed_conditions' created in get_rules.
        classifier = RuleClassifier(rules, algorithm_type=algorithm_type)
        
        print(f"Algorithm Type: {classifier.algorithm_type}")

        path = 'examples/files/initial_model.pkl'
        try:
            with open(path, 'wb') as model_file:
                pickle.dump(classifier, model_file)
            print(f"Classifier file saved: {path}")
        except Exception as e:
            print(f"Warning: Could not save classifier to {path}. Error: {e}")

        return classifier
    
    # Method to create a new classifier
    @staticmethod
    def new_classifier(train_path, test_path, model_parameters, model_path=None, algorithm_type='Random Forest'):
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
            model_save_path = 'examples/files/sklearn_model.pkl'
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

            # Compile native model now that metadata is set
            classifier.update_native_model(classifier.initial_rules)

            # Save initial model (after metadata is set)
            path = 'examples/files/initial_model.pkl'
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
        classifier = RuleClassifier.generate_classifier_model(rules, class_names_map, algorithm_type)

        return classifier