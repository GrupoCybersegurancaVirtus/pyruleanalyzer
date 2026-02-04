import os
import time
import pickle
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Union, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

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

    __slots__ = ['name', 'class_', 'conditions', 'usage_count', 'error_count', 'parsed_conditions']

    def __init__(self, name, class_, conditions):
        """
        Initializes a new Rule instance representing a decision path.

        Args:
            name (str): Unique identifier for the rule.
            class_ (str): Target class label as a string.
            conditions (List[str]): List of conditions required to trigger this rule.
        """
        self.name = name
        self.class_ = class_
        self.conditions = conditions
        self.usage_count = 0
        self.error_count = 0
        self.parsed_conditions = []  # Populated by parse_conditions_static

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
            algorithm_type (str): Type of model ('Decision Tree' or 'Random Forest').
        """
        # Parse raw rules into Rule objects
        self.initial_rules = self.parse_rules(rules, algorithm_type)
        self.algorithm_type = algorithm_type
        
        # Initialize lists
        self.final_rules = []
        self.duplicated_rules = []
        self.specific_rules = []
        
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
        self.native_fn = None
        self.update_native_model(self.initial_rules)

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
                if not line: continue
                
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
                    except:
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
        # Tree structure: {id: {'l': left, 'r': right, 'f': feature, 't': threshold, 'v': value}}
        # 'f': -2 indicates a leaf node
        tree_dict = {0: {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': -1}}
        next_id = 1
        
        for rule in rules_to_compile:
            curr = 0
            for var, op, val in rule.parsed_conditions:
                # If current node is a leaf, transform it into a decision node
                if tree_dict[curr]['f'] == -2:
                    tree_dict[curr].update({'f': var, 't': val, 'l': next_id, 'r': next_id + 1})
                    # Create children
                    tree_dict[next_id] = {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': -1}
                    tree_dict[next_id + 1] = {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': -1}
                    next_id += 2
                
                # Traverse
                if op in ['<=', '<']:
                    curr = tree_dict[curr]['l']
                else:
                    curr = tree_dict[curr]['r']
            
            # Assign leaf class (ensure int if possible)
            try:
                clean_class = int(str(rule.class_).replace('Class', '').strip())
            except:
                clean_class = rule.class_
            tree_dict[curr]['v'] = clean_class

        # Recursive function to build Python code string
        def build_code(node_id, indent):
            node = tree_dict[node_id]
            tab = "    " * indent
            
            # Leaf Node
            if node['f'] == -2: 
                val = node['v']
                # Handle empty leaves (default class)
                if val == -1: 
                    try:
                        val = int(str(self.default_class).replace('Class', '').strip())
                    except:
                        val = self.default_class
                # Return tuple (class, votes_placeholder, proba_placeholder)
                return f"{tab}return {val}, None, None\n"
            
            # Decision Node
            feat_name = node['f']
            code = f"{tab}if sample.get('{feat_name}', 0) <= {node['t']}:\n"
            code += build_code(node['l'], indent + 1)
            code += f"{tab}else:\n"
            code += build_code(node['r'], indent + 1)
            return code

        # Generate the full function code
        func_code = "def native_predict(sample):\n"
        func_code += build_code(0, 1)

        # Compile logic into the local namespace
        context = {}
        try:
            exec(func_code, {}, context)
            self.native_fn = context['native_predict']
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
        # --- FAST PATH: Native Execution ---
        # If we are using initial rules and the native function exists, use it.
        # This bypasses the slow iteration over Rule objects.
        if not final and self.native_fn is not None:
            try:
                # native_fn returns (prediction, votes, proba)
                return self.native_fn(data)
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
        Aggregates votes from individual trees.

        Args:
            data (Dict[str, float]): Instance data.
            rules (List[Rule]): List of rule instances.

        Returns:
            Tuple: (Predicted Label, Votes List, Proba List, Matched Rules List)
        """
        if not rules: 
            return None, [], [], []
        
        # 1. Organize rules by Tree (DT1, DT2...)
        # This creates a virtual "Forest" structure from the flat rule list
        tree_rules = defaultdict(list)
        for rule in rules:
            # Assuming format "DT{id}_..."
            tree_identifier = rule.name.split('_')[0]
            tree_rules[tree_identifier].append(rule)

        votes = []
        all_matched_rules = []

        # 2. Vote: Evaluate each tree
        for _, rules_in_tree in tree_rules.items():
            # Reuse logic from classify_dt for each tree
            matched_rule = RuleClassifier.classify_dt(data, rules_in_tree)
            
            if matched_rule:
                # Extract clean label
                try:
                    label = int(str(matched_rule.class_).replace('Class', '').strip())
                except:
                    label = matched_rule.class_
                
                votes.append(label)
                all_matched_rules.append(matched_rule)

        # 3. Aggregate Results (CPU/NumPy version)
        if not votes:
            return None, [], [], []

        # Calculate probabilities
        counts = Counter(votes)
        total_votes = len(votes)
        
        # Get all unique classes known to this specific rule set
        unique_classes = sorted(list(set(r.class_ for r in rules)))
        
        # Build probability distribution
        # We infer from votes here.
        proba_map = {k: v / total_votes for k, v in counts.items()}
        
        # Determine winner
        predicted_class = counts.most_common(1)[0][0]
        
        # Format probabilities as a list (optional, mostly for compatibility)
        # Using a simple list of values present in the votes for now
        probas_out = list(proba_map.values())

        return predicted_class, votes, probas_out, all_matched_rules

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
            print(f"Merging {len(similar_rules_soft)} pairs of boundary rules...")

        for rule1, rule2 in similar_rules_soft:
            rules_to_remove.add(rule1)
            rules_to_remove.add(rule2)

            # Create a new generalized rule (Parent Logic)
            # We strip the last condition which differentiated the two siblings
            common_conditions = rule1.conditions[:-1]
            
            new_rule_name = f"{rule1.name}_&_{rule2.name}"
            new_rule = Rule(new_rule_name, rule1.class_, common_conditions)
            
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
                    
                    # Create representative rule
                    new_rule = Rule(new_name, representative.class_, representative.conditions)
                    new_rule.parsed_conditions = self.parse_conditions_static(new_rule.conditions)
                    
                    new_generalized_rules.append(new_rule)

        # 3. Rebuild the Rule Set
        # Keep rules that were NOT marked for removal + the newly created generalized rules
        final_list = new_generalized_rules + [r for r in source_rules if r not in rules_to_remove]
        
        return final_list, similar_rules_soft
    
    # Exports the rule set to a standalone native Python classifier file
    def export_to_native_python(self, feature_names=None, filename="fast_classifier.py"):
        """
        Generates a standalone Python file with the decision logic using nested if/else.

        This allows for high-performance inference outside of this package, using only 
        standard Python dictionaries as input. The generated code uses .get() for safety against 
        missing keys.

        Args:
            feature_names (List[str], optional): List of feature names (kept for compatibility, 
                                                 not strictly used as rules already contain names).
            filename (str): Output filename.
        """
        print(f"[EXPORT] Generating native classifier: {filename}")
        
        # Determine which rules to export (Final if available, else Initial)
        rules_to_export = self.final_rules if self.final_rules else self.initial_rules

        # Tree Structure: {id: {'l': left, 'r': right, 'f': feature, 't': threshold, 'v': value}}
        # 'f': -2 indicates a leaf node
        tree_dict = {0: {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': -1}}
        next_id = 1
        
        for rule in rules_to_export:
            curr = 0
            for var, op, val in rule.parsed_conditions:
                # 'var' comes directly from the parsed rule (e.g., 'dur')
                if tree_dict[curr]['f'] == -2:
                    # Update current node to be a decision node
                    tree_dict[curr].update({'f': var, 't': val, 'l': next_id, 'r': next_id + 1})
                    
                    # Create empty children
                    tree_dict[next_id] = {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': -1}
                    tree_dict[next_id + 1] = {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': -1}
                    next_id += 2
                
                # Navigate tree
                if op in ['<=', '<']:
                    curr = tree_dict[curr]['l']
                else:
                    curr = tree_dict[curr]['r']
            
            # Set the leaf value (Class)
            # Ensure we save as integer for consistency
            try:
                clean_class = int(str(rule.class_).replace('Class', '').strip())
            except:
                clean_class = rule.class_
            tree_dict[curr]['v'] = clean_class

        # Recursive function to generate Python code strings
        def build_if_else(node_id, indent):
            node = tree_dict[node_id]
            tab = "    " * indent
            
            # Leaf Node
            if node['f'] == -2: 
                val = node['v']
                # If value wasn't set (unreachable path), use default class
                if val == -1: 
                    try:
                        val = int(str(self.default_class).replace('Class', '').strip())
                    except:
                        val = self.default_class
                return f"{tab}return {val}\n"
            
            # Decision Node
            feat_name = node['f']
            
            # OPTIMIZATION: Use sample.get(key, 0) instead of sample[key].
            # This makes the standalone file robust against missing features in input dicts.
            code = f"{tab}if sample.get('{feat_name}', 0) <= {node['t']}:\n"
            code += build_if_else(node['l'], indent + 1)
            code += f"{tab}else:\n"
            code += build_if_else(node['r'], indent + 1)
            return code

        with open(filename, "w") as f:
            f.write("def predict(sample):\n")
            f.write(build_if_else(0, 1))
        
        print(f"[EXPORT] File '{filename}' generated.")
        
    # Method to execute the rule analysis and identify duplicated rules
    def execute_rule_analysis(self, file_path, remove_duplicates="none", remove_below_n_classifications=-1):
        """
        Executes a full rule evaluation and pruning process on a given dataset.

        This method:
        - Applies optional duplicate rule removal (iteratively until convergence).
        - Recompiles the native Python model for speed.
        - Runs evaluation using the appropriate algorithm.
        - Optionally removes rules used less than or equal to a given threshold.

        Args:
            file_path (str): Path to the CSV file containing data for evaluation.
            remove_duplicates (str): Strategy ("soft", "hard", "custom", "none").
            remove_below_n_classifications (int): Threshold for pruning low-usage rules.
        """
        print("\n" + "*"*80)
        print("EXECUTING RULE ANALYSIS")
        print("*"*80 + "\n")

        # Start with a copy of initial rules
        self.final_rules = list(self.initial_rules)

        # 1. Duplicate Removal Loop
        if remove_duplicates != "none":
            iteration = 1
            while True:
                prev_count = len(self.final_rules)
                self.final_rules, self.duplicated_rules = self.adjust_and_remove_rules(remove_duplicates)
                
                # Convergence check: stop if no duplicates found or count didn't change
                if not self.duplicated_rules or len(self.final_rules) == prev_count:
                    break
                iteration += 1

        # 2. CRITICAL OPTIMIZATION: Update native model immediately
        # This ensures the subsequent analysis steps use the optimized compiled function
        # instead of slow object iteration.
        self.update_native_model(self.final_rules)

        # 3. Route to specific analyzer
        if self.algorithm_type == 'Random Forest':
            self.execute_rule_analysis_rf(file_path, remove_below_n_classifications)
        elif self.algorithm_type == 'Decision Tree':
            self.execute_rule_analysis_dt(file_path, remove_below_n_classifications)

    # Method to execute the rule analysis for Decision Tree
    def execute_rule_analysis_dt(self, file_path, remove_below_n_classifications=-1):
        """
        Evaluates Decision Tree rules on a dataset and logs classification performance.

        This method tests the rules, evaluates performance, removes infrequent rules 
        (if specified), and logs detailed diagnostics to 'examples/files/output_classifier_dt.txt'.

        Args:
            file_path (str): Path to the CSV file.
            remove_below_n_classifications (int): Minimum usage count threshold.
        """
        print(f"Testing Decision Tree Rules on {file_path}...")
        start_time = time.time()

        # 1. Load Data
        _, _, X_test, y_test, _, _, feature_names = self.process_data(".", file_path, is_test_only=True)
        sample_dicts = [dict(zip(feature_names, row)) for row in X_test]
        total_samples = len(y_test)

        # 2. Reset Counters
        for rule in self.final_rules:
            rule.usage_count = 0
            rule.error_count = 0

        # 3. Classification Loop
        # We iterate samples to collect usage stats per rule. 
        # Note: classify_dt is optimized to be fast, but we cannot use native_fn 
        # for stats because native_fn hides which rule triggered.
        y_pred = []
        
        for i, sample in enumerate(sample_dicts):
            true_label = int(y_test[i])
            
            # Find the specific rule that applies
            matched_rule = self.classify_dt(sample, self.final_rules)
            
            if matched_rule:
                # Update Rule Stats
                matched_rule.usage_count += 1
                try:
                    pred_label = int(str(matched_rule.class_).replace('Class', '').strip())
                except:
                    pred_label = matched_rule.class_
                
                if pred_label != true_label:
                    matched_rule.error_count += 1
            else:
                # Fallback
                try:
                    pred_label = int(str(self.default_class).replace('Class', '').strip())
                except:
                    pred_label = self.default_class
            
            y_pred.append(pred_label)

        # 4. Metrics
        y_pred = np.array(y_pred)
        correct = np.sum(y_pred == y_test)
        
        self.display_metrics(y_test, y_pred, correct, total_samples, class_names=self.class_labels)

        # 5. Pruning (Specific Rules)
        if remove_below_n_classifications > -1:
            print(f"\nPruning rules with <= {remove_below_n_classifications} classifications...")
            rules_to_keep = []
            self.specific_rules = []
            
            for rule in self.final_rules:
                if rule.usage_count > remove_below_n_classifications:
                    rules_to_keep.append(rule)
                else:
                    self.specific_rules.append(rule)
            
            self.final_rules = rules_to_keep
            # Recompile native model after pruning
            self.update_native_model(self.final_rules)

        # 6. Generate Report
        self._write_report(
            "examples/files/output_classifier_dt.txt", 
            file_path, correct, total_samples, 
            remove_below_n_classifications
        )

        print(f"Analysis Time: {time.time() - start_time:.3f}s")
        self._save_final_model()

    # Method to execute the rule analysis for Random Forest
    def execute_rule_analysis_rf(self, file_path, remove_below_n_classifications=-1):
        """
        Evaluates Random Forest rules on a dataset and logs classification performance.

        Uses voting logic from classify_rf. Logs results to 'examples/files/output_classifier_rf.txt'.
        """
        print(f"Testing Random Forest Rules on {file_path}...")
        start_time = time.time()

        _, _, X_test, y_test, _, _, feature_names = self.process_data(".", file_path, is_test_only=True)
        sample_dicts = [dict(zip(feature_names, row)) for row in X_test]
        total_samples = len(y_test)

        # Reset counts
        for rule in self.final_rules:
            rule.usage_count = 0
            rule.error_count = 0

        y_pred = []

        # Classification Loop (CPU Optimized)
        for i, sample in enumerate(sample_dicts):
            true_label = int(y_test[i])
            
            # Get prediction and the list of rules that voted
            pred_label, _, _, matched_rules = self.classify_rf(sample, self.final_rules)
            
            # Convert prediction to int
            try:
                pred_label = int(str(pred_label).replace('Class', '').strip())
            except:
                pass
            y_pred.append(pred_label)
            
            # Update stats for ALL rules that participated in the vote
            for rule in matched_rules:
                rule.usage_count += 1
                try:
                    r_class = int(str(rule.class_).replace('Class', '').strip())
                except:
                    r_class = rule.class_
                
                # In RF, "error" for a rule is ambiguous. 
                # We define it here as: the rule voted for the wrong class (regardless of final outcome)
                if r_class != true_label:
                    rule.error_count += 1

        # Metrics
        y_pred = np.array(y_pred)
        correct = np.sum(y_pred == y_test)
        self.display_metrics(y_test, y_pred, correct, total_samples, class_names=self.class_labels)

        # Pruning
        if remove_below_n_classifications > -1:
            print(f"\nPruning rules with <= {remove_below_n_classifications} classifications...")
            rules_to_keep = []
            self.specific_rules = []
            for rule in self.final_rules:
                if rule.usage_count > remove_below_n_classifications:
                    rules_to_keep.append(rule)
                else:
                    self.specific_rules.append(rule)
            self.final_rules = rules_to_keep
            self.update_native_model(self.final_rules)

        # Generate Report
        self._write_report(
            "examples/files/output_classifier_rf.txt", 
            file_path, correct, total_samples, 
            remove_below_n_classifications
        )

        print(f"Analysis Time: {time.time() - start_time:.3f}s")
        self._save_final_model()

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
        precision = precision_score(y_true, y_pred_safe, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred_safe, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred_safe, average='macro', zero_division=0)

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

        This method evaluates both the original (`initial_rules`) and pruned (`final_rules`)
        rule sets on the same dataset, and logs performance metrics such as:
        - Accuracy,
        - Confusion matrices,
        - Divergent predictions between the two rule sets,
        - Interpretability metrics per tree.

        It delegates to algorithm-specific methods based on the classifier type.

        Args:
            file_path (str): Path to the CSV file used for evaluation.
        """
        # Robust data loading using static method
        _, _, X_test, y_test, _, target_column_name, feature_names = RuleClassifier.process_data(".", file_path, is_test_only=True)
        
        # Convert to DataFrame for easier handling in comparisons (optional but kept for structure compatibility)
        df_test = pd.DataFrame(X_test, columns=feature_names)
        df_test[target_column_name] = y_test

        if self.algorithm_type == 'Random Forest':
            self.compare_initial_final_results_rf(df_test, target_column_name)
        elif self.algorithm_type == 'Decision Tree':
            self.compare_initial_final_results_dt(df_test, target_column_name)
        else:
            raise ValueError(f"Unsupported algorithm type: {self.algorithm_type}")

    # Method to compare initial and final results for Decision Tree
    def compare_initial_final_results_dt(self, df_test, target_column_name):
        """
        Compares performance of initial vs final rules for Decision Tree models.
        
        Optimized to use native execution for final rules where possible.
        """
        print("\n" + "*"*80)
        print("RUNNING INITIAL AND FINAL CLASSIFICATIONS (Decision Tree)")
        print("*"*80 + "\n")

        # Prepare Data
        df = df_test.copy()
        y_true = df[target_column_name].astype(int).values
        feature_cols = [c for c in df.columns if c != target_column_name]
        
        # Pre-convert to list of dicts for fast iteration
        sample_dicts = [dict(zip(feature_cols, row)) for row in df[feature_cols].values]
        total_samples = len(y_true)
        indices = df.index

        with open('examples/files/output_final_classifier_dt.txt', 'w') as f:
            f.write("******************** INITIAL VS FINAL DECISION TREE CLASSIFICATION REPORT ********************\n\n")

            # --- 1. Scikit-Learn Model ---
            print("Evaluated Scikit-Learn Model...")
            f.write("\n******************************* SCIKIT-LEARN MODEL *******************************\n")
            try:
                with open('examples/files/sklearn_model.pkl', 'rb') as mf:
                    sk_model = pickle.load(mf)
                y_pred_sk = sk_model.predict(df[feature_cols].values)
                correct_sk = np.sum(y_pred_sk == y_true)
                self.display_metrics(y_true, y_pred_sk, correct_sk, total_samples, f, self.class_labels)
            except Exception as e:
                msg = f"Could not load/evaluate sklearn model: {e}"
                print(msg)
                f.write(msg + "\n")

            # --- 2. Initial Rules (Iterative) ---
            print("\nEvaluating Initial Rules...")
            f.write("\n******************************* INITIAL MODEL *******************************\n")
            
            y_pred_initial = []
            for sample in sample_dicts:
                matched = self.classify_dt(sample, self.initial_rules)
                if matched:
                    try:
                        pred = int(str(matched.class_).replace('Class', '').strip())
                    except:
                        pred = matched.class_
                else:
                    try:
                        pred = int(str(self.default_class).replace('Class', '').strip())
                    except:
                        pred = self.default_class
                y_pred_initial.append(pred)
            
            y_pred_initial = np.array(y_pred_initial)
            correct_initial = np.sum(y_pred_initial == y_true)
            self.display_metrics(y_true, y_pred_initial, correct_initial, total_samples, f, self.class_labels)
            f.write(f"\nNumber of initial rules: {len(self.initial_rules)}\n")

            # --- 3. Final Rules (Native/Fast) ---
            print("\nEvaluating Final Rules...")
            f.write("\n******************************* FINAL MODEL *******************************\n")
            
            y_pred_final = []
            # Use native function if available (it corresponds to final_rules after analysis)
            use_native = (self.native_fn is not None)
            
            for sample in sample_dicts:
                if use_native:
                    try:
                        # native_fn returns (pred, votes, proba)
                        pred, _, _ = self.native_fn(sample)
                    except:
                        # Fallback
                        matched = self.classify_dt(sample, self.final_rules)
                        if matched:
                            pred = int(str(matched.class_).replace('Class', '').strip())
                        else:
                            pred = int(str(self.default_class).replace('Class', '').strip())
                else:
                     matched = self.classify_dt(sample, self.final_rules)
                     if matched:
                        try:
                            pred = int(str(matched.class_).replace('Class', '').strip())
                        except:
                            pred = matched.class_
                     else:
                        pred = int(str(self.default_class).replace('Class', '').strip())
                y_pred_final.append(pred)

            y_pred_final = np.array(y_pred_final)
            correct_final = np.sum(y_pred_final == y_true)
            self.display_metrics(y_true, y_pred_final, correct_final, total_samples, f, self.class_labels)
            f.write(f"\nNumber of final rules: {len(self.final_rules)}\n")

            # --- 4. Divergent Cases ---
            print("\nAnalyzing Divergent Cases...")
            f.write("\n******************************* DIVERGENT CASES *******************************\n")
            
            divergent_count = 0
            for i in range(total_samples):
                if y_pred_initial[i] != y_pred_final[i]:
                    divergent_count += 1
                    f.write(f"Index: {indices[i]}, Initial: {y_pred_initial[i]}, Final: {y_pred_final[i]}, Actual: {y_true[i]}\n")
            
            print(f"Total divergent cases: {divergent_count}")
            f.write(f"Total divergent cases: {divergent_count}\n")
            if divergent_count == 0: f.write("No divergent cases found.\n")

            # --- 5. Interpretability Metrics ---
            print("\nCalculating Interpretability Metrics...")
            f.write("\n******************************* INTERPRETABILITY METRICS *******************************\n")
            
            n_features = len(feature_cols)
            metrics_init = self.calculate_structural_complexity(self.initial_rules, n_features)
            metrics_final = self.calculate_structural_complexity(self.final_rules, n_features)
            
            f.write("\nMetrics (Initial):\n")
            for k, v in metrics_init.items():
                f.write(f"  {k}: {v}\n")
            
            f.write("\nMetrics (Final):\n")
            for k, v in metrics_final.items():
                diff = ""
                if isinstance(v, (int, float)) and metrics_init.get(k, 0) != 0:
                    pct = ((v - metrics_init[k]) / metrics_init[k]) * 100
                    diff = f" ({pct:+.1f}%)"
                f.write(f"  {k}: {v}{diff}\n")

    # Method to compare initial and final results for Random Forest
    def compare_initial_final_results_rf(self, df_test, target_column_name):
        """
        Compares performance of initial vs final rules for Random Forest models.
        """
        print("\n" + "*"*80)
        print("RUNNING INITIAL AND FINAL CLASSIFICATIONS (Random Forest)")
        print("*"*80 + "\n")

        df = df_test.copy()
        y_true = df[target_column_name].astype(int).values
        feature_cols = [c for c in df.columns if c != target_column_name]
        sample_dicts = [dict(zip(feature_cols, row)) for row in df[feature_cols].values]
        total_samples = len(y_true)
        indices = df.index

        with open('examples/files/output_final_classifier_rf.txt', 'w') as f:
            f.write("******************** INITIAL VS FINAL RANDOM FOREST CLASSIFICATION REPORT ********************\n\n")

            # 1. Scikit-Learn
            f.write("\n******************************* SCIKIT-LEARN MODEL *******************************\n")
            try:
                with open('examples/files/sklearn_model.pkl', 'rb') as mf:
                    sk_model = pickle.load(mf)
                y_pred_sk = sk_model.predict(df[feature_cols].values)
                correct_sk = np.sum(y_pred_sk == y_true)
                self.display_metrics(y_true, y_pred_sk, correct_sk, total_samples, f, self.class_labels)
            except Exception:
                f.write("Could not load/evaluate sklearn model.\n")

            # 2. Initial Rules
            print("Evaluating Initial Rules...")
            f.write("\n******************************* INITIAL MODEL *******************************\n")
            y_pred_initial = []
            for sample in sample_dicts:
                pred, _, _, _ = self.classify_rf(sample, self.initial_rules)
                try: pred = int(str(pred).replace('Class', '').strip())
                except: pass
                y_pred_initial.append(pred)
            
            y_pred_initial = np.array(y_pred_initial)
            correct_init = np.sum(y_pred_initial == y_true)
            self.display_metrics(y_true, y_pred_initial, correct_init, total_samples, f, self.class_labels)
            
            # 3. Final Rules
            print("Evaluating Final Rules...")
            f.write("\n******************************* FINAL MODEL *******************************\n")
            y_pred_final = []
            for sample in sample_dicts:
                pred, _, _, _ = self.classify_rf(sample, self.final_rules)
                try: pred = int(str(pred).replace('Class', '').strip())
                except: pass
                y_pred_final.append(pred)
                
            y_pred_final = np.array(y_pred_final)
            correct_final = np.sum(y_pred_final == y_true)
            self.display_metrics(y_true, y_pred_final, correct_final, total_samples, f, self.class_labels)

            # 4. Divergence & Metrics
            divergent_count = np.sum(y_pred_initial != y_pred_final)
            print(f"Total divergent cases: {divergent_count}")
            f.write(f"\n******************************* DIVERGENT CASES *******************************\n")
            f.write(f"Total divergent cases: {divergent_count}\n")
            
            # Metrics
            n_features = len(feature_cols)
            metrics_init = self.calculate_structural_complexity(self.initial_rules, n_features)
            metrics_final = self.calculate_structural_complexity(self.final_rules, n_features)
            
            f.write("\n******************************* INTERPRETABILITY METRICS *******************************\n")
            f.write("Metrics (Final vs Initial):\n")
            for k, v in metrics_final.items():
                init_v = metrics_init.get(k, 0)
                diff = ""
                if isinstance(v, (int, float)) and init_v != 0:
                    pct = ((v - init_v) / init_v) * 100
                    diff = f" ({pct:+.1f}%)"
                f.write(f"  {k}: {v}{diff}\n")

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
                print(f"  Current Conditions:")
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
            except:
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
                        df_test[col] = codes

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
                df_train[col] = le.fit_transform(df_train[col].astype(str))
                
                # Transform Test set if column exists
                if col in df_test.columns:
                    # Handle unseen labels: map them to the first known class to avoid crashes
                    df_test[col] = df_test[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    df_test[col] = le.transform(df_test[col].astype(str))

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
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
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
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
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
                class_idx = np.argmax(tree_.value[node])
                
                # Ensure we handle class names correctly (int or str)
                try:
                    class_label = class_names[class_idx]
                except (IndexError, TypeError):
                    class_label = str(class_idx)

                rule_name = f"Rule{len(rules)}_Class{class_label}"
                
                # Create the Rule object
                new_rule = Rule(rule_name, str(class_label), conditions_str)
                
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
            for estimator in model.estimators_:
                rules.append(RuleClassifier.get_rules(estimator, feature_names, class_names))
        
        elif algorithm_type == 'Decision Tree':
            # Single tree
            rules.append(RuleClassifier.get_rules(model, feature_names, class_names))
        
        else:
            raise ValueError(f"Unsupported algorithm type: {algorithm_type}")
        
        return rules

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
            algorithm_type (str): 'Random Forest' or 'Decision Tree'.

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

        # 2. Train or Load Model
        if model_path:
            print(f"Loading model from: {model_path}")
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)
        else:
            print(f"Training a new Scikit-Learn {algorithm_type}...")
            if algorithm_type == 'Random Forest':
                model = RandomForestClassifier(**model_parameters)
            elif algorithm_type == 'Decision Tree':
                model = DecisionTreeClassifier(**model_parameters)
            else:
                raise ValueError(f"Unsupported algorithm type: {algorithm_type}")
            
            # Fit model
            model.fit(X_train, y_train)

            # Save Sklearn model
            model_save_path = 'examples/files/sklearn_model.pkl'
            with open(model_save_path, 'wb') as model_file: 
                pickle.dump(model, model_file)
            print(f"Trained model saved at: {model_save_path}")

        # 3. Test Scikit-Learn Model (Benchmark)
        print("\nTesting Scikit-Learn Model (Benchmark):")
        y_pred = model.predict(X_test)
        
        correct = np.sum(y_pred == y_test)
        total = len(y_test)
        
        # Display metrics using the extracted class names for better readability
        RuleClassifier.display_metrics(y_test, y_pred, correct, total, class_names=class_names)

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