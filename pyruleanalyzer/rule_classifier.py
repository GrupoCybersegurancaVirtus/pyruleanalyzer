from collections import Counter, defaultdict
import csv
import numpy as np
import pandas as pd
import pickle
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.preprocessing import LabelEncoder

# Class to represent a rule
class Rule:
    """
    Represents a complete ruleset (path from root to leaf) in a decision tree.

    This class is not designed to be operated directly, but by instead using the main :ref:`RuleClassifier<rule_classifier>` class.

    Attributes:
        name (str): Name of the rule (e.g., "DT1_Rule36_Class0")
        class (str): Class that the rule assigns to matching instances (Note: trailing underscore is used because "class" is a Python keyword.)
        conditions (List[str]): List of condition strings (like "v2 > 0.5")
        usage_count (int): Number of times the rule matched during classification
        error_count (int): Number of times the rule matched but the prediction was wrong
        parse_conditions (Tuple[str, str, float]): Provides caching for conditions
    """

    __slots__ = ['name', 'class_', 'conditions', 'usage_count', 'error_count', 'parsed_conditions']

    def __init__(self, name, class_, conditions):
        """
        Initializes a new Rule instance representing a decision path in a tree.

        Args:
            name (str): The name of the rule, including tree and class identifiers, e.g. "DT1_Rule36_Class0".
            class (str): The class label assigned to instances that satisfy the rule's conditions. (Note: trailing underscore is used because "class" is a Python keyword.)
            conditions (List[str]): List of attribute comparison conditions defining the rule.
        """
        self.name = name
        self.class_ = class_
        self.conditions = conditions
        self.usage_count = 0
        self.error_count = 0
        # Parsed conditions will be stored here to avoid repeated parsing
        self.parsed_conditions = []

# Class to handle the rule classification process
class RuleClassifier:
    """
    Represents a rule-based classifier built from decision paths in tree models.
    
    This class supports rule extraction, classification, refinement, and 
    analysis of decision logic derived from scikit-learn Decision Tree and 
    Random Forest models.

    Attributes:
        initial_rules (List[Rule]): List of all parsed rules extracted from the model.
        final_rules (List[Rule]): Filtered rule list after duplicate removal or other processing.
        duplicated_rules (List[Tuple[Rule, Rule]]): List of rules identified as structurally redundant.
        specific_rules (List[Rule]): Rules removed based on low usage or specificity.
        algorithm_type (str): Type of model used to generate the rules ('Decision Tree' or 'Random Forest').
    """
    def __init__(self, rules, algorithm_type='Decision Tree'):
        self.initial_rules = self.parse_rules(rules, algorithm_type)
        self.algorithm_type = algorithm_type
        self.final_rules = []
        self.duplicated_rules = []
        self.specific_rules = []

    # Method to parse the rules from string based on the algorithm type
    def parse_rules(self, rules, algorithm_type):
        """
        Parses a raw rule string into structured Rule objects based on model type.
        
        Depending on whether the rules originate from a Decision Tree or a Random Forest,
        this method delegates to the appropriate parsing logic.

        Args:
            rules (str): Multiline string containing rule definitions.
            algorithm_type (str): The model type ('Decision Tree' or 'Random Forest').

        Returns:
            List[Rule]: A list of Rule objects parsed from the input string.
        """
        rules = rules.replace('"', '').replace('- ', '').strip().split('\n')

        parsed_rule_list = []
        if algorithm_type == 'Random Forest':
            parsed_rule_list = [self.parse_rf_rule(rule) for rule in rules if rule]
        elif algorithm_type == 'Decision Tree':
            parsed_rule_list = [self.parse_dt_rule(rule) for rule in rules if rule]

        # Pre-parse all conditions for efficiency
        for rule in parsed_rule_list:
            rule.parsed_conditions = self.parse_conditions_static(rule.conditions)
        
        return parsed_rule_list

    # Method to parse the rules for Decision Tree
    def parse_dt_rule(self, rule):
        """
        Parses a decision tree rule string into a structured Rule object.
        
        This method processes a rule extracted from a Decision Tree by separating 
        its identifier and its condition list, and then converting it into a Rule instance.

        Args:
            rule (str): A string representing a single rule in the format "RuleName: [condition1, condition2, ...]".

        Returns:
            Rule: A Rule object with the extracted name, class, and condition list.
        """
        rule = rule.strip().split(':', 1)
        rule_name = rule[0].strip()
        class_ = rule_name.split('_')[-1]
        conditions = rule[1].strip().replace('[', '').replace(']', '').split(', ') if len(rule) > 1 and rule[1].strip() != '[]' else []
        return Rule(rule_name, class_, conditions)

    # Method to parse the rules for Random Forest
    def parse_rf_rule(self, rule):
        """
        Parses a random forest rule string into a structured Rule object.
        
        This method processes a rule extracted from Random Forest estimators by separating 
        its identifier and its condition list, and then converting it into a Rule instance.

        Args:
            rule (str): A string representing a Random Forest rule in the format "RuleName: [condition1, condition2, ...]".

        Returns:
            Rule: A Rule object containing the parsed name, class, and condition list.
        """
        rule = rule.split(':')
        rule_name, conditions_str = rule[0].strip(), rule[1].strip()
        class_ = rule_name.split('_')[-1]
        conditions = conditions_str.replace('[', '').replace(']', '').split(', ') if conditions_str != '[]' else []
        return Rule(rule_name, class_, conditions)

    # Method to parse conditions from string to tuple (variable, operator, value)
    @staticmethod
    def parse_conditions_static(conditions):
        """
        Parses a list of condition strings into structured tuples for evaluation.
        
        Converts conditions like "v1 <= 0.5" into a tuple representation 
        ("v1", "<=", 0.5) to facilitate programmatic comparison during classification.

        Args:
            conditions (List[str]): A list of condition strings from a rule.

        Returns:
            List[Tuple[str,str,float]]: A list of parsed conditions, where each
            tuple contains (variable name, operator, numeric threshold).
        """
        parsed_conditions = []
        if not conditions or conditions[0] == '' or conditions == []:
            return []
        for condition in conditions:
            if '<=' in condition:
                parts = condition.split(' <= ')
                if len(parts) == 2:
                    var, value = parts
                    parsed_conditions.append((var, '<=', float(value)))
            elif '>=' in condition:
                parts = condition.split(' >= ')
                if len(parts) == 2:
                    var, value = parts
                    parsed_conditions.append((var, '>=', float(value)))
            elif '<' in condition:
                parts = condition.split(' < ')
                if len(parts) == 2:
                    var, value = parts
                    parsed_conditions.append((var, '<', float(value)))
            elif '>' in condition:
                parts = condition.split(' > ')
                if len(parts) == 2:
                    var, value = parts
                    parsed_conditions.append((var, '>', float(value)))
            else:
                # Ignore or log conditions that do not match expected format
                continue
        return parsed_conditions

    # Method to execute the classification process
    def classify(self, data, final=False):
        """
        Classifies a single data instance using extracted rules.

        This method will delegate the classification logic to the appropriate function based on the algorithm type.

        Args:
            data (Dict[str, float]): A dictionary representing the instance to classify, where keys are feature names (e.g., 'v1', 'v2') and values are the corresponding feature values.
            final (bool): If True, use `final_rules` (post-analysis); otherwise, use `initial_rules`.

        Returns:
            Tuple[int|None,List[int]|None,np.ndarray|None]: A tuple containing
                - Predicted class label (or None if no rule matched),
                - List of votes (Random Forest only, otherwise None),
                - Class probabilities (Random Forest only, otherwise None).
        """
        rules = self.final_rules if final else self.initial_rules

        if self.algorithm_type == 'Random Forest':
            predicted_class, votes, proba, _ = self.classify_rf(data, rules)
            return predicted_class, votes, proba
        if self.algorithm_type == 'Decision Tree':
            matched_rule = self.classify_dt(data, rules)
            if matched_rule:
                # To maintain original return signature, extract class from rule
                for part in matched_rule.name.split('_'):
                    if part.startswith('Class'):
                        return int(part.replace('Class', '')), None, None
            return None, None, None
        return None, None, None

    # Method to classify data using Decision Tree rules
    @staticmethod
    def classify_dt(data, rules):
        """
        Classifies a single data instance using extracted rules from the decision tree model.

        This method applies the rule set to classify a given data instance, it returns the class of the first rule that matches.

        Args:
            data (Dict[str, float]): A dictionary representing the instance to classify, where keys are feature names (e.g., 'v1', 'v2') and values are the corresponding feature values.
            rules: (List[rule]): A list of rule instances.

        Returns:
            Rule|None: The first Rule object that matches the data, or None.
        """
        for rule in rules:
            rule_satisfied = True
            # Use pre-parsed conditions for efficiency
            for var, op, value in rule.parsed_conditions:
                instance_value = data.get(var)
                if instance_value is None:
                    rule_satisfied = False
                    break

                if not ( (op == '<=' and instance_value <= value) or \
                         (op == '>=' and instance_value >= value) or \
                         (op == '<'  and instance_value < value)  or \
                         (op == '>'  and instance_value > value) ):
                    rule_satisfied = False
                    break

            if rule_satisfied:
                return rule # Return the entire rule object

        return None

    # Method to classify data using Random Forest rules
    @staticmethod
    def classify_rf(data, rules):
        """
        Classifies a single data instance using extracted rules from the random forest model.

        This method applies the rule set to classify a given data instance, it returns the class of the first rule that matches.

        Args:
            data (Dict[str, float]): A dictionary representing the instance to classify, where keys are feature names (e.g., 'v1', 'v2') and values are the corresponding feature values.
            rules: (List[rule]): A list of rule instances.

        Returns:
            Tuple[int|None, List[int], List[float], List[Rule]]: A tuple containing
                - Predicted class label (or None if no rule matched),
                - List of votes,
                - Class probabilities,
                - List of all rules that were matched.
        """
        if not rules:
            return None, [], [], []

        class_labels = sorted({int(rule.class_[-1]) for rule in rules})
        if not class_labels:
            return None, [], [], []

        class_to_index = {label: idx for idx, label in enumerate(class_labels)}
        num_classes = len(class_labels)

        tree_rules = defaultdict(list)
        for rule in rules:
            tree_name = rule.name.split('_')[0]
            tree_rules[tree_name].append(rule)

        probas = []
        all_matched_rules = []

        for tree_name, tree_rule_list in tree_rules.items():
            proba_tree = np.zeros(num_classes)
            matched_classes_in_tree = []

            for rule in tree_rule_list:
                # Use pre-parsed conditions
                parsed_conditions = rule.parsed_conditions
                matched = all(
                    var in data and (
                        (op == '<=' and data[var] <= value) or
                        (op == '>=' and data[var] >= value) or
                        (op == '<' and data[var] < value) or
                        (op == '>' and data[var] > value)
                    ) for var, op, value in parsed_conditions
                )

                if matched:
                    class_label = int(rule.class_[-1])
                    matched_classes_in_tree.append(class_label)
                    all_matched_rules.append(rule)

            if matched_classes_in_tree:
                class_counts = Counter(matched_classes_in_tree)
                total = sum(class_counts.values())
                for label, count in class_counts.items():
                    if label in class_to_index:
                        idx = class_to_index[label]
                        proba_tree[idx] = count / total
            probas.append(proba_tree)

        if not probas:
            return None, [], [0.0] * num_classes, []

        avg_proba = np.mean(probas, axis=0)
        predicted_class_index = int(np.argmax(avg_proba))
        predicted_class = class_labels[predicted_class_index]
        votes = [class_labels[int(np.argmax(p))] for p in probas if p.any()]

        return predicted_class, votes, avg_proba.tolist(), all_matched_rules

    # Method to extract variables, operators, and values from conditions
    @staticmethod
    def extract_variables_and_operators(conditions):
        """
        Extracts variable-operator-value triples from a list of rule conditions.

        This helper method parses each condition (e.g., "v1 <= 0.5") and returns a
        normalized list of tuples containing the variable name, the comparison operator,
        and the threshold value. Operators '<=' and '<' are treated equivalently, as are '>=' and '>'.

        Args:
            conditions (List[str]): A list of string conditions from a rule.

        Returns:
            List[Tuple[str, str, float]]: A sorted list of (variable, operator, value) triples, with normalized operators.
        """
        vars_ops_vals = []
        if not conditions or conditions[0] == '':
            return []
        for cond in conditions:
            # Optimized parsing assuming space-separated tokens
            parts = cond.split(' ')
            if len(parts) == 3:
                var, op, value_str = parts
                # Normalize operators
                if op == '<':
                    op = '<='
                elif op == '>':
                    op = '>='

                try:
                    value = float(value_str)
                    vars_ops_vals.append((var, op, value))
                except ValueError:
                    # Handle case where value is not a float, if necessary
                    pass
            else:
                # Fallback for more complex conditions if any
                for op_str in ['<=', '>=', '<', '>']:
                    if op_str in cond:
                        idx = cond.index(op_str)
                        var = cond[:idx].strip()
                        value = cond[idx + len(op_str):].strip()
                        norm_op = '<=' if op_str in ['<=', '<'] else '>='
                        vars_ops_vals.append((var, norm_op, float(value)))
                        break

        return sorted(vars_ops_vals)

    # Method to find similar rules between trees, considering the variables and operators
    def find_duplicated_rules_between_trees(self):
        """
        Identifies semantically similar rules between different rules.

        This method compares rules across the full rule set to find groups that:
        - Use the same set of variables and logical operators (ignoring threshold values),
        - Belong to the same target class.

        Returns:
            List[List[Rule]]: A list of groups, where each group is a list of similar rules.
        """
        rules_by_signature = defaultdict(list)
        # Operate on the current set of final_rules to allow for convergence
        for rule in self.final_rules:
            # Create a canonical signature for the rule's logic
            vars_ops = self.extract_variables_and_operators(rule.conditions)
            # The key includes the class and the logic signature
            signature = (rule.class_, tuple(v[0:2] for v in vars_ops))
            rules_by_signature[signature].append(rule)

        # Return only the groups that have more than one similar rule
        return [group for group in rules_by_signature.values() if len(group) > 1]

    # Method to find duplicated rules in the same tree
    def find_duplicated_rules(self, type='soft'):
        """
        Identifies nearly identical rules within the the same decision tree.

        This method searches for rule pairs that:
        - Have the same class label,
        - Share all conditions except the last,
        - Differ only in the final condition, where one uses a '<=' and the other a '>' (or vice versa).

        Such pairs are considered duplicates due to redundant decision splits at the boundary.

        Returns:
            List[Tuple[Rule,Rule]]: A list of tuples, each representing a pair of duplicated rules.
        """
        # Optimized approach using a hash map to group potential duplicates
        duplicated_rules = []
        rules_by_prefix = defaultdict(list)

        # Group rules by class and all conditions except the last one
        for rule in self.final_rules:
            if not rule.conditions:
                continue
            prefix_key = (rule.class_, tuple(rule.conditions[:-1]))
            rules_by_prefix[prefix_key].append(rule)

        # Check for duplicates only within the smaller groups
        for prefix, candidates in rules_by_prefix.items():
            if len(candidates) < 2:
                continue

            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    rule1 = candidates[i]
                    rule2 = candidates[j]

                    last_cond1 = rule1.conditions[-1]
                    last_cond2 = rule2.conditions[-1]

                    # Efficiently split last conditions
                    parts1 = last_cond1.split(' ')
                    parts2 = last_cond2.split(' ')

                    if len(parts1) != 3 or len(parts2) != 3:
                        continue

                    var1, op1, _ = parts1
                    var2, op2, _ = parts2

                    # Check for opposite operators on the same variable
                    is_duplicate = False
                    if var1 == var2:
                        op_pair = {op1, op2}
                        if type == 'soft':
                            if op_pair in [{'<=', '>'}, {'<', '>='}, {'<', '>'}]:
                                is_duplicate = True
                        elif type == 'medium':
                            if op_pair in [{'<=', '>'}, {'<', '>='}, {'<', '>'}, {'>=', '<'}, {'<=', '<'}]:
                                is_duplicate = True

                    if is_duplicate:
                        duplicated_rules.append((rule1, rule2))

        return duplicated_rules

    # Method to set a custom rule removal function
    def set_custom_rule_removal(self, custom_function):
        """
        Allows the user to override the rule removal logic, by employing their own implementation.

        Args:
            custom_function (Callable[[List[Rule]],Tuple[List[Rule],List[Tuple[Rule,Rule]]]]): A callback that takes a list of Rule instances as argument and returns a tuple containing a new list of rules after removing duplicates and the list of duplicate rule pairs.
        """
        self.custom_rule_removal = custom_function

    # Method to remove rules based on custom logic
    def custom_rule_removal(self, rules):
        """
        Placeholder for custom rule removal logic. Does not alter the rule set.

        Args:
            rules (List[Rule]): List of Rules instances.

        Returns:
            Tuple[List[Rule],List[]]:
            A tuple containing
                - The same rules from the input,
                - An empty list.
        """
        return rules, []

    # Method to adjust and remove duplicated rules
    def adjust_and_remove_rules(self, method):
        """
        Adjusts and removes duplicated rules from the rule set based on the specified method.

        This method analyzes the current rule set to identify and remove duplicated rules. The logic supports three modes:
            - "custom": Uses a user-defined custom function to remove rules.
            - "soft": Detects and removes duplicated rules within the same tree only.
            - "hard": Removes duplicated rules both within the same tree and across different trees.
        
        Args:
            method (str): Strategy for rule refinement. Must be either "custom", "soft" or "hard".

        Returns:
            Tuple[List[Rule],List[Tuple[Rule,Rule]]]: A tuple containing
                - A new list of rules after removing duplicates and adding generalized ones,
                - A list of the identified duplicated rule pairs (from soft check only, for loop condition).
        """
        if method == "custom":
            return self.custom_rule_removal(self.initial_rules)

        print("\nANALYSING DUPLICATED RULES IN THE SAME TREE")
        if method not in ["soft", "medium", "hard", 'custom']:
            raise ValueError(f"Invalid method: {method}. Use 'soft', 'medium', 'hard' or 'custom'.")

        # This list of pairs is only from the soft check and is used to decide if the loop should break.
        similar_rules_soft = self.find_duplicated_rules(type=method if method in ['soft', 'medium'] else 'soft')

        duplicated_rules = set()
        unique_rules = []

        for rule1, rule2 in similar_rules_soft:
            duplicated_rules.add(rule1)
            duplicated_rules.add(rule2)
            print(f"\nDuplicated rules from the same tree: {rule1.name} == {rule2.name}")
            print(f"{rule1.name}: {rule1.conditions}")
            print(f"{rule2.name}: {rule2.conditions}")

            # Create a new rule that generalizes the two duplicated ones
            common_conditions = rule1.conditions[:-1]
            new_rule_name = f"{rule1.name}_&_{rule2.name}"
            new_rule_class = rule1.class_
            new_rule = Rule(new_rule_name, new_rule_class, common_conditions)
            # Also parse the new rule's conditions
            new_rule.parsed_conditions = self.parse_conditions_static(new_rule.conditions)
            print(f"New rule created: {new_rule.name} with conditions: {new_rule.conditions}")
            unique_rules.append(new_rule)

        if method == "hard":
            print("\nANALYSING DUPLICATED RULES BETWEEN TREES")
            if self.algorithm_type == 'Random Forest':
                # Find groups of similar rules across different trees
                similar_rule_groups = self.find_duplicated_rules_between_trees()
                for group in similar_rule_groups:
                    # Add all rules in the group to the removal set
                    for rule in group:
                        duplicated_rules.add(rule)

                    # Create ONE representative rule from the group.
                    # We take the first rule's conditions as representative.
                    representative_rule = group[0]
                    new_rule_name = "_&_".join(sorted([r.name for r in group]))
                    new_rule_class = representative_rule.class_
                    new_rule_conditions = representative_rule.conditions

                    new_rule = Rule(new_rule_name, new_rule_class, new_rule_conditions)
                    new_rule.parsed_conditions = self.parse_conditions_static(new_rule.conditions)

                    print(f"\nDuplicated group of {len(group)} rules found. Generalizing to {new_rule.name}")

                    print(f"New rule conditions: {new_rule.conditions}")
                    unique_rules.append(new_rule)

        # Construct the final list: the new generalized rules + the old rules that were not duplicates
        unique_rules.extend(rule for rule in self.final_rules if rule not in duplicated_rules)

        # The break condition for the main loop is based only on the soft check duplicates.
        return unique_rules, similar_rules_soft

    # Method to execute the rule analysis and identify duplicated rules
    def execute_rule_analysis(self, file_path, remove_duplicates="none", remove_below_n_classifications=-1):
        """
        Executes a full rule evaluation and pruning process on a given dataset.

        This method:
        - Applies optional duplicate rule removal,
        - Prints and logs final rule structure,
        - Runs evaluation using the appropriate algorithm (Decision Tree or Random Forest),
        - Optionally removes rules used less than or equal to a given threshold.

        Args:
            file_path (str): Path to the CSV file containing data for evaluation.
            remove_duplicates (str): Method for removing duplicate rules, can be either "soft", "hard", "custom" or "none".
            remove_below_n_classifications (int): Threshold for rule usage count. If set to -1, no filtering is applied.
        """
        print("\n*********************************************************************************************************")
        print("**************************************** EXECUTING RULE ANALYSIS ****************************************")
        print("*********************************************************************************************************\n")

        self.final_rules = self.initial_rules

        if remove_duplicates != "none":
            while True:
                self.final_rules, self.duplicated_rules = self.adjust_and_remove_rules(remove_duplicates)
                if not self.duplicated_rules:
                    print("\nNo more duplicated rules found.")
                    break

        # The specific analysis functions will now handle data loading robustly.
        if self.algorithm_type == 'Random Forest':
            self.execute_rule_analysis_rf(file_path, remove_below_n_classifications)
        elif self.algorithm_type == 'Decision Tree':
            self.execute_rule_analysis_dt(file_path, remove_below_n_classifications)
        else:
            raise ValueError(f"Unsupported algorithm type: {self.algorithm_type}")

    # Method to execute the rule analysis for Decision Tree
    def execute_rule_analysis_dt(self, file_path, remove_below_n_classifications=-1):
        """
        Evaluates Decision Tree rules on a dataset and logs classification performance.

        This method tests the decision tree rules on a CSV dataset, evaluates rule performance, removes infrequent rules (if specified), and logs classification results, errors, usage counts, and rule effectiveness into an output file.

        Outputs are written to 'examples/files/output_classifier_dt.txt'.

        Args:
            file_path (str): Path to the CSV file containing the dataset to evaluate.
            remove_below_n_classifications (int): Minimum usage count required to retain a rule.
        """

        print("\n******************************** TESTING RULES AFTER ANALYSIS ***************************************")
        start_time = time.time()

        # Robust data loading
        _, _, X_test, y_test, _, target_column_name, feature_names = RuleClassifier.process_data(".", file_path, is_test_only=True)
        df_test = pd.DataFrame(X_test, columns=feature_names)
        df_test[target_column_name] = y_test

        # Reset rule counts
        for rule in self.final_rules:
            rule.usage_count = 0
            rule.error_count = 0

        y_pred = pd.Series(index=df_test.index, dtype=int)

        # Apply each rule to the entire dataframe at once
        for rule in self.final_rules:
            if not rule.conditions or rule.conditions[0] == '':
                # Empty rule matches everything not yet classified
                condition_query = y_pred.isnull()
            else:
                # Build pandas query
                query_parts = [f"`{var}` {op} {val}" for var, op, val in rule.parsed_conditions]
                query = " & ".join(query_parts)

                # Find rows matching the rule AND not yet predicted
                unpredicted_indices = y_pred[y_pred.isnull()].index
                try:
                    matched_indices = df_test.loc[unpredicted_indices].query(query).index
                    condition_query = pd.Series(df_test.index.isin(matched_indices), index=df_test.index)
                except Exception as e:
                    print(f"Error in rule query {rule.name}: {e}")
                    continue

            # Update predictions for matched rows
            predicted_class = int(rule.class_.replace('Class', ''))
            y_pred.loc[condition_query] = predicted_class

            # Update usage and error counts
            matches = df_test[condition_query]
            rule.usage_count = len(matches)
            if not matches.empty:
                errors = matches[matches[target_column_name] != predicted_class]
                rule.error_count = len(errors)

        y_pred.fillna(-1, inplace=True) # Fill remaining as error
        y_true = df_test[target_column_name]

        with open('examples/files/output_classifier_dt.txt', 'w') as f:
            correct = (y_pred == y_true).sum()
            total = len(y_true)
            errors_df = df_test[y_pred != y_true]
            errors_str = "\n".join([f"Index: {idx}, Predicted: {pred}, Actual: {actual}" 
                        for idx, pred, actual in zip(errors_df.index, y_pred[errors_df.index], y_true[errors_df.index])])

            if remove_below_n_classifications != -1:
                rules_to_keep = []
                f.write(f"\nRules removed with usage count below {remove_below_n_classifications}:\n")
                for rule in self.final_rules:
                    if rule.usage_count > remove_below_n_classifications:
                        rules_to_keep.append(rule)
                    else:
                        f.write(f"Rule: {rule.name}, Count: {rule.usage_count}\n")
                        self.specific_rules.append(rule)
                self.final_rules = rules_to_keep

            f.write("\nFinal Rules:\n")
            for rule in self.final_rules:
                f.write(f"Rule: {rule.name}, Class: {rule.class_}, Conditions: {rule.conditions}\n")

            print("\n******************************* RESULTS SUMMARY *******************************\n")
            f.write("\n******************************* RESULTS SUMMARY *******************************\n")

            print(f"\nTotal Initial Rules: {len(self.initial_rules)}")
            f.write(f"\nTotal Initial Rules: {len(self.initial_rules)}\n")
            print(f"Total Final Rules: {len(self.final_rules)}")
            f.write(f"Total Final Rules: {len(self.final_rules)}\n")

            print(f"\nTotal Duplicated Rules: {len(self.initial_rules) - len(self.final_rules) - len(self.specific_rules)}")
            f.write(f"\nTotal Duplicated Rules: {len(self.initial_rules) - len(self.final_rules) - len(self.specific_rules)}\n")

            if remove_below_n_classifications > -1:
                print(f"\nTotal Specific Rules: {len(self.specific_rules)} (<= {remove_below_n_classifications} classifications)")
                f.write(f"\nTotal Specific Rules: {len(self.specific_rules)} (<= {remove_below_n_classifications} classifications)\n")

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"\nTime elapsed in executing rule analysis and adjustment: {elapsed_time:.3f} seconds")
            f.write(f"\nTime elapsed in executing rule analysis and adjustment: {elapsed_time:.3f} seconds\n")

        # Save the final classifier to a pickle file
        with open('examples/files/final_model.pkl', 'wb') as model_file:
            pickle.dump(self, model_file)
        print("Final classifier saved to 'examples/files/final_model.pkl'.")

        return self

    # Method to execute the rule analysis for Random Forest
    def execute_rule_analysis_rf(self, file_path, remove_below_n_classifications=-1):
        """
        Evaluates Random Forest rules on a dataset and logs classification performance.

        This method evaluates the rule-based classifier on test data using extracted random forest rules.
        It logs predictions, voting behavior, rule usage, errors, confusion matrix, and other diagnostics.
        It can also filter out rarely used rules if a threshold is specified.

        Outputs are written to 'examples/files/output_classifier.txt'.

        Args:
            file_path (str): Path to the CSV file containing the dataset to evaluate.
            remove_below_n_classifications (int): Minimum rule usage required to retain a rule.
        """
        print("\n******************************** TESTING RULES AFTER ANALYSIS ***************************************")
        start_time = time.time()

        # Robust data loading
        _, _, X_test, y_test, _, target_column_name, feature_names = RuleClassifier.process_data(".", file_path, is_test_only=True)
        df_test = pd.DataFrame(X_test, columns=feature_names)
        df_test[target_column_name] = y_test

        # Group rules by tree
        tree_rules = defaultdict(list)
        for rule in self.final_rules:
            rule.usage_count = 0
            rule.error_count = 0
            tree_name = rule.name.split('_')[0]
            tree_rules[tree_name].append(rule)

        class_labels = sorted({int(r.class_[-1]) for r in self.final_rules})
        class_to_index = {label: i for i, label in enumerate(class_labels)}

        # DataFrame to store probabilities of each tree for each sample
        tree_probas = pd.DataFrame(0, index=df_test.index, columns=tree_rules.keys())

        all_probas = np.zeros((len(df_test), len(class_labels)))

        # Iterate over each tree
        for i, (tree_name, rules) in enumerate(tree_rules.items()):
            tree_class_votes = np.zeros((len(df_test), len(class_labels)))

            # For each rule in the tree, find the samples it classifies
            for rule in rules:
                if not rule.conditions or rule.conditions[0] == '':
                    matched_indices = df_test.index
                else:
                    query = " & ".join([f"`{var}` {op} {val}" for var, op, val in rule.parsed_conditions])
                    try:
                        matched_indices = df_test.query(query).index
                    except Exception:
                        continue

                if not matched_indices.empty:
                    predicted_class = int(rule.class_.replace('Class', ''))
                    class_idx = class_to_index.get(predicted_class)
                    if class_idx is not None:
                        tree_class_votes[matched_indices, class_idx] += 1

                    # Update usage count
                    rule.usage_count = len(matched_indices)

            # Normalize votes to get tree probability
            vote_sums = tree_class_votes.sum(axis=1, keepdims=True)
            np.divide(tree_class_votes, vote_sums, where=vote_sums!=0, out=tree_class_votes)
            all_probas += tree_class_votes

        # Calculate average probability and final prediction
        avg_probas = all_probas / len(tree_rules)
        y_pred_indices = np.argmax(avg_probas, axis=1)
        y_pred = np.array(class_labels)[y_pred_indices]

        y_true = df_test[target_column_name].values

        # Update error count
        error_indices = np.where(y_pred != y_true)[0]
        for rule in self.final_rules:
            if rule.usage_count > 0:
                query = " & ".join([f"`{var}` {op} {val}" for var, op, val in rule.parsed_conditions]) if rule.conditions else None
                if query:
                    matched_indices = df_test.query(query).index
                    rule.error_count = len(np.intersect1d(matched_indices, error_indices))

        with open('examples/files/output_classifier_rf.txt', 'w') as f:
            correct = (y_pred == y_true).sum()
            total = len(y_true)
            if remove_below_n_classifications != -1:
                f.write(f"\nRules removed with usage count below {remove_below_n_classifications}:\n")
                rules_to_keep = []
                for rule in self.final_rules:
                    if rule.usage_count > remove_below_n_classifications:
                        rules_to_keep.append(rule)
                    else:
                        f.write(f"Rule: {rule.name}, Count: {rule.usage_count}\n")
                        self.specific_rules.append(rule)
                self.final_rules = rules_to_keep

            f.write("\nInitial Rules:\n")
            for rule in self.initial_rules:
                f.write(f"Rule: {rule.name}, Class: {rule.class_}, Conditions: {rule.conditions}\n")

            f.write("\nFinal Rules:\n")
            for rule in self.final_rules:
                f.write(f"Rule: {rule.name}, Class: {rule.class_}, Conditions: {rule.conditions}\n")

            initial_tree_rule_counts = Counter(rule.name.split('_')[0] for rule in self.initial_rules)
            f.write("\nInitial Tree Rule Counts:\n")
            for tree_name, count in sorted(initial_tree_rule_counts.items()):
                f.write(f"Tree: {tree_name}, Rule Count: {count}\n")

            final_tree_rule_counts = Counter(rule.name.split('_')[0] for rule in self.final_rules)
            f.write("\nFinal Tree Rule Counts:\n")
            for tree_name, count in sorted(final_tree_rule_counts.items()):
                f.write(f"Tree: {tree_name}, Rule Count: {count}\n")

            f.write("\nRules with most Classifications:\n")
            sorted_by_usage = sorted((r for r in self.final_rules if r.usage_count > 0), key=lambda r: r.usage_count, reverse=True)
            for rule in sorted_by_usage:
                f.write(f"- {rule.name}, Classifications: {rule.usage_count}\n")

            f.write("\nRules with most Errors:\n")
            sorted_rules = sorted((r for r in self.final_rules if r.error_count > 0), key=lambda r: r.error_count, reverse=True)
            for rule in sorted_rules:
                error_percentage = (rule.error_count / rule.usage_count) * 100 if rule.usage_count > 0 else 0
                f.write(f"- {rule.name}, Errors: {rule.error_count} / {rule.usage_count} classifications ({error_percentage:.2f}%)\n")

            error_percentages = [
                (rule, (rule.error_count / rule.usage_count) * 100)
                for rule in self.final_rules if rule.usage_count > 0
            ]
            f.write("\nRules with most Error Percentage:\n")
            if error_percentages:
                avg_error_percentage = np.mean([ep for _, ep in error_percentages])
                for rule, error_percentage in sorted(error_percentages, key=lambda x: x[1], reverse=True):
                    if error_percentage > avg_error_percentage:
                        f.write(f"- {rule.name}, Errors: {rule.error_count} / {rule.usage_count} ({error_percentage:.2f}%)\n")
            else:
                f.write("No rules with usage > 0 to calculate error percentage.\n")

            print("\n******************************* RESULTS SUMMARY *******************************\n")
            f.write("\n******************************* RESULTS SUMMARY *******************************\n")

            print(f"\nTotal Initial Rules: {len(self.initial_rules)}")
            f.write(f"\nTotal Initial Rules: {len(self.initial_rules)}\n")
            print(f"Total Final Rules: {len(self.final_rules)}")
            f.write(f"Total Final Rules: {len(self.final_rules)}\n")

            print(f"\nTotal Duplicated Rules: {len(self.initial_rules) - len(self.final_rules) - len(self.specific_rules)}")
            f.write(f"\nTotal Duplicated Rules: {len(self.initial_rules) - len(self.final_rules) - len(self.specific_rules)}\n")

            if remove_below_n_classifications > -1:
                print(f"\nTotal Specific Rules: {len(self.specific_rules)} (<= {remove_below_n_classifications} classifications)")
                f.write(f"\nTotal Specific Rules: {len(self.specific_rules)} (<= {remove_below_n_classifications} classifications)\n")

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"\nTime elapsed in executing rule analysis and adjustment: {elapsed_time:.3f} seconds")
            f.write(f"\nTime elapsed in executing rule analysis and adjustment: {elapsed_time:.3f} seconds\n")

        # Save the final classifier to a pickle file
        with open('examples/files/final_model.pkl', 'wb') as model_file:
            pickle.dump(self, model_file)
        print("Final classifier saved to 'examples/files/final_model.pkl'.")

        return self

    @staticmethod
    def calculate_sparsity_interpretability(rules, n_features_total):
        """
        Computes sparsity and interpretability metrics for a given rule set.

        This method measures how concise and generalizable the rules are by evaluating:
        - The proportion of total features actually used,
        - The total number of rules,
        - Rule depth statistics (max and mean),
        - A combined Sparsity Interpretability (SI) score.

        Args:
            rules (List[Rule]): A list of Rule objects to analyze.
            n_features_total (int): Total number of available features in the dataset.

        Returns:
            Dict[str,Any]: A dictionary containing
                - features_used (int): Number of unique features used in rules,
                - total_features (int): Total number of features in the dataset,
                - sparsity (float): 1 - (features_used / total_features),
                - total_rules (int): Total number of rules,
                - max_depth (int): Maximum number of conditions in a single rule,
                - mean_rule_depth (float): Average number of conditions per rule,
                - sparsity_interpretability_score (float): Combined interpretability score (higher is better).
        """
        if not rules:
            return {
                "features_used": 0, "total_features": n_features_total, "sparsity": 1.0,
                "total_rules": 0, "max_depth": 0, "mean_rule_depth": 0.0,
                "sparsity_interpretability_score": float('inf')
            }

        features_used = set()
        for rule in rules:
            for condition in rule.conditions:
                feature = condition.split(' ')[0]
                features_used.add(feature)

        n_features_used = len(features_used)
        sparsity = 1 - (n_features_used / n_features_total) if n_features_total > 0 else 0

        rule_depths = [len(rule.conditions) for rule in rules]
        max_depth = max(rule_depths) if rule_depths else 0
        mean_rule_depth = np.mean(rule_depths) if rule_depths else 0

        total_rules = len(rules)

        alpha, beta, gamma = 1, 1, 1
        denominator = alpha * max_depth + beta * mean_rule_depth + gamma * total_rules
        SI = 100 / denominator if denominator > 0 else float('inf')

        return {
            "features_used": n_features_used,
            "total_features": n_features_total,
            "sparsity": sparsity,
            "total_rules": total_rules,
            "max_depth": max_depth,
            "mean_rule_depth": mean_rule_depth,
            "sparsity_interpretability_score": SI,
        }

    @staticmethod
    def display_metrics(y_true, y_pred, correct, total, file=None):
        """
        Computes and displays classification performance metrics.

        This method calculates standard evaluation metrics including accuracy, precision,
        recall, F1 score, specificity, and the confusion matrix. The results are printed to
        the console and optionally written to a file.

        Args:
            y_true (List[int]): List of true class labels.
            y_pred (List[int]): List of predicted class labels.
            correct (int): Number of correct predictions.
            total (int): Total number of predictions.
            file (Optional[TextIO]): File object to write the metrics to. If None, metrics are only printed.
        """
        y_pred_safe = [p if p is not None else -1 for p in y_pred] # Use a placeholder for None

        tp = sum(1 for yt, yp in zip(y_true, y_pred_safe) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred_safe) if yt != 1 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred_safe) if yt == 0 and yp == 0)
        fn = sum(1 for yt, yp in zip(y_true, y_pred_safe) if yt == 1 and yp == 0)

        accuracy = correct / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f'\nCorrect: {correct}, Errors: {total - correct}, Total: {total}')
        print(f'Accuracy: {accuracy:.5f}')
        print(f'Precision: {precision:.5f}')
        print(f'Recall: {recall:.5f}')
        print(f'F1 Score: {f1:.5f}')
        print(f'Specificity: {specificity:.5f}')

        # Ensure labels are plain Python ints, not numpy types
        labels = sorted(int(l) for l in set(y_true) | set(y for y in y_pred_safe if y != -1))
        cm = confusion_matrix(y_true, y_pred_safe, labels=labels)

        print("\nConfusion Matrix with Labels:")
        print("Labels:", labels)
        print(cm)

        if file:
            file.write(f'\nCorrect: {correct}, Errors: {total - correct}, Total: {total}\n')
            file.write(f'Accuracy: {accuracy:.5f}\n')
            file.write(f'Precision: {precision:.5f}\n')
            file.write(f'Recall: {recall:.5f}\n')
            file.write(f'F1 Score: {f1:.5f}\n')
            file.write(f'Specificity: {specificity:.5f}\n')
            file.write("\nConfusion Matrix with Labels:\n")
            file.write(f"Labels: {labels}\n")
            file.write(f"{cm}\n")

    # Method to compare initial and final results
    def compare_initial_final_results(self, file_path):
        """
        Compares the classification performance of the initial and final rule sets.

        This method evaluates both the original (`initial_rules`) and pruned (`final_rules`) rule sets on the same dataset, and logs performance metrics such as:
        - Accuracy,
        - Confusion matrices,
        - Divergent predictions between the two rule sets,
        - Interpretability metrics per tree.

        It delegates to algorithm-specific methods based on the classifier type.

        Args:
            file_path (str): Path to the CSV file used for evaluation.
        """
        # Robust data loading
        _, _, X_test, y_test, _, target_column_name, feature_names = RuleClassifier.process_data(".", file_path, is_test_only=True)
        df_test = pd.DataFrame(X_test, columns=feature_names)
        df_test[target_column_name] = y_test

        if self.algorithm_type == 'Random Forest':
            test_data = df_test.to_dict('records')
            self.compare_initial_final_results_rf(test_data, target_column_name)
        elif self.algorithm_type == 'Decision Tree':
            self.compare_initial_final_results_dt(df_test, target_column_name)
        else:
            raise ValueError(f"Unsupported algorithm type: {self.algorithm_type}")

    def compare_initial_final_results_dt(self, df_test, target_column_name):
        """
        Compares the performance of the initial and final rule sets for a Decision Tree model using a vectorized methodology similar to Random Forest.
        This method evaluates both the initial and final rule sets on a given test DataFrame, computes classification metrics, identifies divergent cases (where predictions differ between the two rule sets), and calculates interpretability metrics such as sparsity. Results are printed to the console and saved to 'examples/files/output_final_classifier_dt.txt'.

        Args:
            df_test (pd.DataFrame): The test dataset to be classified.
            target_column_name (str): The name of the column containing the true class labels.

        Side Effects:
            - Prints classification results, divergent cases, and interpretability metrics to the console.
            - Writes the same information to 'examples/files/output_final_classifier_dt.txt'.

        Notes:
            - This method is intended to be used as part of the RuleClassifier class.
            - Rule usage and error counts are updated for both initial and final rule sets.
            - Divergent cases include the index, feature values, initial and final predictions, and the actual class.
            - Interpretability metrics include sparsity and feature usage statistics for both rule sets.
        """
        print("\n*********************************************************************************************************")
        print("******************************* RUNNING INITIAL AND FINAL CLASSIFICATIONS *******************************")
        print("*********************************************************************************************************\n")

        df = df_test.copy()
        y_true = df[target_column_name].astype(int)
        indices = df.index

        with open('examples/files/output_final_classifier_dt.txt', 'w') as f:

            print("\n******************************* INITIAL MODEL *******************************\n")
            f.write("\n******************************* INITIAL MODEL *******************************\n")
            start_time_initial = time.time()

            # Initialize predictions as None
            y_pred_initial = pd.Series([None]*len(df), index=df.index)
            for rule in self.initial_rules:
                rule.usage_count = 0
                rule.error_count = 0
                if not rule.conditions or rule.conditions[0] == '':
                    condition_query = y_pred_initial.isnull()
                else:
                    query = " & ".join([f"`{var}` {op} {val}" for var, op, val in rule.parsed_conditions])
                    unpredicted_indices = y_pred_initial[y_pred_initial.isnull()].index
                    try:
                        matched_indices = df.loc[unpredicted_indices].query(query).index
                        condition_query = pd.Series(df.index.isin(matched_indices), index=df.index)
                    except Exception:
                        continue
                predicted_class = int(rule.class_.replace('Class', ''))
                y_pred_initial.loc[condition_query] = predicted_class
                matches = df[condition_query]
                rule.usage_count = len(matches)
                if not matches.empty:
                    errors = matches[matches[target_column_name] != predicted_class]
                    rule.error_count = len(errors)
            y_pred_initial = y_pred_initial.infer_objects(copy=False).astype(int)
            correct_initial = (y_pred_initial == y_true).sum()
            RuleClassifier.display_metrics(y_true.tolist(), y_pred_initial.tolist(), correct_initial, len(y_true), f)
            end_time_initial = time.time()

            print(f"\nNumber of initial rules: {len(self.initial_rules)}")
            f.write(f"\nNumber of initial rules: {len(self.initial_rules)}\n")

            print(f"\nTime elapsed in executing initial model classifications: {end_time_initial - start_time_initial:.3f} seconds")
            f.write(f"\nTime elapsed in executing initial model classifications: {end_time_initial - start_time_initial:.3f} seconds\n")

            print("\n******************************* FINAL MODEL *******************************\n")
            f.write("\n******************************* FINAL MODEL *******************************\n")
            start_time_final = time.time()

            y_pred_final = pd.Series([None]*len(df), index=df.index)
            for rule in self.final_rules:
                rule.usage_count = 0
                rule.error_count = 0
                if not rule.conditions or rule.conditions[0] == '':
                    condition_query = y_pred_final.isnull()
                else:
                    query = " & ".join([f"`{var}` {op} {val}" for var, op, val in rule.parsed_conditions])
                    unpredicted_indices = y_pred_final[y_pred_final.isnull()].index
                    try:
                        matched_indices = df.loc[unpredicted_indices].query(query).index
                        condition_query = pd.Series(df.index.isin(matched_indices), index=df.index)
                    except Exception:
                        continue
                predicted_class = int(rule.class_.replace('Class', ''))
                y_pred_final.loc[condition_query] = predicted_class
                matches = df[condition_query]
                rule.usage_count = len(matches)
                if not matches.empty:
                    errors = matches[matches[target_column_name] != predicted_class]
            y_pred_final = y_pred_final.infer_objects(copy=False).fillna(-1).astype(int)
            correct_final = (y_pred_final == y_true).sum()
            RuleClassifier.display_metrics(y_true.tolist(), y_pred_final.tolist(), correct_final, len(y_true), f)
            end_time_final = time.time()

            print(f"\nNumber of final rules: {len(self.final_rules)}")
            f.write(f"\nNumber of final rules: {len(self.final_rules)}\n")

            print(f"\nTime elapsed in executing final model classifications: {end_time_final - start_time_final:.3f} seconds")
            f.write(f"\nTime elapsed in executing final model classifications: {end_time_final - start_time_final:.3f} seconds\n")

            print("\n******************************* DIVERGENT CASES *******************************\n")
            f.write("\n******************************* DIVERGENT CASES *******************************\n")
            divergent_cases = []
            for idx in indices:
                init_pred = y_pred_initial.at[idx]
                final_pred = y_pred_final.at[idx]
                actual = y_true.at[idx]
                if init_pred != final_pred:
                    case = {
                        'index': idx + 1,
                        'data': {k: v for k, v in df.loc[idx].items() if k != target_column_name},
                        'initial_class': init_pred,
                        'final_class': final_pred,
                        'actual_class': actual
                    }
                    divergent_cases.append(case)
                    print(f"Index: {case['index']}, Data: {case['data']}, Initial Class: {case['initial_class']}, "
                          f"Final Class: {case['final_class']}, Actual Class: {case['actual_class']}")
                    f.write(f"Index: {case['index']}, Data: {case['data']}, Initial Class: {case['initial_class']}, "
                            f"Final Class: {case['final_class']}, Actual Class: {case['actual_class']}\n")
            if not divergent_cases:
                print("No divergent cases found.")
                f.write("No divergent cases found.\n")

            # --- Interpretability Metrics ---
            print("\n******************************* INTERPRETABILITY METRICS *******************************\n")
            f.write("\n******************************* INTERPRETABILITY METRICS *******************************\n")
            all_features = set(k for k in df.columns if k != target_column_name)
            n_features_total = len(all_features)

            print("\nMetrics (Initial):")
            f.write("\nMetrics (Initial):\n")
            sparsity_info_initial = RuleClassifier.calculate_sparsity_interpretability(self.initial_rules, n_features_total)
            for key, value in sparsity_info_initial.items():
                if isinstance(value, float):
                    # Use scientific notation for very small or large numbers
                    if abs(value) < 1e-3 or abs(value) > 1e4:
                        print(f"  {key.replace('_', ' ').title()}: {value:.2e}")
                        f.write(f"  {key.replace('_', ' ').title()}: {value:.2e}\n")
                    else:
                        print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
                        f.write(f"  {key.replace('_', ' ').title()}: {value:.2f}\n")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}")
                    f.write(f"  {key.replace('_', ' ').title()}: {value}\n")

            print("\nMetrics (Final):")
            f.write("\nMetrics (Final):\n")
            sparsity_info_final = RuleClassifier.calculate_sparsity_interpretability(self.final_rules, n_features_total)
            # Calculate percentage difference for each metric
            for key, value in sparsity_info_final.items():
                initial_value = sparsity_info_initial.get(key)
                percent_diff_str = ""
                # Only compute percentage if initial_value is not None and is a number
                if isinstance(value, (int, float)) and isinstance(initial_value, (int, float)) and initial_value != 0:
                    percent_diff = ((value - initial_value) / initial_value) * 100
                    sign = "+" if percent_diff >= 0 else ""
                    percent_diff_str = f" ({sign}{percent_diff:.1f}%)"
                # Formatting
                if isinstance(value, float):
                    if abs(value) < 1e-1 or abs(value) > 1e5:
                        print(f"  {key.replace('_', ' ').title()}: {value:.2e}{percent_diff_str}")
                        f.write(f"  {key.replace('_', ' ').title()}: {value:.2e}{percent_diff_str}\n")
                    else:
                        print(f"  {key.replace('_', ' ').title()}: {value:.2f}{percent_diff_str}")
                        f.write(f"  {key.replace('_', ' ').title()}: {value:.2f}{percent_diff_str}\n")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}{percent_diff_str}")
                    f.write(f"  {key.replace('_', ' ').title()}: {value}{percent_diff_str}\n")

    # Method to compare initial and final results for Random Forest using vectorized methodology
    def compare_initial_final_results_rf(self, df_test, target_column_name):
        """
        Compares the performance of the initial and final rule sets for a Random Forest model using a vectorized methodology.
        This method evaluates both the initial and final rule sets on a given test DataFrame, computes classification metrics, identifies divergent cases (where predictions differ between the two rule sets), and calculates interpretability metrics such as sparsity. Results are printed to the console and saved to 'examples/files/output_final_classifier_rf.txt'.

        Args:
            df_test (pd.DataFrame or list of dict): The test dataset to be classified. Can be a DataFrame or a list of dictionaries.
            target_column_name (str): The name of the column containing the true class labels.

        Side Effects:
            - Prints classification results, divergent cases, and interpretability metrics to the console.
            - Writes the same information to 'examples/files/output_final_classifier_rf.txt'.

        Notes:
            - This method is intended to be used as part of the RuleClassifier class.
            - Rule usage and error counts are updated for both initial and final rule sets.
            - Divergent cases include the index, feature values, initial and final predictions, and the actual class.
            - Interpretability metrics include sparsity and feature usage statistics for both rule sets.
        """
        print("\n*********************************************************************************************************")
        print("******************************* RUNNING INITIAL AND FINAL CLASSIFICATIONS *******************************")
        print("*********************************************************************************************************\n")

        # Convert list of dicts to DataFrame if necessary
        if isinstance(df_test, list):
            df = pd.DataFrame(df_test)
        else:
            df = df_test.copy()
        y_true = df[target_column_name].astype(int)
        indices = df.index

        with open('examples/files/output_final_classifier_rf.txt', 'w') as f:
            print("\n******************************* INITIAL MODEL *******************************\n")
            f.write("\n******************************* INITIAL MODEL *******************************\n")
            start_time_initial = time.time()

            # Group rules by tree
            tree_rules = defaultdict(list)
            for rule in self.initial_rules:
                rule.usage_count = 0
                rule.error_count = 0
                tree_name = rule.name.split('_')[0]
                tree_rules[tree_name].append(rule)

            class_labels = sorted({int(r.class_[-1]) for r in self.initial_rules})
            class_to_index = {label: i for i, label in enumerate(class_labels)}
            all_probas = np.zeros((len(df), len(class_labels)))

            for i, (tree_name, rules) in enumerate(tree_rules.items()):
                tree_class_votes = np.zeros((len(df), len(class_labels)))
                for rule in rules:
                    if not rule.conditions or rule.conditions[0] == '':
                        matched_indices = df.index
                    else:
                        query = " & ".join([f"`{var}` {op} {val}" for var, op, val in rule.parsed_conditions])
                        try:
                            matched_indices = df.query(query).index
                        except Exception:
                            continue
                    if not matched_indices.empty:
                        predicted_class = int(rule.class_.replace('Class', ''))
                        class_idx = class_to_index.get(predicted_class)
                        if class_idx is not None:
                            tree_class_votes[matched_indices, class_idx] += 1
                        rule.usage_count = len(matched_indices)
                vote_sums = tree_class_votes.sum(axis=1, keepdims=True)
                np.divide(tree_class_votes, vote_sums, where=vote_sums != 0, out=tree_class_votes)
                all_probas += tree_class_votes

            avg_probas = all_probas / len(tree_rules) if len(tree_rules) > 0 else all_probas
            y_pred_indices = np.argmax(avg_probas, axis=1)
            y_pred_initial = np.array(class_labels)[y_pred_indices]
            correct_initial = (y_pred_initial == y_true.values).sum()
            RuleClassifier.display_metrics(y_true.tolist(), y_pred_initial.tolist(), correct_initial, len(y_true), f)
            end_time_initial = time.time()

            print(f"\nNumber of initial rules: {len(self.initial_rules)}")
            f.write(f"\nNumber of initial rules: {len(self.initial_rules)}\n")

            print(f"\nTime elapsed in executing initial model classifications: {end_time_initial - start_time_initial:.3f} seconds")
            f.write(f"\nTime elapsed in executing initial model classifications: {end_time_initial - start_time_initial:.3f} seconds\n")

            print("\n******************************* FINAL MODEL *******************************\n")
            f.write("\n******************************* FINAL MODEL *******************************\n")
            start_time_final = time.time()

            tree_rules_final = defaultdict(list)
            for rule in self.final_rules:
                rule.usage_count = 0
                rule.error_count = 0
                tree_name = rule.name.split('_')[0]
                tree_rules_final[tree_name].append(rule)

            class_labels_final = sorted({int(r.class_[-1]) for r in self.final_rules})
            class_to_index_final = {label: i for i, label in enumerate(class_labels_final)}
            all_probas_final = np.zeros((len(df), len(class_labels_final)))

            for i, (tree_name, rules) in enumerate(tree_rules_final.items()):
                tree_class_votes = np.zeros((len(df), len(class_labels_final)))
                for rule in rules:
                    if not rule.conditions or rule.conditions[0] == '':
                        matched_indices = df.index
                    else:
                        query = " & ".join([f"`{var}` {op} {val}" for var, op, val in rule.parsed_conditions])
                        try:
                            matched_indices = df.query(query).index
                        except Exception:
                            continue
                    if not matched_indices.empty:
                        predicted_class = int(rule.class_.replace('Class', ''))
                        class_idx = class_to_index_final.get(predicted_class)
                        if class_idx is not None:
                            tree_class_votes[matched_indices, class_idx] += 1
                        rule.usage_count = len(matched_indices)
                vote_sums = tree_class_votes.sum(axis=1, keepdims=True)
                np.divide(tree_class_votes, vote_sums, where=vote_sums != 0, out=tree_class_votes)
                all_probas_final += tree_class_votes

            avg_probas_final = all_probas_final / len(tree_rules_final) if len(tree_rules_final) > 0 else all_probas_final
            y_pred_indices_final = np.argmax(avg_probas_final, axis=1)
            y_pred_final = np.array(class_labels_final)[y_pred_indices_final]
            correct_final = (y_pred_final == y_true.values).sum()
            RuleClassifier.display_metrics(y_true.tolist(), y_pred_final.tolist(), correct_final, len(y_true), f)
            end_time_final = time.time()

            print(f"\nNumber of final rules: {len(self.final_rules)}")
            f.write(f"\nNumber of final rules: {len(self.final_rules)}\n")

            print(f"\nTime elapsed in executing final model classifications: {end_time_final - start_time_final:.3f} seconds")
            f.write(f"\nTime elapsed in executing final model classifications: {end_time_final - start_time_final:.3f} seconds\n")

            print("\n******************************* DIVERGENT CASES *******************************\n")
            f.write("\n******************************* DIVERGENT CASES *******************************\n")
            divergent_cases = []
            for idx in indices:
                init_pred = y_pred_initial[idx] if idx < len(y_pred_initial) else None
                final_pred = y_pred_final[idx] if idx < len(y_pred_final) else None
                actual = y_true.at[idx]
                if init_pred != final_pred:
                    case = {
                        'index': idx + 1,
                        'data': {k: v for k, v in df.loc[idx].items() if k != target_column_name},
                        'initial_class': init_pred,
                        'final_class': final_pred,
                        'actual_class': actual
                    }
                    divergent_cases.append(case)
                    print(f"Index: {case['index']}, Data: {case['data']}, Initial Class: {case['initial_class']}, "
                          f"Final Class: {case['final_class']}, Actual Class: {case['actual_class']}")
                    f.write(f"Index: {case['index']}, Data: {case['data']}, Initial Class: {case['initial_class']}, "
                            f"Final Class: {case['final_class']}, Actual Class: {case['actual_class']}\n")
            if not divergent_cases:
                print("No divergent cases found.")
                f.write("No divergent cases found.\n")

            print("\n******************************* INTERPRETABILITY METRICS *******************************\n")
            f.write("\n******************************* INTERPRETABILITY METRICS *******************************\n")
            all_features = set(k for k in df.columns if k != target_column_name)
            n_features_total = len(all_features)

            print("\nMetrics (Initial):")
            f.write("\nMetrics (Initial):\n")
            sparsity_info_initial = RuleClassifier.calculate_sparsity_interpretability(self.initial_rules, n_features_total)
            for key, value in sparsity_info_initial.items():
                if isinstance(value, float):
                    # Use scientific notation for very small or large numbers
                    if abs(value) < 1e-1 or abs(value) > 1e5:
                        print(f"  {key.replace('_', ' ').title()}: {value:.4e}")
                        f.write(f"  {key.replace('_', ' ').title()}: {value:.4e}\n")
                    else:
                        print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
                        f.write(f"  {key.replace('_', ' ').title()}: {value:.4f}\n")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}")
                    f.write(f"  {key.replace('_', ' ').title()}: {value}\n")

            print("\nMetrics (Final):")
            f.write("\nMetrics (Final):\n")
            sparsity_info_final = RuleClassifier.calculate_sparsity_interpretability(self.final_rules, n_features_total)
            # Calculate percentage difference for each metric
            for key, value in sparsity_info_final.items():
                initial_value = sparsity_info_initial.get(key)
                percent_diff_str = ""
                # Only compute percentage if initial_value is not None and is a number
                if isinstance(value, (int, float)) and isinstance(initial_value, (int, float)) and initial_value != 0:
                    percent_diff = ((value - initial_value) / initial_value) * 100
                    sign = "+" if percent_diff >= 0 else ""
                    percent_diff_str = f" ({sign}{percent_diff:.1f}%)"
                # Formatting
                if isinstance(value, float):
                    if abs(value) < 1e-1 or abs(value) > 1e5:
                        print(f"  {key.replace('_', ' ').title()}: {value:.2e}{percent_diff_str}")
                        f.write(f"  {key.replace('_', ' ').title()}: {value:.2e}{percent_diff_str}\n")
                    else:
                        print(f"  {key.replace('_', ' ').title()}: {value:.2f}{percent_diff_str}")
                        f.write(f"  {key.replace('_', ' ').title()}: {value:.2f}{percent_diff_str}\n")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}{percent_diff_str}")
                    f.write(f"  {key.replace('_', ' ').title()}: {value}{percent_diff_str}\n")

    # ************************ RULE EDITING MODULE ************************

    # Add this helper method inside the RuleClassifier class
    def _validate_and_parse_condition(self, condition_str):
        """
        Validates and parses a single rule condition string entered by the user.

        This helper method ensures that a condition string follows the expected format "variable operator value" (e.g., "v5 > 10.5"). It validates the operator, attempts to cast the value to a float, and returns both a formatted condition string and a parsed tuple representation. If the condition is invalid, it returns None.

        Args:
            condition_str (str): The raw condition string provided by the user.

        Returns:
            Tuple[str,Tuple[str,str,float]]|None:
                - If valid: A tuple containing formatted condition: string (e.g., "v5 > 10.5"), parsed representation (variable, operator, value) (e.g., ("v5", ">", 10.5)).
                - If invalid: None.
        """
        try:
            parts = condition_str.split(' ')
            if len(parts) != 3:
                return None

            var, op, value_str = parts
            value = float(value_str)

            if op not in ['<=', '>=', '<', '>']:
                return None

            # Returns the condition as a string and as a parsed tuple
            return f"{var} {op} {value}", (var, op, value)
        except (ValueError, IndexError):
            return None

    # Method to edit rules manually
    def edit_rules(self):
        """
        Starts an interactive terminal session to inspect and manually edit final rules.

        This method provides a simple REPL-like interface to:
        - List all current **final** rules with their names, predicted class labels, and conditions;
        - Select a rule by number or exact name;
        - Add or remove conditions to/from the selected rule;
        - Change the predicted class of the selected rule;
        - Persist the edits and return to the main menu.

        On save, the method re-parses the rule's conditions to keep the cached `parsed_conditions` in sync and appends an "_edited" suffix to the rule's name (once) to indicate the rule was modified manually. The entire classifier (self) is serialized to ``'examples/files/edited_model.pkl'``.

        Notes:
            - Edits are applied to ``self.final_rules``. If you have not run the analysis or otherwise populated ``final_rules``, this method will exit early with a message.
            - Condition format must be exactly: ``"<variable> <operator> <value>"`` where ``operator`` ∈ {``<=``, ``>=``, ``<``, ``>``} and ``value`` is numeric.
        """

        print("\n*********************************************************************************************************")
        print("************************************* MANUAL RULE EDITING MODE *************************************")
        print("*********************************************************************************************************\n")

        # Always work on the list of final rules
        if not self.final_rules:
            print("No final rules to edit. Please run the analysis first.")
            return

        while True:
            print("\n--- Current Rules ---")
            for i, rule in enumerate(self.final_rules):
                print(f"  [{i+1}] {rule.name}: Class={rule.class_}, Conditions={rule.conditions}")

            print("\nEnter the NUMBER or NAME of the rule you want to edit (or 'exit' to finish):")
            user_input = input("> ").strip()

            if user_input.lower() == 'exit':
                break

            selected_rule = None
            # Try to interpret as number (1-based index)
            try:
                rule_index = int(user_input) - 1
                if 0 <= rule_index < len(self.final_rules):
                    selected_rule = self.final_rules[rule_index]
                else:
                    print(f"ERROR: Number '{user_input}' is out of range. Try again.")
                    continue
            except ValueError:
                # Not a number, try to match by rule name (case-insensitive)
                for rule in self.final_rules:
                    if rule.name.lower() == user_input.lower():
                        selected_rule = rule
                        break
                if not selected_rule:
                    print(f"ERROR: No rule found with name '{user_input}'. Try again.")
                    continue
            # Editing loop for the selected rule
            while True:
                print(f"\n--- Editing Rule: {selected_rule.name} ---")
                print(f"  Current Class: {selected_rule.class_}")
                print(f"  Current Conditions:")
                for i, cond in enumerate(selected_rule.conditions):
                    print(f"    [{i}] {cond}")

                print("\nEdit Options:")
                print("  [a]dd condition")
                print("  [r]emove condition")
                print("  [c]lass (change the prediction class)")
                print("  [s]ave and return to main menu")

                action = input("Choose an action > ").strip().lower()

                if action == 'a':
                    new_cond_str = input("Enter the new condition (format: 'variable operator value', e.g., 'v5 > 10.5'): ")
                    validation_result = self._validate_and_parse_condition(new_cond_str)
                    if validation_result:
                        formatted_cond, parsed_cond = validation_result
                        selected_rule.conditions.append(formatted_cond)
                        print(f"Condition '{formatted_cond}' added.")
                    else:
                        print("ERROR: Invalid condition format. Use 'variable operator value' (e.g., v1 <= 0.5).")

                elif action == 'r':
                    if not selected_rule.conditions:
                        print("This rule has no conditions to remove.")
                        continue
                    try:
                        idx_to_remove = int(input("Enter the index of the condition to remove: "))
                        if 0 <= idx_to_remove < len(selected_rule.conditions):
                            removed = selected_rule.conditions.pop(idx_to_remove)
                            print(f"Condition '{removed}' removed.")
                        else:
                            print("ERROR: Index out of range.")
                    except ValueError:
                        print("ERROR: Please enter a valid index number.")

                elif action == 'c':
                    new_class = input("Enter the new class for this rule (e.g., 'Class1'): ")
                    selected_rule.class_ = new_class
                    print(f"Rule class changed to '{new_class}'.")

                elif action == 's':
                    # *** Crucial step: Re-parse the conditions to maintain consistency ***
                    selected_rule.parsed_conditions = self.parse_conditions_static(selected_rule.conditions)

                    # Add a mark to indicate it was manually edited
                    if not selected_rule.name.endswith("_edited"):
                        selected_rule.name += "_edited"

                    print(f"Rule '{selected_rule.name}' successfully updated.")
                    # Save the edited classifier to a file for later access
                    with open('examples/files/edited_model.pkl', 'wb') as f_out:
                        pickle.dump(self, f_out)
                    print("Edited classifier saved to 'examples/files/edited_model.pkl'.")
                    break # Exit the editing loop for the current rule

                else:
                    print("Invalid option. Try again.")

        print("\n************************************** END OF EDITING MODE **************************************\n")

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

    # ************************ GENERATING SCIKIT-LEARN MODEL ************************
    @staticmethod
    def process_data (train_path, test_path, is_test_only=False):
        """
        Loads and processes training and testing data from CSV files.

        This method:
        - Reads training and test datasets,
        - Splits features and labels,
        - Encodes class labels using scikit-learn's LabelEncoder.

        Args:
            train_path (str): File path to the training CSV dataset.
            test_path (str): File path to the testing CSV dataset.
            is_test_only (bool): If True, only processes the test set and train_path is ignored.

        Returns:
            A tuple containing processed data and metadata.
        """

        # Check if the file has a header
        def has_header(file_path):
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    first_line = f.readline().strip()
                # If the first line is empty or contains non-numeric characters (except separators), assume it's a header
                has_header = any(c.isalpha() for c in first_line)
                return has_header
            except Exception:
                return False # Default to assuming no header on error

        df_train = None
        if not is_test_only:
            header_train = 0 if has_header(train_path) else None
            df_train = pd.read_csv(train_path, header=header_train, encoding='latin-1', on_bad_lines='skip')
            if header_train != 0:
                num_features = df_train.shape[1] - 1
                df_train.columns = [f'v{i+1}' for i in range(num_features)] + ['class']
            else:
                # Remove leading/trailing spaces from column names
                df_train.columns = [col.strip() for col in df_train.columns]

        header_test = 0 if has_header(test_path) else None
        df_test = pd.read_csv(test_path, header=header_test, encoding='latin-1', on_bad_lines='skip')
        if header_test != 0:
            num_features = df_test.shape[1] - 1
            df_test.columns = [f'v{i+1}' for i in range(num_features)] + ['class']
        else:
            # Remove leading/trailing spaces from column names
            df_test.columns = [col.strip() for col in df_test.columns]

        if is_test_only:
            target_column_name = df_test.columns[-1]
            feature_names = df_test.columns[:-1].tolist()
            X_test = df_test.iloc[:, :-1].values.astype(float)
            y_test = df_test.iloc[:, -1].values.astype(float)
            return None, None, X_test, y_test, None, target_column_name, feature_names

        # Encode all string columns (categorical features and label)
        for col in df_train.columns:
            if df_train[col].dtype == 'object':
                le = LabelEncoder()
                df_train[col] = le.fit_transform(df_train[col].astype(str))
                if col in df_test.columns:
                    # Apply the same encoding to the test set, handling unseen labels
                    df_test[col] = df_test[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    df_test[col] = le.transform(df_test[col].astype(str))

        target_column_name = df_train.columns[-1]
        feature_names = df_train.columns[:-1].tolist()
        class_names = df_train[target_column_name].unique()

        print("Number of classes:", len(class_names))
        print("Classes names:", class_names)
        print("Number of samples in training set:", df_train.shape[0])
        print("Number of samples in test set:", df_test.shape[0])

        X_train = df_train.iloc[:, :-1].values.astype(float)
        y_train = df_train.iloc[:, -1].values.astype(float)
        X_test = df_test.iloc[:, :-1].values.astype(float)
        y_test = df_test.iloc[:, -1].values.astype(float)

        return X_train, y_train, X_test, y_test, class_names, target_column_name, feature_names

    # Method to extract rules from a tree model
    @staticmethod
    def get_rules(tree, feature_names, class_names):
        """
        Extracts human-readable decision rules from a scikit-learn DecisionTreeClassifier.

        This method traverses the tree structure to generate logical condition paths from root to leaf,
        and organizes them by predicted class.

        Args:
            tree (DecisionTreeClassifier): A trained scikit-learn decision tree model.
            feature_names (List[str]): A list of feature names corresponding to the tree input features.
            class_names (List[str]): A list of class names corresponding to output labels.

        Returns:
            Dict[str,List[str]]: A dictionary mapping each class name to a list of rule strings that lead to predictions for that class.
        """

        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        paths = []

        def recurse(node, path):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                # Left child
                recurse(tree_.children_left[node], path + [f"{name} <= {np.round(threshold, 3)}"])
                # Right child
                recurse(tree_.children_right[node], path + [f"{name} > {np.round(threshold, 3)}"])
            else:
                # It's a leaf node
                class_distribution = tree_.value[node][0]
                predicted_class_index = np.argmax(class_distribution)
                paths.append((path, predicted_class_index))

        recurse(0, [])

        # Ensure class_names are strings for mapping
        class_names_str = [str(c) for c in class_names]
        rules_by_class = {class_name: [] for class_name in class_names_str}

        for path, class_idx in paths:
            if class_idx < len(class_names_str):
                class_name = class_names_str[class_idx]
                rule_str = f"[{', '.join(path)}]" if path else "[]"
                rules_by_class[class_name].append(rule_str)

        return rules_by_class

    # Method to extract rules from a Random Forest model
    @staticmethod
    def get_tree_rules(model, feature_names, class_names, algorithm_type='Random Forest'):
        """
        Extracts rules from a trained scikit-learn model (Decision Tree or Random Forest).

        For Decision Trees, this returns one rule set. For Random Forests, it aggregates rule sets
        from all individual decision trees.

        Args:
            model (Union[DecisionTreeClassifier, RandomForestClassifier]): The trained model.
            feature_names (List[str]): List of feature names.
            class_names (List[str]): List of class names.
            algorithm_type (str): Type of model; either 'Decision Tree' or 'Random Forest'.

        Returns:
            List[Dict[str,List[str]]]: A list of rule sets, each as a dictionary mapping class names to rule strings.
        """
        print("Feature names:", feature_names)

        rules = []
        if algorithm_type == 'Random Forest':
            for estimator in model.estimators_:
                rules.append(RuleClassifier.get_rules(estimator, feature_names, class_names))
        elif algorithm_type == 'Decision Tree':
            rules.append(RuleClassifier.get_rules(model, feature_names, class_names))
        else:
            raise ValueError(f"Unsupported algorithm type: {algorithm_type}")
        return rules

    # Method to generate a classifier model based on rules
    @staticmethod
    def generate_classifier_model(rules, class_names_map, algorithm_type='Random Forest'):
        """
        Converts a list of extracted rule sets into a RuleClassifier instance.

        This method formats rule sets into a standardized string format and initializes
        a RuleClassifier object with it. The resulting classifier is saved to 'files/initial_model.pkl'.

        Args:
            rules (List[Dict[str, List[str]]]): A list of rule dictionaries, each mapping class names to rule strings.
            class_names_map (Dict[str, int]): A map from class name to class index.
            algorithm_type (str): The type of model the rules originated from ('Random Forest' or 'Decision Tree').

        Returns:
            RuleClassifier: A RuleClassifier instance initialized with the given rules.
        """
        rules_text = ""        
        for i, rule_set in enumerate(rules):
            rule_index_counter = 1
            for class_name, class_rules in rule_set.items():
                class_index = class_names_map.get(str(class_name), -1)
                for rule in class_rules:
                    rules_text += f"DT{i+1}_Rule{rule_index_counter}_Class{class_index}: {rule}\n"
                    rule_index_counter += 1

        classifier = RuleClassifier(rules_text, algorithm_type=algorithm_type)
        print(f"Algorithm Type: {classifier.algorithm_type}")

        path = 'examples/files/initial_model.pkl'
        with open(path, 'wb') as model_file:
            pickle.dump(classifier, model_file)
        print(f"Classifier file saved: {path}")
        return classifier

    @staticmethod
    def new_classifier(train_path, test_path, model_parameters, model_path=None, algorithm_type='Random Forest'):
        """
        Trains or loads a model, extracts decision rules, and builds a rule-based classifier.

        This method either loads an existing scikit-learn model or trains a new one using the provided
        training dataset and model parameters. It evaluates the model on test data, saves it, extracts
        decision rules, and constructs a corresponding `RuleClassifier` object.

        Args:
            train_path (str): Path to the training CSV file. Each row should contain features and the target label.
            test_path (str): Path to the test CSV file. Each row should contain features and the target label.
            model_parameters (dict): Parameters to initialize the scikit-learn model. Must match the accepted parameters of either ``sklearn.tree.DecisionTreeClassifier`` or ``sklearn.ensemble.RandomForestClassifier``, depending on the value of ``algorithm_type``.
            model_path (Optional[str]): Path to a pre-trained model file (.pkl). If provided, skips training.
            algorithm_type (str, optional): Type of model to use ('Random Forest' or 'Decision Tree').
                Defaults to 'Random Forest'.

        Returns:
            RuleClassifier: A rule-based classifier instance constructed from the trained or loaded model.
        """

        print("\n*********************************************************************************************************")
        print("************************************** GENERATING A NEW CLASSIFIER **************************************")
        print("*********************************************************************************************************\n")

        print("\nDatabase details:")
        X_train, y_train, X_test, y_test, class_names, _, feature_names = RuleClassifier.process_data(train_path, test_path)

        if model_path:
            print(f"Loading model from: {model_path}")
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)
        else:
            print("Training a new Scikit-Learn model")
            if algorithm_type == 'Random Forest':
                model = RandomForestClassifier(**model_parameters)
            elif algorithm_type == 'Decision Tree':
                model = DecisionTreeClassifier(**model_parameters)
            else:
                raise ValueError(f"Unsupported algorithm type: {algorithm_type}")
            model.fit(X_train, y_train)

        print("\nTesting model:")
        y_pred = model.predict(X_test)
        print("\nRESULTS SUMMARY:")
        correct = np.sum(y_pred == y_test)
        total = len(y_test)
        RuleClassifier.display_metrics(y_test, y_pred, correct, total)

        # Create a mapping from class names to their integer labels
        class_names_map = {str(name): i for i, name in enumerate(np.unique(y_train))}

        rules = RuleClassifier.get_tree_rules(model, feature_names, np.unique(y_train), algorithm_type=algorithm_type)

        print("\nGenerating classifier model:")
        classifier = RuleClassifier.generate_classifier_model(rules, class_names_map, algorithm_type)

        return classifier
