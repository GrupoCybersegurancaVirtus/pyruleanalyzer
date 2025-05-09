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

# Class to represent a rule
class Rule:
    def __init__(self, name, class_, conditions):
        self.name = name
        self.class_ = class_
        self.conditions = conditions
        self.usage_count = 0
        self.error_count = 0

# Class to handle the rule classification process
class RuleClassifier:
    def __init__(self, rules, algorithm_type='Decision Tree'):
        self.initial_rules = self.parse_rules(rules, algorithm_type)
        self.algorithm_type = algorithm_type
        self.final_rules, self.duplicated_rules = [], [] 
        self.specific_rules = []

    # Method to parse the rules from string based on the algorithm type
    def parse_rules(self, rules, algorithm_type):
        rules = rules.replace('"', '').replace('- ','').strip().split('\n')
        
        if algorithm_type == 'Random Forest':
            return [self.parse_rf_rule(rule) for rule in rules if rule]
        if algorithm_type == 'Decision Tree':
            return [self.parse_dt_rule(rule) for rule in rules if rule]

    # Method to parse the rules for Decision Tree
    def parse_dt_rule(self, rule):
        rule = rule.strip().split(':', 1)
        rule_name = rule[0].strip()
        class_ = rule_name.split('_')[-1]
        conditions = rule[1].strip().replace('[', '').replace(']', '').split(', ') if len(rule) > 1 else []
        return Rule(rule_name, class_, conditions)

    # Method to parse the rules for Random Forest
    def parse_rf_rule(self, rule):
        rule = rule.split(':')
        rule_name, conditions = rule[0].strip(), rule[1].strip()
        class_ = rule_name.split('_')[-1]
        conditions = conditions.replace('[', '').replace(']', '').split(', ')
        return Rule(rule_name, class_, conditions)
    
    # Method to parse conditions from string to tuple (variable, operator, value)
    def parse_conditions(self, conditions):
        parsed_conditions = []
        for condition in conditions:
            if '<=' in condition:
                var, value = condition.split(' <= ')
                parsed_conditions.append((var, '<=', float(value)))
            elif '>=' in condition:
                var, value = condition.split(' >= ')
                parsed_conditions.append((var, '>=', float(value)))
            elif '<' in condition:
                var, value = condition.split(' < ')
                parsed_conditions.append((var, '<', float(value)))
            elif '>' in condition:
                var, value = condition.split(' > ')
                parsed_conditions.append((var, '>', float(value)))
        return parsed_conditions

    # Method to execute the classification process
    def classify(self, data, final=False):
        rules = self.final_rules if final else self.initial_rules

        if self.algorithm_type == 'Random Forest':
            return self.classify_rf(data, rules)

        if self.algorithm_type == 'Decision Tree':
            return self.classify_dt(data, rules)
    
    # Method to classify data using Decision Tree rules
    def classify_dt(self, data, rules):
        for rule in rules:
                    rule_satisfied = True
                    parsed_conditions = self.parse_conditions(rule.conditions)
                    for (var, op, value) in parsed_conditions:
                        instance_value = data[var]
                        if instance_value is None:
                            rule_satisfied = False
                            break
                        
                        if op == '<=' and not (instance_value <= value):
                            rule_satisfied = False
                            break
                        elif op == '>=' and not (instance_value >= value):
                            rule_satisfied = False
                            break
                        elif op == '<' and not (instance_value < value):
                            rule_satisfied = False
                            break
                        elif op == '>' and not (instance_value > value):
                            rule_satisfied = False
                            break
                    
                    if rule_satisfied:
                        parts = rule.name.split('_')
                        rule.usage_count += 1  # Increment rule usage count
                        for part in parts:
                            if part.startswith('Class'):
                                return int(part.replace('Class', '')), None, None  # Return class label
                
        # If no rule satisfied, return None
        return None, None, None

    # Method to classify data using Random Forest rules    
    def classify_rf(self,data, rules):
        # Identify unique class labels and create a mapping
            class_labels = sorted({int(rule.class_[-1]) for rule in rules})
            class_to_index = {label: idx for idx, label in enumerate(class_labels)}
            num_classes = len(class_labels)

            tree_rules = defaultdict(list)
            for rule in rules:
                tree_name = rule.name.split('_')[0]
                tree_rules[tree_name].append(rule)

            probas = []

            for rules in tree_rules.values():
                proba_tree = np.zeros(num_classes)

                matched_classes = []

                # Verify if the conditions of the rules are satisfied
                for rule in rules:
                    parsed_conditions = self.parse_conditions(rule.conditions)
                    if all(var in data and (
                            data[var] <= float(value) if op == '<=' else
                            data[var] >= float(value) if op == '>=' else
                            data[var] < float(value) if op == '<' else
                            data[var] > float(value)
                    ) for var, op, value in parsed_conditions):
                        class_label = int(rule.class_[-1])
                        matched_classes.append(class_label)
                        rule.usage_count += 1
                
                if matched_classes:
                    class_counts = Counter(matched_classes)
                    total = sum(class_counts.values())
                    for label, count in class_counts.items():
                        idx = class_to_index[label]
                        proba_tree[idx] = count / total
                probas.append(proba_tree)

            if not probas:
                return None, [], [0.0] * num_classes

            # Calculate the average probabilities across trees (soft voting)
            avg_proba = np.mean(probas, axis=0)

            # Class with highest average probability
            predicted_class_index = int(np.argmax(avg_proba))
            predicted_class = class_labels[predicted_class_index]

            votes = [class_labels[int(np.argmax(p))] for p in probas]

            return predicted_class, votes, avg_proba.tolist()
    
    # Method to extract variables and operators from conditions
    def extract_variables_and_operators(self, conditions):
        vars_ops = []
        for cond in conditions:
            parts = cond.split(' ')
            if len(parts) >= 3:  # Granting that the condition is well-formed (e.g., "var op value")
                var = parts[0]
                op = parts[1]
                # Normalizes similar operators (<= and < are treated as equivalent, > and >= also)
                if op in ['<=', '<']:
                    op = '<='
                elif op in ['>=', '>']:
                    op = '>='
                vars_ops.append((var, op))
        # Sort by variable name and operator
        return sorted(vars_ops)

    # Method to find similar rules between trees, considering the variables and operators
    def find_duplicated_rules_between_trees(self):
        similar_rules = []
        for i, rule1 in enumerate(self.initial_rules):
            for j, rule2 in enumerate(self.initial_rules):
                if i >= j:
                    continue
                vars_ops1 = self.extract_variables_and_operators(rule1.conditions)
                vars_ops2 = self.extract_variables_and_operators(rule2.conditions)
                
                # Verify if the variables and operators are the same (values may differ) and if the resulting classes are equal
                if vars_ops1 == vars_ops2 and rule1.class_ == rule2.class_:
                    similar_rules.append((rule1, rule2))
        return similar_rules
    
    # Method to find duplicated rules in the same tree
    def find_duplicated_rules(self):
        duplicated_rules = []
        for i, rule1 in enumerate(self.final_rules):
            for j, rule2 in enumerate(self.final_rules):
                if i >= j:
                    continue
                # Compare conditions up to the penultimate condition
                if len(rule1.conditions) == len(rule2.conditions) and rule1.class_ == rule2.class_:
                    if rule1.conditions[:-1] == rule2.conditions[:-1]:
                        # Check if the last condition differs only by the operator and value
                        last_cond1 = rule1.conditions[-1]
                        last_cond2 = rule2.conditions[-1]
                        if ('<=' in last_cond1 and '>' in last_cond2 or
                            '>' in last_cond1 and '<=' in last_cond2):
                                duplicated_rules.append((rule1, rule2))
        return duplicated_rules
    
    # Method to set a custom rule removal function
    def set_custom_rule_removal(self, custom_function):
        self.custom_rule_removal = custom_function

    # Method to remove rules based on custom logic
    def custom_rule_removal(self, rules):
        # Example custom logic to remove rules with specific conditions
        # This can be customized based on your requirements
        return rules, [] # Placeholder for custom logic

    # Method to adjust and remove duplicated rules
    def adjust_and_remove_rules(self, method):
        if method == "custom":
            return self.custom_rule_removal(self.initial_rules)
        
        print("\nANALYSING DUPLICATED RULES IN THE SAME TREE")
        similar_rules = self.find_duplicated_rules()

        unique_rules = []
        duplicated_rules = set()

        # Find duplicated rules in the same tree
        for rule1, rule2 in similar_rules:
            duplicated_rules.add(rule1)
            duplicated_rules.add(rule2)
            print(f"\nDuplicated rules from the same tree: {rule1.name} == {rule2.name}")
            print(f"{rule1.name}: {rule1.conditions}")
            print(f"{rule2.name}: {rule2.conditions}")

            # Create a new rule based on the common conditions of the duplicated rules
            common_conditions = rule1.conditions[:-1]  # Use the common conditions up to the penultimate condition
            new_rule_name = f"{rule1.name}_&_{rule2.name}"
            new_rule_class = rule1.class_  # Assuming both rules have the same class
            new_rule = Rule(new_rule_name, new_rule_class, common_conditions)

            print(f"New rule created: {new_rule.name} with conditions: {new_rule.conditions}")
 
            # Add the new rule to the unique rules list
            unique_rules.append(new_rule)

        unique_rules_between_trees = []
        duplicated_rules_between_trees = set()

        if method == "hard":

            print("\nANALYSING DUPLICATED RULES BETWEEN TREES")

            if self.algorithm_type == 'Decision Tree':
                # Remove duplicated rules from the same tree
                print("\nThere is only one tree, so no duplicated rules between trees.")

            similar_rules_between_trees = self.find_duplicated_rules_between_trees()
            for rule1, rule2 in similar_rules_between_trees:

                duplicated_rules_between_trees.add(rule1)
                duplicated_rules_between_trees.add(rule2)
                print(f"\nDuplicated rules between trees: {rule1.name} == {rule2.name}")
                print(f"{rule1.name}: {rule1.conditions}")
                print(f"{rule2.name}: {rule2.conditions}")

                # Create a new rule based on the common conditions of the duplicated rules
                common_conditions = rule1.conditions[:-1]  # Use the common conditions up to the penultimate condition
                new_rule_name = f"{rule1.name}_&_{rule2.name}"
                new_rule_class = rule1.class_  # Assuming both rules have the same class
                new_rule = Rule(new_rule_name, new_rule_class, common_conditions)

                print(f"New rule created: {new_rule.name} with conditions: {new_rule.conditions}")
    
                # Add the new rule to the unique rules list
                unique_rules_between_trees.append(new_rule)

            # Combine duplicated rules from the same tree and between trees
            duplicated_rules = list(duplicated_rules.union(duplicated_rules_between_trees))
            
        for rule in self.final_rules:
            if rule not in duplicated_rules:
                unique_rules.append(rule)
        
        return unique_rules, similar_rules
    
    # Method to execute the rule analysis and identify duplicated rules
    # remove_duplicates = "soft" (in the same tree, probably does not affect the final metrics), "hard" (between trees, may affect the final metrics), "custom" (custom function to remove duplicates) or "none" (no removal)
    # remove_below_n_classifications = -1 (no removal), 0 (removal of rules with 0 classifications), or any other integer (removal of rules with equal or less than n classifications)
    def execute_rule_analysis(self, file_path, remove_duplicates="none", remove_below_n_classifications=-1):
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

        # Print the final rules after analysis
        print("\nFinal Rules After Analysis:")
        for rule in self.final_rules:
            print(f"Rule: {rule.name}, Class: {rule.class_}, Conditions: {rule.conditions}")


        if self.algorithm_type == 'Random Forest':
            self.execute_rule_analysis_rf(file_path, remove_below_n_classifications)
        elif self.algorithm_type == 'Decision Tree':
            self.execute_rule_analysis_dt(file_path, remove_below_n_classifications)
        else:
            raise ValueError(f"Unsupported algorithm type: {self.algorithm_type}")

    # Method to execute the rule analysis for RDecision Tree
    def execute_rule_analysis_dt(self, file_path, remove_below_n_classifications=-1):
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        with open('examples/files/output_classifier_dt.txt', 'w') as f:
            with open(file_path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                i = 1
                errors = ""
                print("\nTESTING RULES AFTER ANALYSIS")
                for row in reader:
                    print(f'\nIndex: {i}')
                    f.write(f'\nIndex: {i}\n')
                    i += 1
                    data = {f'v{i+1}': float(value) for i, value in enumerate(row[:-1])}
                    predicted_class = self.classify(data, final=True)[0]
                    actual_class = int(row[-1])
                    y_true.append(actual_class)
                    y_pred.append(predicted_class)
                    if predicted_class != actual_class:
                        print(f'ERROR: Predicted: {predicted_class}, Actual: {actual_class}')
                        f.write(f'ERROR: Predicted: {predicted_class}, Actual: {actual_class}\n')
                        errors += f'\nIndex: {i-1}\nERROR: Predicted: {predicted_class}, Actual: {actual_class}\n'
                        for rule in self.final_rules:
                            parsed_conditions = self.parse_conditions(rule.conditions)
                            if all(var in data and (
                                    data[var] <= value if op == '<=' else
                                    data[var] >= value if op == '>=' else
                                    data[var] < value if op == '<' else
                                    data[var] > value
                            ) for var, op, value in parsed_conditions):
                                rule.usage_count += 1
                                if predicted_class != actual_class:
                                    rule.error_count = getattr(rule, 'error_count', 0) + 1
                    else:
                        print(f'Predicted: {predicted_class}, Actual: {actual_class}')
                        f.write(f'Predicted: {predicted_class}, Actual: {actual_class}\n')
                        correct += 1
                    total += 1

            if remove_below_n_classifications != -1:
                f.write(f"\nRules removed with usage count below {remove_below_n_classifications}:\n")
                for rule in self.final_rules[:]:
                    if rule.usage_count <= remove_below_n_classifications:
                        f.write(f"Rule: {rule.name}, Count: {rule.usage_count}\n")
                        self.specific_rules.append(rule)
                        self.final_rules.remove(rule)

            accuracy = correct / total if total > 0 else 0
            
            print("\nRESULTS SUMMARY:")
            f.write("\nRESULTS SUMMARY:\n")

            print(f'\nCorrect: {correct}, Errors: {total - correct}, Accuracy: {accuracy:.5f}')
            f.write(f'\nCorrect: {correct}, Errors: {total - correct}, Accuracy: {accuracy:.5f}\n')

            # Compute confusion matrix
            labels = sorted(set(y_true))
            cm = confusion_matrix(y_true, y_pred, labels=labels) 

            # Print confusion matrix with labels
            print("\nConfusion Matrix with Labels:")
            f.write("\nConfusion Matrix with Labels:\n")
            print("Labels:", labels)
            f.write(f"Labels: {labels}\n")
            print(cm)
            f.write(f"{cm}\n")

            print("\nErrors: \n" + errors + "\n")
            f.write("\nErrors: \n" + errors + "\n")

            # Print each rule with its usage count
            print("\nRule Usage Counts:")
            f.write("\nRule Usage Counts:\n")
            for rule in self.initial_rules:
                print(f"Rule: {rule.name}, Count: {rule.usage_count}")
                f.write(f"Rule: {rule.name}, Count: {rule.usage_count}\n")

            # Print the rules with the most errors
            print("\nRules with most Errors:")
            f.write("\nRules with most Errors:\n")
            sorted_rules = sorted(self.final_rules, key=lambda r: r.error_count, reverse=True)
            for rule in sorted_rules:
                if rule.error_count > 0:
                    print(f"Rule: {rule.name}, Errors: {rule.error_count}")
                    f.write(f"Rule: {rule.name}, Errors: {rule.error_count}\n")

            # Print the final rules
            f.write("\nFinal Rules:\n")
            for rule in self.final_rules:
                f.write(f"Rule: {rule.name}, Class: {rule.class_}, Conditions: {rule.conditions}\n")

            print("\n******************************* SUMMARY *******************************\n")

            # Print the total number of initial and final rules
            print(f"\nTotal Initial Rules: {len(self.initial_rules)}")
            f.write(f"\nTotal Initial Rules: {len(self.initial_rules)}\n")
            print(f"Total Final Rules: {len(self.final_rules)}")
            f.write(f"Total Final Rules: {len(self.final_rules)}\n")

            # Print the total number of duplicated rules
            print(f"\nTotal Duplicated Rules: {len(self.initial_rules) - len(self.final_rules) + len(self.specific_rules)}")
            f.write(f"\nTotal Duplicated Rules: {len(self.initial_rules) - len(self.final_rules) + len(self.specific_rules)}\n")

            # Print the total number of specific rules
            if remove_below_n_classifications > -1:
                print(f"\nTotal Specific Rules: {len(self.specific_rules)} (<= {remove_below_n_classifications} classifications)")
                f.write(f"\nTotal Specific Rules: {len(self.specific_rules)} (<= {remove_below_n_classifications} classifications)\n")

        return self
    
    # Method to execute the rule analysis for Random Forest
    def execute_rule_analysis_rf(self, file_path,remove_below_n_classifications=-1):
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        with open('examples/files/output_classifier.txt', 'w') as f:
                with open(file_path, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    i=1
                    errors = ""
                    print("\nTESTING RULES AFTER ANALYSIS")
                    for row in reader:
                        print(f'\nIndex: {i}') 
                        f.write(f'\nIndex: {i}\n')
                        i+=1
                        data = {f'v{i+1}': float(value) for i, value in enumerate(row[:-1])}
                        predicted_class, votes, proba = self.classify(data, final=True)
                        if predicted_class != int(row[-1]):
                            for rule in self.final_rules:
                                parsed_conditions = self.parse_conditions(rule.conditions)
                                if all(var in data and (data[var] <= value if op == '<=' else
                                                        data[var] >= value if op == '>=' else
                                                        data[var] < value if op == '<' else
                                                        data[var] > value) for var, op, value in parsed_conditions):
                                    rule.usage_count += 1
                                    rule.error_count = getattr(rule, 'error_count', 0) + 1
                        class_vote_counts = {cls: votes.count(cls) for cls in set(votes)}
                        print(f'Votes: {votes}\nClass Votes: {class_vote_counts}\nNumber of classifications: {len(votes)}')
                        f.write(f'Votes: {votes}\nClass Votes: {class_vote_counts}\nNumber of classifications: {len(votes)}\n')
                        print(f"Probabilities: {proba}")
                        f.write(f"Probabilities: {proba}\n")
                        actual_class = int(row[-1])
                        y_true.append(actual_class)
                        y_pred.append(predicted_class)
                        if predicted_class != actual_class:
                            print(f'ERROR: Predicted: {predicted_class}, Actual: {actual_class}')
                            f.write(f'ERROR: Predicted: {predicted_class}, Actual: {actual_class}\n')
                            errors += f'\nIndex: {i-1}\nVotes: {votes}\nClass Votes: {class_vote_counts}\nNumber of classifications: {len(votes)}\nProbabilities: {proba}\nERRO: Predicted: {predicted_class}, Actual: {actual_class}\n'
                        if predicted_class == actual_class:
                            print(f'Predicted: {predicted_class}, Actual: {actual_class}')
                            f.write(f'Predicted: {predicted_class}, Actual: {actual_class}\n')
                            correct += 1
                        total += 1

                if remove_below_n_classifications != -1:
                    print(f"\nRules removed with usage count below {remove_below_n_classifications}:")
                    f.write(f"\nRules removed with usage count below {remove_below_n_classifications}:\n")
                    for rule in self.final_rules[:]:
                        if rule.usage_count <= remove_below_n_classifications:
                            print(f"Rule: {rule.name}, Count: {rule.usage_count}")
                            f.write(f"Rule: {rule.name}, Count: {rule.usage_count}\n")
                            self.specific_rules.append(rule)
                            self.final_rules.remove(rule)

                accuracy = correct / total if total > 0 else 0

                print("\nRESULTS SUMMARY:")
                f.write("\nRESULTS SUMMARY:\n")

                print(f'\nCorrect: {correct}, Errors: {total - correct}, Accuracy: {accuracy:.5f}')
                f.write(f'\nCorrect: {correct}, Errors: {total - correct}, Accuracy: {accuracy:.5f}\n')
                
                # Filter out None values from y_true and y_pred
                y_true_filtered = [y for y, y_p in zip(y_true, y_pred) if y_p is not None]
                y_pred_filtered = [y_p for y_p in y_pred if y_p is not None]

                # Compute confusion matrix
                labels = sorted(set(y_true_filtered))
                cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=labels)
                
                # Print confusion matrix with labels
                print("\nConfusion Matrix with Labels:")
                f.write("\nConfusion Matrix with Labels:\n")
                print("Labels:", labels)
                f.write(f"Labels: {labels}\n")
                print(cm)
                f.write(f"{cm}\n")

                print("\nErrors: \n" + errors + "\n")
                f.write("\nErrors: \n" + errors + "\n")

                # Print each rule with its usage count
                print("\nRule Usage Counts:")
                f.write("\nRule Usage Counts:\n")
                for rule in self.initial_rules:
                    print(f"Rule: {rule.name}, Count: {rule.usage_count}")
                    f.write(f"Rule: {rule.name}, Count: {rule.usage_count}\n")

                # Sum the usage counts by tree
                tree_usage_counts = {}
                for rule in self.initial_rules:
                    tree_name = rule.name.split('_')[0]
                    if tree_name not in tree_usage_counts:
                        tree_usage_counts[tree_name] = 0
                    tree_usage_counts[tree_name] += rule.usage_count

                # Print the usage counts for each tree
                print("\nTree Usage Counts:")
                f.write("\nTree Usage Counts:\n")
                for tree_name, count in tree_usage_counts.items():
                    print(f"Tree: {tree_name}, Total Usage Count: {count}")
                    f.write(f"Tree: {tree_name}, Total Usage Count: {count}\n")

                # Print the initial rules
                print("\nInitial Rules:")
                f.write("\nInitial Rules:\n")
                for rule in self.initial_rules:
                    print(f"Rule: {rule.name}, Class: {rule.class_}, Conditions: {rule.conditions}")
                    f.write(f"Rule: {rule.name}, Class: {rule.class_}, Conditions: {rule.conditions}\n")
                
                # Print the final rules
                print("\nFinal Rules:")
                f.write("\nFinal Rules:\n")
                for rule in self.final_rules:
                    print(f"Rule: {rule.name}, Class: {rule.class_}, Conditions: {rule.conditions}")
                    f.write(f"Rule: {rule.name}, Class: {rule.class_}, Conditions: {rule.conditions}\n")

                # Count the number of rules for each tree in initial rules
                initial_tree_rule_counts = {}
                for rule in self.initial_rules:
                    tree_name = rule.name.split('_')[0]
                    if tree_name not in initial_tree_rule_counts:
                        initial_tree_rule_counts[tree_name] = 0
                    initial_tree_rule_counts[tree_name] += 1

                # Print the number of rules for each tree in initial rules
                print("\nInitial Tree Rule Counts:")
                f.write("\nInitial Tree Rule Counts:\n")
                for tree_name, count in initial_tree_rule_counts.items():
                    print(f"Tree: {tree_name}, Rule Count: {count}")
                    f.write(f"Tree: {tree_name}, Rule Count: {count}\n")

                # Count the number of rules for each tree in final rules
                final_tree_rule_counts = {}
                for rule in self.final_rules:
                    tree_name = rule.name.split('_')[0]
                    if tree_name not in final_tree_rule_counts:
                        final_tree_rule_counts[tree_name] = 0
                    final_tree_rule_counts[tree_name] += 1

                # Print the number of rules for each tree in final rules
                print("\nFinal Tree Rule Counts:")
                f.write("\nFinal Tree Rule Counts:\n")
                for tree_name, count in final_tree_rule_counts.items():
                    print(f"Tree: {tree_name}, Rule Count: {count}")
                    f.write(f"Tree: {tree_name}, Rule Count: {count}\n")

                # Print the rules with errors and their error counts
                print("\nRules with most Errors:")
                f.write("\nRules with most Errors:\n")
                sorted_rules = sorted(self.final_rules, key=lambda r: r.error_count, reverse=True)
                for rule in sorted_rules:
                    if rule.error_count > 0:
                        print(f"Rule: {rule.name}, Errors: {rule.error_count}")
                        f.write(f"Rule: {rule.name}, Errors: {rule.error_count}\n")

                print("\n******************************* SUMMARY *******************************\n")

                # Print the total number of initial and final rules 
                print(f"\nTotal Initial Rules: {len(self.initial_rules)}")
                f.write(f"\nTotal Initial Rules: {len(self.initial_rules)}\n")
                print(f"Total Final Rules: {len(self.final_rules)}")
                f.write(f"Total Final Rules: {len(self.final_rules)}\n")
                
                # Print the total number of duplicated rules
                print(f"\nTotal Duplicated Rules: {len(self.initial_rules) - len(self.final_rules) + len(self.specific_rules)}")
                f.write(f"\nTotal Duplicated Rules: {len(self.initial_rules) - len(self.final_rules) + len(self.specific_rules)}\n")

                # Print the total number of specific rules
                if remove_below_n_classifications > -1:
                    print(f"\nTotal Specific Rules: {len(self.specific_rules)} (<= {remove_below_n_classifications} classifications)")
                    f.write(f"\nTotal Specific Rules: {len(self.specific_rules)} (<= {remove_below_n_classifications} classifications)\n")

                # Save the initial model to a .pkl file
                with open('examples/files/final_model.pkl', 'wb') as file:
                    pickle.dump(self, file)
        return self

    def calculate_sparsity_interpretability(rules, n_features_total):
        # Extract unique features used in the rules
        features_used = set()
        for rule in rules:
            for condition in rule.conditions:
                feature = condition.split(' ')[0]  # Extract the feature name
                features_used.add(feature)

        # Compute sparsity
        n_features_used = len(features_used)
        sparsity = 1 - (n_features_used / n_features_total)

        # Calculate rule depths (number of conditions per rule)
        rule_depths = [len(rule.conditions) for rule in rules]
        max_depth = max(rule_depths) if rule_depths else 0
        mean_rule_depth = np.mean(rule_depths) if rule_depths else 0

        # Total number of rules
        total_rules = len(rules)

        # Sparsity Interpretability Score (SI)
        alpha, beta, gamma = 1, 1, 1  # Adjustable weights
        SI = 100 / (alpha * max_depth + beta * mean_rule_depth + gamma * total_rules)

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
        # Calculate accuracy, precision, recall, F1 score and specificity
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp and yt == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != yp and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp and yt == 0)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt != yp and yp == 0)
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Print metrics
        print(f'\nCorrect: {correct}, Errors: {total - correct}, Total: {total}')
        print(f'Accuracy: {accuracy:.5f}')
        print(f'Precision: {precision:.5f}')
        print(f'Recall: {recall:.5f}')
        print(f'F1 Score: {f1:.5f}')
        print(f'Specificity: {specificity:.5f}')

        # Compute confusion matrix
        labels = sorted(set(y_true))
        # Compute confusion matrix manually
        cm = [[0 for _ in labels] for _ in labels]
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        for yt, yp in zip(y_true, y_pred):
            if yt in label_to_index and yp in label_to_index:
                cm[label_to_index[yt]][label_to_index[yp]] += 1

        print("\nConfusion Matrix with Labels:")
        print("Labels:", labels)
        for row in cm:
            print(row)
        
        # Write metrics to file if provided
        if file:
            file.write(f'\nCorrect: {correct}, Errors: {total - correct}\n')
            file.write(f'Accuracy: {accuracy:.5f}\n')
            file.write(f'Precision: {precision:.5f}\n')
            file.write(f'Recall: {recall:.5f}\n')
            file.write(f'F1 Score: {f1:.5f}\n')
            file.write(f'Specificity: {specificity:.5f}\n')

            file.write("\nConfusion Matrix with Labels:\n")
            file.write(f"Labels: {labels}\n")
            for row in cm:
                file.write(f"{row}\n")

    # Method to compare initial and final results
    def compare_initial_final_results(self, file_path):
        if self.algorithm_type == 'Random Forest':
            self.compare_initial_final_results_rf(file_path)
        elif self.algorithm_type == 'Decision Tree':
            self.compare_initial_final_results_dt(file_path)
        else:
            raise ValueError(f"Unsupported algorithm type: {self.algorithm_type}")
        
    # Method to compare initial and final results for Decision Tree
    def compare_initial_final_results_dt(self, file_path):
        print("\n*********************************************************************************************************")
        print("******************************* RUNNING INITIAL AND FINAL CLASSIFICATIONS *******************************")
        print("*********************************************************************************************************\n")
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        with open('examples/files/output_final_classifier_dt.txt', 'w') as f:
            with open(file_path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                print("\n******************************* INITIAL MODEL *******************************\n")
                f.write("\n******************************* INITIAL MODEL *******************************\n")
                i = 1
                for row in reader:
                    data = {f'v{i+1}': float(value) for i, value in enumerate(row[:-1])}
                    predicted_class, _, _ = self.classify(data)
                    actual_class = int(row[-1])
                    y_true.append(actual_class)
                    y_pred.append(predicted_class)
                    if predicted_class == actual_class:
                        correct += 1
                    total += 1
            RuleClassifier.display_metrics(y_true, y_pred, correct, total, f)

            print("\n******************************* FINAL MODEL *******************************\n")
            f.write("\n******************************* FINAL MODEL *******************************\n")

            correct_final = 0
            total_final = 0
            y_true_final = []
            y_pred_final = []
            with open(file_path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                i = 1
                for row in reader:
                    data = {f'v{i+1}': float(value) for i, value in enumerate(row[:-1])}
                    predicted_class, _, _ = self.classify(data, final=True)
                    actual_class = int(row[-1])
                    y_true_final.append(actual_class)
                    y_pred_final.append(predicted_class)
                    if predicted_class == actual_class:
                        correct_final += 1
                    total_final += 1

            RuleClassifier.display_metrics(y_true_final, y_pred_final, correct_final, total_final, f)

            print("\n******************************* DIVERGENT CASES *******************************\n")
            f.write("\n******************************* DIVERGENT CASES *******************************\n")

            divergent_cases = []
            with open(file_path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                i = 1
                for row in reader:
                    data = {f'v{i+1}': float(value) for i, value in enumerate(row[:-1])}
                    initial_predicted_class, _, _ = self.classify(data)
                    final_predicted_class, _, _ = self.classify(data, final=True)
                    if initial_predicted_class != final_predicted_class:
                        divergent_cases.append({
                            'index': i,
                            'data': data,
                            'initial_class': initial_predicted_class,
                            'final_class': final_predicted_class,
                            'actual_class': int(row[-1])                           
                    })
                    i += 1

                if not divergent_cases:
                    print("No divergent cases found.")
                    f.write("No divergent cases found.\n")
                else:
                    for case in divergent_cases:
                        print(f"Index: {case['index']}, Data: {case['data']}, Initial Class: {case['initial_class']}, "
                            f"Final Class: {case['final_class']}, Actual Class: {case['actual_class']}")
                        f.write(f"Index: {case['index']}, Data: {case['data']}, Initial Class: {case['initial_class']}, "
                                f"Final Class: {case['final_class']}, Actual Class: {case['actual_class']}\n")
                
            print("\n******************************* INTERPRETABILITY METRICS *******************************\n")
            f.write("\n******************************* INTERPRETABILITY METRICS *******************************\n")
            # Calculate sparsity and interpretability for each tree
            # Calculate sparsity and interpretability for initial rules
            tree_sparsity_info = {}
            for rule in self.initial_rules:
                tree_name = rule.name.split('_')[0]
                if tree_name not in tree_sparsity_info:
                    tree_sparsity_info[tree_name] = []
                tree_sparsity_info[tree_name].append(rule)

            for tree_name, rules in tree_sparsity_info.items():
                n_features_total = len(set(cond.split(' ')[0] for rule in rules for cond in rule.conditions))
                sparsity_info = RuleClassifier.calculate_sparsity_interpretability(rules, n_features_total)
                print(f"\nTree (Initial):")
                print(f"  Features Used: {sparsity_info['features_used']}/{sparsity_info['total_features']}")
                print(f"  Sparsity: {sparsity_info['sparsity']:.2f}")
                print(f"  Total Rules: {sparsity_info['total_rules']}")
                print(f"  Max Rule Depth: {sparsity_info['max_depth']}")
                print(f"  Mean Rule Depth: {sparsity_info['mean_rule_depth']:.2f}")
                print(f"  Sparsity Interpretability Score: {sparsity_info['sparsity_interpretability_score']:.2f}")
                f.write(f"Tree (Initial): {tree_name}\n")
                f.write(f"  Features Used: {sparsity_info['features_used']}/{sparsity_info['total_features']}\n")
                f.write(f"  Sparsity: {sparsity_info['sparsity']:.2f}\n")
                f.write(f"  Total Rules: {sparsity_info['total_rules']}\n")
                f.write(f"  Max Rule Depth: {sparsity_info['max_depth']}\n")
                f.write(f"  Mean Rule Depth: {sparsity_info['mean_rule_depth']:.2f}\n")
                f.write(f"  Sparsity Interpretability Score: {sparsity_info['sparsity_interpretability_score']:.2f}\n")

            # Calculate sparsity and interpretability for final rules
            tree_sparsity_info = {}
            for rule in self.final_rules:
                tree_name = rule.name.split('_')[0]
                if tree_name not in tree_sparsity_info:
                    tree_sparsity_info[tree_name] = []
                tree_sparsity_info[tree_name].append(rule)

            for tree_name, rules in tree_sparsity_info.items():
                n_features_total = len(set(cond.split(' ')[0] for rule in rules for cond in rule.conditions))
                sparsity_info = RuleClassifier.calculate_sparsity_interpretability(rules, n_features_total)
                print(f"\nTree (Final):")
                print(f"  Features Used: {sparsity_info['features_used']}/{sparsity_info['total_features']}")
                print(f"  Sparsity: {sparsity_info['sparsity']:.2f}")
                print(f"  Total Rules: {sparsity_info['total_rules']}")
                print(f"  Max Rule Depth: {sparsity_info['max_depth']}")
                print(f"  Mean Rule Depth: {sparsity_info['mean_rule_depth']:.2f}")
                print(f"  Sparsity Interpretability Score: {sparsity_info['sparsity_interpretability_score']:.2f}")
                f.write(f"Tree (Final): {tree_name}\n")
                f.write(f"  Features Used: {sparsity_info['features_used']}/{sparsity_info['total_features']}\n")
                f.write(f"  Sparsity: {sparsity_info['sparsity']:.2f}\n")
                f.write(f"  Total Rules: {sparsity_info['total_rules']}\n")
                f.write(f"  Max Rule Depth: {sparsity_info['max_depth']}\n")
                f.write(f"  Mean Rule Depth: {sparsity_info['mean_rule_depth']:.2f}\n")
                f.write(f"  Sparsity Interpretability Score: {sparsity_info['sparsity_interpretability_score']:.2f}\n")
                    
    # Method to compare initial and final results for Random Forest            
    def compare_initial_final_results_rf(self, file_path):
        print("\n*********************************************************************************************************")
        print("******************************* RUNNING INITIAL AND FINAL CLASSIFICATIONS *******************************")
        print("*********************************************************************************************************\n")
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        with open('examples/files/output_final_classifier.txt', 'w') as f:
                with open(file_path, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    print("\n******************************* INITIAL MODEL *******************************\n")
                    f.write("\n******************************* INITIAL MODEL *******************************\n")
                    i=1
                    errors = ""
                    for row in reader:
                        i+=1
                        data = {f'v{i+1}': float(value) for i, value in enumerate(row[:-1])}
                        predicted_class, votes, proba = self.classify(data) 
                        class_vote_counts = {cls: votes.count(cls) for cls in set(votes)}
                        actual_class = int(row[-1])
                        y_true.append(actual_class)
                        y_pred.append(predicted_class)
                        if predicted_class != actual_class:
                            errors += f'\nIndex: {i-1}\nVotes: {votes}\nClass Votes: {class_vote_counts}\nNumber of classifications: {len(votes)}\nProbabilities: {proba}\nERRO: Predicted: {predicted_class}, Actual: {actual_class}\n'
                        if predicted_class == actual_class:
                            correct += 1
                        total += 1

                RuleClassifier.display_metrics(y_true, y_pred, correct, total, f)

                # Count the number of rules for each tree in initial rules
                initial_tree_rule_counts = {}
                for rule in self.initial_rules:
                    tree_name = rule.name.split('_')[0]
                    if tree_name not in initial_tree_rule_counts:
                        initial_tree_rule_counts[tree_name] = 0
                    initial_tree_rule_counts[tree_name] += 1

                # Print the number of rules for each tree in initial rules
                total_rules = sum(initial_tree_rule_counts.values())
                print(f"Total Rules: {total_rules}")
                f.write(f"Total Rules: {total_rules}\n")


                print("\n******************************* FINAL MODEL *******************************\n")
                f.write("\n******************************* FINAL MODEL *******************************\n")

                correct_final = 0
                total_final = 0
                y_true_final = []
                y_pred_final = []
                with open(file_path, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    i=1
                    errors = ""
                    for row in reader:
                        i+=1
                        data = {f'v{i+1}': float(value) for i, value in enumerate(row[:-1])}
                        predicted_class, votes, proba = self.classify(data, final=True)
                        class_vote_counts = {cls: votes.count(cls) for cls in set(votes)}
                        actual_class = int(row[-1])
                        y_true_final.append(actual_class)
                        y_pred_final.append(predicted_class)
                        if predicted_class != actual_class:
                            errors += f'\nIndex: {i-1}\nVotes: {votes}\nClass Votes: {class_vote_counts}\nNumber of classifications: {len(votes)}\nProbabilities: {proba}\nERRO: Predicted: {predicted_class}, Actual: {actual_class}\n'
                        if predicted_class == actual_class:
                            correct_final += 1
                        total_final += 1

                RuleClassifier.display_metrics(y_true_final, y_pred_final, correct_final, total_final, f)

                final_tree_rule_counts = {}
                for rule in self.final_rules:
                    tree_name = rule.name.split('_')[0]
                    if tree_name not in final_tree_rule_counts:
                        final_tree_rule_counts[tree_name] = 0
                    final_tree_rule_counts[tree_name] += 1

                # Print the number of rules for each tree in final rules
                total_rules = sum(final_tree_rule_counts.values())
                print(f"Total Rules: {total_rules}")
                f.write(f"Total Rules: {total_rules}\n")

                # Track cases where the initial classification diverged from the final classification

                print("\n******************************* DIVERGENT CASES *******************************\n")
                f.write("\n******************************* DIVERGENT CASES *******************************\n")
                
                divergent_cases = []
                with open(file_path, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    i = 1
                    for row in reader:
                        data = {f'v{i+1}': float(value) for i, value in enumerate(row[:-1])}
                        initial_predicted_class, _, _ = self.classify(data)
                        final_predicted_class, _, _ = self.classify(data, final=True)
                        if initial_predicted_class != final_predicted_class:
                            divergent_cases.append({
                                'index': i,
                                'data': data,
                                'initial_class': initial_predicted_class,
                                'final_class': final_predicted_class,
                                'actual_class': int(row[-1])
                            })
                        i += 1


                if not divergent_cases:
                    print("No divergent cases found.")
                    f.write("No divergent cases found.\n")
                else:
                    for case in divergent_cases:
                        print(f"Index: {case['index']}, Data: {case['data']}, Initial Class: {case['initial_class']}, "
                            f"Final Class: {case['final_class']}, Actual Class: {case['actual_class']}")
                        f.write(f"Index: {case['index']}, Data: {case['data']}, Initial Class: {case['initial_class']}, "
                                f"Final Class: {case['final_class']}, Actual Class: {case['actual_class']}\n")

                print("\n******************************* INTERPRETABILITY METRICS *******************************\n")
                f.write("\n******************************* INTERPRETABILITY METRICS *******************************\n")

                # Calculate sparsity and interpretability for each tree in Random Forest
                tree_sparsity_info = {}
                for rule in self.initial_rules:
                    tree_name = rule.name.split('_')[0]
                    if tree_name not in tree_sparsity_info:
                        tree_sparsity_info[tree_name] = []
                    tree_sparsity_info[tree_name].append(rule)

                total_features_used = 0
                total_features = 0
                total_rules = 0
                total_max_depth = 0
                total_mean_rule_depth = 0
                total_sparsity_interpretability_score = 0
                tree_count = len(tree_sparsity_info)

                for tree_name, rules in tree_sparsity_info.items():
                    n_features_total = len(set(cond.split(' ')[0] for rule in rules for cond in rule.conditions))
                    sparsity_info = RuleClassifier.calculate_sparsity_interpretability(rules, n_features_total)
                    total_features_used += sparsity_info['features_used']
                    total_features += sparsity_info['total_features']
                    total_rules += sparsity_info['total_rules']
                    total_max_depth += sparsity_info['max_depth']
                    total_mean_rule_depth += sparsity_info['mean_rule_depth']
                    total_sparsity_interpretability_score += sparsity_info['sparsity_interpretability_score']

                # Calculate and print averages
                if tree_count > 0:
                    avg_features_used = total_features_used / tree_count
                    avg_features = total_features / tree_count
                    avg_rules = total_rules / tree_count
                    avg_max_depth = total_max_depth / tree_count
                    avg_mean_rule_depth = total_mean_rule_depth / tree_count
                    avg_sparsity_interpretability_score = total_sparsity_interpretability_score / tree_count

                    print("\nAverage Metrics Across Trees (Initial Rules):")
                    print(f"  Average Features Used: {avg_features_used:.2f}/{avg_features:.2f}")
                    print(f"  Average Total Rules: {avg_rules:.2f}")
                    print(f"  Average Max Rule Depth: {avg_max_depth:.2f}")
                    print(f"  Average Mean Rule Depth: {avg_mean_rule_depth:.2f}")
                    print(f"  Average Sparsity Interpretability Score: {avg_sparsity_interpretability_score:.2f}")
                    f.write("\nAverage Metrics Across Trees (Initial Rules):\n")
                    f.write(f"  Average Features Used: {avg_features_used:.2f}/{avg_features:.2f}\n")
                    f.write(f"  Average Total Rules: {avg_rules:.2f}\n")
                    f.write(f"  Average Max Rule Depth: {avg_max_depth:.2f}\n")
                    f.write(f"  Average Mean Rule Depth: {avg_mean_rule_depth:.2f}\n")
                    f.write(f"  Average Sparsity Interpretability Score: {avg_sparsity_interpretability_score:.2f}\n")

                # Calculate sparsity and interpretability for each tree in Random Forest
                tree_sparsity_info = {}
                for rule in self.final_rules:
                    tree_name = rule.name.split('_')[0]
                    if tree_name not in tree_sparsity_info:
                        tree_sparsity_info[tree_name] = []
                    tree_sparsity_info[tree_name].append(rule)

                total_features_used = 0
                total_features = 0
                total_rules = 0
                total_max_depth = 0
                total_mean_rule_depth = 0
                total_sparsity_interpretability_score = 0
                tree_count = len(tree_sparsity_info)

                for tree_name, rules in tree_sparsity_info.items():
                    n_features_total = len(set(cond.split(' ')[0] for rule in rules for cond in rule.conditions))
                    sparsity_info = RuleClassifier.calculate_sparsity_interpretability(rules, n_features_total)
                    total_features_used += sparsity_info['features_used']
                    total_features += sparsity_info['total_features']
                    total_rules += sparsity_info['total_rules']
                    total_max_depth += sparsity_info['max_depth']
                    total_mean_rule_depth += sparsity_info['mean_rule_depth']
                    total_sparsity_interpretability_score += sparsity_info['sparsity_interpretability_score']

                # Calculate and print averages
                if tree_count > 0:
                    avg_features_used = total_features_used / tree_count
                    avg_features = total_features / tree_count
                    avg_rules = total_rules / tree_count
                    avg_max_depth = total_max_depth / tree_count
                    avg_mean_rule_depth = total_mean_rule_depth / tree_count
                    avg_sparsity_interpretability_score = total_sparsity_interpretability_score / tree_count

                    print("\nAverage Metrics Across Trees (Final Rules):")
                    print(f"  Average Features Used: {avg_features_used:.2f}/{avg_features:.2f}")
                    print(f"  Average Total Rules: {avg_rules:.2f}")
                    print(f"  Average Max Rule Depth: {avg_max_depth:.2f}")
                    print(f"  Average Mean Rule Depth: {avg_mean_rule_depth:.2f}")
                    print(f"  Average Sparsity Interpretability Score: {avg_sparsity_interpretability_score:.2f}")
                    f.write("\nAverage Metrics Across Trees (Final Rules):\n")
                    f.write(f"  Average Features Used: {avg_features_used:.2f}/{avg_features:.2f}\n")
                    f.write(f"  Average Total Rules: {avg_rules:.2f}\n")
                    f.write(f"  Average Max Rule Depth: {avg_max_depth:.2f}\n")
                    f.write(f"  Average Mean Rule Depth: {avg_mean_rule_depth:.2f}\n")
                    f.write(f"  Average Sparsity Interpretability Score: {avg_sparsity_interpretability_score:.2f}\n")

    # ************************  GENERATING SCIKIT-LEARN MODEL ************************

    def process_data (train_path, test_path):
        # Data loading
        df_train = pd.read_csv(train_path, header=None, encoding='latin-1')
        data_train = df_train.values

        df_test = pd.read_csv(test_path, header=None, encoding='latin-1')
        data_test = df_test.values

        colunas = df_train.shape[1]
        print("Number of collumns:", colunas)

        classes = df_train.iloc[:, -1].nunique()
        print("Number of classes:", classes)
        print("Classes names:", df_train.iloc[:, -1].unique())
        print("Number of samples in training set:", data_train.shape[0])
        print("Number of samples in test set:", data_test.shape[0])

        # Separating features and labels
        X_train = data_train[:, :-1]
        y_train = data_train[:, -1]

        X_test = data_test[:, :-1]
        y_test = data_test[:, -1]

        # Encode categorical labels to integers
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)

        return X_train, y_train, X_test, y_test

    # Method to extract rules from a tree model
    def get_rules(tree, feature_names, class_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        paths = []
        path = []

        def recurse(node, path, paths):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                p1, p2 = list(path), list(path)
                p1 += [f"{name} <= {np.round(threshold, 3)}"]
                recurse(tree_.children_left[node], p1, paths)
                p2 += [f"{name} > {np.round(threshold, 3)}"]
                recurse(tree_.children_right[node], p2, paths)
            else:
                path += [(tree_.value[node], tree_.n_node_samples[node])]
                paths.append(path)

        recurse(0, path, paths)

        # Sorting paths by number of samples
        samples_count = [p[-1][1] for p in paths]
        ii = list(np.argsort(samples_count))
        paths = [paths[i] for i in reversed(ii)]

        # Sorting paths by class
        rules_by_class = {class_name: [] for class_name in class_names}

        for path in paths:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            class_name = class_names[l]
            rule = ", ".join(p for p in path[:-1])
            rules_by_class[class_name].append(f"[{rule}]")
        
        return rules_by_class
    
    # Method to extract rules from a Random Forest model
    def get_tree_rules(model, lst, lst_class, algorithm_type='Random Forest'):
        feature = [f'v{i}' for i in lst]  # Supondo que estas são suas características
        print(feature)

        class_names = lst_class  # Substitua pelos nomes reais das classes

        rules = []
        if algorithm_type == 'Random Forest':
            for estimator in model.estimators_:
                rules.append(RuleClassifier.get_rules(estimator, feature, class_names))
        elif algorithm_type == 'Decision Tree':
            rules.append(RuleClassifier.get_rules(model, feature, class_names))
        else:
            raise ValueError(f"Unsupported algorithm type: {algorithm_type}")
        return rules
    
    # Method to save rules in a text file
    def save_tree_rules(rules, lst, lst_class):

        # Salvando a saída em um arquivo
        output_path = 'examples/files/rules_sklearn.txt'  # Defina o caminho do arquivo de saída
        with open(output_path, 'w') as file:
            for i, rule_set in enumerate(rules):
                for class_index, (class_name, class_rules) in enumerate(rule_set.items()):
                    for rule_index, rule in enumerate(class_rules, 1):
                        rule_name = f"DT{i+1}_Rule{rule_index}_Class{class_index}"
                        file.write(f"{rule_name}: {rule}\n")

        print(f"Rules file saved: {output_path}")

        return rules
    
    # Method to save the Scikit-Learn model
    def save_sklearn_model(model):
        path = 'examples/files/sklearn_model.pkl'
        with open(path, 'wb') as model_file:
            pickle.dump(model, model_file)
        print(f"Sklearn file saved: {path}")

    # Method to generate a classifier model based on rules
    def generate_classifier_model(rules, algorithm_type='Random Forest'):

        rules_text = ""        
        for i, rule_set in enumerate(rules):
                for class_index, (class_name, class_rules) in enumerate(rule_set.items()):
                    for rule_index, rule in enumerate(class_rules, 1):
                        rules_text = rules_text + f"DT{i+1}_Rule{rule_index}_Class{class_index}: {rule}" + "\n"
                        
        classifier = RuleClassifier(rules_text, algorithm_type=algorithm_type)

        print(f"Algorith Type: {classifier.algorithm_type}")

        path = 'examples/files/initial_model.pkl'
        with open(path, 'wb') as model_file:
                    pickle.dump(classifier, model_file)
        print(f"Classifier file saved: {path}")

        return classifier

    def new_classifier(train_path, test_path, model_parameters, model_path=None, algorithm_type='Random Forest'):
        print("\n*********************************************************************************************************")
        print("************************************** GENERATING A NEW CLASSIFIER **************************************")
        print("*********************************************************************************************************\n")
        if model_path:
            # Load the model from the provided path
            print(f"Loading model from: {model_path}")
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)
        else:
            # Train a new model with dynamic parameters
            print("Training a new model with dynamic parameters")
            if algorithm_type == 'Random Forest':
                model = RandomForestClassifier(**model_parameters)
            elif algorithm_type == 'Decision Tree':
                model = DecisionTreeClassifier(**model_parameters)
            else:
                raise ValueError(f"Unsupported algorithm type: {algorithm_type}")

        print("\nDatabase details:")
        X_train, y_train, X_test, y_test = RuleClassifier.process_data(train_path, test_path)
        model.fit(X_train, y_train)

        # Predictions and evaluations
        print("\nTesting model:")
        y_pred = model.predict(X_test)

        print("\nRESULTS SUMMARY:")

        print(f'Correct: {sum(y_pred == y_test)}, Errors: {sum(y_pred != y_test)}, Total: {len(y_test)}')
        # Calculate precision, recall, and F1 score
        # Calculate metrics manually using tp, fp, tn, fn
        tp = sum(1 for yt, yp in zip(y_test, y_pred) if yt == yp and yt == 1)
        fp = sum(1 for yt, yp in zip(y_test, y_pred) if yt != yp and yp == 1)
        tn = sum(1 for yt, yp in zip(y_test, y_pred) if yt == yp and yt == 0)
        fn = sum(1 for yt, yp in zip(y_test, y_pred) if yt != yp and yp == 0)

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Print metrics
        print(f"Accuracy: {accuracy:.5f}")
        print(f"Precision: {precision:.5f}")
        print(f"Recall: {recall:.5f}")
        print(f"F1 Score: {f1:.5f}")

        # Compute confusion matrix
        labels = sorted(set(y_test))
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        # Print confusion matrix with labels
        print("\nConfusion Matrix with Labels:")
        print("Labels:", labels)
        print(cm)

        print("\nSaving Scikit-Learn model:")
        RuleClassifier.save_sklearn_model(model)

        # Generate trees and extract decision rules
        feature_names = [f'feature_{i+1}' for i in range(X_train.shape[1])]
        class_names = np.unique(y_train).astype(str)

        lst = list(range(1, X_train.shape[1]+1))
        feature = [f'v{i}' for i in lst] 

        rules = RuleClassifier.get_tree_rules(model, lst, class_names, algorithm_type=algorithm_type)

        RuleClassifier.save_tree_rules(rules, lst, class_names)

        print("\nGenerating classifier model:")
        classifier = RuleClassifier.generate_classifier_model(rules, algorithm_type)

        return classifier