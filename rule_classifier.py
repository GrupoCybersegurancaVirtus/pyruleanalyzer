import csv
import numpy as np
import pandas as pd
import pickle
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import export_text, _tree
import tkinter as tk
from tkinter import filedialog

class Rule:
    def __init__(self, name, class_, conditions):
        self.name = name
        self.class_ = class_
        self.conditions = conditions
        self.usage_count = 0
        self.error_count = 0

class RuleClassifier:
    def __init__(self, rules, algorithm_type='Decision Tree'):
        self.initial_rules = self.parse_rules(rules, algorithm_type)
        self.algorithm_type = algorithm_type
        self.final_rules, self.duplicated_rules = self.adjust_and_remove_rules()
        self.specific_rules = []

    def parse_rules(self, rules, algorithm_type):
        rules = rules.replace('"', '').replace('- ','').strip().split('\n')
        
        if algorithm_type == 'Random Forest':
            return [self.parse_rf_rule(rule) for rule in rules if rule]
        if algorithm_type == 'Decision Tree':
            return [self.parse_rule(rule) for rule in rules if rule]

    def parse_rule(self, rule):
        rule = rule.strip().split(':')
        rule_name = rule[0].strip()
        class_, conditions = rule_name.split('_')
        conditions = rule[1].strip().replace('[', '').replace(']', '').split(', ')
        return Rule(rule_name, class_, conditions)

    def parse_rf_rule(self, rule):
        rule = rule.split(':')
        rule_name, conditions = rule[0].strip(), rule[1].strip()
        class_ = rule_name.split('_')[-1]
        conditions = conditions.replace('[', '').replace(']', '').split(', ')
        return Rule(rule_name, class_, conditions)

    @staticmethod
    def compute_predict_proba(predictions):
        n_classes = 2 # Número de classes
        class_counts = np.bincount(predictions, minlength=n_classes)
        probabilities = class_counts / len(predictions)
        return probabilities
    
    @staticmethod
    def predict_from_proba(proba_matrix, classes):
        return classes[np.argmax(proba_matrix, axis=0)]

    def classify(self, data, final=False):
        rules_to_use = self.final_rules if final else self.initial_rules

        if self.algorithm_type == 'Random Forest':
            votes = []
            for rule in rules_to_use:
                parsed_conditions = self.parse_conditions(rule.conditions)
                if all(var in data and (data[var] <= float(value) if op == '<=' else
                                        data[var] >= float(value) if op == '>=' else
                                        data[var] < float(value) if op == '<' else
                                        data[var] > float(value)) for var, op, value in parsed_conditions):
                    votes.append(int(rule.class_[-1]))
                    rule.usage_count += 1  # Increment rule usage count
            if votes:
                proba = RuleClassifier.compute_predict_proba(votes)
                class_labels = [0, 1]
                if proba[0] == proba[1]:
                    return class_labels[1], votes, proba

                return class_labels[np.argmax(proba)], votes, proba

        if self.algorithm_type == 'Decision Tree':
            class_counts = {}
            for rule in rules_to_use:
                parsed_conditions = self.parse_conditions(rule.conditions)
                if all(var in data and (data[var] <= value if op == '<=' else
                                        data[var] >= value if op == '>=' else
                                        data[var] < value if op == '<' else
                                        data[var] > value) for var, op, value in parsed_conditions):
                    rule.usage_count += 1  # Increment rule usage count
                    return int(rule.class_[-1])
                class_counts[rule.class_] = class_counts.get(rule.class_, 0) + len(rule.conditions)
            # If no rule matches, return the class with the most conditions covered
            return int(max(class_counts, key=class_counts.get)[-1])
        return None

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
    

    def find_similar_rules(self):
        similar_rules = []
        for i, rule1 in enumerate(self.initial_rules):
            for j, rule2 in enumerate(self.initial_rules):
                if i >= j:
                    continue
                if self.compare_similar_rules(rule1.conditions, rule2.conditions):
                    similar_rules.append((rule1.name, rule2.name))
        return similar_rules

    def compare_similar_rules(self, conditions1, conditions2):
        if len(conditions1) != len(conditions2):
            return False
        for cond1, cond2 in zip(conditions1, conditions2):
            var1, op1, val1 = cond1.split(' ')
            var2, op2, val2 = cond2.split(' ')
            if var1 != var2:
                return False
            if (op1 in ['<=', '<'] and op2 in ['>', '>=']) or (op1 in ['>', '>='] and op2 in ['<=', '<']):
                continue
            if val1 != val2:
                return False
        return True
    
    def adjust_and_remove_rules(self):
        similar_rules = self.find_similar_rules()
        unique_rules = []
        duplicated_rules = set()

        for rule in self.initial_rules:
            if rule.name not in duplicated_rules:
                unique_rules.append(rule)
            for similar_rule in similar_rules:
                if rule.name in similar_rule:
                    duplicated_rules.add(similar_rule[1] if rule.name == similar_rule[0] else similar_rule[0])

        return unique_rules, duplicated_rules

    def execute_rule_analysis(self, file_path,remove_below_n_classifications=-1):

        print("\n*********************************************************************************************************")
        print("**************************************** EXECUTING RULE ANALYSIS ****************************************")
        print("*********************************************************************************************************\n")
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        with open('files/output_classifier.txt', 'w') as f:
                with open(file_path, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    i=1
                    errors = ""
                    for row in reader:
                        print(f'\nIndex: {i}') 
                        f.write(f'\nIndex: {i}\n')
                        i+=1
                        data = {f'v{i+1}': float(value) for i, value in enumerate(row[:-1])}
                        predicted_class, votes, proba = self.classify(data)
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
                    # print(f"\nRules removed with usage count below {remove_below_n_classifications}:")
                    f.write(f"\nRules removed with usage count below {remove_below_n_classifications}:\n")
                    for rule in self.final_rules[:]:
                        if rule.usage_count <= remove_below_n_classifications:
                            # print(f"Rule: {rule.name}, Count: {rule.usage_count}")
                            f.write(f"Rule: {rule.name}, Count: {rule.usage_count}\n")
                            self.specific_rules.append(rule)
                            self.final_rules.remove(rule)

                accuracy = correct / total if total > 0 else 0
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
                # print("\nInitial Rules:")
                f.write("\nInitial Rules:\n")
                for rule in self.initial_rules:
                    # print(f"Rule: {rule.name}, Class: {rule.class_}, Conditions: {rule.conditions}")
                    f.write(f"Rule: {rule.name}, Class: {rule.class_}, Conditions: {rule.conditions}\n")
                
                # Print the final rules
                # print("\nFinal Rules:")
                f.write("\nFinal Rules:\n")
                for rule in self.final_rules:
                    # print(f"Rule: {rule.name}, Class: {rule.class_}, Conditions: {rule.conditions}")
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
                print(f"\nTotal Duplicated Rules: {len(self.duplicated_rules)}")
                f.write(f"\nTotal Duplicated Rules: {len(self.duplicated_rules)}\n")

                # Print the total number of specific rules
                print(f"\nTotal Specific Rules: {len(self.specific_rules)} (<= {remove_below_n_classifications} classifications)")
                f.write(f"\nTotal Specific Rules: {len(self.specific_rules)} (<= {remove_below_n_classifications} classifications)\n")

                # # Print the total number of incorrect rules
                # print(f"\nTotal Incorrect Rules: {len(incorrect_rules)}")
                # f.write(f"\nTotal Incorrect Rules: {len(incorrect_rules)}\n")

                # Save the initial model to a .pkl file
                with open('files/final_model.pkl', 'wb') as file:
                    pickle.dump(self, file)
        return self

                
    def compare_initial_final_results(self, file_path):
        print("\n*********************************************************************************************************")
        print("******************************* RUNNING INITIAL AND FINAL CLASSIFICATIONS *******************************")
        print("*********************************************************************************************************\n")
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        with open('files/output_final_classifier.txt', 'w') as f:
                with open(file_path, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    print("\n******************************* INITIAL MODEL *******************************\n")
                    f.write("\n******************************* INITIAL MODEL *******************************\n")
                    i=1
                    errors = ""
                    for row in reader:
                        # print(f'\nIndex: {i}') 
                        # f.write(f'\nIndex: {i}\n')
                        i+=1
                        data = {f'v{i+1}': float(value) for i, value in enumerate(row[:-1])}
                        predicted_class, votes, proba = self.classify(data)
                        class_vote_counts = {cls: votes.count(cls) for cls in set(votes)}
                        # print(f'Votes: {votes}\nClass Votes: {class_vote_counts}\nNumber of classifications: {len(votes)}')
                        # f.write(f'Votes: {votes}\nClass Votes: {class_vote_counts}\nNumber of classifications: {len(votes)}\n')
                        # print(f"Probabilities: {proba}")
                        # f.write(f"Probabilities: {proba}\n")
                        actual_class = int(row[-1])
                        y_true.append(actual_class)
                        y_pred.append(predicted_class)
                        if predicted_class != actual_class:
                            # print(f'ERROR: Predicted: {predicted_class}, Actual: {actual_class}')
                            # f.write(f'ERROR: Predicted: {predicted_class}, Actual: {actual_class}\n')
                            errors += f'\nIndex: {i-1}\nVotes: {votes}\nClass Votes: {class_vote_counts}\nNumber of classifications: {len(votes)}\nProbabilities: {proba}\nERRO: Predicted: {predicted_class}, Actual: {actual_class}\n'
                        if predicted_class == actual_class:
                            # print(f'Predicted: {predicted_class}, Actual: {actual_class}')
                            # f.write(f'Predicted: {predicted_class}, Actual: {actual_class}\n')
                            correct += 1
                        total += 1

                accuracy = correct / total if total > 0 else 0
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

                # print("\nErrors: \n" + errors + "\n")
                # f.write("\nErrors: \n" + errors + "\n")

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
                # for tree_name, count in initial_tree_rule_counts.items():
                #     print(f"Tree: {tree_name}, Rule Count: {count}")
                #     f.write(f"Tree: {tree_name}, Rule Count: {count}\n")



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
                        # print(f'\nIndex: {i}') 
                        # f.write(f'\nIndex: {i}\n')
                        i+=1
                        data = {f'v{i+1}': float(value) for i, value in enumerate(row[:-1])}
                        predicted_class, votes, proba = self.classify(data, final=True)
                        class_vote_counts = {cls: votes.count(cls) for cls in set(votes)}
                        # print(f'Votes: {votes}\nClass Votes: {class_vote_counts}\nNumber of classifications: {len(votes)}')
                        # f.write(f'Votes: {votes}\nClass Votes: {class_vote_counts}\nNumber of classifications: {len(votes)}\n')
                        # print(f"Probabilities: {proba}")
                        # f.write(f"Probabilities: {proba}\n")
                        actual_class = int(row[-1])
                        y_true_final.append(actual_class)
                        y_pred_final.append(predicted_class)
                        if predicted_class != actual_class:
                            # print(f'ERROR: Predicted: {predicted_class}, Actual: {actual_class}')
                            # f.write(f'ERROR: Predicted: {predicted_class}, Actual: {actual_class}\n')
                            errors += f'\nIndex: {i-1}\nVotes: {votes}\nClass Votes: {class_vote_counts}\nNumber of classifications: {len(votes)}\nProbabilities: {proba}\nERRO: Predicted: {predicted_class}, Actual: {actual_class}\n'
                        if predicted_class == actual_class:
                            # print(f'Predicted: {predicted_class}, Actual: {actual_class}')
                            # f.write(f'Predicted: {predicted_class}, Actual: {actual_class}\n')
                            correct_final += 1
                        total_final += 1

                accuracy_final = correct_final / total_final if total_final > 0 else 0
                print(f'\nCorrect: {correct_final}, Errors: {total_final - correct_final}, Accuracy: {accuracy_final:.5f}')
                f.write(f'\nCorrect: {correct_final}, Errors: {total_final - correct_final}, Accuracy: {accuracy_final:.5f}\n')
                
                # Filter out None values from y_true_final and y_pred_final
                y_true_final_filtered = [y for y, y_p in zip(y_true_final, y_pred_final) if y_p is not None]
                y_pred_final_filtered = [y_p for y_p in y_pred_final if y_p is not None]

                # Compute confusion matrix
                labels_final = sorted(set(y_true_final_filtered))
                cm_final = confusion_matrix(y_true_final_filtered, y_pred_final_filtered, labels=labels_final)
                
                # Print confusion matrix with labels
                print("\nConfusion Matrix with Labels (Final):")
                f.write("\nConfusion Matrix with Labels (Final):\n")
                print("Labels:", labels_final)
                f.write(f"Labels: {labels_final}\n")
                print(cm_final)
                f.write(f"{cm_final}\n")

                # print("\nErrors: \n" + errors + "\n")
                # f.write("\nErrors: \n" + errors + "\n")
                # Count the number of rules for each tree in final rules

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
                # for tree_name, count in final_tree_rule_counts.items():
                #     print(f"Tree: {tree_name}, Rule Count: {count}")
                #     f.write(f"Tree: {tree_name}, Rule Count: {count}\n")

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


                for case in divergent_cases:
                    print(f"Index: {case['index']}, Data: {case['data']}, Initial Class: {case['initial_class']}, "
                          f"Final Class: {case['final_class']}, Actual Class: {case['actual_class']}")
                    f.write(f"Index: {case['index']}, Data: {case['data']}, Initial Class: {case['initial_class']}, "
                            f"Final Class: {case['final_class']}, Actual Class: {case['actual_class']}\n")

    @staticmethod
    def select_csv_and_test(classifier):
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            classifier.execute_with_csv(file_path)
        # Save the classifier to a .pkl file
        # file_name = 'rule_classifier_rf.pkl' if classifier.algorithm_type == 'Random Forest' else 'rule_classifier_dt.pkl'
        # with open(file_name, 'wb') as file:
        #     pickle.dump(classifier, file)


    @staticmethod
    def select_csv_and_final_test(classifier):
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            classifier.compare_initial_final_results(file_path)

    # ************************  SKLEARN MODEL ************************

    def process_data (train_path, test_path):
        # Carregamento dos dados
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

        # Separação dos dados
        X_train = data_train[:, :-1]
        y_train = data_train[:, -1]

        X_test = data_test[:, :-1]
        y_test = data_test[:, -1]

        # Codificação das classes
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)

        return X_train, y_train, X_test, y_test

    # Função para extrair regras de uma árvore de decisão
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

        # Ordenando por contagem de amostras
        samples_count = [p[-1][1] for p in paths]
        ii = list(np.argsort(samples_count))
        paths = [paths[i] for i in reversed(ii)]

        # Organizando regras por classe
        rules_by_class = {class_name: [] for class_name in class_names}

        for path in paths:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            class_name = class_names[l]
            rule = ", ".join(p for p in path[:-1])
            rules_by_class[class_name].append(f"[{rule}]")
        
        return rules_by_class
    
    def get_tree_rules (model,lst,lst_class):
        feature = [f'v{i}' for i in lst]  # Supondo que estas são suas características
        print(feature)

        class_names = lst_class  # Substitua pelos nomes reais das classes

        rules = []
        for estimator in model.estimators_:
            rules.append(RuleClassifier.get_rules(estimator, feature, class_names))
        return rules
    
    def save_tree_rules(rules, lst, lst_class):

        # Salvando a saída em um arquivo
        output_path = 'files/rules_sklearn.txt'  # Defina o caminho do arquivo de saída
        rules_text = ""
        with open(output_path, 'w') as file:
            for i, rule_set in enumerate(rules):
                for class_index, (class_name, class_rules) in enumerate(rule_set.items()):
                    for rule_index, rule in enumerate(class_rules, 1):
                        rule_name = f"DT{i+1}_Rule{rule_index}_Class{class_index}"
                        file.write(f"{rule_name}: {rule}\n")

        print(f"Rules file saved: {output_path}")

        return rules
    
    def save_sklearn_model(model):
        path = 'files/sklearn_model.pkl'
        with open(path, 'wb') as model_file:
            pickle.dump(model, model_file)
        print(f"Sklearn file saved: {path}")

    def generate_classifier_model(rules, algorithm_type='Random Forest'):

        rules_text = ""        
        for i, rule_set in enumerate(rules):
                for class_index, (class_name, class_rules) in enumerate(rule_set.items()):
                    for rule_index, rule in enumerate(class_rules, 1):
                        rules_text = rules_text + f"DT{i+1}_Rule{rule_index}_Class{class_index}: {rule}" + "\n"
                        
        classifier = RuleClassifier(rules_text, algorithm_type=algorithm_type)

        print(f"Algorith Type: {classifier.algorithm_type}")

        #print(f"Rules: {classifier.rules}")

        # Serializa o modelo em um arquivo .pkl
        path = 'files/initial_model.pkl'
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
            model = RandomForestClassifier(**model_parameters) if algorithm_type == 'Random Forest' else None
            if model is None:
                raise ValueError(f"Unsupported algorithm type: {algorithm_type}")
            X_train, y_train, X_test, y_test = RuleClassifier.process_data(train_path, test_path)
            model.fit(X_train, y_train)
        model = RandomForestClassifier(**model_parameters) if algorithm_type == 'Random Forest' else None
        if model is None:
            raise ValueError(f"Unsupported algorithm type: {algorithm_type}")

        print("\nDatabase details:")
        X_train, y_train, X_test, y_test = RuleClassifier.process_data(train_path, test_path)
        model.fit(X_train, y_train)

        # Predições e avaliações
        print("\nTesting model:")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
        print(f'Accuracy: {accuracy:.2f}%')

        print("\nSaving Scikit-Learn model:")
        RuleClassifier.save_sklearn_model(model)

        # Geração das árvores e extração das regras de decisão
        feature_names = [f'feature_{i+1}' for i in range(X_train.shape[1])]
        class_names = np.unique(y_train).astype(str)

        lst = list(range(1, X_train.shape[1]+1))
        feature = [f'v{i}' for i in lst] 

        rules = RuleClassifier.get_tree_rules(model, lst, class_names)

        RuleClassifier.save_tree_rules(rules, lst, class_names)

        print("\nGenerating classifier model:")
        classifier = RuleClassifier.generate_classifier_model(rules, algorithm_type)

        return classifier
    
# # ************************ EXEMPLO DE EXECUÇÃO ************************

# train_path = "G:/Meu Drive/Mestrado/VIRTUS/Código/Novo RuleXtract/data/rapid_balanceado_treinamento.csv"
# test_path = "G:/Meu Drive/Mestrado/VIRTUS/Código/Novo RuleXtract/data/rapid_balanceado_teste.csv"

# model_parameters = {
#     'n_estimators': 101,
#     'min_samples_leaf': 2,
#     'min_samples_split': 2,
#     'max_depth': None,
#     'max_features': 'sqrt',
#     'random_state': 42
# }

# # Execute the new classifier
# classifier = RuleClassifier.new_classifier(train_path, test_path, model_parameters, algorithm_type='Random Forest')

# classifier.execute_rule_analysis(test_path, remove_below_n_classifications=-1)
