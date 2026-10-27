"""Reports, metrics and interactive editing.

Provides ``ReportingMixin``: writing the final model and the refinement
report, the structural complexity metrics, classification metrics, the
initial-versus-final comparison (delegated to the per-algorithm analyzers)
and the interactive rule editor.
"""

import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, List
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from ._rules import Rule


class ReportingMixin:
    """Reports, metrics, comparisons and rule editing."""

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
