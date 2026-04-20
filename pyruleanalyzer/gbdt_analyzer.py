"""Gradient Boosting Decision Trees Analyzer module.

Provides ``GBDTAnalyzer``, a class that encapsulates GBDT-specific
redundancy analysis and initial-vs-final comparison logic.  It operates
**on** a :class:`RuleClassifier` instance (composition, not inheritance)
so that the base classifier remains algorithm-agnostic.

Typical usage::

    from pyruleanalyzer import RuleClassifier, GBDTAnalyzer

    clf = RuleClassifier.new_classifier(train, test, params,
                                        algorithm_type='Gradient Boosting Decision Trees')
    analyzer = GBDTAnalyzer(clf)
    analyzer.execute_rule_refinement(test_path, remove_below_n_classifications=0)
    analyzer.compare_initial_final_results(test_path)
"""

import os
import sys
import time
import pickle
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .rule_classifier import RuleClassifier


class GBDTAnalyzer:
    """Gradient-Boosting-specific rule analysis and comparison.

    This class extracts the GBDT analysis pipeline out of ``RuleClassifier``
    and adds **redundancy breakdown tracking** so the output shows:

    * ``intra_tree``  -- boundary-redundant sibling pairs merged (within each tree)
    * ``low_impact``  -- rules with negligible contribution removed
    * ``low_usage``   -- rules removed because of low classification count

    Attributes:
        classifier: The underlying ``RuleClassifier`` instance.
        redundancy_counts: Dict mapping redundancy type to count.
    """

    # Method to initialize the GBDTAnalyzer.
    def __init__(self, classifier: RuleClassifier) -> None:
        """Initializes the GBDTAnalyzer.

        Args:
            classifier: The RuleClassifier instance to analyze.
        """
        if classifier.algorithm_type != 'Gradient Boosting Decision Trees':
            raise ValueError(
                f"GBDTAnalyzer requires a Gradient Boosting Decision Trees classifier, "
                f"got '{classifier.algorithm_type}'."
            )
        self.classifier = classifier
        self.redundancy_counts: Dict[str, int] = {
            'intra_tree': 0,
            'low_impact': 0,
            'low_usage': 0,
        }

    # ------------------------------------------------------------------
    # Redundancy summary
    # ------------------------------------------------------------------

    # Method to print the redundancy summary.
    def print_redundancy_summary(self) -> None:
        """Prints the redundancy breakdown and rule reduction summary."""
        clf = self.classifier
        initial_count = len(clf.initial_rules)
        final_count = len(clf.final_rules) if clf.final_rules else initial_count
        total_redundancies = sum(self.redundancy_counts.values())
        reduction = initial_count - final_count
        pct = (reduction / initial_count * 100) if initial_count > 0 else 0.0

        print(f'\nTotal redundancies found: {total_redundancies}')
        for rtype, count in self.redundancy_counts.items():
            print(f'  {rtype}: {count}')
        print(f'\nInitial total rules: {initial_count}')
        print(f'Refined total rules:  {final_count}')
        print(f'Reduction: {reduction} rules ({pct:.1f}%)')

    # ------------------------------------------------------------------
    # Track redundancies from adjust_and_remove_rules
    # ------------------------------------------------------------------

    # Method to update redundancy counters after rule removal.
    def track_from_adjust_and_remove(
        self,
        method: str,
        intra_tree_pairs: list,
    ) -> None:
        """Updates redundancy counters after an adjust_and_remove cycle.

        Called by :meth:`RuleClassifier.execute_rule_refinement` after the
        duplicate-removal loop finishes for GBDT.

        Args:
            method: The removal method used ('soft', 'hard').
            intra_tree_pairs: Pairs found by ``find_duplicated_rules``.
        """
        # Each pair = 2 rules merged into 1 generalized -> 1 redundancy
        self.redundancy_counts['intra_tree'] += len(intra_tree_pairs)

    # ------------------------------------------------------------------
    # Main analysis pipeline
    # ------------------------------------------------------------------

    # Method to evaluate rules, detect redundancies, and refine the model.
    def execute_rule_refinement(
        self,
        file_path: str = None,
        X=None,
        y=None,
        remove_below_n_classifications: int = -1,
        refine_between_trees: bool = False,
        save_final_model: bool = True,
        save_report: bool = True,
    ) -> None:
        """Evaluates GBDT rules on a dataset, detects redundancies, and refines.

        This method:
        1. Classifies every sample to gather per-rule usage stats.
        2. Optionally removes low-usage rules (with sibling promotion).
        3. Tracks redundancy counts by type.
        4. Writes a report and saves the final model.

        Args:
            file_path: Path to the CSV test file.
            X: Dataframe or array for test data.
            y: True labels.
            remove_below_n_classifications: Threshold for low-usage refinement
                (-1 disables).
            save_final_model: Whether to save the final model to 'final_model.pkl'.
                Default is True.
            save_report: Whether to save the analysis report to 'output_classifier_gbdt.txt'.
                Default is True.
        """
        clf = self.classifier
        data_name = os.path.basename(file_path) if file_path else "DataFrames/Arrays"
        print(f'Testing GBDT Rules on {data_name}...')
        start_time = time.time()

        # 1. Load Data
        X_test, y_test, feature_names = RuleClassifier._prepare_test_data(file_path, X, y, clf)
        sample_dicts = [dict(zip(feature_names, row)) for row in X_test]
        total_samples = len(y_test)

        # 2. Reset Counters
        clf.final_rules = list(clf.initial_rules)
        for rule in clf.final_rules:
            rule.usage_count = 0
            rule.error_count = 0

        # 3. Classification Loop
        y_pred: List[Any] = []

        for i, sample in enumerate(sample_dicts):
            true_label = int(y_test[i])

            # Use classify_gbdt which returns (predicted_class, matched_rules, None)
            pred_label, matched_rules, _ = RuleClassifier.classify_gbdt(
                sample, clf.final_rules,
                clf._gbdt_init_scores, clf._gbdt_is_binary, clf._gbdt_classes,
            )

            # Convert prediction to int
            try:
                pred_label = int(str(pred_label).replace('Class', '').strip())
            except (ValueError, AttributeError):
                pass

            y_pred.append(pred_label)

            # Update stats for ALL matched rules (one per tree per class group)
            if matched_rules:
                for rule in matched_rules:
                    rule.usage_count += 1
                    if pred_label != true_label:
                        rule.error_count += 1

            # Progress bar
            if i % 100 == 0 or i == total_samples - 1:
                current_time = time.time()
                elapsed = current_time - start_time
                if i > 0:
                    rate = i / elapsed if elapsed > 0 else 0
                    remaining = (total_samples - i) / rate if rate > 0 else 0
                else:
                    remaining = 0

                rem_str = time.strftime('%H:%M:%S', time.gmtime(remaining))
                percent = (i + 1) / total_samples
                bar_len = 30
                filled_len = int(bar_len * percent)
                bar = '=' * filled_len + '-' * (bar_len - filled_len)

                sys.stdout.write(f'\r[{bar}] {percent:.1%} | ETA: {rem_str}')
                sys.stdout.flush()

        print()  # Newline after progress bar

        # 4. Metrics
        y_pred_arr = np.array(y_pred)
        correct = np.sum(y_pred_arr == y_test)

        # Detect and remove intra-tree duplicated rules (soft mode)
        # This is equivalent to the old remove_duplicates="soft" behavior
        duplicated_pairs = clf.find_duplicated_rules(type='soft')
        intra_tree_count = len(duplicated_pairs)
        self.redundancy_counts["intra_tree"] = intra_tree_count
        
        if duplicated_pairs:
            print(f'Found {intra_tree_count} duplicated rule pairs (intra-tree).')
            # Create generalized rules by merging the siblings
            rules_to_remove_ids = set()
            new_generalized_rules = []
            for rule1, rule2 in duplicated_pairs:
                rules_to_remove_ids.add(id(rule1))
                rules_to_remove_ids.add(id(rule2))
                common_conditions = rule1.conditions[:-1]
                new_rule_name = f"{rule1.name}_&_{rule2.name}"
                
                # For GBDT it is important to pass the correct metadata
                new_rule = __import__('pyruleanalyzer.rule_classifier').rule_classifier.Rule(
                    new_rule_name, rule1.class_, common_conditions,
                    leaf_value=getattr(rule1, 'leaf_value', 0.0),
                    learning_rate=getattr(rule1, 'learning_rate', 0.1),
                    class_group=getattr(rule1, 'class_group', 0)
                )
                if hasattr(rule1, 'parsed_conditions') and rule1.parsed_conditions:
                    new_rule.parsed_conditions = rule1.parsed_conditions[:-1]
                new_generalized_rules.append(new_rule)
                
            clf.final_rules = [r for r in clf.final_rules if id(r) not in rules_to_remove_ids] + new_generalized_rules
            print(f'Rules after removing duplicates: {len(clf.final_rules)}')
            clf.update_native_model(clf.final_rules)

        # Low-usage refinement with sibling promotion
        if remove_below_n_classifications > -1:
            print(f'\nRefining rules with <= {remove_below_n_classifications} classifications...')
            clf.specific_rules = []

            for rule in clf.final_rules:
                # Skip init rules (no conditions) -- they are always needed
                if rule.contribution is not None and not rule.conditions:
                    continue
                if rule.usage_count <= remove_below_n_classifications:
                    clf.specific_rules.append(rule)

            low_usage_count = len(clf.specific_rules)
            self.redundancy_counts['low_usage'] = low_usage_count
            print(f'Identified {low_usage_count} specific (low-usage) rules.')

            if clf.specific_rules:
                clf.final_rules = clf._promote_siblings(clf.specific_rules, clf.final_rules)
                print(f'Rules after promotion: {len(clf.final_rules)}')

            clf.update_native_model(clf.final_rules)

        # Detect and merge semantically identical rules across trees
        if refine_between_trees:
            print("\nAnalyzing duplicated rules between trees...")
            similar_rule_groups = clf.find_duplicated_rules_between_trees()
            inter_tree_count = len(similar_rule_groups)
            self.redundancy_counts["inter_tree"] = inter_tree_count
            
            if similar_rule_groups:
                print(f"Found {inter_tree_count} groups of semantically duplicated rules (inter-tree).")
                rules_to_remove_ids = set()
                new_generalized_rules = []
                for group in similar_rule_groups:
                    for rule in group:
                        rules_to_remove_ids.add(id(rule))
                    representative = group[0]
                    new_name = "_&_".join(sorted([r.name for r in group]))
                    
                    # For GBDT, we sum the leaf values
                    merged_leaf_value = sum(getattr(r, 'leaf_value', 0.0) for r in group)
                    
                    new_rule = __import__('pyruleanalyzer.rule_classifier').rule_classifier.Rule(
                        new_name, representative.class_, representative.conditions,
                        leaf_value=merged_leaf_value,
                        learning_rate=getattr(representative, 'learning_rate', 0.1),
                        class_group=getattr(representative, 'class_group', 0)
                    )
                    if hasattr(representative, 'parsed_conditions') and representative.parsed_conditions:
                        new_rule.parsed_conditions = representative.parsed_conditions
                    new_generalized_rules.append(new_rule)
                
                clf.final_rules = [r for r in clf.final_rules if id(r) not in rules_to_remove_ids] + new_generalized_rules
                print(f"Rules after merging inter-tree duplicates: {len(clf.final_rules)}")
                clf.update_native_model(clf.final_rules)

        # Generate Report
        if save_report:
            clf._write_report(
                'files/output_classifier_gbdt.txt',
                file_path, correct, total_samples,
                remove_below_n_classifications,
            )

        print(f'Analysis Time: {time.time() - start_time:.3f}s')
        if save_final_model:
            clf._save_final_model()

        # 7. Print redundancy summary
        self.print_redundancy_summary()

    # ------------------------------------------------------------------
    # Initial vs Final comparison
    # ------------------------------------------------------------------

    # Method to compare initial and final rules performance.
    def compare_initial_final_results(self, file_path: str = None, X = None, y = None) -> None:
        """Compares performance of initial vs final rules for GBDT.

        Evaluates both rule sets on the test data, displays metrics, logs
        divergent cases, and writes a detailed report.

        Args:
            file_path: Path to the CSV test file.
            X: Dataframe or array for test data.
            y: True labels.
        """
        X_test, y_test, feature_names = RuleClassifier._prepare_test_data(file_path, X, y, self.classifier)
        target_column_name = "Class"
        df_test = pd.DataFrame(X_test, columns=feature_names)
        df_test[target_column_name] = y_test

        self._compare_gbdt(df_test, target_column_name)

    # Method to perform the internal GBDT comparison logic.
    def _compare_gbdt(self, df_test: pd.DataFrame, target_column_name: str) -> None:
        """Internal GBDT comparison logic.

        Args:
            df_test: DataFrame containing the test data.
            target_column_name: The name of the target column.
        """
        clf = self.classifier

        print('\n' + '*' * 80)
        print('RUNNING INITIAL AND FINAL CLASSIFICATIONS (Gradient Boosting Decision Trees)')
        print('*' * 80 + '\n')

        df = df_test.copy()
        y_true = df[target_column_name].astype(int).values
        feature_cols = [c for c in df.columns if c != target_column_name]
        X_test_np = df[feature_cols].values.astype(np.float64)
        feature_names_list = list(feature_cols)
        total_samples = len(y_true)

        with open('files/output_final_classifier_gbdt.txt', 'w') as f:
            f.write('******************** INITIAL VS FINAL GBDT CLASSIFICATION REPORT ********************\n\n')

            # 1. Scikit-Learn Model
            print('Evaluated Scikit-Learn Model (Benchmark)...')
            f.write('\n******************************* SCIKIT-LEARN MODEL *******************************\n')
            try:
                with open('files/sklearn_model.pkl', 'rb') as mf:
                    sk_model = pickle.load(mf)
                
                # Since models are now always trained with DataFrames (with column names),
                # predict() must receive a DataFrame to avoid UserWarnings/Errors.
                X_test_predict = pd.DataFrame(X_test_np, columns=feature_names_list)
                
                y_pred_sk = sk_model.predict(X_test_predict)
                correct_sk = np.sum(y_pred_sk == y_true)
                RuleClassifier.display_metrics(y_true, y_pred_sk, correct_sk, total_samples, f, clf.class_labels)
            except Exception as e:
                msg = f'Could not load/evaluate sklearn model: {e}'
                print(msg)
                f.write(msg + '\n')

            # 2. Compare Rules — Vectorized batch prediction
            print('\nComparing Initial vs Final Rules...')
            start_time = time.time()

            # Compile arrays for initial rules, predict, then for final rules
            clf.compile_tree_arrays(rules=clf.initial_rules, feature_names=feature_names_list)
            y_pred_initial_arr = clf.predict_batch(X_test_np, feature_names=feature_names_list)

            clf.compile_tree_arrays(rules=clf.final_rules, feature_names=feature_names_list)
            y_pred_final_arr = clf.predict_batch(X_test_np, feature_names=feature_names_list)

            elapsed = time.time() - start_time
            print(f'  Batch prediction complete in {elapsed:.3f}s')

            # 3. Output Metrics
            print('Evaluation Complete. Generating Report...')

            # Initial Metrics
            f.write('\n******************************* INITIAL MODEL *******************************\n')
            correct_initial = np.sum(y_pred_initial_arr == y_true)

            print('\n--- Initial Rules Metrics ---')
            RuleClassifier.display_metrics(y_true, y_pred_initial_arr, correct_initial, total_samples, f, clf.class_labels)
            print(f'Number of initial rules: {len(clf.initial_rules)}')
            f.write(f'\nNumber of initial rules: {len(clf.initial_rules)}\n')

            # Final Metrics
            f.write('\n******************************* FINAL MODEL *******************************\n')
            correct_final = np.sum(y_pred_final_arr == y_true)

            print('\n--- Final Rules Metrics ---')
            RuleClassifier.display_metrics(y_true, y_pred_final_arr, correct_final, total_samples, f, clf.class_labels)
            print(f'Number of final rules: {len(clf.final_rules)}')
            f.write(f'\nNumber of final rules: {len(clf.final_rules)}\n')

            # 4. Divergent Cases
            f.write('\n******************************* DIVERGENT CASES *******************************\n')

            divergent_count = int(np.sum(y_pred_initial_arr != y_pred_final_arr))
            print(f'\nTotal divergent cases: {divergent_count}')
            f.write(f'Total divergent cases: {divergent_count}\n')

            if divergent_count == 0:
                f.write('No divergent cases found.\n')
            else:
                divergent_indices = np.where(y_pred_initial_arr != y_pred_final_arr)[0]
                for idx in divergent_indices:
                    f.write(
                        f'Index: {idx}, Initial: {y_pred_initial_arr[idx]}, '
                        f'Final: {y_pred_final_arr[idx]}, Actual: {y_true[idx]}\n'
                    )

            # 5. Interpretability Metrics
            print('Calculating Interpretability Metrics...')
            f.write('\n******************************* INTERPRETABILITY METRICS *******************************\n')

            n_features = len(feature_cols)
            metrics_init = RuleClassifier.calculate_structural_complexity(clf.initial_rules, n_features)
            metrics_final = RuleClassifier.calculate_structural_complexity(clf.final_rules, n_features)

            f.write('\nMetrics (Initial):\n')
            for k, v in metrics_init.items():
                f.write(f'  {k}: {v}\n')

            f.write('\nMetrics (Final):\n')
            for k, v in metrics_final.items():
                diff = ''
                if isinstance(v, (int, float)) and metrics_init.get(k, 0) != 0:
                    pct = ((v - metrics_init[k]) / metrics_init[k]) * 100
                    diff = f' ({pct:+.1f}%)'
                f.write(f'  {k}: {v}{diff}\n')

            # Print SCS metrics to stdout
            col_m = 30
            col_v = 14
            print(f"\n{'=' * 80}")
            print('STRUCTURAL COMPLEXITY METRICS (SCS)')
            print(f"{'=' * 80}")
            print(f"{'METRIC':<{col_m}} | {'INITIAL':>{col_v}} | {'FINAL':>{col_v}} | {'CHANGE':>{col_v}}")
            print(f"{'-' * 80}")
            for k in metrics_init:
                v_init = metrics_init[k]
                v_final = metrics_final.get(k, 0)
                if isinstance(v_init, float):
                    s_init = f'{v_init:.4f}'
                    s_final = f'{v_final:.4f}'
                else:
                    s_init = str(v_init)
                    s_final = str(v_final)
                if isinstance(v_init, (int, float)) and v_init != 0:
                    pct = ((v_final - v_init) / v_init) * 100
                    s_pct = f'{pct:+.1f}%'
                else:
                    s_pct = 'N/A'
                print(f'  {k:<{col_m}} | {s_init:>{col_v}} | {s_final:>{col_v}} | {s_pct:>{col_v}}')
            print(f"{'=' * 80}")

        # Print redundancy summary at the end
        self.print_redundancy_summary()
