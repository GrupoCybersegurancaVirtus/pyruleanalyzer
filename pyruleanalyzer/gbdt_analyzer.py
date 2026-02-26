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
    analyzer.execute_rule_analysis(test_path, remove_below_n_classifications=0)
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

    def __init__(self, classifier: RuleClassifier) -> None:
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

    def track_from_adjust_and_remove(
        self,
        method: str,
        intra_tree_pairs: list,
    ) -> None:
        """Updates redundancy counters after an adjust_and_remove cycle.

        Called by :meth:`RuleClassifier.execute_rule_analysis` after the
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

    def execute_rule_analysis(
        self,
        file_path: str,
        remove_below_n_classifications: int = -1,
    ) -> None:
        """Evaluates GBDT rules on a dataset, detects redundancies, and refines.

        This method:
        1. Classifies every sample to gather per-rule usage stats.
        2. Optionally removes low-usage rules (with sibling promotion).
        3. Tracks redundancy counts by type.
        4. Writes a report and saves the final model.

        Args:
            file_path: Path to the CSV test file.
            remove_below_n_classifications: Threshold for low-usage pruning
                (-1 disables).
        """
        clf = self.classifier
        print(f'Testing GBDT Rules on {os.path.basename(file_path)}...')
        start_time = time.time()

        # 1. Load Data
        _, _, X_test, y_test, _, _, feature_names = RuleClassifier.process_data(
            '.', file_path, is_test_only=True
        )
        sample_dicts = [dict(zip(feature_names, row)) for row in X_test]
        total_samples = len(y_test)

        # 2. Reset Counters
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

        # 5. Low-usage pruning with sibling promotion
        if remove_below_n_classifications > -1:
            print(f'\nPruning rules with <= {remove_below_n_classifications} classifications...')
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

        # 6. Generate Report
        clf._write_report(
            'examples/files/output_classifier_gbdt.txt',
            file_path, correct, total_samples,
            remove_below_n_classifications,
        )

        print(f'Analysis Time: {time.time() - start_time:.3f}s')
        clf._save_final_model()

        # 7. Print redundancy summary
        self.print_redundancy_summary()

    # ------------------------------------------------------------------
    # Initial vs Final comparison
    # ------------------------------------------------------------------

    def compare_initial_final_results(self, file_path: str) -> None:
        """Compares performance of initial vs final rules for GBDT.

        Evaluates both rule sets on the test data, displays metrics, logs
        divergent cases, and writes a detailed report.

        Args:
            file_path: Path to the CSV test file.
        """
        _, _, X_test, y_test, _, target_column_name, feature_names = (
            RuleClassifier.process_data('.', file_path, is_test_only=True)
        )
        df_test = pd.DataFrame(X_test, columns=feature_names)
        df_test[target_column_name] = y_test

        self._compare_gbdt(df_test, target_column_name)

    def _compare_gbdt(self, df_test: pd.DataFrame, target_column_name: str) -> None:
        """Internal GBDT comparison logic."""
        clf = self.classifier

        print('\n' + '*' * 80)
        print('RUNNING INITIAL AND FINAL CLASSIFICATIONS (Gradient Boosting Decision Trees)')
        print('*' * 80 + '\n')

        df = df_test.copy()
        y_true = df[target_column_name].astype(int).values
        feature_cols = [c for c in df.columns if c != target_column_name]
        sample_dicts = [dict(zip(feature_cols, row)) for row in df[feature_cols].values]
        total_samples = len(y_true)

        with open('examples/files/output_final_classifier_gbdt.txt', 'w') as f:
            f.write('******************** INITIAL VS FINAL GBDT CLASSIFICATION REPORT ********************\n\n')

            # 1. Scikit-Learn Model
            print('Evaluated Scikit-Learn Model (Benchmark)...')
            f.write('\n******************************* SCIKIT-LEARN MODEL *******************************\n')
            try:
                with open('examples/files/sklearn_model.pkl', 'rb') as mf:
                    sk_model = pickle.load(mf)
                y_pred_sk = sk_model.predict(df[feature_cols].values)
                correct_sk = np.sum(y_pred_sk == y_true)
                RuleClassifier.display_metrics(y_true, y_pred_sk, correct_sk, total_samples, f, clf.class_labels)
            except Exception as e:
                msg = f'Could not load/evaluate sklearn model: {e}'
                print(msg)
                f.write(msg + '\n')

            # 2. Compare Initial vs Final Rules
            print('\nComparing Initial vs Final Rules...')
            start_time = time.time()

            y_pred_initial: List[Any] = []
            y_pred_final: List[Any] = []

            for i, sample in enumerate(sample_dicts):
                # A) Initial Prediction
                pred_init, _, _ = RuleClassifier.classify_gbdt(
                    sample, clf.initial_rules,
                    clf._gbdt_init_scores, clf._gbdt_is_binary, clf._gbdt_classes,
                )
                try:
                    pred_init = int(str(pred_init).replace('Class', '').strip())
                except (ValueError, AttributeError):
                    pass
                y_pred_initial.append(pred_init)

                # B) Final Prediction (use native_fn if available)
                if clf.native_fn is not None:
                    try:
                        pred_final, _, _ = clf.native_fn.classify(sample)  # type: ignore
                    except Exception:
                        pred_final, _, _ = RuleClassifier.classify_gbdt(
                            sample, clf.final_rules,
                            clf._gbdt_init_scores, clf._gbdt_is_binary, clf._gbdt_classes,
                        )
                else:
                    pred_final, _, _ = RuleClassifier.classify_gbdt(
                        sample, clf.final_rules,
                        clf._gbdt_init_scores, clf._gbdt_is_binary, clf._gbdt_classes,
                    )
                try:
                    pred_final = int(str(pred_final).replace('Class', '').strip())
                except (ValueError, AttributeError):
                    pass
                y_pred_final.append(pred_final)

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

            print()  # Newline

            # 3. Output Metrics
            print('Evaluation Complete. Generating Report...')

            # Initial Metrics
            f.write('\n******************************* INITIAL MODEL *******************************\n')
            y_pred_initial_arr = np.array(y_pred_initial)
            correct_initial = np.sum(y_pred_initial_arr == y_true)

            print('\n--- Initial Rules Metrics ---')
            RuleClassifier.display_metrics(y_true, y_pred_initial_arr, correct_initial, total_samples, f, clf.class_labels)
            print(f'Number of initial rules: {len(clf.initial_rules)}')
            f.write(f'\nNumber of initial rules: {len(clf.initial_rules)}\n')

            # Final Metrics
            f.write('\n******************************* FINAL MODEL *******************************\n')
            y_pred_final_arr = np.array(y_pred_final)
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
                for i in range(total_samples):
                    if y_pred_initial_arr[i] != y_pred_final_arr[i]:
                        f.write(
                            f'Index: {i}, Initial: {y_pred_initial_arr[i]}, '
                            f'Final: {y_pred_final_arr[i]}, Actual: {y_true[i]}\n'
                        )

            # 5. Interpretability Metrics
            print('Calculating Interpretability Metrics...')
            f.write('\n******************************* INTERPRETABILITY METRICS *******************************\n')

            n_features = len([c for c in df.columns if c != target_column_name])
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

        # Print redundancy summary at the end
        self.print_redundancy_summary()
