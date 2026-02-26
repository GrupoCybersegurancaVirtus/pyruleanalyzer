"""Random Forest Analyzer module.

Provides ``RFAnalyzer``, a class that encapsulates Random-Forest-specific
redundancy analysis and initial-vs-final comparison logic.  It operates
**on** a :class:`RuleClassifier` instance (composition, not inheritance)
so that the base classifier remains algorithm-agnostic.

Typical usage::

    from pyruleanalyzer import RuleClassifier, RFAnalyzer

    clf = RuleClassifier.new_classifier(train, test, params, algorithm_type='Random Forest')
    analyzer = RFAnalyzer(clf)
    analyzer.execute_rule_analysis(test_path, remove_below_n_classifications=1)
    analyzer.compare_initial_final_results(test_path)
"""

import os
import sys
import time
import pickle
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .rule_classifier import RuleClassifier


class RFAnalyzer:
    """Random-Forest-specific rule analysis and comparison.

    This class extracts the RF analysis pipeline out of ``RuleClassifier``
    and adds **redundancy breakdown tracking** so the output shows:

    * ``intra_tree``  – boundary-redundant sibling pairs merged (within each tree)
    * ``inter_tree``  – semantically duplicate rules across trees
    * ``low_usage``   – rules removed because of low classification count

    Attributes:
        classifier: The underlying ``RuleClassifier`` instance.
        redundancy_counts: Dict mapping redundancy type to count.
    """

    def __init__(self, classifier: RuleClassifier) -> None:
        if classifier.algorithm_type != 'Random Forest':
            raise ValueError(
                f"RFAnalyzer requires a Random Forest classifier, "
                f"got '{classifier.algorithm_type}'."
            )
        self.classifier = classifier
        self.redundancy_counts: Dict[str, int] = {
            "intra_tree": 0,
            "inter_tree": 0,
            "low_usage": 0,
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

        print(f"\nTotal redundancies found: {total_redundancies}")
        for rtype, count in self.redundancy_counts.items():
            print(f"  {rtype}: {count}")
        print(f"\nInitial total rules: {initial_count}")
        print(f"Refined total rules:  {final_count}")
        print(f"Reduction: {reduction} rules ({pct:.1f}%)")

    # ------------------------------------------------------------------
    # Track redundancies from adjust_and_remove_rules
    # ------------------------------------------------------------------

    def track_from_adjust_and_remove(
        self,
        method: str,
        intra_tree_pairs: list,
        inter_tree_groups: Optional[list] = None,
    ) -> None:
        """Updates redundancy counters after an adjust_and_remove cycle.

        Called by :meth:`RuleClassifier.execute_rule_analysis` after the
        duplicate-removal loop finishes for RF.

        Args:
            method: The removal method used ('soft', 'medium', 'hard').
            intra_tree_pairs: Pairs found by ``find_duplicated_rules``.
            inter_tree_groups: Groups found by
                ``find_duplicated_rules_between_trees`` (only for 'hard').
        """
        # Each pair = 2 rules merged into 1 generalized -> 1 redundancy
        self.redundancy_counts["intra_tree"] += len(intra_tree_pairs)
        if inter_tree_groups:
            # Each group of N rules -> (N - 1) redundancies removed
            for group in inter_tree_groups:
                self.redundancy_counts["inter_tree"] += len(group) - 1

    # ------------------------------------------------------------------
    # Main analysis pipeline
    # ------------------------------------------------------------------

    def execute_rule_analysis(
        self,
        file_path: str,
        remove_below_n_classifications: int = -1,
    ) -> None:
        """Evaluates RF rules on a dataset, detects redundancies, and refines.

        This method:
        1. Classifies every sample to gather per-rule usage stats.
        2. Optionally removes low-usage rules (with sibling promotion).
        3. Tracks redundancy counts by type (intra_tree, inter_tree, low_usage).
        4. Writes a report and saves the final model.

        Args:
            file_path: Path to the CSV test file.
            remove_below_n_classifications: Threshold for low-usage pruning
                (-1 disables).
        """
        clf = self.classifier
        print(f"Testing Random Forest Rules on {os.path.basename(file_path)}...")
        start_time = time.time()

        _, _, X_test, y_test, _, _, feature_names = RuleClassifier.process_data(
            ".", file_path, is_test_only=True
        )
        sample_dicts = [dict(zip(feature_names, row)) for row in X_test]
        total_samples = len(y_test)

        # Reset counts
        for rule in clf.final_rules:
            rule.usage_count = 0
            rule.error_count = 0

        y_pred: List[Any] = []

        # Pre-compile tree lookups for RF
        tree_rules_map: Dict[str, list] = defaultdict(list)
        for rule in clf.final_rules:
            tree_identifier = rule.name.split('_')[0]
            tree_rules_map[tree_identifier].append(rule)

        tree_lookups: Dict[str, tuple] = {}
        for tree_id, rules in tree_rules_map.items():
            lookup_fn = clf._compile_tree_lookup(rules)
            if lookup_fn:
                tree_lookups[tree_id] = (lookup_fn, rules)

        # Main Classification Loop (Soft Voting)
        for i, sample in enumerate(sample_dicts):
            true_label = int(y_test[i])

            # Fast RF Voting — collect distributions for soft voting
            matched_rules: list = []
            tree_probas: list = []
            has_distributions = True

            for tree_id, (lookup_fn, rules) in tree_lookups.items():
                try:
                    rule_idx = lookup_fn(sample)
                    if rule_idx != -1:
                        matched_rule = rules[rule_idx]
                        matched_rules.append(matched_rule)

                        # Collect distribution for soft voting
                        if matched_rule.class_distribution is not None:
                            dist = matched_rule.class_distribution
                            total_c = sum(dist)
                            if total_c > 0:
                                tree_probas.append([c / total_c for c in dist])
                            else:
                                tree_probas.append(dist)
                        else:
                            has_distributions = False
                except Exception:
                    pass

            # Aggregate using soft voting
            if matched_rules:
                if has_distributions and tree_probas:
                    n_cls = len(tree_probas[0])
                    avg_proba = [0.0] * n_cls
                    for proba in tree_probas:
                        for j in range(n_cls):
                            avg_proba[j] += proba[j]
                    n_trees_voted = len(tree_probas)
                    avg_proba = [p / n_trees_voted for p in avg_proba]
                    pred_label = avg_proba.index(max(avg_proba))
                else:
                    # Fallback: hard voting
                    votes: List[Any] = []
                    for rule in matched_rules:
                        try:
                            label = int(str(rule.class_).replace('Class', '').strip())
                        except Exception:
                            label = rule.class_
                        votes.append(label)
                    counts = Counter(votes)
                    pred_label = counts.most_common(1)[0][0]
            else:
                pred_label = clf.default_class
                try:
                    pred_label = int(str(pred_label).replace('Class', '').strip())
                except Exception:
                    pass

            y_pred.append(pred_label)

            # Update stats for ALL rules that participated in the vote
            for rule in matched_rules:
                rule.usage_count += 1
                try:
                    r_class = int(str(rule.class_).replace('Class', '').strip())
                except Exception:
                    r_class = rule.class_
                if r_class != true_label:
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

                rem_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
                percent = (i + 1) / total_samples
                bar_len = 30
                filled_len = int(bar_len * percent)
                bar = '=' * filled_len + '-' * (bar_len - filled_len)

                sys.stdout.write(f"\r[{bar}] {percent:.1%} | ETA: {rem_str}")
                sys.stdout.flush()

        print()  # Newline after progress bar

        # Metrics
        y_pred_arr = np.array(y_pred)
        correct = np.sum(y_pred_arr == y_test)

        # Low-usage pruning with sibling promotion
        if remove_below_n_classifications > -1:
            print(f"\nPruning rules with <= {remove_below_n_classifications} classifications...")
            clf.specific_rules = []

            for rule in clf.final_rules:
                if rule.usage_count <= remove_below_n_classifications:
                    clf.specific_rules.append(rule)

            low_usage_count = len(clf.specific_rules)
            self.redundancy_counts["low_usage"] = low_usage_count
            print(f"Identified {low_usage_count} specific (low-usage) rules.")

            if clf.specific_rules:
                clf.final_rules = clf._promote_siblings(clf.specific_rules, clf.final_rules)
                print(f"Rules after promotion: {len(clf.final_rules)}")

            clf.update_native_model(clf.final_rules)

        # Generate Report
        clf._write_report(
            "examples/files/output_classifier_rf.txt",
            file_path, correct, total_samples,
            remove_below_n_classifications,
        )

        print(f"Analysis Time: {time.time() - start_time:.3f}s")
        clf._save_final_model()

        # Print redundancy summary
        self.print_redundancy_summary()

    # ------------------------------------------------------------------
    # Initial vs Final comparison
    # ------------------------------------------------------------------

    def compare_initial_final_results(self, file_path: str) -> None:
        """Compares performance of initial vs final rules for a Random Forest.

        Evaluates both rule sets on the test data, displays metrics, logs
        divergent cases, and writes a detailed report.

        Args:
            file_path: Path to the CSV test file.
        """
        _, _, X_test, y_test, _, target_column_name, feature_names = (
            RuleClassifier.process_data(".", file_path, is_test_only=True)
        )
        df_test = pd.DataFrame(X_test, columns=feature_names)
        df_test[target_column_name] = y_test

        self._compare_rf(df_test, target_column_name)

    def _compare_rf(self, df_test: pd.DataFrame, target_column_name: str) -> None:
        """Internal RF comparison logic."""
        clf = self.classifier

        print("\n" + "*" * 80)
        print("RUNNING INITIAL AND FINAL CLASSIFICATIONS (Random Forest)")
        print("*" * 80 + "\n")

        df = df_test.copy()
        y_true = df[target_column_name].astype(int).values
        feature_cols = [c for c in df.columns if c != target_column_name]
        sample_dicts = [dict(zip(feature_cols, row)) for row in df[feature_cols].values]
        total_samples = len(y_true)

        with open('examples/files/output_final_classifier_rf.txt', 'w') as f:
            f.write("******************** INITIAL VS FINAL RANDOM FOREST CLASSIFICATION REPORT ********************\n\n")

            # 1. Scikit-Learn
            print("Evaluated Scikit-Learn Model (Benchmark)...")
            f.write("\n******************************* SCIKIT-LEARN MODEL *******************************\n")
            try:
                with open('examples/files/sklearn_model.pkl', 'rb') as mf:
                    sk_model = pickle.load(mf)
                y_pred_sk = sk_model.predict(df[feature_cols].values)
                correct_sk = np.sum(y_pred_sk == y_true)
                RuleClassifier.display_metrics(y_true, y_pred_sk, correct_sk, total_samples, f, clf.class_labels)
            except Exception:
                f.write("Could not load/evaluate sklearn model.\n")

            # 2. Compare Rules (Merged Loop)
            print("\nComparing Initial vs Final Rules...")
            start_time = time.time()

            y_pred_initial: List[Any] = []
            y_pred_final: List[Any] = []

            # Build optimized lookups for Final Rules
            tree_rules_map_final: Dict[str, list] = defaultdict(list)
            for rule in clf.final_rules:
                tree_id = rule.name.split('_')[0]
                tree_rules_map_final[tree_id].append(rule)

            tree_lookups_final: Dict[str, tuple] = {}
            for tid, rules in tree_rules_map_final.items():
                fn = clf._compile_tree_lookup(rules)
                if fn:
                    tree_lookups_final[tid] = (fn, rules)

            # Build optimized lookups for Initial Rules
            tree_rules_map_init: Dict[str, list] = defaultdict(list)
            for rule in clf.initial_rules:
                tree_id = rule.name.split('_')[0]
                tree_rules_map_init[tree_id].append(rule)

            tree_lookups_init: Dict[str, tuple] = {}
            for tid, rules in tree_rules_map_init.items():
                fn = clf._compile_tree_lookup(rules)
                if fn:
                    tree_lookups_init[tid] = (fn, rules)

            # Helper for Fast RF Prediction (Soft Voting)
            def predict_fast_rf(sample: dict, lookups: dict, default_cls: Any) -> Any:
                tree_probas: list = []
                has_distributions = True
                matched_any = False

                for _, (lookup_fn, rules) in lookups.items():
                    try:
                        idx = lookup_fn(sample)
                        if idx != -1:
                            r = rules[idx]
                            matched_any = True
                            if r.class_distribution is not None:
                                dist = r.class_distribution
                                total_c = sum(dist)
                                if total_c > 0:
                                    tree_probas.append([c / total_c for c in dist])
                                else:
                                    tree_probas.append(dist)
                            else:
                                has_distributions = False
                    except Exception:
                        pass

                if matched_any and has_distributions and tree_probas:
                    n_cls = len(tree_probas[0])
                    avg = [0.0] * n_cls
                    for proba in tree_probas:
                        for j in range(n_cls):
                            avg[j] += proba[j]
                    n = len(tree_probas)
                    avg = [p / n for p in avg]
                    return avg.index(max(avg))

                if matched_any:
                    # Fallback: hard voting
                    votes: List[Any] = []
                    for _, (lookup_fn, rules) in lookups.items():
                        try:
                            idx = lookup_fn(sample)
                            if idx != -1:
                                r = rules[idx]
                                try:
                                    val = int(str(r.class_).replace('Class', '').strip())
                                except Exception:
                                    val = r.class_
                                votes.append(val)
                        except Exception:
                            pass
                    if votes:
                        return Counter(votes).most_common(1)[0][0]

                try:
                    return int(str(default_cls).replace('Class', '').strip())
                except Exception:
                    return default_cls

            for i, sample in enumerate(sample_dicts):
                # Initial
                if tree_lookups_init:
                    pred_init = predict_fast_rf(sample, tree_lookups_init, clf.default_class)
                else:
                    pred_init, _, _, _ = RuleClassifier.classify_rf(sample, clf.initial_rules)
                    try:
                        pred_init = int(str(pred_init).replace('Class', '').strip())
                    except Exception:
                        pass
                y_pred_initial.append(pred_init)

                # Final
                if tree_lookups_final:
                    pred_final = predict_fast_rf(sample, tree_lookups_final, clf.default_class)
                else:
                    pred_final, _, _, _ = RuleClassifier.classify_rf(sample, clf.final_rules)
                    try:
                        pred_final = int(str(pred_final).replace('Class', '').strip())
                    except Exception:
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

                    rem_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
                    percent = (i + 1) / total_samples
                    bar_len = 30
                    filled_len = int(bar_len * percent)
                    bar = '=' * filled_len + '-' * (bar_len - filled_len)

                    sys.stdout.write(f"\r[{bar}] {percent:.1%} | ETA: {rem_str}")
                    sys.stdout.flush()

            print()  # Newline

            # Metrics Output
            print("Evaluation Complete. Generating Report...")

            # Initial
            f.write("\n******************************* INITIAL MODEL *******************************\n")
            y_pred_initial_arr = np.array(y_pred_initial)
            correct_init = np.sum(y_pred_initial_arr == y_true)

            print("\n--- Initial Rules Metrics ---")
            RuleClassifier.display_metrics(y_true, y_pred_initial_arr, correct_init, total_samples, f, clf.class_labels)
            print(f"Number of initial rules: {len(clf.initial_rules)}")

            # Final
            f.write("\n******************************* FINAL MODEL *******************************\n")
            y_pred_final_arr = np.array(y_pred_final)
            correct_final = np.sum(y_pred_final_arr == y_true)

            print("\n--- Final Rules Metrics ---")
            RuleClassifier.display_metrics(y_true, y_pred_final_arr, correct_final, total_samples, f, clf.class_labels)
            print(f"Number of final rules: {len(clf.final_rules)}")

            # Divergence
            divergent_count = np.sum(y_pred_initial_arr != y_pred_final_arr)
            print(f"\nTotal divergent cases: {divergent_count}")
            f.write("\n******************************* DIVERGENT CASES *******************************\n")
            f.write(f"Total divergent cases: {divergent_count}\n")

            # Metrics
            n_features = len(feature_cols)
            metrics_init = RuleClassifier.calculate_structural_complexity(clf.initial_rules, n_features)
            metrics_final = RuleClassifier.calculate_structural_complexity(clf.final_rules, n_features)

            f.write("\n******************************* INTERPRETABILITY METRICS *******************************\n")
            f.write("Metrics (Final vs Initial):\n")
            for k, v in metrics_final.items():
                init_v = metrics_init.get(k, 0)
                diff = ""
                if isinstance(v, (int, float)) and init_v != 0:
                    pct = ((v - init_v) / init_v) * 100
                    diff = f" ({pct:+.1f}%)"
                f.write(f"  {k}: {v}{diff}\n")

        # Print redundancy summary at the end
        self.print_redundancy_summary()
