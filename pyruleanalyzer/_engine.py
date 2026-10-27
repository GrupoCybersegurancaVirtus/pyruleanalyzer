"""Iterative (rule by rule) prediction engine.

Provides ``IterativeEngineMixin``: ``classify``, the single-sample entry
point (native function first, then compiled arrays), and the reference
engines that walk the ``Rule`` objects themselves -- ``classify_dt``,
``classify_rf`` (soft voting) and ``classify_gbdt`` (additive scores).

`RuleClassifier` is defined after its mixins, so the methods that name it import
it locally.
"""

import numpy as np
from collections import Counter, defaultdict


class IterativeEngineMixin:
    """Single-sample classification over the Rule objects."""

    # Method to execute the classification process
    def classify(self, data, final=False):
        """
        Classifies a single data instance using extracted rules.

        This method delegates the classification logic to the compiled tree
        arrays, to the native function when it was compiled from the requested
        rule set, or to iterative evaluation. All three answer with the rule
        set `final` selects.

        Args:
            data (Dict[str, float]): The instance to classify.
            final (bool): If True, uses `final_rules` (post-analysis; the
                initial rules when refinement has not run). If False, uses
                `initial_rules`.

        Returns:
            Tuple[int, List[int]|None, List[float]|None]: 
                - Predicted class label (int).
                - List of votes (Random Forest only).
                - Class probabilities (Random Forest only).
        """
        # Every path below must answer with the rule set `final` selects:
        # final_rules, or initial_rules when there are none yet.
        rules = self._rules_for(final)

        # Checked up front: the fast paths below swallow exceptions, so this
        # must not be left to predict_batch or it would be silently masked.
        # NaN is answered (by every path) only when the rules record where
        # each split sends it.
        for _v in data.values():
            if _v != _v:   # NaN, whatever the numeric type
                self._check_missing_values(
                    np.array([np.nan]),
                    self._rules_route_missing(rules) if rules
                    else self._trees_route_missing(self._tree_arrays))
                break

        # A missing feature is an error on every engine (also checked before
        # the fast paths, which would swallow it or read it as 0).
        self._assert_features_present(data, rules)

        # --- FAST PATH 1: Array-based single-sample prediction ---
        # predict_batch selects (and, the first time, compiles) the arrays of
        # the requested rule set, so this path cannot answer with the other.
        if self._arrays_compiled:
            try:
                row = np.array(
                    [[data.get(f, 0.0) for f in self._array_feature_names]],
                    dtype=np.float64,
                )
                # predict_batch applies the configured input precision itself.
                pred = self.predict_batch(
                    row, feature_names=list(self._array_feature_names),
                    use_final=final)[0]
                clean_class = int(pred)
                return clean_class, None, None
            except Exception:
                pass  # Fallback to slower paths

        if self.input_dtype == 'float32':
            data = {
                k: (float(np.float32(v)) if isinstance(v, (int, float)) else v)
                for k, v in data.items()
            }

        # --- FAST PATH 2: Native function (exec-compiled) ---
        # Only valid for the rule set it was compiled from (update_native_model
        # is also called on the refined rules, and on unpickling).
        if (self.native_fn is not None
                and getattr(self, '_native_source', None) is rules):
            try:
                # native_fn returns (prediction, votes, proba)
                return self.native_fn(data) # pyright: ignore[reportCallIssue]
            except Exception:
                # Fallback to standard execution if something goes wrong (e.g., missing keys)
                pass

        # --- SLOW PATH: Iterative Execution ---

        predicted_class = self.default_class
        votes = None
        proba = None

        if self.algorithm_type == 'Random Forest':
            n_dist = next((len(r.class_distribution) for r in rules
                           if r.class_distribution is not None), 0)
            predicted_class, votes, proba, _ = self.classify_rf(
                data, rules, labels=self._rf_labels(n_dist))
            if predicted_class is None:
                # No tree had a matching rule: uncovered region in every tree.
                predicted_class = self.default_class
                self.fallback_activations += 1

        elif self.algorithm_type == 'Gradient Boosting Decision Trees':
            predicted_class, _, _ = self.classify_gbdt(
                data, rules, self._gbdt_init_scores,
                self._gbdt_is_binary, self._gbdt_classes,
            )

        elif self.algorithm_type == 'Decision Tree':
            matched_rule = self.classify_dt(data, rules)
            if matched_rule:
                predicted_class = matched_rule.class_
            else:
                # Uncovered region: `predicted_class` stays at default_class.
                self.fallback_activations += 1

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
            for cond in rule.parsed_conditions:
                var, op, value = cond
                # Use data.get() to avoid KeyErrors, defaulting to None implies failure
                instance_value = data.get(var)

                if instance_value is None:
                    rule_satisfied = False
                    break

                if instance_value != instance_value:
                    # NaN: satisfied only on the side its split sends
                    # missing values to (unknown routing: not satisfied).
                    if getattr(cond, 'missing', False):
                        continue
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
    def classify_rf(data, rules, labels=None):
        """
        Classifies a single data instance using extracted rules from a Random Forest.
        Uses soft probability averaging (like sklearn) when class distributions are
        available, otherwise falls back to hard majority voting.

        Args:
            data (Dict[str, float]): Instance data.
            rules (List[Rule]): List of rule instances.
            labels (List[int], optional): The label of each distribution
                column. Without it, soft voting answers the column index.

        Returns:
            Tuple: (Predicted Label, Votes List, Proba List, Matched Rules List)
        """
        from .rule_classifier import RuleClassifier
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
            predicted_class = (labels[best_idx]
                               if labels is not None and len(labels) == n_classes
                               else best_idx)
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
        classification takes the positive class when the score is >= 0 (as
        scikit-learn does); multiclass uses argmax.

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
                    for cond in rule.parsed_conditions:
                        var, op, value = cond
                        instance_value = data.get(var)
                        if instance_value is None:
                            matched = False
                            break
                        if instance_value != instance_value:
                            # NaN follows the routing of its split
                            if getattr(cond, 'missing', False):
                                continue
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

            # Decide on the raw score, as scikit-learn does (raw >= 0); see
            # update_native_model for why not sigmoid(score) >= 0.5.
            predicted_class = int(classes[1]) if score >= 0.0 else int(classes[0])

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
                    for cond in rule.parsed_conditions:
                        var, op, value = cond
                        instance_value = data.get(var)
                        if instance_value is None:
                            matched = False
                            break
                        if instance_value != instance_value:
                            # NaN follows the routing of its split
                            if getattr(cond, 'missing', False):
                                continue
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
