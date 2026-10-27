"""Rule refinement.

Provides ``RefinementMixin``: detection of overlapping and duplicated rules,
merging of sibling leaves that predict the same class (boundary
redundancy), removal of low-usage rules with promotion of their siblings,
and the error-aware evaluator that bounds what refinement may cost.
"""

import copy
import numpy as np
from collections import defaultdict
from typing import Dict
from ._rules import condition_signature, Rule


class RefinementMixin:
    """Redundancy analysis and rule refinement."""

    # Method to detect rules whose regions overlap
    def find_overlapping_rules(self, rules=None, per_tree=True, max_rules_per_tree=20000):
        """
        Find pairs of rules whose feature-space regions intersect.

        Each rule's conditions are axis-aligned inequalities, so its region is
        a box (a product of intervals). Two rules overlap exactly when their
        boxes intersect on every feature. Rules extracted from a tree never
        overlap — the leaves partition the space — but refinement can break
        that: promoting a sibling drops a condition and may extend the rule
        over ground already covered by other rules.

        This matters because an overlap is invisible to accuracy. The iterative
        engine resolves it by taking the first matching rule, which is an
        implicit priority order that a tree does not have and that the CPN model
        does not reproduce. A rule set with overlaps is a *rule list*, not a
        decision tree.

        Args:
            rules (List[Rule], optional): Rules to check. Defaults to
                final_rules if available, else initial_rules.
            per_tree (bool): Compare only rules belonging to the same tree
                (the name prefix before the first '_'). Rules from different
                trees of an ensemble are expected to overlap and are skipped.
            max_rules_per_tree (int): Trees with more rules than this are
                skipped, with their id reported under 'skipped_trees', to keep
                the quadratic comparison bounded.

        Returns:
            Dict with keys:
                'overlaps': list of (name_i, name_j, same_class) tuples;
                'n_overlaps': total number of overlapping pairs;
                'n_conflicting': pairs that also disagree on the predicted
                    class (these change the prediction, not just the shape);
                'trees_affected': sorted list of tree ids containing overlaps;
                'skipped_trees': tree ids skipped for exceeding the size limit.
        """
        if rules is None:
            rules = self.final_rules if self.final_rules else self.initial_rules

        # A Decision Tree is a single tree even though its rule names contain
        # '_' ("Rule0_Class1"); splitting on it would put every rule in its own
        # group and compare nothing.
        group_by_tree = per_tree and self.algorithm_type != 'Decision Tree'

        groups: Dict[str, list] = defaultdict(list)
        for rule in rules:
            tid = (rule.name.split('_')[0]
                   if (group_by_tree and '_' in rule.name) else '_all')
            groups[tid].append(rule)

        overlaps = []
        skipped = []
        for tid, group in groups.items():
            if len(group) > max_rules_per_tree:
                skipped.append(tid)
                continue

            # Bounds per rule: lo/hi arrays plus inclusiveness flags.
            feats = sorted({v for r in group for v, _, _ in r.parsed_conditions})
            if not feats:
                continue
            col = {f: i for i, f in enumerate(feats)}
            n, d = len(group), len(feats)

            lo = np.full((n, d), -np.inf)
            hi = np.full((n, d), np.inf)
            lo_open = np.zeros((n, d), dtype=bool)   # lower bound is strict (>)
            hi_open = np.zeros((n, d), dtype=bool)   # upper bound is strict (<)

            for i, rule in enumerate(group):
                for var, op, val in rule.parsed_conditions:
                    c = col[var]
                    if op in ('<=', '<'):
                        if val < hi[i, c] or (val == hi[i, c] and op == '<'):
                            hi[i, c] = val
                            hi_open[i, c] = (op == '<')
                    else:
                        if val > lo[i, c] or (val == lo[i, c] and op == '>'):
                            lo[i, c] = val
                            lo_open[i, c] = (op == '>')

            for i in range(n - 1):
                j = slice(i + 1, n)
                # Intervals [lo_a, hi_a] and [lo_b, hi_b] intersect when
                # max(lo) < min(hi), or they touch at a point that both sides
                # include.
                lo_max = np.maximum(lo[i], lo[j])
                hi_min = np.minimum(hi[i], hi[j])
                touching = lo_max == hi_min
                # Inclusiveness comes from whichever bound is binding: a looser
                # bound being strict says nothing about the tighter one. Only
                # when both coincide do they both have to be inclusive.
                lo_incl = np.where(
                    lo[i] > lo[j], ~lo_open[i],
                    np.where(lo[j] > lo[i], ~lo_open[j],
                             ~(lo_open[i] | lo_open[j])))
                hi_incl = np.where(
                    hi[i] < hi[j], ~hi_open[i],
                    np.where(hi[j] < hi[i], ~hi_open[j],
                             ~(hi_open[i] | hi_open[j])))
                per_feat = (lo_max < hi_min) | (touching & lo_incl & hi_incl)
                hits = np.nonzero(per_feat.all(axis=1))[0]
                for k in hits:
                    other = group[i + 1 + int(k)]
                    overlaps.append((
                        group[i].name, other.name,
                        str(group[i].class_) == str(other.class_),
                    ))

        trees = sorted({name.split('_')[0] for name, _, _ in overlaps})
        return {
            'overlaps': overlaps,
            'n_overlaps': len(overlaps),
            'n_conflicting': sum(1 for _, _, same in overlaps if not same),
            'trees_affected': trees,
            'skipped_trees': skipped,
        }

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
            # A rule without conditions is a GBDT init score (or a tree that
            # is a single leaf): it is not a region shared between trees, and
            # summing it with a leaf value would mix an unscaled init score
            # with learning-rate-scaled contributions.
            if not rule.parsed_conditions:
                continue
            # Create a canonical signature for the rule's logic
            # 1. We use parsed_conditions directly (no string parsing)
            # 2. We sort the conditions so that logic order doesn't matter
            # 3. We use the class label
            
            # signature format: (class_label, tuple_of_sorted_conditions)
            # condition format: (var, op, value) - already float from parsing
            # The missing-value routing is part of the region: two trees
            # with the same thresholds may send NaN different ways.
            sorted_conditions = tuple(sorted(
                (condition_signature(c) for c in rule.parsed_conditions),
                key=lambda s: (s[0], s[1], s[2], str(s[3]))))
            signature = (rule.class_, sorted_conditions)
            
            rules_by_signature[signature].append(rule)
            
        # Return only the groups that have more than one similar rule
        return [group for group in rules_by_signature.values() if len(group) > 1]
    
    # Method to find duplicated rules within the same tree (Boundary Redundancy)
    def find_duplicated_rules(self):
        """
        Identifies nearly identical rules within the same decision tree context.

        This method searches for rule pairs that:
        - Have the same class label.
        - Share all conditions except the last one (same path parent).
        - Differ only in the final condition, which splits the same variable at
          the same threshold with complementary operators (``<=``/``>`` or
          ``<``/``>=``), e.g. v1 <= 5 vs v1 > 5.

        Such pairs are considered duplicates because they imply the split at that
        boundary was unnecessary for the final classification outcome.

        Returns:
            List[Tuple[Rule,Rule]]: A list of tuples, each representing a pair of duplicated rules.
        """
        duplicated_rules = []
        rules_by_prefix = defaultdict(list)

        # Target rules to analyze
        target_rules = self.final_rules if self.final_rules else self.initial_rules

        # Boundary redundancy is a property of a *single* tree: the two rules
        # must be the two children of one split. Ensemble trees often split on
        # the same feature at the same threshold, so the signature has to
        # include the tree id -- otherwise leaves of different trees can be
        # merged as if they were siblings, silently dropping a leaf from one of
        # them. A Decision Tree is one tree even though its rule names contain
        # '_' ("Rule0_Class1"), so it is grouped as a whole.
        group_by_tree = self.algorithm_type != 'Decision Tree'

        # 1. Group rules by tree, class and their "Prefix" (all conditions except the last)
        for rule in target_rules:
            if not rule.parsed_conditions:
                continue

            tree_id = (rule.name.split('_')[0]
                       if (group_by_tree and '_' in rule.name) else '_all')

            # Prefix signature: (tree, class, conditions[0:-1])
            # We assume order matters here because it represents a tree path
            prefix_key = (tree_id, rule.class_, tuple(rule.parsed_conditions[:-1]))
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
                    if val1 != val2 and not abs(val1 - val2) <= 1e-9:
                        continue
                    
                    # Check 3: Complementary operators, (<= vs >) or (< vs >=),
                    # so the two rules split one region exactly in two.
                    is_duplicate = {op1, op2} in ({'<=', '>'}, {'<', '>='})

                    # GBDT: siblings must also add exactly the same value to
                    # the score (v_i = v_j). The merged rule keeps rule1's
                    # value, so any tolerance here would move the score of
                    # rule2's region and could flip a decision near 0.
                    if is_duplicate and self.algorithm_type == 'Gradient Boosting Decision Trees':
                        if rule1.contribution != rule2.contribution:
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
    def adjust_and_remove_rules(self, method="refine_duplicates"):
        """
        Adjusts and removes duplicated rules from the rule set based on the specified method.

        With ``"refine_duplicates"`` it finds sibling leaves whose split is redundant
        (:meth:`find_duplicated_rules`) and merges each pair into their parent,
        one round at a time. With ``"custom"`` it delegates to the function set
        via :meth:`set_custom_rule_removal`.

        Args:
            method (str): ``"refine_duplicates"`` (default) or ``"custom"``.

        Returns:
            Tuple[List[Rule], List[Tuple[Rule, Rule]]]: 
                - A new list of rules (merged + remaining).
                - A list of identified duplicated pairs (used for convergence checks).
        """
        if method == "custom":
            return self.custom_rule_removal(self.initial_rules)
        
        if method != "refine_duplicates":
            raise ValueError(f"Invalid method: {method}. Use 'refine_duplicates' or 'custom'.")

        # Determine source rules: Use final_rules if populated (iteration n), else initial (iteration 0)
        source_rules = self.final_rules if self.final_rules else self.initial_rules

        # 1. Boundary redundancy within the same tree: siblings that can be
        # merged into their parent
        sibling_pairs = self.find_duplicated_rules()

        rules_to_remove = set()
        new_generalized_rules = []

        if sibling_pairs:
            print(f"Merging {len(sibling_pairs)} pairs of duplicated rules...")

        for rule1, rule2 in sibling_pairs:
            rules_to_remove.add(rule1)
            rules_to_remove.add(rule2)

            # Create a new generalized rule (Parent Logic): the last condition,
            # which differentiated the two siblings, is dropped.
            new_generalized_rules.append(self._merge_sibling_pair(rule1, rule2))

        # 2. Rebuild the Rule Set
        # Keep rules that were NOT marked for removal + the newly created generalized rules
        final_list = new_generalized_rules + [r for r in source_rules if r not in rules_to_remove]
        
        return final_list, sibling_pairs
    
    # Method to name the class at an index of a class distribution
    def _label_at(self, idx: int, n: int) -> str:
        """The class label at index `idx` of an `n`-class distribution.

        Distributions are indexed like the estimator's ``classes_``, which are
        the labels only when those are 0..K-1; `class_labels` holds the real
        ones when it covers every class.

        Args:
            idx (int): Index into the distribution.
            n (int): Length of the distribution.

        Returns:
            str: The label, as the rules store it.
        """
        labels = getattr(self, 'class_labels', None)
        if labels is not None and len(labels) == n:
            return str(labels[idx])
        return str(idx)

    # Method to merge two sibling leaves into their parent
    def _merge_sibling_pair(self, rule1, rule2):
        """The rule of the parent of two sibling leaves with the same output.

        `find_duplicated_rules` only pairs siblings whose outputs agree (same
        class; for GBDT the same leaf value, for RF proportional
        distributions), so the parent rule predicts exactly what both did.

        Args:
            rule1 (Rule): One leaf.
            rule2 (Rule): Its sibling.

        Returns:
            Rule: The rule of the parent node. Its usage and error counts are
            the sums of the children's: its region is their union, and the
            low-usage stage decides on these counts.
        """
        common_conditions = rule1.conditions[:-1]
        name = f"{rule1.name}_&_{rule2.name}"
        if self.algorithm_type == 'Gradient Boosting Decision Trees':
            merged = Rule(name, rule1.class_, common_conditions,
                          leaf_value=rule1.leaf_value,
                          learning_rate=rule1.learning_rate,
                          class_group=rule1.class_group)
            merged.contribution = rule1.contribution  # == rule2.contribution
        else:
            d1 = rule1.class_distribution
            d2 = rule2.class_distribution
            if d1 is not None and d2 is not None:
                dist = [a + b for a, b in zip(d1, d2)]
            elif d1 is not None or d2 is not None:
                dist = list(d1 if d1 is not None else d2)
            else:
                dist = None
            # Both leaves predict rule1.class_; so does their parent.
            merged = Rule(name, rule1.class_, common_conditions,
                          class_distribution=dist)
        merged.parsed_conditions = list(rule1.parsed_conditions[:-1])
        merged.usage_count = rule1.usage_count + rule2.usage_count
        merged.error_count = rule1.error_count + rule2.error_count
        return merged

    # Method to remove boundary redundancy until none is left
    def merge_boundary_redundancy(self):
        """
        Merge sibling leaves with the same output, repeatedly, up to the root.

        This is the boundary-redundancy stage of the refinement: two sibling
        leaves that produce the same output are replaced by their parent, and
        since that may make the parent redundant with its own sibling, the
        stage repeats until no pair is left. The model's predictions do not
        change. Operates on `final_rules` (seeded from `initial_rules`).

        Returns:
            List[Tuple[Rule, Rule]]: Every pair merged, over all rounds.
        """
        if not self.final_rules:
            self.final_rules = list(self.initial_rules)
        merged_pairs = []
        while True:
            pairs = self.find_duplicated_rules()
            if not pairs:
                return merged_pairs
            remove = set()
            new_rules = []
            for rule1, rule2 in pairs:
                if id(rule1) in remove or id(rule2) in remove:
                    continue
                remove.update((id(rule1), id(rule2)))
                new_rules.append(self._merge_sibling_pair(rule1, rule2))
                merged_pairs.append((rule1, rule2))
            self.final_rules = [r for r in self.final_rules
                                if id(r) not in remove] + new_rules

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
            if (c_var == last_var and c_op == expected_op
                    and (c_val == last_val or abs(c_val - last_val) < 1e-9)):
                return candidate

        return None

    # Method to promote sibling when removing specific rules
    def _promote_siblings(self, rules_to_remove, all_rules, accept=None,
                          fill_holes=None):
        """
        Removes rules and hands each removed region to what is left of its tree.

        Removing a leaf removes the split above it: the parent node is replaced
        by the leaf's other child, which now answers for the whole parent
        region. Concretely, every rule on the other side of that split drops
        the split's condition (the condition at the parent's depth):

        * **Sibling leaf** (promotion): the sibling drops its last condition
          and becomes the parent. Its class distribution absorbs the removed
          rule's, and its class is the majority of the sum.
        * **Sibling sub-branch** (grafting): every live leaf of the sub-branch
          drops the parent's condition, so the sub-branch is raised into the
          parent's place and its own splits route the removed region. Leaf
          outputs are unchanged (the removed rule's training distribution
          cannot be attributed to individual leaves without data).

        Either way the rules of each tree remain a partition: no region is
        left uncovered (no `default_class` fallback) and no two rules overlap.
        A rule that is the only rule of its tree (no conditions) is never
        removed, since that would leave the tree without an answer anywhere.

        Usage counts move with promotion (the sibling covers the union of the
        two regions). After grafting, the removed rule's samples spread over
        the raised leaves, which cannot be known without data: the analyzers
        recount usage and errors on the refinement set afterwards
        (`_refinement_evaluator(...).recount()`).

        **Gradient Boosting** (``fill_holes=False``, its default): a removed
        leaf whose sibling is a sub-branch is removed without grafting, and the
        tree adds 0.0 to the score on its region. In an additive model a tree
        that does not answer simply contributes nothing (it is not a fallback
        to `default_class`; the other trees and the init score still decide),
        whereas grafting would apply leaf values fitted to the residuals of a
        different region.

        **Critical**: Rules are processed from deepest to shallowest, so deeper
        removals and promotions happen first and may create the sibling a
        shallower removal needs.

        **Performance**: sibling leaves are found through a hash index keyed by
        (tree, prefix, variable, operator, threshold); only the sub-branch case
        scans the rules of one tree.

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

            Removing only A instead grafts the [z <= 7] sub-branch: B becomes
            "x<=5 AND z<=7" and C "x<=5 AND z>7", and A's region is routed by
            the z split instead of falling back to default_class.

        Args:
            rules_to_remove (List[Rule]): Rules identified for removal.
            all_rules (List[Rule]): The complete set of current rules.
            accept (Callable[[Rule, List[Rule], List[Rule]], bool], optional):
                Called before each removal with the rule to remove, the rules
                whose regions the removal changes (as they are now) and those
                rules as they would become; the removal happens only if it
                returns True. Used for error-aware refinement (see
                `_refinement_evaluator`). Default: every removal happens.
            fill_holes (bool, optional): Graft sub-branches (True) or leave the
                region uncovered (False). Default: True, except for Gradient
                Boosting.

        Returns:
            List[Rule]: The adjusted rule set (copies; the input rules are not
            modified).
        """
        complement = {'<=': '>', '>': '<=', '<': '>=', '>=': '<'}
        if fill_holes is None:
            fill_holes = self.algorithm_type != 'Gradient Boosting Decision Trees'

        # Siblings only exist *within* one tree. Ensemble trees routinely split
        # on the same feature at the same threshold, so a key that ignores the
        # tree would match a leaf of one tree as the sibling of a leaf of
        # another and promote it -- generalizing a rule over ground still
        # covered by its own tree's leaves (overlapping guards). Group by tree
        # id, using the same convention as find_overlapping_rules: a Decision
        # Tree is a single tree even though its rule names contain '_'.
        group_by_tree = self.algorithm_type != 'Decision Tree'

        # Method to _tree_id.
        def _tree_id(r):
            """Tree the rule belongs to (prefix before the first '_')."""
            if group_by_tree and '_' in r.name:
                return r.name.split('_')[0]
            return '_all'

        # Method to _make_key.
        def _make_key(r):
            """Index key: (tree_id, prefix, var, op, threshold_rounded)."""
            if not r.parsed_conditions:
                return None
            prefix = tuple(r.parsed_conditions[:-1])
            var, op, val = r.parsed_conditions[-1]
            return (_tree_id(r), prefix, var, op, round(val, 9))

        # Promotion rewrites a rule's conditions, class and name. `final_rules`
        # and `initial_rules` hold the *same* Rule objects, so mutating them in
        # place would silently rewrite the unrefined model too: its rules would
        # gain the generalized regions, start overlapping each other, and stop
        # reproducing the original tree. Clone first, so refinement only ever
        # touches its own copies.
        def _clone(r):
            c = copy.copy(r)
            c.conditions = list(r.conditions)
            c.parsed_conditions = list(r.parsed_conditions)
            dist = getattr(r, 'class_distribution', None)
            if dist is not None:
                c.class_distribution = list(dist)
            return c

        clones = {id(r): _clone(r) for r in all_rules}
        all_rules = [clones[id(r)] for r in all_rules]
        rules_to_remove = [clones[id(r)] for r in rules_to_remove if id(r) in clones]

        # --- Build sibling index: key -> list of rules with that key ---
        sibling_index: dict = {}
        rules_of_tree: dict = defaultdict(list)
        for r in all_rules:
            key = _make_key(r)
            if key is not None:
                sibling_index.setdefault(key, []).append(r)
            rules_of_tree[_tree_id(r)].append(r)

        # Method to _unindex.
        def _unindex(r):
            """Drop `r` from the sibling index (before its conditions change)."""
            key = _make_key(r)
            if key is not None and key in sibling_index:
                lst = [x for x in sibling_index[key] if x is not r]
                if lst:
                    sibling_index[key] = lst
                else:
                    del sibling_index[key]

        # Method to _reindex.
        def _reindex(r):
            """Insert `r` in the sibling index under its current conditions."""
            key = _make_key(r)
            if key is not None:
                sibling_index.setdefault(key, []).append(r)

        # Work on a mutable copy
        working_rules = list(all_rules)
        promoted_count = 0
        grafted_count = 0
        kept_count = 0
        rejected_count = 0

        # Sort rules to remove by depth (deepest first)
        sorted_to_remove = sorted(
            rules_to_remove,
            key=lambda r: len(r.parsed_conditions),
            reverse=True
        )

        # Track which rules have been removed by identity
        removed_ids: set = set()

        for rule in sorted_to_remove:
            if id(rule) in removed_ids:
                continue

            if not rule.parsed_conditions:
                # The rule is its whole tree: nothing could take over.
                kept_count += 1
                continue

            tid = _tree_id(rule)
            prefix = tuple(rule.parsed_conditions[:-1])
            depth = len(prefix)
            last_var, last_op, last_val = rule.parsed_conditions[-1]
            expected_op = complement.get(last_op)

            # --- O(1) sibling lookup via index ---
            sibling = None
            branch = []
            if expected_op is not None:
                # The sibling has the complementary operator, in the same tree
                sib_key = (tid, prefix, last_var, expected_op, round(last_val, 9))
                for c in sibling_index.get(sib_key, []):
                    if c is not rule and id(c) not in removed_ids:
                        sibling = c
                        break
                if sibling is None and fill_holes:
                    # The other child is a sub-branch: its live leaves extend
                    # the prefix with the complementary condition.
                    branch = [
                        r for r in rules_of_tree[tid]
                        if r is not rule and id(r) not in removed_ids
                        and len(r.parsed_conditions) > depth + 1
                        and tuple(r.parsed_conditions[:depth]) == prefix
                        and r.parsed_conditions[depth][0] == last_var
                        and r.parsed_conditions[depth][1] == expected_op
                        and (r.parsed_conditions[depth][2] == last_val
                             or abs(r.parsed_conditions[depth][2] - last_val) < 1e-9)
                    ]

            if sibling is None and not branch:
                if fill_holes:
                    # No other side to hand the region to (the rules are not
                    # the leaves of a tree here): keep the rule rather than
                    # open a hole.
                    kept_count += 1
                    continue
                # Gradient Boosting: the tree abstains (adds 0.0) on the
                # removed region.
                if accept is not None and not accept(rule, [rule], []):
                    rejected_count += 1
                    continue
                _unindex(rule)
                removed_ids.add(id(rule))
                continue

            # Proposal: the rules on the other side of the split, without the
            # split's condition.
            if sibling is not None:
                new = _clone(sibling)
                new.conditions = sibling.conditions[:-1]
                new.parsed_conditions = sibling.parsed_conditions[:-1]
                # Combine class_distribution: the promoted sibling now covers
                # the region of the removed rule as well, so add their counts
                if (getattr(rule, 'class_distribution', None) is not None
                        and getattr(sibling, 'class_distribution', None) is not None):
                    new.class_distribution = [
                        a + b for a, b in zip(sibling.class_distribution, rule.class_distribution)
                    ]
                    # Update class_ to match the new majority class (a label,
                    # not the index into the distribution)
                    dist = new.class_distribution
                    new.class_ = self._label_at(dist.index(max(dist)), len(dist))
                if "_promoted" not in new.name:
                    new.name = f"{new.name}_promoted"
                plan = [(sibling, new)]
            else:
                plan = []
                for r in branch:
                    new = _clone(r)
                    new.conditions = r.conditions[:depth] + r.conditions[depth + 1:]
                    new.parsed_conditions = (r.parsed_conditions[:depth]
                                             + r.parsed_conditions[depth + 1:])
                    if "_grafted" not in new.name:
                        new.name = f"{new.name}_grafted"
                    plan.append((r, new))

            if accept is not None and not accept(
                    rule, [rule] + [old for old, _ in plan], [new for _, new in plan]):
                rejected_count += 1
                continue

            for old, new in plan:
                _unindex(old)
                old.conditions = new.conditions
                old.parsed_conditions = new.parsed_conditions
                old.class_distribution = new.class_distribution
                old.class_ = new.class_
                old.name = new.name
                _reindex(old)

            if sibling is not None:
                sibling.usage_count += rule.usage_count
                sibling.error_count += rule.error_count
                promoted_count += 1
            else:
                grafted_count += 1

            # Remove the rule from the index; mark removed (deferred filtering)
            _unindex(rule)
            removed_ids.add(id(rule))

        # Single-pass filtering instead of per-iteration list rebuild
        working_rules = [r for r in working_rules if id(r) not in removed_ids]

        if promoted_count > 0:
            print(f"Promoted {promoted_count} sibling rule(s) after specific rule removal.")
        if grafted_count > 0:
            print(f"Grafted {grafted_count} sibling sub-branch(es) into the "
                  "place of their parent after specific rule removal.")
        if kept_count > 0:
            print(f"Kept {kept_count} low-usage rule(s) that no other rule of "
                  "their tree could take over.")
        if rejected_count > 0:
            print(f"Kept {rejected_count} low-usage rule(s) whose removal "
                  "would add errors on the refinement set.")
        if promoted_count or grafted_count:
            self._warn_promotions_that_overlap(working_rules, tree_of=_tree_id)

        return working_rules

    # Method to run the low-usage stage on the refinement set
    def _remove_low_usage(self, rules_to_remove, X, y, feature_names,
                          max_error_increase=None):
        """Remove `rules_to_remove` from `final_rules`, then recount usage.

        The low-usage stage of the analyzers. Each removal hands the rule's
        region to the rest of its tree (`_promote_siblings`); with
        `max_error_increase` set, a removal happens only while the
        refinement-set errors stay within that many samples of the errors
        before the stage (0: reduced-error pruning, errors never increase).
        Usage and error counts of the resulting rules are then recounted on
        the refinement set, since grafting spreads a removed region over
        several leaves.

        Args:
            rules_to_remove (List[Rule]): Candidates (rules of `final_rules`).
            X (np.ndarray): Refinement samples (from `_prepare_test_data`).
            y (array-like): True labels.
            feature_names (List[str]): Column names of `X`.
            max_error_increase (int, optional): Error budget; None (default)
                removes every candidate (usage-only criterion).

        Returns:
            List[Rule]: The refined rule set (also stored in `final_rules`).
        """
        accept = None
        if max_error_increase is not None:
            if max_error_increase < 0:
                raise ValueError("max_error_increase must be >= 0 or None.")
            evaluator = self._refinement_evaluator(
                X, y, feature_names, self.final_rules, max_error_increase)
            accept = evaluator.accept
        self.final_rules = self._promote_siblings(
            rules_to_remove, self.final_rules, accept=accept)
        if max_error_increase is not None:
            print(f"Refinement-set errors: {evaluator.base_errors} -> "
                  f"{evaluator.errors()} (budget +{max_error_increase}).")
        self._refinement_evaluator(
            X, y, feature_names, self.final_rules).recount(self.final_rules)
        return self.final_rules

    # Method to build an evaluator of rule-set changes on the refinement set
    def _refinement_evaluator(self, X, y, feature_names, rules, max_error_increase=0):
        """Predictions and errors of `rules` on a refinement set, kept up to date.

        Backs error-aware refinement (reduced-error pruning on the refinement
        set) and the recount of usage/error statistics after refinement. The
        semantics are those of the compiled engines: DT, the label of the
        matching rule (`default_class` when none matches); RF, the argmax of
        the sum of the matching leaves' normalised distributions
        (`default_class` when every tree abstains); GBDT, init score plus the
        matching contributions (binary: score >= 0).

        Each change touches one tree, and only on the samples that the changed
        rules cover, so a candidate is evaluated on those samples alone.

        Args:
            X (np.ndarray): Refinement samples, already prepared like the
                analyzers do (`_prepare_test_data`, which applies the input
                precision).
            y (array-like): True labels.
            feature_names (List[str]): Column names of `X`.
            rules (List[Rule]): Current rule set.
            max_error_increase (int): How many more refinement samples the
                accepted changes may misclassify, in total, than `rules` do.

        Returns:
            types.SimpleNamespace: ``accept(rule, old_rules, new_rules)``, the
            callback for `_promote_siblings` (commits the change when the
            error budget allows it); ``recount(rules)``, which sets
            `usage_count`/`error_count` of `rules` from the current state (call
            on the rules the state describes); ``errors()``, the current
            number of misclassified samples; ``base_errors``.
        """
        import types

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n = len(y)
        col = {f: i for i, f in enumerate(feature_names)}
        ops = {'<=': np.less_equal, '>': np.greater, '<': np.less, '>=': np.greater_equal}
        algo = self.algorithm_type
        is_gbdt = algo == 'Gradient Boosting Decision Trees'
        default = self._default_class_int()

        # Method to _lab.
        def _lab(v):
            """A rule label as the analyzers compare it with y."""
            try:
                return int(str(v).replace('Class', '').strip())
            except (ValueError, AttributeError):
                return v

        y = np.array([_lab(v) for v in y])

        # Method to _tree_id.
        def _tree_id(r):
            """Tree of a rule (a Decision Tree is one tree)."""
            if algo == 'Decision Tree':
                return '_all'
            return r.name.split('_')[0] if '_' in r.name else r.name

        # Method to _is_init.
        def _is_init(r):
            """GBDT init score: not a leaf (scored via `_gbdt_init_scores`)."""
            return is_gbdt and not r.parsed_conditions

        # Method to _mask.
        def _mask(r, rows):
            """Which of `rows` satisfy every condition of `r` (a NaN
            satisfies the side its split sends missing values to)."""
            m = np.ones(len(rows), dtype=bool)
            for cond in r.parsed_conditions:
                var, op, thr = cond
                values = X[rows, col[var]]
                hit = ops[op](values, thr)
                if getattr(cond, 'missing', False):
                    hit |= np.isnan(values)
                m &= hit
            return m

        if algo == 'Random Forest':
            k = next((len(r.class_distribution) for r in rules
                      if r.class_distribution is not None), self.num_classes or 0)
            rf_labels = np.asarray(self._rf_labels(k))
        elif is_gbdt:
            classes = list(self._gbdt_classes or [])
            k = len(classes)
            gcol = {c: i for i, c in enumerate(classes)}
            init = np.array([self._gbdt_init_scores.get(c, 0.0) for c in classes],
                            dtype=np.float64)
        else:
            k = 1

        # Method to _output.
        def _output(tree_rules, rows):
            """One tree's output on `rows`: (len(rows), k), zero = abstains."""
            out = np.zeros((len(rows), k), dtype=np.float64)
            label = np.full(len(rows), default, dtype=object) if algo == 'Decision Tree' else None
            for r in tree_rules:
                m = _mask(r, rows)
                if not m.any():
                    continue
                if algo == 'Decision Tree':
                    label[m] = _lab(r.class_)
                elif algo == 'Random Forest':
                    d = np.asarray(r.class_distribution if r.class_distribution is not None
                                   else [0.0] * k, dtype=np.float64)
                    if r.class_distribution is None:
                        # No distribution: a vote for the rule's class.
                        hit = np.flatnonzero(rf_labels == _lab(r.class_))
                        if len(hit):
                            d[hit[0]] = 1.0
                    s = d.sum()
                    out[m] = d / s if s > 0 else d
                else:
                    out[m, gcol[r.class_group]] = r.contribution or 0.0
            return label if algo == 'Decision Tree' else out

        # Method to _predict.
        def _predict(agg):
            """Labels from the aggregated outputs of every tree."""
            if algo == 'Decision Tree':
                return agg
            if algo == 'Random Forest':
                pred = rf_labels[np.argmax(agg, axis=1)].astype(object)
                pred[agg.sum(axis=1) == 0.0] = default
                return pred
            score = agg + init
            if self._gbdt_is_binary:
                return np.where(score[:, 1] >= 0, _lab(classes[1]),
                                _lab(classes[0])).astype(object)
            return np.array([_lab(classes[i]) for i in np.argmax(score, axis=1)],
                            dtype=object)

        all_rows = np.arange(n)
        trees: dict = defaultdict(list)
        for r in rules:
            if not _is_init(r):
                trees[_tree_id(r)].append(r)
        outs = {t: _output(rs, all_rows) for t, rs in trees.items()}
        if algo == 'Decision Tree':
            agg = outs.get('_all', np.full(n, default, dtype=object))
        else:
            agg = np.zeros((n, k), dtype=np.float64)
            for o in outs.values():
                agg += o
        state = {'agg': agg, 'pred': _predict(agg)}
        state['errors'] = int(np.sum(state['pred'] != y))
        base_errors = state['errors']
        limit = base_errors + max_error_increase

        # Method to accept.
        def accept(rule, old_rules, new_rules):
            """Commit the change if the error budget allows it."""
            covered = np.zeros(n, dtype=bool)
            for r in old_rules:
                covered[_mask(r, all_rows)] = True
            rows = np.flatnonzero(covered)
            if len(rows) == 0:
                return True
            tid = _tree_id(rule)
            new_out = _output(new_rules, rows)
            if algo == 'Decision Tree':
                new_agg = new_out
            else:
                new_agg = state['agg'][rows] - outs[tid][rows] + new_out
            new_pred = _predict(new_agg)
            delta = (int(np.sum(new_pred != y[rows]))
                     - int(np.sum(state['pred'][rows] != y[rows])))
            if state['errors'] + delta > limit:
                return False
            outs[tid][rows] = new_out
            state['agg'][rows] = new_agg
            state['pred'][rows] = new_pred
            state['errors'] += delta
            return True

        # Method to recount.
        def recount(target_rules):
            """Usage = samples matched; errors as the analyzers count them."""
            for r in target_rules:
                if _is_init(r):
                    continue
                m = _mask(r, all_rows)
                r.usage_count = int(m.sum())
                if is_gbdt:
                    r.error_count = int(np.sum(state['pred'][m] != y[m]))
                else:
                    r.error_count = int(np.sum(y[m] != _lab(r.class_)))

        return types.SimpleNamespace(
            accept=accept, recount=recount,
            errors=lambda: state['errors'], base_errors=base_errors)

    # Method to report promotions that swallowed still-live rules
    @staticmethod
    def _warn_promotions_that_overlap(rules, tree_of=None) -> int:
        """
        Report promoted rules whose region now contains other live rules.

        Promotion drops a rule's last condition so it covers the region of the
        removed sibling. When that sibling was a whole sub-branch with leaves
        still in the rule set, the promoted rule spreads over them and the rule
        set stops being a partition: two rules match the same sample. Accuracy
        does not reveal this (the iterative engine just takes the first match),
        but the CPN model becomes nondeterministic and the set is no longer a
        decision tree.

        Detection is a prefix test: rules from a tree are prefix-structured, so
        a rule lies inside a promoted rule's region exactly when its conditions
        extend the promoted rule's conditions. Sorting makes those a contiguous
        range, giving O(n log n) instead of a quadratic comparison.

        Only rules of the same tree can overlap in this sense: the trees of an
        ensemble all cover the whole space and often split on the same
        thresholds, so comparing across trees would flag every promotion whose
        conditions happen to prefix a path of another tree.

        Args:
            rules (List[Rule]): The rule set after promotion.
            tree_of (Callable[[Rule], str], optional): Tree a rule belongs to.
                Defaults to one tree for all rules.

        Returns:
            int: How many promoted rules swallowed at least one live rule.
        """
        import bisect

        promoted = [r for r in rules
                    if r.name.endswith('_promoted') or r.name.endswith('_grafted')]
        if not promoted:
            return 0

        if tree_of is None:
            def tree_of(_r):
                return '_all'
        keys_by_tree: dict = defaultdict(list)
        for r in rules:
            keys_by_tree[tree_of(r)].append(tuple(r.parsed_conditions))
        for keys in keys_by_tree.values():
            keys.sort()
        affected = 0
        for r in promoted:
            keys = keys_by_tree[tree_of(r)]
            p = tuple(r.parsed_conditions)
            i = bisect.bisect_right(keys, p)
            if i < len(keys) and keys[i][:len(p)] == p:
                affected += 1

        if affected:
            # Plain ASCII: this runs with stdout redirected to files whose
            # encoding is the system codepage (cp1252 on Windows).
            print(
                f'  WARNING: {affected} promoted rule(s) now overlap rules '
                'still in the set: the refined rule set is no longer a '
                'partition. Predictions become order-dependent and the CPN '
                'model becomes nondeterministic. Inspect with '
                'find_overlapping_rules().'
            )
        return affected
