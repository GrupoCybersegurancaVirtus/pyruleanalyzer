"""Compiled tree arrays and vectorized batch prediction.

Provides ``ArrayEngineMixin``: the rules of each tree rebuilt as flat
parallel arrays (the layout of scikit-learn's ``tree_``), one compiled
state per rule set, and the batch engines ``predict_batch`` and
``predict_batch_proba`` that traverse them with the C extension (or its
numpy fallback, see :mod:`pyruleanalyzer._accel`).

`RuleClassifier` is defined after its mixins, so the methods that name it import
it locally.
"""

import re
import numpy as np
from collections import defaultdict
from typing import Dict, Optional
from pyruleanalyzer._accel import (
    traverse_tree_batch as _accel_traverse,
    traverse_tree_batch_multi as _accel_traverse_multi,
)
from ._rules import _tree_sort_key, condition_missing


class ArrayEngineMixin:
    """Flat tree arrays and batch prediction."""

    # =========================================================================
    # ARRAY-BASED VECTORIZED PREDICTION (High-Performance Path)
    # =========================================================================
    #
    # The methods below convert rules into flat numpy arrays (mirroring sklearn's
    # internal tree representation) and use vectorized numpy operations for
    # batch prediction.  This eliminates:
    #   - Linear scans over Rule objects
    #   - Per-sample Python dict lookups
    #   - Per-call tree grouping with string splitting
    #   - Pure-Python voting loops
    #
    # Array layout per tree (parallel arrays, one element per node):
    #   feature_idx[i]    int32   feature column index (-2 = leaf)
    #   threshold[i]      float64 split threshold
    #   children_left[i]  int32   left child node index (-1 = none)
    #   children_right[i] int32   right child node index (-1 = none)
    #   missing_left[i]   uint8   1 if a NaN goes left at this split (only
    #                             when the rules record the routing)
    #   value[i]          varies  leaf payload:
    #       DT:   int32   class label
    #       RF:   float64 array of shape (n_classes,) — raw sample counts
    #       GBDT: float64 contribution (learning_rate * leaf_value)
    # =========================================================================

    # Method to rebuild the tree a rule set was read from
    @staticmethod
    def _tree_from_rules(rules: list, validate: bool = True) -> Dict[int, dict]:
        """
        Rebuild the binary tree that one tree's rules were read from.

        Every compiled engine (tree arrays, binary, C header, native function)
        routes a sample down this tree instead of testing the rules one by one,
        so it only computes what the rules say if the rules really are the
        leaves of one tree. Four ways they can fail to be, each of which would
        otherwise make a compiled engine answer differently from the iterative
        engine and from the CPN, without any error:

        - two rules want different splits at a node they share;
        - a rule stops at a node other rules go past (its region contains
          theirs), or goes past a node where another rule stops;
        - two rules stop at the same node (identical regions);
        - two rules disagree on where a shared split sends missing values.

        The second and third are overlaps: the iterative engine takes the first match,
        the tree keeps one of them, and the CPN becomes nondeterministic.

        Args:
            rules (List[Rule]): The rules of one tree.
            validate (bool): Raise on any of the conditions above.

        Returns:
            Dict[int, dict]: Nodes keyed by id; internal nodes carry 'f', 't',
            'l', 'r' and 'm' (whether NaN goes left: True/False, None when
            the conditions do not record it); a leaf has ``f == -2`` and
            'rule' set to the rule that ends there, or None for a region no
            rule covers.

        Raises:
            ValueError: If `validate` is set and the rules are not the leaves
                of one tree.
        """
        nodes: Dict[int, dict] = {
            0: {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'rule': None}
        }
        next_id = 1

        def _name(r):
            return getattr(r, 'name', '?')

        for rule in rules:
            curr = 0
            for cond in rule.parsed_conditions:
                var, op, val = cond
                goes_left = op in ('<=', '<')
                # Where the split sends NaN: True (left), False (right) or
                # None (unknown: the condition is a plain tuple).
                miss = condition_missing(cond)
                m = None if miss is None else (miss if goes_left else not miss)
                node = nodes[curr]
                if node['f'] == -2:
                    if validate and node['rule'] is not None:
                        raise ValueError(
                            'Rule set is not a tree and cannot be compiled: rule '
                            f'{_name(rule)!r} continues past the leaf of rule '
                            f'{_name(node["rule"])!r}, whose region therefore '
                            'contains it (overlapping rules).'
                        )
                    node.update({'f': var, 't': val, 'm': m,
                                 'l': next_id, 'r': next_id + 1})
                    for child in (next_id, next_id + 1):
                        nodes[child] = {'l': -1, 'r': -1, 'f': -2, 't': -2.0,
                                        'rule': None}
                    next_id += 2
                elif validate and (node['f'] != var
                                   or float(node['t']) != float(val)):
                    raise ValueError(
                        'Rule set is not prefix-closed and cannot be compiled '
                        'into tree arrays: rule '
                        f'{_name(rule)!r} requires the split '
                        f'({var} {op} {val}) at a node already split on '
                        f'({node["f"]} @ {node["t"]}). '
                        'Rules sharing a prefix of conditions must agree on the '
                        'split at every shared node. This usually means a custom '
                        'rule-removal function produced a rule set that is no '
                        'longer a tree.'
                    )
                elif node.get('m') != m:
                    if validate and node.get('m') is not None and m is not None:
                        raise ValueError(
                            'Rule set is not a tree and cannot be compiled: rule '
                            f'{_name(rule)!r} sends missing values (NaN) to the '
                            f'other side of the split ({var} {op} {val}) than '
                            'the rules sharing it.'
                        )
                    node['m'] = None   # unknown routing
                curr = node['l'] if goes_left else node['r']

            node = nodes[curr]
            if validate and node['f'] != -2:
                raise ValueError(
                    'Rule set is not a tree and cannot be compiled: rule '
                    f'{_name(rule)!r} stops at an internal node, so its region '
                    'contains the regions of other rules (overlapping rules).'
                )
            if validate and node['rule'] is not None:
                raise ValueError(
                    'Rule set is not a tree and cannot be compiled: rules '
                    f'{_name(node["rule"])!r} and {_name(rule)!r} cover the same '
                    'region.'
                )
            node['rule'] = rule
        return nodes

    # Method to execute _build_single_tree_arrays
    @staticmethod
    def _build_single_tree_arrays(
        rules: list,
        feature_name_to_idx: Dict[str, int],
        algorithm_type: str,
        n_classes: int = 0,
        default_class: int = 0,
        validate_structure: bool = True,
        class_labels: Optional[list] = None,
    ) -> dict:
        """
        Convert a list of rules (all from one tree) into flat numpy arrays.

        The reconstruction walks each rule from the root (see
        `_tree_from_rules`), so it requires the rules to be the leaves of one
        tree: rules sharing a prefix of conditions must agree on the split at
        every shared node, and no rule's region may contain another's. This
        holds for rules extracted from a tree and is preserved by the built-in
        refinement operations, but a custom removal function (see
        `set_custom_rule_removal`) can break it.  When that happens rules land
        in the wrong leaf, silently; `validate_structure` turns that silent
        corruption into an error.

        Args:
            rules (List[Rule]): Rules for a single tree.
            feature_name_to_idx (Dict[str, int]): Mapping feature_name -> column index.
            algorithm_type (str): 'Decision Tree', 'Random Forest', or
                'Gradient Boosting Decision Trees'.
            n_classes (int): Number of classes (needed for RF leaf distributions).
            default_class (int): Class assigned to uncovered regions (DT only).
            validate_structure (bool): Raise if the rules are not the leaves
                of one tree.
            class_labels (List[int], optional): The label of each RF
                distribution column (see `_rf_labels`). A rule without a
                distribution is one-hot encoded in its label's column.

        Returns:
            Dict with keys: 'feature_idx', 'threshold', 'children_left',
            'children_right', 'value', 'max_depth', 'is_hole', and
            'missing_left' (uint8, 1 where a split sends NaN left) when the
            rules record the missing-value routing.  Each array is
            indexed by node id; 'is_hole' marks leaves that no rule reached,
            i.e. uncovered regions of the feature space.

        Raises:
            ValueError: If `validate_structure` is set and the rules are not
                the leaves of one tree (conflicting split or overlap).
        """
        from .rule_classifier import RuleClassifier
        # 1. Reconstruct tree structure
        tree_dict = RuleClassifier._tree_from_rules(rules, validate_structure)

        for curr, node in tree_dict.items():
            rule = node.pop('rule')
            node['v'] = None
            if rule is None:
                continue

            # Set leaf value based on algorithm type
            if algorithm_type == 'Gradient Boosting Decision Trees':
                tree_dict[curr]['v'] = float(rule.contribution) if rule.contribution is not None else 0.0
            elif algorithm_type == 'Random Forest':
                if rule.class_distribution is not None:
                    tree_dict[curr]['v'] = list(rule.class_distribution)
                else:
                    # Fallback: one-hot encode the class label, in the
                    # column that label stands for (labels 10/20/30 are
                    # columns 0/1/2, not 10/20/30).
                    try:
                        label = int(str(rule.class_).replace('Class', '').strip())
                    except (ValueError, AttributeError):
                        label = None
                    if class_labels is not None and label in class_labels:
                        cls_idx = list(class_labels).index(label)
                    elif label is not None:
                        cls_idx = label   # unknown label: read as a column
                    else:
                        cls_idx = 0
                    one_hot = [0.0] * max(n_classes, cls_idx + 1)
                    one_hot[cls_idx] = 1.0
                    tree_dict[curr]['v'] = one_hot
            else:
                # Decision Tree -- the rule's own label, read exactly as the
                # iterative engine, the native function and the CPN read it.
                # (argmax of the distribution is its *index*, which is the
                # label only when the labels happen to be 0..K-1.)
                try:
                    tree_dict[curr]['v'] = int(str(rule.class_).replace('Class', '').strip())
                except (ValueError, AttributeError):
                    dist = getattr(rule, 'class_distribution', None)
                    m = re.search(r'(\d+)$', str(rule.class_))
                    if m:
                        tree_dict[curr]['v'] = int(m.group(1))
                    elif dist is not None:
                        tree_dict[curr]['v'] = int(np.argmax(dist))
                    else:
                        tree_dict[curr]['v'] = 0

        # 2. Convert to flat numpy arrays
        n_nodes = len(tree_dict)
        feature_idx = np.full(n_nodes, -2, dtype=np.int32)
        threshold = np.zeros(n_nodes, dtype=np.float64)
        children_left = np.full(n_nodes, -1, dtype=np.int32)
        children_right = np.full(n_nodes, -1, dtype=np.int32)

        if algorithm_type == 'Random Forest':
            value = np.zeros((n_nodes, n_classes), dtype=np.float64)
        elif algorithm_type == 'Gradient Boosting Decision Trees':
            value = np.zeros(n_nodes, dtype=np.float64)
        else:
            value = np.full(n_nodes, int(default_class), dtype=np.int32)

        # Leaves that no rule reached: uncovered regions left behind by
        # refinement.  They resolve to `default_class` (DT), to an abstention
        # (zero probability vector for RF, zero contribution for GBDT), and are
        # flagged here so callers can report how often they are hit.
        is_hole = np.zeros(n_nodes, dtype=bool)

        # Where each split sends a missing value (1 = left). Kept only when
        # every split knows it; leaves are 0 (both children are the leaf).
        missing_left = np.zeros(n_nodes, dtype=np.uint8)
        routing_known = True

        for node_id, node in tree_dict.items():
            feat = node['f']
            if feat == -2:
                # Leaf node — self-referencing children so traversal
                # samples that reached a leaf simply stay in place.
                # We also set feature_idx to 0 (any valid column) so that
                # the gather X[samples, feature_idx[node_ids]] always reads
                # a valid column.  The threshold is set to -inf so the
                # comparison (feat_val <= threshold) is always False, making
                # the traversal pick children_right which points back here.
                feature_idx[node_id] = 0  # valid column (not -2)
                threshold[node_id] = -np.inf
                children_left[node_id] = node_id   # self-loop
                children_right[node_id] = node_id   # self-loop
                is_hole[node_id] = node['v'] is None
                if algorithm_type == 'Random Forest':
                    v = node['v']
                    if v is not None:
                        for ci in range(min(len(v), n_classes)):
                            value[node_id, ci] = v[ci]
                elif algorithm_type == 'Gradient Boosting Decision Trees':
                    value[node_id] = float(node['v']) if node['v'] is not None else 0.0
                else:
                    value[node_id] = (
                        int(node['v']) if node['v'] is not None
                        else int(default_class)
                    )
            else:
                feature_idx[node_id] = feature_name_to_idx.get(feat, -2)
                threshold[node_id] = float(node['t'])
                children_left[node_id] = node['l']
                children_right[node_id] = node['r']
                if node.get('m') is None:
                    routing_known = False
                else:
                    missing_left[node_id] = 1 if node['m'] else 0

        # Compute tree depth by BFS from root
        max_depth = 0
        stack = [(0, 0)]  # (node_id, depth)
        while stack:
            nid, d = stack.pop()
            if nid < 0 or nid >= n_nodes:
                continue
            node = tree_dict.get(nid)
            if node is None:
                continue
            if node['f'] == -2:
                # Leaf
                if d > max_depth:
                    max_depth = d
            else:
                stack.append((node['l'], d + 1))
                stack.append((node['r'], d + 1))

        out = {
            'feature_idx': feature_idx,
            'threshold': threshold,
            'children_left': children_left,
            'children_right': children_right,
            'value': value,
            'max_depth': max_depth,
            'is_hole': is_hole,
        }
        if routing_known:
            out['missing_left'] = missing_left
        return out

    # Method to pick the rule set a prediction should use
    def _rules_for(self, use_final: bool) -> list:
        """The rule set a prediction with `use_final` must use.

        Args:
            use_final (bool): True for the refined model, False for the one
                extracted from the trained estimator.

        Returns:
            List[Rule]: `final_rules` when requested and present, otherwise
            `initial_rules` (before refinement they are the same model).
        """
        if use_final and self.final_rules:
            return self.final_rules
        return self.initial_rules

    # Method to execute compile_tree_arrays
    def compile_tree_arrays(self, rules: Optional[list] = None, feature_names: Optional[list] = None) -> None:
        """
        Compile rules into numpy tree arrays for vectorized prediction.

        After calling this method, `predict_batch(X)` becomes available.
        This is called automatically by `update_native_model` but can also be
        called explicitly after rule removal to refresh the arrays.

        The compiled set becomes the *current* one, used by `predict_batch`
        and the binary/C/Arduino exports when no rule set is named. Each set
        compiled is also kept, so `predict_batch(use_final=...)` can switch
        between the initial and the refined model without recompiling.
        Compile again after editing a rule list in place.

        Args:
            rules (List[Rule], optional): Rules to compile. Defaults to
                final_rules if available, else initial_rules.
            feature_names (List[str], optional): Ordered feature names matching
                the columns of X that will be passed to predict_batch.
                If None, they are inferred from the rules.
        """
        if rules is None:
            rules = self.final_rules if self.final_rules else self.initial_rules

        state = self._compile_array_state(rules, feature_names)
        self._set_array_state(state)
        states = [st for st in getattr(self, '_array_states', [])
                  if st['rules'] is not rules]
        self._array_states = (states + [state])[-4:]

    # Method to compile one rule set into tree arrays
    def _compile_array_state(self, rules: list, feature_names: Optional[list] = None) -> dict:
        """
        Compile one rule set into tree arrays without making it current.

        Args:
            rules (List[Rule]): Rules to compile.
            feature_names (List[str], optional): Column order of X. If None,
                inferred from the rules (sorted).

        Returns:
            dict: 'rules' (the list compiled, kept by identity), 'n_rules',
            'feature_names', 'trees', 'class_groups', 'is_init'.
        """
        # --- Infer feature names from rules if not provided ---
        if feature_names is None:
            feat_set: set = set()
            for rule in rules:
                for var, _, _ in rule.parsed_conditions:
                    feat_set.add(var)
            feature_names = sorted(feat_set)

        feature_name_to_idx = {name: i for i, name in enumerate(feature_names)}

        n_classes = self.num_classes
        class_groups = None
        is_init = None

        # --- Group rules by tree ---
        if self.algorithm_type == 'Decision Tree':
            # Single tree — all rules belong to the same tree
            tree_arrays = [
                self._build_single_tree_arrays(
                    rules, feature_name_to_idx, self.algorithm_type, n_classes,
                    default_class=self._default_class_int(),
                )
            ]

        elif self.algorithm_type == 'Random Forest':
            tree_rules_map: Dict[str, list] = defaultdict(list)
            for rule in rules:
                tid = rule.name.split('_')[0] if '_' in rule.name else 'Tree0'
                tree_rules_map[tid].append(rule)

            tree_arrays = []
            for tid in sorted(tree_rules_map.keys(), key=_tree_sort_key):
                arr = self._build_single_tree_arrays(
                    tree_rules_map[tid], feature_name_to_idx,
                    self.algorithm_type, n_classes,
                    default_class=self._default_class_int(),
                    class_labels=self._rf_labels(n_classes),
                )
                tree_arrays.append(arr)

        elif self.algorithm_type == 'Gradient Boosting Decision Trees':
            tree_rules_map = defaultdict(list)
            for rule in rules:
                tid = rule.name.split('_')[0] if '_' in rule.name else rule.name
                tree_rules_map[tid].append(rule)

            tree_arrays = []
            class_groups = []
            is_init = []

            for tid in sorted(tree_rules_map.keys(), key=_tree_sort_key):
                trules = tree_rules_map[tid]
                if not trules:
                    continue
                # Check if this is an init rule (single rule, no conditions)
                if (len(trules) == 1 and not trules[0].parsed_conditions
                        and trules[0].contribution is not None):
                    is_init.append(True)
                    class_groups.append(trules[0].class_group)
                    # Store init score directly — no tree structure needed
                    tree_arrays.append({
                        'init_score': float(trules[0].contribution),
                    })
                else:
                    is_init.append(False)
                    class_groups.append(trules[0].class_group)
                    arr = self._build_single_tree_arrays(
                        trules, feature_name_to_idx,
                        self.algorithm_type, n_classes,
                        default_class=self._default_class_int(),
                    )
                    tree_arrays.append(arr)
        else:
            raise ValueError(f'Unsupported algorithm_type: {self.algorithm_type}')

        return {
            'rules': rules,
            'n_rules': len(rules),
            'feature_names': list(feature_names),
            'trees': tree_arrays,
            'class_groups': class_groups,
            'is_init': is_init,
        }

    # Method to make a compiled rule set the current one
    def _set_array_state(self, state: dict) -> None:
        """Install `state` (from `_compile_array_state`) as the current arrays.

        Args:
            state (dict): A compiled rule set.
        """
        self._array_feature_names = list(state['feature_names'])
        self._tree_arrays = state['trees']
        self._tree_class_groups = state['class_groups']
        self._tree_is_init = state['is_init']
        self._arrays_source = state['rules']
        self._arrays_source_len = state['n_rules']
        self._arrays_compiled = True

    # Method to get the compiled arrays of the current rule set
    def _current_array_state(self) -> dict:
        """The current arrays as a state dict (see `_compile_array_state`).

        Returns:
            dict: The arrays `predict_batch` uses when no rule set is named.
        """
        return {
            'rules': getattr(self, '_arrays_source', None),
            'n_rules': getattr(self, '_arrays_source_len', None),
            'feature_names': list(self._array_feature_names),
            'trees': self._tree_arrays,
            'class_groups': self._tree_class_groups,
            'is_init': self._tree_is_init,
        }

    # Method to get the compiled arrays of the rule set a prediction needs
    def _array_state_for(self, use_final: Optional[bool],
                         feature_names: Optional[list] = None) -> dict:
        """The compiled arrays of the rule set selected by `use_final`.

        Args:
            use_final (bool | None): None for the current arrays (whatever
                `compile_tree_arrays` compiled last); True/False for the
                refined/initial rule set, compiled on first use.
            feature_names (List[str], optional): Column order of X, used only
                when the set must be compiled and nothing was compiled before.

        Returns:
            dict: A compiled rule set (see `_compile_array_state`).

        Raises:
            RuntimeError: If `use_final` is None and nothing was compiled.
        """
        compiled = getattr(self, '_arrays_compiled', False)
        rules = None
        if use_final is not None and (self.initial_rules or self.final_rules):
            rules = self._rules_for(use_final)
        if rules is None:
            # No rule set named, or no rules at all (a model loaded with
            # load_binary holds only its arrays): use the current arrays.
            if not compiled:
                raise RuntimeError(
                    'Tree arrays not compiled. Call compile_tree_arrays() first.'
                )
            return self._current_array_state()
        names = (list(self._array_feature_names)
                 if getattr(self, '_array_feature_names', None) else feature_names)
        return self._array_state_for_rules(rules, names)

    # Method to get (compiling if needed) the arrays of one rule list
    def _array_state_for_rules(self, rules: list,
                               feature_names: Optional[list] = None) -> dict:
        """The compiled arrays of `rules`, compiled and kept on first use.

        Args:
            rules (List[Rule]): The rule list (matched by identity).
            feature_names (List[str], optional): Column order when compiling.

        Returns:
            dict: A compiled rule set (see `_compile_array_state`).
        """
        if (getattr(self, '_arrays_compiled', False)
                and getattr(self, '_arrays_source', None) is rules
                and getattr(self, '_arrays_source_len', None) == len(rules)):
            return self._current_array_state()
        states = getattr(self, '_array_states', [])
        for st in states:
            if st['rules'] is rules and st['n_rules'] == len(rules):
                return st

        state = self._compile_array_state(rules, feature_names)
        states = [st for st in states if st['rules'] is not rules]
        self._array_states = (states + [state])[-4:]
        return state

    # Method to execute _traverse_tree_batch
    @staticmethod
    def _traverse_tree_batch(
        X: np.ndarray,
        feature_idx: np.ndarray,
        threshold: np.ndarray,
        children_left: np.ndarray,
        children_right: np.ndarray,
        max_depth: int = 50,
    ) -> np.ndarray:
        """
        Traverse a single tree for all samples simultaneously using numpy indexing.

        This is the core vectorized operation:  O(depth) steps, each step processes
        ALL N samples in parallel via numpy fancy indexing -- no masking, no branching.

        Leaf nodes are set up as self-loops (children point to themselves) with
        threshold = -inf, so samples that have already reached a leaf simply stay
        in place each iteration.

        Args:
            X (np.ndarray): Input data, shape (n_samples, n_features).
            feature_idx (np.ndarray): Per-node feature index, shape (n_nodes,).
                Leaf nodes have a valid column index (not -2) since we read
                from X unconditionally.
            threshold (np.ndarray): Per-node threshold, shape (n_nodes,).
                Leaf nodes have -inf so (val <= -inf) is always False.
            children_left (np.ndarray): Per-node left child, shape (n_nodes,).
                Leaf nodes point to themselves.
            children_right (np.ndarray): Per-node right child, shape (n_nodes,).
                Leaf nodes point to themselves.
            max_depth (int): Maximum depth of the tree. The traversal loop runs
                exactly max_depth iterations.

        Returns:
            np.ndarray: Array of leaf node indices, shape (n_samples,).
        """
        n_samples = X.shape[0]
        node_ids = np.zeros(n_samples, dtype=np.int32)  # Start at root (node 0)
        sample_idx = np.arange(n_samples)  # Precompute once

        for _ in range(max_depth):
            # Gather the split feature column and threshold for every sample
            feat = feature_idx[node_ids]          # shape (n_samples,)

            # Gather the feature value from X using fancy indexing
            # For leaf nodes: feat is a valid column (0), thresh is -inf,
            # so go_left is False, and children_right[leaf] = leaf (no-op).
            go_left = X[sample_idx, feat] <= threshold[node_ids]

            node_ids = np.where(go_left, children_left[node_ids],
                                children_right[node_ids])

        return node_ids

    # Method to execute predict_batch
    def predict_batch(
        self,
        X: np.ndarray,
        feature_names: Optional[list] = None,
        use_final: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Vectorized batch prediction over a numpy array.

        This is the high-performance prediction method.  It uses the pre-compiled
        tree arrays (from compile_tree_arrays) and numpy vectorized traversal.

        Args:
            X (np.ndarray): Input data, shape (n_samples, n_features).
                Columns must be ordered to match the feature_names used during
                compile_tree_arrays (or the feature_names argument here).
            feature_names (list, optional): Feature names corresponding to columns
                of X.  If provided and different from compiled order, X columns
                are reordered accordingly.
            use_final (bool, optional): True predicts with final_rules (the
                refined model; initial_rules if there are none), False with
                initial_rules, each compiled on first use and kept. None (the
                default) uses whatever `compile_tree_arrays` compiled last.

        Returns:
            np.ndarray: Predicted class labels, shape (n_samples,), dtype int.
        """
        st = self._array_state_for(use_final, feature_names)
        trees, groups, is_init = st['trees'], st['class_groups'], st['is_init']

        # Reorder columns if feature_names differ from compiled order
        if feature_names is not None and list(feature_names) != st['feature_names']:
            col_order = [list(feature_names).index(f) for f in st['feature_names']]
            X = X[:, col_order]

        self._check_missing_values(X, self._trees_route_missing(trees))
        X = self._quantize_input(X)
        n_samples = X.shape[0]

        # --- Decision Tree ---
        if self.algorithm_type == 'Decision Tree':
            tree = trees[0]
            leaf_ids = _accel_traverse(
                X, tree['feature_idx'], tree['threshold'],
                tree['children_left'], tree['children_right'],
                tree['max_depth'], tree.get('missing_left'),
            )
            # Uncovered regions already carry `default_class` in `value`; count
            # them so the loss of coverage is reported, not silently absorbed.
            holes = tree.get('is_hole')
            if holes is not None:
                self.fallback_activations += int(holes[leaf_ids].sum())
            predictions = tree['value'][leaf_ids]
            return predictions

        # --- Random Forest (Soft Voting) ---
        elif self.algorithm_type == 'Random Forest':
            n_classes = self.num_classes
            n_trees = len(trees)

            # Build list of tree tuples for multi-tree C traversal
            tree_tuples = [
                (t['feature_idx'], t['threshold'],
                 t['children_left'], t['children_right'], t['max_depth'],
                 t.get('missing_left'))
                for t in trees
            ]
            # all_leaf_ids: shape (n_trees, n_samples), int32
            all_leaf_ids = _accel_traverse_multi(X, tree_tuples)

            # Accumulate probability contributions from each tree
            prob_sum = np.zeros((n_samples, n_classes), dtype=np.float64)
            for t_idx in range(n_trees):
                leaf_ids = all_leaf_ids[t_idx]
                # Get raw counts at leaves
                raw_counts = trees[t_idx]['value'][leaf_ids]
                # Normalize to probabilities per sample
                totals = raw_counts.sum(axis=1, keepdims=True)
                totals = np.where(totals == 0, 1.0, totals)
                probas = raw_counts / totals
                prob_sum += probas

            # Average and argmax
            avg_proba = prob_sum / n_trees
            # argmax is a column of the distribution; answer its label
            labels = np.asarray(self._rf_labels(n_classes), dtype=np.int32)
            predictions = labels[np.argmax(avg_proba, axis=1)]

            # Every tree abstained (refinement left this sample uncovered in
            # all of them): the argmax would silently return class 0 with no
            # evidence behind it.  Resolve to `default_class` and count it.
            abstained = prob_sum.sum(axis=1) == 0.0
            n_abstained = int(abstained.sum())
            if n_abstained:
                predictions[abstained] = np.int32(self._default_class_int())
                self.fallback_activations += n_abstained
            return predictions

        # --- GBDT (Additive Scoring) ---
        elif self.algorithm_type == 'Gradient Boosting Decision Trees':
            classes = self._gbdt_classes or []
            is_binary = self._gbdt_is_binary
            assert groups is not None
            assert is_init is not None

            # Separate init trees from real trees and batch-traverse real trees
            real_tree_indices = []
            tree_tuples = []
            for i, tree in enumerate(trees):
                if not is_init[i]:
                    real_tree_indices.append(i)
                    tree_tuples.append((
                        tree['feature_idx'], tree['threshold'],
                        tree['children_left'], tree['children_right'],
                        tree['max_depth'], tree.get('missing_left'),
                    ))

            # Batch traverse all real trees at once
            if tree_tuples:
                all_leaf_ids = _accel_traverse_multi(X, tree_tuples)
            else:
                all_leaf_ids = np.empty((0, n_samples), dtype=np.int32)

            # A stage whose region is uncovered contributes 0.0 (abstention).
            # Track, per sample, whether any stage abstained.
            touched_hole = np.zeros(n_samples, dtype=bool)

            if is_binary and len(classes) >= 2:
                # Binary: accumulate score for positive class
                pos_class = classes[1]
                scores = np.zeros(n_samples, dtype=np.float64)

                # Add init scores
                for i, tree in enumerate(trees):
                    if is_init[i]:
                        if groups[i] == pos_class:
                            scores += tree['init_score']

                # Add real tree contributions
                for batch_idx, orig_idx in enumerate(real_tree_indices):
                    if groups[orig_idx] != pos_class:
                        continue
                    leaf_ids = all_leaf_ids[batch_idx]
                    scores += trees[orig_idx]['value'][leaf_ids]
                    holes = trees[orig_idx].get('is_hole')
                    if holes is not None:
                        touched_hole |= holes[leaf_ids]

                # Decide on the raw score, as scikit-learn does
                # (raw_predictions >= 0); see update_native_model.
                predictions = np.where(
                    scores >= 0.0, int(classes[1]), int(classes[0])
                ).astype(np.int32)
                self.fallback_activations += int(touched_hole.sum())
                return predictions

            else:
                # Multiclass: accumulate scores per class
                class_to_col = {cl: idx for idx, cl in enumerate(classes)}
                n_cls = len(classes)
                scores = np.zeros((n_samples, n_cls), dtype=np.float64)

                # Add init scores
                for i, tree in enumerate(trees):
                    if is_init[i]:
                        cg = groups[i]
                        col = class_to_col.get(cg)
                        if col is not None:
                            scores[:, col] += tree['init_score']

                # Add real tree contributions
                for batch_idx, orig_idx in enumerate(real_tree_indices):
                    cg = groups[orig_idx]
                    col = class_to_col.get(cg)
                    if col is None:
                        continue
                    leaf_ids = all_leaf_ids[batch_idx]
                    scores[:, col] += trees[orig_idx]['value'][leaf_ids]
                    holes = trees[orig_idx].get('is_hole')
                    if holes is not None:
                        touched_hole |= holes[leaf_ids]

                # Argmax -> class label
                best_col = np.argmax(scores, axis=1)
                class_arr = np.array([int(cl) for cl in classes], dtype=np.int32)
                predictions = class_arr[best_col]
                self.fallback_activations += int(touched_hole.sum())
                return predictions

        raise ValueError(f'Unsupported algorithm_type: {self.algorithm_type}')

    # Method to execute predict_batch_proba
    def predict_batch_proba(
        self,
        X: np.ndarray,
        feature_names: Optional[list] = None,
        use_final: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Vectorized batch probability prediction.

        Args:
            X (np.ndarray): Input data, shape (n_samples, n_features).
            feature_names (list, optional): Feature names for column reordering.
            use_final (bool, optional): Rule set to use, as in `predict_batch`.

        Returns:
            np.ndarray: Predicted probabilities, shape (n_samples, n_classes).
                For DT, this is a one-hot array.
                For RF, this is the averaged probability across trees.
                For GBDT binary, shape (n_samples, 2) with sigmoid probabilities.
                For GBDT multiclass, shape (n_samples, n_classes) with softmax probabilities.
        """
        st = self._array_state_for(use_final, feature_names)
        trees, groups, is_init = st['trees'], st['class_groups'], st['is_init']

        if feature_names is not None and list(feature_names) != st['feature_names']:
            col_order = [list(feature_names).index(f) for f in st['feature_names']]
            X = X[:, col_order]

        # Same input handling as predict_batch, so the probabilities describe
        # the leaves predict_batch routes the samples to.
        self._check_missing_values(X, self._trees_route_missing(trees))
        X = self._quantize_input(X)
        n_samples = X.shape[0]

        # --- Decision Tree ---
        if self.algorithm_type == 'Decision Tree':
            tree = trees[0]
            leaf_ids = _accel_traverse(
                X, tree['feature_idx'], tree['threshold'],
                tree['children_left'], tree['children_right'],
                tree['max_depth'], tree.get('missing_left'),
            )
            preds = tree['value'][leaf_ids]
            # One-hot encode, in the column of each predicted *label*
            # (classes_ order): labels 1..3 are columns 0..2, and labels
            # 10/20/30 are not column indices at all.
            n_classes = self.num_classes
            proba = np.zeros((n_samples, n_classes), dtype=np.float64)
            col_of = {lab: j for j, lab in enumerate(self._rf_labels(n_classes))}
            uniq, inverse = np.unique(preds, return_inverse=True)
            cols = np.array([col_of.get(int(u), -1) for u in uniq],
                            dtype=np.int64)[inverse.reshape(-1)]
            rows = np.flatnonzero(cols >= 0)
            proba[rows, cols[rows]] = 1.0
            return proba

        # --- Random Forest ---
        elif self.algorithm_type == 'Random Forest':
            n_classes = self.num_classes
            prob_sum = np.zeros((n_samples, n_classes), dtype=np.float64)
            n_trees = len(trees)

            # Batch traverse all trees at once
            tree_tuples = [
                (t['feature_idx'], t['threshold'],
                 t['children_left'], t['children_right'], t['max_depth'],
                 t.get('missing_left'))
                for t in trees
            ]
            all_leaf_ids = _accel_traverse_multi(X, tree_tuples)

            for t_idx in range(n_trees):
                leaf_ids = all_leaf_ids[t_idx]
                raw_counts = trees[t_idx]['value'][leaf_ids]
                totals = raw_counts.sum(axis=1, keepdims=True)
                totals = np.where(totals == 0, 1.0, totals)
                probas = raw_counts / totals
                prob_sum += probas
            return prob_sum / n_trees

        # --- GBDT ---
        elif self.algorithm_type == 'Gradient Boosting Decision Trees':
            classes = self._gbdt_classes or []
            is_binary = self._gbdt_is_binary
            assert groups is not None
            assert is_init is not None
            class_to_col = {cl: idx for idx, cl in enumerate(classes)}
            n_cls = len(classes)
            scores = np.zeros((n_samples, n_cls), dtype=np.float64)

            # Separate init vs real trees, batch traverse real ones
            real_tree_indices = []
            tree_tuples_gbdt = []
            for i, tree in enumerate(trees):
                if is_init[i]:
                    cg = groups[i]
                    col = class_to_col.get(cg)
                    if col is not None:
                        scores[:, col] += tree['init_score']
                else:
                    real_tree_indices.append(i)
                    tree_tuples_gbdt.append((
                        tree['feature_idx'], tree['threshold'],
                        tree['children_left'], tree['children_right'],
                        tree['max_depth'], tree.get('missing_left'),
                    ))

            if tree_tuples_gbdt:
                all_leaf_ids = _accel_traverse_multi(X, tree_tuples_gbdt)
                for batch_idx, orig_idx in enumerate(real_tree_indices):
                    cg = groups[orig_idx]
                    col = class_to_col.get(cg)
                    if col is None:
                        continue
                    leaf_ids = all_leaf_ids[batch_idx]
                    scores[:, col] += trees[orig_idx]['value'][leaf_ids]

            if is_binary and n_cls == 2:
                # Sigmoid on positive class score
                prob_pos = 1.0 / (1.0 + np.exp(-scores[:, 1]))
                proba = np.column_stack([1.0 - prob_pos, prob_pos])
                return proba
            else:
                # Softmax
                exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
                proba = exp_scores / exp_scores.sum(axis=1, keepdims=True)
                return proba

        raise ValueError(f'Unsupported algorithm_type: {self.algorithm_type}')
