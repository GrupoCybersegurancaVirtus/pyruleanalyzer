"""Python code generation from the rules.

Provides ``NativePythonMixin``: rebuilding each tree from its rules as
nested ``if``/``else`` Python source, compiled in memory
(``update_native_model``, the fast single-sample path) or written as a
standalone module (``export_to_native_python``). Both use the same source
builders (``_python_model_source``); the in-memory function returns the
3-tuple of ``classify`` while the standalone ``predict`` returns the label.
"""

from collections import Counter
from ._pycodegen import (function_source, group_by_tree, label_value,
                         set_leaf_values)


# Function to tell the init-score rule of a GBDT "tree"
def _is_gbdt_init(rules):
    """Whether `rules` is the single unconditional init-score rule.

    Args:
        rules (List[Rule]): The rules of one tree.

    Returns:
        bool: True for the init-score pseudo-tree.
    """
    return (len(rules) == 1 and not rules[0].parsed_conditions
            and rules[0].contribution is not None)


# Function to compute the value a forest leaf returns
def _rf_leaf_value(rule):
    """The normalized class distribution of a forest leaf, or its label.

    Raw counts become probabilities at compile time for a fast runtime.

    Args:
        rule (Rule | None): The leaf's rule (None: uncovered region, the tree
            abstains).

    Returns:
        List[float] | int | Any | None: The distribution, the label when the
        rule has none, or None.
    """
    if rule is None:
        return None
    if rule.class_distribution is not None:
        raw = rule.class_distribution
        total_c = sum(raw)
        return [c / total_c for c in raw] if total_c > 0 else list(raw)
    return label_value(rule.class_)


class NativePythonMixin:
    """Native Python code generation (in-memory function and standalone export)."""

    # Helper method to compile a list of rules into a fast lookup function
    def _compile_tree_lookup(self, rules_to_compile):
        """
        Compiles a list of rules into a function that returns the index of the matched rule.
        Used for optimizing Random Forest analysis.
        """
        # Build tree structure. Raises if the rules are not the leaves of one
        # tree: a lookup through a wrong tree would credit the usage of one
        # rule to another, and the low-usage refinement acts on those counts.
        tree_dict = self._tree_from_rules(rules_to_compile)
        # Each leaf returns the index of its rule in `rules_to_compile`
        # (-1: uncovered); the order of the rules is never changed.
        index_of = {id(r): i for i, r in enumerate(rules_to_compile)}
        set_leaf_values(tree_dict, lambda rule: index_of[id(rule)]
                        if rule is not None else -1)
        func_code = function_source('lookup', tree_dict, str)

        context = {}
        try:
            exec(func_code, {}, context)
            return context['lookup']
        except Exception:
            return None

    # Method to rebuild one tree for the native function
    def _native_tree(self, rules, stacklevel=3):
        """Rebuild one tree for `update_native_model`, or None if impossible.

        The native function is only an accelerator: when the rules are not the
        leaves of a tree (see `_tree_from_rules`) it is not built, and
        `classify` falls back to the iterative engine instead of routing
        samples through a tree that does not compute the rules.

        Args:
            rules (List[Rule]): The rules of one tree.
            stacklevel (int): Passed to `warnings.warn`, so that the warning
                points at the code that called `update_native_model`.

        Returns:
            Dict[int, dict] | None: The nodes, or None when the rules are not
            a tree (a warning says why).
        """
        try:
            return self._tree_from_rules(rules)
        except ValueError as exc:
            import warnings
            warnings.warn(f'Native model not compiled: {exc}', RuntimeWarning,
                          stacklevel=stacklevel)
            self.native_fn = None
            return None

    # Method to rebuild one tree with the value each of its leaves returns
    def _codegen_tree(self, rules, export, payload):
        """Rebuild the tree of `rules` and store its leaf values.

        Args:
            rules (List[Rule]): The rules of one tree.
            export (bool): True for the standalone export, which raises when
                the rules are not a tree; False for the in-memory function,
                which warns and gives up instead (see `_native_tree`).
            payload (Callable[[Rule | None], Any]): See `set_leaf_values`.

        Returns:
            Dict[int, dict] | None: The nodes, or None (in memory only) when
            the rules are not a tree.
        """
        if export:
            tree = self._tree_from_rules(rules)
        else:
            # Frames: update_native_model -> _python_model_source ->
            # _*_python_source -> here -> _native_tree: level 6 is the caller
            # of update_native_model, as when it called _native_tree itself.
            tree = self._native_tree(rules, stacklevel=6)
            if tree is None:
                return None
        return set_leaf_values(tree, payload)

    # Method to generate the Python source of the whole model
    def _python_model_source(self, rules, write, export):
        """Write the Python source of the model made of `rules`.

        The in-memory function (``native_predict``) returns the 3-tuple of
        `classify`; the standalone one (``predict``) returns the label alone.

        Args:
            rules (List[Rule]): The rules to compile.
            write (Callable[[str], Any]): Receives the source, piece by piece
                (a file's write, or a list's append).
            export (bool): True for the standalone export, False for
                `update_native_model`.

        Returns:
            bool: False when the in-memory function cannot be built (the
            rules are not trees; a warning says why), else True.
        """
        if self.algorithm_type == 'Gradient Boosting Decision Trees':
            return self._gbdt_python_source(rules, write, export)
        if self.algorithm_type == 'Random Forest':
            return self._rf_python_source(rules, write, export)
        return self._dt_python_source(rules, write, export)

    # Method to generate the Python source of a GBDT (additive scoring)
    def _gbdt_python_source(self, rules, write, export):
        """GBDT source: one function per tree, summed into per-class scores.

        Args:
            rules (List[Rule]): The rules to compile.
            write (Callable[[str], Any]): See `_python_model_source`.
            export (bool): See `_python_model_source`.

        Returns:
            bool: See `_python_model_source`.
        """
        tree_rules_map = group_by_tree(rules)

        # The init rules hold the initial score of their class
        init_scores = {}
        for tree_rules in tree_rules_map.values():
            if _is_gbdt_init(tree_rules):
                init_scores[tree_rules[0].class_group] = tree_rules[0].contribution

        write('import math\n\n' if export else 'import math\n')

        # Identifiers come from positions, never from rule names or class
        # labels, which may hold anything (a '-' in "-1" already breaks
        # them, and a crafted name would be executed).
        fn_of = {tid: f'predict_{i}' for i, tid in enumerate(tree_rules_map)}

        for tid, tree_rules in tree_rules_map.items():
            if _is_gbdt_init(tree_rules):
                continue
            # Uncovered region: the stage abstains (adds 0.0).
            tree = self._codegen_tree(
                tree_rules, export,
                lambda rule: (rule.contribution if rule is not None
                              and rule.contribution is not None else 0.0))
            if tree is None:
                return False
            # GradientBoostingClassifier refuses NaN: no routing emitted.
            write(function_source(fn_of[tid], tree,
                                  lambda v: repr(float(v)), nan=False))
            write('\n')

        # The boosting stages (tree id, class group), init rules left out
        stages = [(tid, tree_rules[0].class_group)
                  for tid, tree_rules in tree_rules_map.items()
                  if not (len(tree_rules) == 1 and not tree_rules[0].parsed_conditions)]

        is_binary = self._gbdt_is_binary
        classes = self._gbdt_classes or []

        write(f"def {'predict' if export else 'native_predict'}(sample):\n")

        if is_binary and len(classes) >= 2:
            pos_class = classes[1]
            write(f'    score = {repr(init_scores.get(pos_class, 0.0))}\n')
            for tid, class_group in stages:
                if class_group == pos_class:
                    write(f'    score += {fn_of[tid]}(sample)\n')
            # Decide on the raw score, as scikit-learn does
            # (raw_predictions >= 0). sigmoid(score) >= 0.5 is not the same
            # test in floating point: sigmoid rounds to exactly 0.5 for
            # negative scores of magnitude below ~1e-16 and would pick the
            # positive class.
            if export:
                write(f'    return {int(classes[1])} if score >= 0.0 '
                      f'else {int(classes[0])}\n')
            else:
                write('    if score >= 0.0:\n')
                write(f'        return {int(classes[1])}, None, None\n')
                write('    else:\n')
                write(f'        return {int(classes[0])}, None, None\n')
        else:
            # Multiclass: one score per class, named by its position
            col = {cl: i for i, cl in enumerate(classes)}
            for class_label in classes:
                write(f'    score_{col[class_label]} = '
                      f'{repr(init_scores.get(class_label, 0.0))}\n')
            for tid, class_group in stages:
                if class_group in col:
                    write(f'    score_{col[class_group]} += {fn_of[tid]}(sample)\n')

            # Argmax
            score_vars = [f'score_{col[cl]}' for cl in classes]
            write(f'    scores = [{", ".join(score_vars)}]\n')
            write(f'    classes = {[int(cl) for cl in classes]}\n')
            if export:
                write('    return classes[scores.index(max(scores))]\n')
            else:
                write('    best_idx = scores.index(max(scores))\n')
                write('    return classes[best_idx], None, None\n')
        return True

    # Method to generate the Python source of a Random Forest (soft voting)
    def _rf_python_source(self, rules, write, export):
        """Random Forest source: one function per tree and a soft vote.

        Each tree returns the class distribution of its leaf (None where it
        abstains); the vote averages them and answers the label of the
        argmax column, or falls back to a hard vote without distributions.

        Args:
            rules (List[Rule]): The rules to compile.
            write (Callable[[str], Any]): See `_python_model_source`.
            export (bool): See `_python_model_source`.

        Returns:
            bool: See `_python_model_source`.
        """
        tree_rules_map = group_by_tree(rules, default_tid='Tree0')

        # Determine number of classes from distributions
        n_classes = 0
        for rule in rules:
            if rule.class_distribution is not None:
                n_classes = len(rule.class_distribution)
                break

        tree_func_names = []
        for t_pos, tree_rules in enumerate(tree_rules_map.values()):
            # Named by position: rule names are not valid identifiers
            func_name = f'predict_{t_pos}'
            tree_func_names.append(func_name)
            tree = self._codegen_tree(tree_rules, export, _rf_leaf_value)
            if tree is None:
                return False
            write(function_source(func_name, tree, repr))
            write('\n')

        # Hard vote over the tree results, at indentation `tab`
        def hard_vote(tab):
            if export:
                write(f'{tab}from collections import Counter\n')
                write(f'{tab}return Counter(tree_results).most_common(1)[0][0]\n')
            else:
                write(f'{tab}pred = Counter(tree_results).most_common(1)[0][0]\n')
                write(f'{tab}return pred, tree_results, None\n')

        write(f"def {'predict' if export else 'native_predict'}(sample):\n")
        write('    tree_results = []\n')
        for func in tree_func_names:
            write(f'    v = {func}(sample)\n')
            write('    if v is not None: tree_results.append(v)\n')
        default = repr(label_value(self.default_class))
        write(f"    if not tree_results: return {default}{'' if export else ', [], None'}\n")

        if n_classes > 0:
            # The argmax is a column of the distribution; answer its label
            write(f'    labels = {self._rf_labels(n_classes)!r}\n')
            write('    # Soft voting: average probability distributions\n' if export
                  else '    # Check if results are probability lists (soft voting)\n')
            write('    if isinstance(tree_results[0], list):\n')
            write(f'        n_classes = {n_classes}\n')
            write('        avg = [0.0] * n_classes\n')
            write('        for proba in tree_results:\n')
            write('            for j in range(n_classes):\n')
            write('                avg[j] += proba[j]\n')
            write('        n = len(tree_results)\n')
            write('        avg = [p / n for p in avg]\n')
            if export:
                write('        return labels[avg.index(max(avg))]\n')
            else:
                write('        best = avg.index(max(avg))\n')
                write('        return labels[best], None, avg\n')
            write('    else:\n')
            if not export:
                write('        # Fallback: hard voting (distributions unavailable)\n')
            hard_vote('        ')
        else:
            # No distributions at all: hard voting
            hard_vote('    ')
        return True

    # Method to generate the Python source of a single Decision Tree
    def _dt_python_source(self, rules, write, export):
        """Decision Tree source: one function answering the leaf's label.

        Args:
            rules (List[Rule]): The rules to compile.
            write (Callable[[str], Any]): See `_python_model_source`.
            export (bool): See `_python_model_source`.

        Returns:
            bool: See `_python_model_source`.
        """
        # Uncovered region -> default_class. None, not -1: -1 is a
        # legitimate label (y in {-1, 1}).
        tree = self._codegen_tree(
            rules, export,
            lambda rule: None if rule is None else label_value(rule.class_))
        if tree is None:
            return False
        tail = '' if export else ', None, None'

        # The returned expression of a leaf
        def leaf(val):
            if val is None:
                val = label_value(self.default_class)
            return f'{val!r}{tail}'

        write(function_source('predict' if export else 'native_predict', tree, leaf))
        return True

    # Method to compile rules into a native Python function
    def update_native_model(self, rules_to_compile):
        """
        Compiles a list of Rules into an optimized in-memory Python function.

        This method rebuilds the decision tree structure from linear rules and
        uses `exec()` to create a `predict(sample)` function that runs at
        native Python speed, bypassing slow Rule object iteration.

        Args:
            rules_to_compile (List[Rule]): The rules to be compiled.
        """
        # `classify` may only answer from native_fn for the rule set it was
        # compiled from; remember which one that is.
        self.native_fn = None
        self._native_source = rules_to_compile

        parts = []
        if not self._python_model_source(rules_to_compile, parts.append, export=False):
            return
        func_code = ''.join(parts)

        if self.algorithm_type == 'Gradient Boosting Decision Trees':
            import math as _math
            context = {'math': _math}
            try:
                exec(func_code, context)
                self.native_fn = context['native_predict']
            except Exception as e:
                print(f'Error compiling GBDT native model: {e}')
                self.native_fn = None
            return

        # Compile logic into the local namespace
        context = {'Counter': Counter}
        try:
            exec(func_code, context)
            self.native_fn = context.get('native_predict')
            if self.native_fn is None:
                print("Error: 'native_predict' function not found in compiled code.")
        except Exception as e:
            print(f"Error compiling native model: {e}")
            self.native_fn = None

    # Exports the rule set to a standalone native Python classifier file
    def export_to_native_python(self, feature_names=None, filename="files/fast_classifier.py",
                               rules=None):
        """
        Generates a standalone Python file with the decision logic.

        For Decision Trees, it exports a single nested if/else function.
        For Random Forests, it exports multiple functions (one per tree) and a voting aggregator.

        Args:
            feature_names (List[str], optional): Kept for compatibility.
            filename (str): Output filename.
            rules (List[Rule], optional): Rules to export. Defaults to
                final_rules if available, else initial_rules.
        """
        print(f"[EXPORT] Generating native classifier: {filename}")

        rules_to_export = rules if rules is not None else (
            self.final_rules if self.final_rules else self.initial_rules)

        with open(filename, "w") as f:
            self._python_model_source(rules_to_export, f.write, export=True)

        f32 = getattr(self, 'input_dtype', None) == 'float32'
        table = getattr(self, 'label_table', None)
        if f32 or table is not None:
            # scikit-learn casts X to float32 before comparing it with the
            # thresholds; the standalone file must too (without numpy). And
            # the trees answer label codes; a table names the original labels.
            with open(filename) as f:  # same encoding it was written with
                code = f.read()
            assert code.count("def predict(sample):\n") == 1
            code = code.replace("def predict(sample):\n", "def _predict(sample):\n")
            call = "_predict(sample)"
            if f32:
                code += (
                    "\n\nimport struct as _struct\n\n\n"
                    "def _f32(v):\n"
                    "    \"\"\"Round v to float32, as scikit-learn does before comparing.\"\"\"\n"
                    "    try:\n"
                    "        return _struct.unpack('f', _struct.pack('f', v))[0]\n"
                    "    except (OverflowError, TypeError, _struct.error):\n"
                    "        return v\n"
                )
                call = "_predict({k: _f32(v) for k, v in sample.items()})"
            if table is not None:
                # ascii(): the file is written in the platform encoding.
                code += (
                    "\n\n# Original class label of each code the trees answer.\n"
                    f"_LABELS = {ascii(list(table))}\n"
                )
                call = f"_LABELS[{call}]"
            code += f"\n\ndef predict(sample):\n    return {call}\n"
            with open(filename, "w") as f:
                f.write(code)

        print(f"[EXPORT] File '{filename}' generated.")
