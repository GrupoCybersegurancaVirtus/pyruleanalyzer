"""Data loading and rule extraction from scikit-learn models.

Provides ``ExtractionMixin``: reading train/test CSV files, extracting the
rules of fitted ``DecisionTreeClassifier``, ``RandomForestClassifier`` and
``GradientBoostingClassifier`` models (with their missing-value routing),
and the factories that train a model and build a classifier from it.

`RuleClassifier` is defined after its mixins, so the methods that name it import
it locally.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from ._rules import make_condition, Rule


class ExtractionMixin:
    """Data processing, rule extraction and classifier creation."""

    # Method to apply the input precision to test rows
    @staticmethod
    def _quantize_rows(X_test, clf):
        """Apply `clf`'s input precision to the refinement's test rows.

        The refinement counts rule usage with the iterative engine; without
        this, a sample within one float32 ULP of a threshold would be counted
        on one side while every prediction engine routes it to the other.

        Args:
            X_test (list | np.ndarray): Test rows.
            clf (RuleClassifier | None): The classifier.

        Returns:
            The rows, float32-rounded when ``clf.input_dtype == 'float32'``.
        """
        if clf is None or getattr(clf, 'input_dtype', None) != 'float32':
            return X_test
        try:
            return np.asarray(X_test, dtype=np.float32).astype(np.float64).tolist()
        except (TypeError, ValueError):
            return X_test

    # Method to prepare test data
    @staticmethod
    def _prepare_test_data(file_path=None, X=None, y=None, clf=None):
        """
        Helper method to prepare test data either from a CSV file or direct arrays/DataFrames.
        Returns:
            X_test (list or array), y_test (list or array), feature_names (list)
        """
        from .rule_classifier import RuleClassifier
        import pandas as pd
        import numpy as np

        if X is not None and y is not None:
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
                X_test = X.values.tolist()
            else:
                feature_names = clf._array_feature_names if clf and hasattr(clf, '_array_feature_names') else [f"Feature_{i}" for i in range(len(X[0]))]
                X_test = X.tolist() if hasattr(X, 'tolist') else list(X)
            y_test = y.tolist() if hasattr(y, 'tolist') else list(y)
            if clf is not None:
                # The rules answer label codes: compare y as codes too.
                if getattr(clf, 'label_table', None) is not None:
                    y_test = clf._encode_labels(y_test).tolist()
                # NaN only where every engine can answer it like the model.
                try:
                    arr = np.asarray(X_test, dtype=np.float64)
                except (TypeError, ValueError):
                    arr = None
                if arr is not None:
                    rules = clf.final_rules or clf.initial_rules
                    clf._check_missing_values(arr, clf._rules_route_missing(rules))
            return RuleClassifier._quantize_rows(X_test, clf), y_test, feature_names

        if file_path:
            _, _, X_test, y_test, _, _, feature_names = RuleClassifier.process_data(
                '.', file_path, is_test_only=True
            )
            return RuleClassifier._quantize_rows(X_test, clf), y_test, feature_names
            
        raise ValueError("Must provide either 'file_path' or both 'X' and 'y'.")

    # Method to execute process_data
    @staticmethod
    def process_data(train_path, test_path, is_test_only=False):
        """
        Loads and preprocesses training and testing datasets from CSV files.

        Handles header detection automatically. If no header is found, columns are named
        v1, v2, ..., class. Performs Label Encoding on categorical features to ensure
        all data is numeric for the classifier.

        Args:
            train_path (str): Path to training CSV (ignored if is_test_only=True).
            test_path (str): Path to testing CSV.
            is_test_only (bool): If True, skips loading training data and returns None for train artifacts.

        Returns:
            Tuple containing (X_train, y_train, X_test, y_test, class_names, target_column, feature_names).
        """
        
        # Helper function to consolidate loading logic
        def load_csv_robust(path):
            """
            load_csv_robust method for RuleClassifier.
            Args:
                path: Description of path.
            Returns:
                Description of return value.
            """
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            
            # 1. Detect Header
            has_header = False
            try:
                with open(path, 'r', encoding='latin-1') as f:
                    first_line = f.readline().strip()
                    # Heuristic: if line contains letters, assume it's a header
                    has_header = any(c.isalpha() for c in first_line)
            except Exception:
                pass

            header_row = 0 if has_header else None
            
            # 2. Read CSV
            df = pd.read_csv(path, header=header_row, encoding='latin-1', on_bad_lines='skip')
            
            # 3. Standardize Column Names
            if has_header:
                df.columns = [str(col).strip() for col in df.columns]
            else:
                # Assign generic names: v1, v2, ..., class
                df.columns = [f'v{i+1}' for i in range(df.shape[1]-1)] + ['class']
            
            return df

        # --- LOAD TEST DATA ---
        df_test = load_csv_robust(test_path)

        # Case 1: Test Only (Analysis Mode)
        if is_test_only:
            feature_names = df_test.columns[:-1].tolist()
            target_name = df_test.columns[-1]
            
            # Pre-processing for Test Only mode (Minimal Encoding)
            # Since we don't have the train set's encoder, we try to force numeric conversion
            for col in feature_names:
                if df_test[col].dtype == 'object':
                    try:
                        # Try simple numeric conversion first
                        df_test[col] = pd.to_numeric(df_test[col])
                    except ValueError:
                        # Fallback: simple factorization if it's a string category
                        codes, _ = pd.factorize(df_test[col])
                        df_test[col] = pd.Series(codes, index=df_test.index)

            X_test = df_test.iloc[:, :-1].values.astype(float)
            y_test = df_test.iloc[:, -1].values.astype(int) # Assumes target is interpretable as int
            
            return None, None, X_test, y_test, None, target_name, feature_names

        # --- LOAD TRAIN DATA ---
        df_train = load_csv_robust(train_path)

        # --- DATA PROCESSING (Label Encoding) ---
        # We fit on Train and transform both Train and Test to ensure consistency
        for col in df_train.columns:
            # Check if feature is categorical (object type)
            if df_train[col].dtype == 'object':
                le = LabelEncoder()
                # Convert to string to handle mixed types safely
                df_train[col] = pd.Series(le.fit_transform(df_train[col].astype(str)), index=df_train.index)
                
                # Transform Test set if column exists
                if col in df_test.columns:
                    # Handle unseen labels: map them to the first known class to avoid crashes
                    df_test[col] = df_test[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    df_test[col] = pd.Series(le.transform(df_test[col].astype(str)), index=df_test.index)

        # Extract Metadata
        feature_names = df_train.columns[:-1].tolist()
        target_name = df_train.columns[-1]
        
        # Create robust class names list
        unique_classes = sorted(df_train[target_name].unique())
        class_names = [str(c) for c in unique_classes]

        # Convert to NumPy arrays
        X_train = df_train.iloc[:, :-1].values.astype(float)
        y_train = df_train.iloc[:, -1].values.astype(int)
        
        # Ensure Test set aligns with Train set features
        # If test set is missing columns, fill with 0; if extra, ignore (basic alignment)
        X_test = df_test[feature_names].values.astype(float)
        y_test = df_test[target_name].values.astype(int)

        return X_train, y_train, X_test, y_test, class_names, target_name, feature_names

    # Method to read where each split of a fitted tree sends missing values
    @staticmethod
    def _missing_go_to_left(tree_):
        """The per-node missing-value routing of a fitted scikit-learn tree.

        scikit-learn >= 1.3 sends a NaN to the left child of a split where
        ``missing_go_to_left`` is set. A split learned from data with NaN
        records the side chosen during training; any other split sends NaN
        to the child that received more training samples. Every split has
        a side, so a NaN is never left unrouted.

        Args:
            tree_: A fitted ``sklearn.tree._tree.Tree``.

        Returns:
            np.ndarray | None: One bool per node, or None for an older
            scikit-learn without the attribute (the routing stays unknown
            and NaN inputs are refused).
        """
        flags = getattr(tree_, 'missing_go_to_left', None)
        if flags is None:
            return None
        return np.asarray(flags).astype(bool)

    # Method to extract rules from a tree model
    @staticmethod
    def get_rules(tree, feature_names, class_names):
        """
        Extracts human-readable decision rules from a scikit-learn DecisionTreeClassifier.

        This method traverses the tree structure to generate logical condition paths from root to leaf.
        
        Optimization:
        It builds both the string representation (for display) and the parsed tuple representation 
        (for calculation) simultaneously. This avoids re-parsing strings later and preserves 
        exact floating-point precision.

        Args:
            tree (DecisionTreeClassifier): A trained scikit-learn decision tree model.
            feature_names (List[str]): A list of feature names corresponding to the tree input features.
            class_names (List[str]): A list of class names corresponding to output labels.

        Returns:
            List[Rule]: A list of extracted Rule objects.
        """
        from .rule_classifier import RuleClassifier
        tree_ = tree.tree_ if hasattr(tree, 'tree_') else tree

        # Mapping feature indices to names
        feature_name = [
            feature_names[i] if i != -2 else "undefined!"
            for i in tree_.feature
        ]
        # Where each split sends a missing value (scikit-learn >= 1.3); the
        # conditions record it (see `Condition`).
        missing_left = RuleClassifier._missing_go_to_left(tree_)

        rules = []

        # We pass two lists down the recursion:
        # 1. conditions_str: ["v1 <= 0.5", ...] (For Human Readability/Display)
        # 2. conditions_parsed: [("v1", "<=", 0.5), ...] (For Logic/Native Execution)
        def recurse(node, conditions_str, conditions_parsed):
            """
            Recursively traverses the decision tree and extracts rules.

            Args:
                node (int): Current node index in the tree structure.
                conditions_str (List[str]): List of condition strings accumulated so far (e.g., ["v1 <= 0.5", "v2 > 0.3"]).
                conditions_parsed (List[Tuple[str, str, float]]): List of parsed conditions as tuples (variable, operator, value).
            """
            if tree_.feature[node] != -2:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                
                # CRITICAL: Use str(threshold) to preserve precision in the string representation
                # e.g., prevents 1.16e-7 from being rounded to 0.000000 in text output
                threshold_str = f"{threshold}"
                nan_left = None if missing_left is None else bool(missing_left[node])

                # Left child (<=)
                # We append the exact float 'threshold' to the parsed list, avoiding string-to-float conversion errors later
                recurse(
                    tree_.children_left[node],
                    conditions_str + [f"{name} <= {threshold_str}"],
                    conditions_parsed + [make_condition(name, '<=', threshold, nan_left)]
                )

                # Right child (>)
                recurse(
                    tree_.children_right[node],
                    conditions_str + [f"{name} > {threshold_str}"],
                    conditions_parsed + [make_condition(
                        name, '>', threshold, None if nan_left is None else not nan_left)]
                )
            else:
                # Leaf node
                value_array = tree_.value[node].flatten()
                class_idx = np.argmax(value_array)
                
                # Store weighted sample counts per class for RF soft voting.
                # sklearn's tree_.value stores normalized probabilities (sum=1),
                # so we multiply by the weighted sample count at this leaf to
                # recover the actual (weighted) sample counts.
                # Raw counts allow correct merging (element-wise addition)
                # during adjust_and_remove_rules; normalization to probabilities
                # happens at classification time in classify_rf / native model.
                weighted_n = tree_.weighted_n_node_samples[node]
                class_dist = (value_array * weighted_n).tolist()
                
                # Ensure we handle class names correctly (int or str)
                try:
                    class_label = class_names[class_idx]
                except (IndexError, TypeError):
                    class_label = str(class_idx)

                rule_name = f"Rule{len(rules)}_Class{class_label}"
                
                # Create the Rule object with class distribution
                new_rule = Rule(rule_name, str(class_label), conditions_str,
                                class_distribution=class_dist)
                
                # OPTIMIZATION: Assign the pre-parsed conditions immediately.
                # This bypasses 'parse_conditions_static' later, saving time and keeping precision.
                new_rule.parsed_conditions = conditions_parsed
                
                rules.append(new_rule)

        recurse(0, [], [])
        return rules

    # Method to extract rules from a Random Forest model
    @staticmethod
    def get_tree_rules(model, feature_names, class_names, algorithm_type='Random Forest'):
        """
        Extracts rules from a trained scikit-learn model (Decision Tree or Random Forest).

        For Decision Trees, this returns one list of rules. 
        For Random Forests, it returns a list of lists (one list of rules per estimator).

        Args:
            model (Union[DecisionTreeClassifier, RandomForestClassifier]): The trained model.
            feature_names (List[str]): List of feature names.
            class_names (List[str]): List of class names.
            algorithm_type (str): Type of model; either 'Decision Tree' or 'Random Forest'.

        Returns:
            Union[List[Rule], List[List[Rule]]]: The extracted rules.
        """
        from .rule_classifier import RuleClassifier
        # Remove debug print to clean up output
        # print("Feature names:", feature_names)

        rules = []
        if algorithm_type == 'Random Forest':
            # Iterate over all trees in the forest
            for i, estimator in enumerate(model.estimators_):
                tree_rules = RuleClassifier.get_rules(estimator, feature_names, class_names)
                # Add Tree Identifier to Rule Names (DT{i}_...)
                for rule in tree_rules:
                    rule.name = f"DT{i}_{rule.name}"
                rules.append(tree_rules)
        
        elif algorithm_type == 'Decision Tree':
            # Single tree
            rules.append(RuleClassifier.get_rules(model, feature_names, class_names))
        
        else:
            raise ValueError(f"Unsupported algorithm type: {algorithm_type}")
        
        return rules

    # Method to extract rules from a Gradient Boosting model
    @staticmethod
    def get_gbdt_rules(model, feature_names, class_names):
        """
        Extracts rules from a trained GradientBoostingClassifier into standard Rule objects.

        Each leaf in each boosting tree becomes a Rule with GBDT-specific metadata
        (leaf_value, learning_rate, contribution, class_group). An additional
        init rule (no conditions) is created per class group to represent the
        initial score from the prior estimator.

        Args:
            model (GradientBoostingClassifier): A fitted sklearn GBDT model.
            feature_names (List[str]): Feature names used during training.
            class_names (List[str]): Class label strings.

        Returns:
            Tuple[List[Rule], Dict[str, float], bool, List[str]]:
                - List of all Rule objects (init rules + tree rules).
                - Dict mapping class_group -> init_score.
                - Whether the model is binary classification.
                - List of class label strings.
        """
        from .rule_classifier import RuleClassifier
        classes = RuleClassifier._label_names(model.classes_)
        is_binary = len(classes) == 2
        learning_rate = model.get_params()['learning_rate']

        # The init score is read once, as a constant (one rule per class).
        # That holds for 'zero' and for scikit-learn's default prior
        # (a DummyClassifier); any other init estimator scores depend on x,
        # and rules built from a single constant would be silently wrong.
        from sklearn.dummy import DummyClassifier
        init = getattr(model, 'init_', 'zero')
        if not (isinstance(init, str) and init == 'zero'):
            if (not isinstance(init, DummyClassifier)
                    or getattr(init, 'strategy', 'prior') == 'stratified'):
                raise ValueError(
                    'Unsupported GradientBoostingClassifier init estimator '
                    f'{type(init).__name__}: only init=None (the default '
                    "constant prior) or init='zero' can be expressed as "
                    'rules, because any other init gives an initial score '
                    'that depends on the input.'
                )

        all_rules = []
        init_scores = {}

        # For binary: only positive class (index 1) has trees, stored in estimators_[:, 0]
        # For multiclass: each class has its own trees, stored in estimators_[:, class_idx]
        class_iter = (
            [(1, classes[1])] if is_binary
            else list(enumerate(classes))
        )

        for class_idx, class_label in class_iter:
            # Compute init score from the prior estimator (DummyClassifier)
            if isinstance(init, str):
                init_score = 0.0
            else:
                n_features = len(feature_names)
                dummy_sample = np.zeros((1, n_features))
                init_preds = np.asarray(init.predict_proba(dummy_sample))
                if is_binary:
                    # Binary GBDT: sklearn uses log-odds (logit) of
                    # the positive class as the initial raw score.
                    p = float(init_preds[0, 1])
                    p = np.clip(p, 1e-15, 1 - 1e-15)
                    init_score = float(np.log(p / (1 - p)))
                else:
                    # Multiclass GBDT: sklearn uses centered log-priors
                    # log(p_k) - mean(log(p_j) for all j).
                    log_priors = np.log(init_preds[0] + 1e-15)
                    mean_log = float(np.mean(log_priors))
                    init_score = float(log_priors[class_idx] - mean_log)

            init_scores[class_label] = init_score

            # Create init rule (no conditions, just a constant score)
            init_rule_name = f'GBDT{class_label}T0_Init_Class{class_label}'
            init_rule = Rule(
                init_rule_name, str(class_label), [],
                leaf_value=init_score, learning_rate=None,
                class_group=str(class_label),
            )
            init_rule.parsed_conditions = []
            all_rules.append(init_rule)

            # Extract rules from each estimator tree
            estimator_col = class_idx if not is_binary else 0
            for tree_idx, estimator in enumerate(model.estimators_[:, estimator_col]):
                tree_ = estimator.tree_
                missing_left = RuleClassifier._missing_go_to_left(tree_)
                tree_rules = []

                # Method to recurse.
                def recurse(node_id, conditions_str, conditions_parsed):
                    """
                    recurse method for RuleClassifier.
                    Args:
                        node_id: Description of node_id.
                        conditions_str: Description of conditions_str.
                        conditions_parsed: Description of conditions_parsed.
                    Returns:
                        Description of return value.
                    """
                    left = int(tree_.children_left[node_id])
                    right = int(tree_.children_right[node_id])

                    if left == right:  # leaf
                        leaf_value = float(tree_.value[node_id, 0, 0])
                        leaf_idx = len(tree_rules)
                        rule_name = (
                            f'GBDT{class_label}T{tree_idx + 1}'
                            f'_Rule{leaf_idx}_Class{class_label}'
                        )
                        rule = Rule(
                            rule_name, str(class_label), list(conditions_str),
                            leaf_value=leaf_value, learning_rate=learning_rate,
                            class_group=str(class_label),
                        )
                        rule.parsed_conditions = list(conditions_parsed)
                        tree_rules.append(rule)
                        return

                    feat_idx = int(tree_.feature[node_id])
                    feat_name = feature_names[feat_idx]
                    threshold = float(tree_.threshold[node_id])
                    threshold_str = f'{threshold}'
                    # Recorded although GradientBoostingClassifier refuses
                    # NaN: the regions are those of the fitted trees.
                    nan_left = (None if missing_left is None
                                else bool(missing_left[node_id]))

                    # Left child (<=)
                    recurse(
                        left,
                        conditions_str + [f'{feat_name} <= {threshold_str}'],
                        conditions_parsed + [make_condition(
                            feat_name, '<=', threshold, nan_left)],
                    )
                    # Right child (>)
                    recurse(
                        right,
                        conditions_str + [f'{feat_name} > {threshold_str}'],
                        conditions_parsed + [make_condition(
                            feat_name, '>', threshold,
                            None if nan_left is None else not nan_left)],
                    )

                recurse(0, [], [])
                all_rules.extend(tree_rules)

        return all_rules, init_scores, is_binary, classes

    # Method to generate a classifier model based on rules
    @staticmethod
    def generate_classifier_model(rules, class_names_map, algorithm_type='Random Forest', save_initial_model=True):
        """
        Instantiates a RuleClassifier from extracted rules and optionally saves it.

        Optimization: 
        This method now passes the Rule objects directly to the constructor, avoiding 
        the intermediate string serialization that caused floating-point precision loss.

        Args:
            rules (Union[List[Rule], List[List[Rule]]]): The extracted rules.
            class_names_map (Dict[str, int]): Map of class names to indices (kept for compatibility).
            algorithm_type (str): 'Random Forest' or 'Decision Tree'.
            save_initial_model (bool): Whether to save the initial model to 'initial_model.pkl'. Default is True.

        Returns:
            RuleClassifier: The initialized classifier.
        """
        from .rule_classifier import RuleClassifier
        # Direct instantiation using the list of Rule objects.
        # This preserves the high-precision 'parsed_conditions' created in get_rules.
        classifier = RuleClassifier(rules, algorithm_type=algorithm_type)
        
        print(f"Algorithm Type: {classifier.algorithm_type}")

        if save_initial_model:
            path = 'files/initial_model.pkl'
            try:
                with open(path, 'wb') as model_file:
                    pickle.dump(classifier, model_file)
                print(f"Classifier file saved: {path}")
            except Exception as e:
                print(f"Warning: Could not save classifier to {path}. Error: {e}")

        return classifier
    
    # Method to execute create
    @staticmethod
    def create(train, test, model="Decision Tree", parameters=None, algorithm_type=None):
        """
        Creates a new rule-based classifier from CSV data files.

        Args:
            train (str): Path to training CSV file.
            test (str): Path to test CSV file.
            model (str): Model type - 'Decision Tree', 'Random Forest', or 
                        'Gradient Boosting Decision Trees'. Default is 'Decision Tree'.
            parameters (dict, optional): Model hyperparameters. Default is None.
            algorithm_type (str, optional): Deprecated. Use 'model' instead.

        Returns:
            RuleClassifier: The created classifier.
        """
        from .rule_classifier import RuleClassifier
        if algorithm_type is not None:
            import warnings
            warnings.warn(
                "algorithm_type is deprecated, use 'model' instead",
                DeprecationWarning, stacklevel=2
            )
            model = algorithm_type
        
        if parameters is None:
            parameters = {"random_state": 42}
        
        return RuleClassifier.new_classifier(
            train_path=train,
            test_path=test,
            model_parameters=parameters,
            algorithm_type=model,
            save_initial_model=False,
            save_sklearn_model=False
        )
    
    # Method to execute export
    def export(self, base_name="model", formats=None, feature_names=None, use_final=None):
        """
        Exports the classifier to one or more file formats.

        Args:
            base_name (str): Base name for exported files (without extension).
            formats (list, optional): List of formats to export to.
                                    Options: "python", "binary", "c".
                                    If None, exports to python and binary only.
            feature_names (list, optional): Feature names for export.
            use_final (bool, optional): Export final_rules (True) or
                initial_rules (False). Defaults to final_rules if available,
                else initial_rules. Every format exports this same rule set.

        Returns:
            dict: Dictionary with export results for each format.
        """
        rules = (self._rules_for(use_final) if use_final is not None
                 else (self.final_rules if self.final_rules else self.initial_rules))
        if formats is None:
            formats = ["python", "binary"]
        
        formats = [f.lower() for f in formats]
        
        export_python = "python" in formats
        export_binary = "binary" in formats or "bin" in formats
        export_c = "c" in formats or "header" in formats
        export_cpn = "cpn" in formats or "cpntools" in formats or "hcpn" in formats
        
        print("\n" + "=" * 60)
        print("EXPORTING CLASSIFIER")
        print("=" * 60)
        
        if '/' not in base_name and '\\' not in base_name:
            base_path = f"files/{base_name}"
        else:
            base_path = base_name
        
        os.makedirs(os.path.dirname(base_path) or '.', exist_ok=True)

        results = {}

        if export_python:
            py_filename = f"{base_path}.py"
            self.export_to_native_python(feature_names=feature_names, filename=py_filename,
                                         rules=rules)
            results["python"] = py_filename
            print(f"  [OK] Python export: {py_filename}")

        if export_binary or export_c:
            # The binary and C exports serialize the *current* arrays, which
            # are whatever was compiled last -- possibly the other rule set.
            # Compile the exported one for the duration of the export.
            saved = self._current_array_state() if self._arrays_compiled else None
            names = feature_names or (list(self._array_feature_names)
                                      if self._array_feature_names else None)
            self._set_array_state(self._array_state_for_rules(rules, names))
            try:
                if export_binary:
                    bin_filename = f"{base_path}.bin"
                    self.export_to_binary(filepath=bin_filename)
                    results["binary"] = bin_filename
                    print(f"  [OK] Binary export: {bin_filename}")

                if export_c:
                    c_filename = f"{base_path}.h"
                    self.export_to_c_header(filepath=c_filename)
                    results["c"] = c_filename
                    print(f"  [OK] C Header export: {c_filename}")
            finally:
                if saved is not None:
                    self._set_array_state(saved)

        if export_cpn:
            cpn_filename = f"{base_path}.cpn"
            self.to_cpn_tools(filepath=cpn_filename, rules=rules,
                              feature_names=feature_names,
                              use_final=rules is not self.initial_rules)
            results["cpn"] = cpn_filename
            print(f"  [OK] CPN Tools export: {cpn_filename}")

        print("=" * 60 + "\n")
        
        return results
    
    # Method to create a new classifier
    @staticmethod
    def new_classifier(train_path, test_path, model_parameters, model_path=None, algorithm_type='Random Forest', save_initial_model=True, save_sklearn_model=True):
        """
        Orchestrates the creation of a new classifier from scratch.

        Pipeline:
        1. Load and Process Data (using robust CSV loader).
        2. Train (or Load) Scikit-Learn Model.
        3. Evaluate Scikit-Learn Model (Benchmark).
        4. Extract Rules from Tree(s).
        5. Generate RuleClassifier.

        Args:
            train_path (str): Path to training CSV.
            test_path (str): Path to testing CSV.
            model_parameters (dict): Arguments for the sklearn classifier.
            model_path (Optional[str]): Path to existing .pkl model to skip training.
            algorithm_type (str): 'Random Forest', 'Decision Tree', or 'Gradient Boosting Decision Trees'.
            save_initial_model (bool): Whether to save the initial model to 'initial_model.pkl'. Default is True.
            save_sklearn_model (bool): Whether to save the sklearn model to 'sklearn_model.pkl'. Default is True.

        Returns:
            RuleClassifier: The final rule-based model.
        """
        from .rule_classifier import RuleClassifier
        
        print("\n" + "*"*80)
        print("GENERATING A NEW CLASSIFIER")
        print("*"*80 + "\n")
        
        print("Processing Database...")
        # 1. Process Data
        # Returns correctly encoded data and class names
        X_train, y_train, X_test, y_test, class_names, _, feature_names = RuleClassifier.process_data(train_path, test_path)

        # Ensure class_names is always a list (process_data may return None in test-only mode)
        if class_names is None:
            class_names = [str(c) for c in sorted(np.unique(np.asarray(y_test)))]

        # 2. Train or Load Model
        if model_path:
            print(f"Loading model from: {model_path}")
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)
        else:
            if X_train is None or y_train is None:
                raise ValueError("Training data could not be loaded. Please check the train_path parameter.")
            
            print(f"Training a new Scikit-Learn {algorithm_type}...")
            if algorithm_type == 'Random Forest':
                model = RandomForestClassifier(**model_parameters)
            elif algorithm_type == 'Decision Tree':
                model = DecisionTreeClassifier(**model_parameters)
            elif algorithm_type == 'Gradient Boosting Decision Trees':
                model = GradientBoostingClassifier(**model_parameters)
            else:
                raise ValueError(f"Unsupported algorithm type: {algorithm_type}")
            
            # Fit model
            # For GBDT, fit with DataFrame so feature_names_in_ is populated
            if algorithm_type == 'Gradient Boosting Decision Trees':
                X_train_fit = pd.DataFrame(X_train, columns=feature_names)
                model.fit(X_train_fit, np.asarray(y_train))
            else:
                model.fit(X_train, np.asarray(y_train))

            # Save Sklearn model
            if save_sklearn_model:
                model_save_path = 'files/sklearn_model.pkl'
                with open(model_save_path, 'wb') as model_file: 
                    pickle.dump(model, model_file)
                print(f"Trained model saved at: {model_save_path}")

        # 3. Test Scikit-Learn Model (Benchmark)
        print("\nTesting Scikit-Learn Model (Benchmark):")
        # For GBDT, use DataFrame to avoid feature name warnings
        if algorithm_type == 'Gradient Boosting Decision Trees':
            X_test_predict = pd.DataFrame(X_test, columns=feature_names)
        else:
            X_test_predict = X_test
        y_pred = model.predict(X_test_predict)
        
        correct = np.sum(y_pred == y_test)
        total = len(y_test)
        
        # Display metrics using the extracted class names for better readability
        RuleClassifier.display_metrics(y_test, y_pred, correct, total, class_names=class_names)

        # --- GBDT: Extract standard Rule objects, same as DT/RF ---
        if algorithm_type == 'Gradient Boosting Decision Trees':
            print("\nExtracting GBDT rules into standard Rule objects...")
            rules, init_scores, is_binary, gbdt_classes = RuleClassifier.get_gbdt_rules(
                model, feature_names, class_names
            )
            print(f"Extracted {len(rules)} GBDT rules.")

            # Build RuleClassifier using standard constructor
            # (constructor defers native compilation for GBDT)
            classifier = RuleClassifier(rules, algorithm_type=algorithm_type)

            # Set GBDT-specific metadata
            classifier._gbdt_init_scores = init_scores
            classifier._gbdt_is_binary = is_binary
            classifier._gbdt_classes = gbdt_classes
            # scikit-learn compares float32(x) with the thresholds
            classifier.input_dtype = 'float32'

            # Compile native model now that metadata is set
            classifier.update_native_model(classifier.initial_rules)

            # Save initial model (after metadata is set)
            if save_initial_model:
                path = 'files/initial_model.pkl'
                try:
                    with open(path, 'wb') as model_file:
                        pickle.dump(classifier, model_file)
                    print(f"Classifier file saved: {path}")
                except Exception as e:
                    print(f"Warning: Could not save classifier to {path}. Error: {e}")

            print(f"Algorithm Type: {classifier.algorithm_type}")
            return classifier

        # 4. Extract Rules
        # We pass class_names so the rules are generated with "Class_Normal" instead of "Class_0"
        print(f"\nExtracting rules from {algorithm_type}...")
        rules = RuleClassifier.get_tree_rules(model, feature_names, class_names, algorithm_type=algorithm_type)

        # 5. Generate RuleClassifier
        # We assume mapping corresponds to the sorted unique labels from process_data
        class_names_map = {str(name): i for i, name in enumerate(class_names)}
        
        print("Initializing RuleClassifier...")
        classifier = RuleClassifier.generate_classifier_model(rules, class_names_map, algorithm_type, save_initial_model=save_initial_model)
        # scikit-learn compares float32(x) with the thresholds
        classifier.input_dtype = 'float32'

        return classifier
