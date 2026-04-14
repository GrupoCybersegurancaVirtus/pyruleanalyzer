import numpy as np
from .rule_classifier import Rule

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
    tree_ = tree.tree_ if hasattr(tree, 'tree_') else tree
    
    # Mapping feature indices to names
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]
    
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
            
            # Left child (<=)
            # We append the exact float 'threshold' to the parsed list, avoiding string-to-float conversion errors later
            recurse(
                tree_.children_left[node], 
                conditions_str + [f"{name} <= {threshold_str}"],
                conditions_parsed + [(name, '<=', threshold)] 
            )
            
            # Right child (>)
            recurse(
                tree_.children_right[node], 
                conditions_str + [f"{name} > {threshold_str}"],
                conditions_parsed + [(name, '>', threshold)]
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
    # Remove debug print to clean up output
    # print("Feature names:", feature_names)

    rules = []
    if algorithm_type == 'Random Forest':
        # Iterate over all trees in the forest
        for i, estimator in enumerate(model.estimators_):
            tree_rules = get_rules(estimator, feature_names, class_names)
            # Add Tree Identifier to Rule Names (DT{i}_...)
            for rule in tree_rules:
                rule.name = f"DT{i}_{rule.name}"
            rules.append(tree_rules)
    
    elif algorithm_type == 'Decision Tree':
        # Single tree
        rules.append(get_rules(model, feature_names, class_names))
    
    else:
        raise ValueError(f"Unsupported algorithm type: {algorithm_type}")
    
    return rules

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
    classes = [str(c) for c in model.classes_]
    is_binary = len(classes) == 2
    learning_rate = model.get_params()['learning_rate']

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
        init = model.init_
        if init == 'zero':
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
            tree_rules = []

            def recurse(node_id, conditions_str, conditions_parsed):
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

                # Left child (<=)
                recurse(
                    left,
                    conditions_str + [f'{feat_name} <= {threshold_str}'],
                    conditions_parsed + [(feat_name, '<=', threshold)],
                )
                # Right child (>)
                recurse(
                    right,
                    conditions_str + [f'{feat_name} > {threshold_str}'],
                    conditions_parsed + [(feat_name, '>', threshold)],
                )

            recurse(0, [], [])
            all_rules.extend(tree_rules)

    return all_rules, init_scores, is_binary, classes