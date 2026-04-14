"""
Simple API for pyruleanalyzer

This module provides a high-level, easy-to-use interface for the entire
classifier lifecycle: create, refine, predict, and export.

Example:
    from pyruleanalyzer import PyRuleAnalyzer
    
    # Create and refine a classifier in one step
    analyzer = PyRuleAnalyzer.create(
        train_path="data/train.csv",
        test_path="data/test.csv",
        model="Decision Tree",
        params={"max_depth": 5},
        refine=True
    )
    
    # Predict
    predictions = analyzer.predict(X_test)
    
    # Export
    analyzer.export("my_model", formats=["python", "binary"])
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union, Any

from .rule_classifier import RuleClassifier


class PyRuleAnalyzer:
    """
    High-level interface for creating, refining, and deploying classifiers.
    
    This class simplifies the pyruleanalyzer workflow into a few intuitive
    methods, handling all the complexity internally.
    
    Attributes:
        classifier (RuleClassifier): The underlying RuleClassifier instance.
        feature_names (list): Names of features used by the model.
        class_names (list): Names of target classes.
    
    Example:
        >>> analyzer = PyRuleAnalyzer.create(
        ...     train_path="data/train.csv",
        ...     test_path="data/test.csv",
        ...     model="Random Forest",
        ...     params={"n_estimators": 100, "random_state": 42},
        ...     refine=True
        ... )
        >>> predictions = analyzer.predict(X_test)
        >>> analyzer.export("my_model")
    """
    
    def __init__(self, classifier: RuleClassifier, feature_names: List[str], class_names: List[str]):
        """
        Initialize a PyRuleAnalyzer with an existing RuleClassifier.
        
        Args:
            classifier: A trained RuleClassifier instance.
            feature_names: List of feature names.
            class_names: List of class names.
        """
        self.classifier = classifier
        self.feature_names = feature_names
        self.class_names = class_names
    
    # ==========================================================================
    # FACTORY METHODS
    # ==========================================================================
    
    @staticmethod
    def create(
        train_path: str,
        test_path: str,
        model: str = "Decision Tree",
        params: Optional[Dict] = None,
        refine: bool = False,
        refine_params: Optional[Dict] = None,
        save_models: bool = False
    ) -> "PyRuleAnalyzer":
        """
        Create a new classifier from CSV data files.
        
        This is the main entry point for using pyruleanalyzer. It handles:
        1. Loading and preprocessing data
        2. Training the sklearn model
        3. Extracting rules
        4. Optionally refining rules
        
        Args:
            train_path: Path to training CSV file.
            test_path: Path to test CSV file (used for refinement).
            model: Model type - "Decision Tree", "Random Forest", or
                  "Gradient Boosting Decision Trees". Default is "Decision Tree".
            params: Model hyperparameters. Default is None (uses sensible defaults).
            refine: If True, automatically refine rules after creation.
                   Default is False.
            refine_params: Parameters for refinement. Ignored if refine=False.
                         See PyRuleAnalyzer.refine() for details.
            save_models: If True, save intermediate models to files/. Default is False.
        
        Returns:
            PyRuleAnalyzer: A configured PyRuleAnalyzer instance ready for prediction.
        
        Example:
            >>> analyzer = PyRuleAnalyzer.create(
            ...     train_path="data/train.csv",
            ...     test_path="data/test.csv",
            ...     model="Decision Tree",
            ...     params={"max_depth": 5},
            ...     refine=True
            ... )
        """
        # Set default parameters if not provided
        if params is None:
            params = PyRuleAnalyzer._get_default_params(model)
        
        # Create the RuleClassifier
        classifier = RuleClassifier.new_classifier(
            train_path=train_path,
            test_path=test_path,
            model_parameters=params,
            algorithm_type=model,
            save_initial_model=save_models,
            save_sklearn_model=save_models
        )
        
        # Extract feature names and class names
        feature_names = classifier._array_feature_names if hasattr(classifier, '_array_feature_names') else []
        class_names = classifier.class_labels
        
        # Create PyRuleAnalyzer instance
        analyzer = PyRuleAnalyzer(classifier, feature_names, class_names)
        
        # Optionally refine
        if refine:
            if refine_params is None:
                refine_params = {}
            analyzer.refine(test_path, **refine_params)
        
        return analyzer
    
    @staticmethod
    def new_model(
        model: str = "Decision Tree",
        params: Optional[Dict] = None
    ) -> "PyRuleAnalyzer":
        """
        Create a new, empty PyRuleAnalyzer ready to be trained.
        
        Args:
            model: Model type - "Decision Tree", "Random Forest", or
                  "Gradient Boosting Decision Trees". Default is "Decision Tree".
            params: Model hyperparameters. Default is None (uses sensible defaults).
        
        Returns:
            PyRuleAnalyzer: A configured but untrained PyRuleAnalyzer instance.
            
        Example:
            >>> analyzer = PyRuleAnalyzer.new_model(model="Decision Tree")
            >>> analyzer.fit(X_train, y_train)
        """
        # Set default parameters if not provided
        if params is None:
            params = PyRuleAnalyzer._get_default_params(model)
            
        # Create empty RuleClassifier
        classifier = RuleClassifier([], algorithm_type=model)
        
        # We need to monkey-patch the model parameters into the classifier
        # so it knows how to initialize the underlying sklearn model when fit() is called
        classifier.model_parameters = params
        
        return PyRuleAnalyzer(classifier, [], [])
        
    def fit(self, X, y) -> "PyRuleAnalyzer":
        """
        Fit the model according to the given training data.
        
        Args:
            X: Training vectors (e.g. pandas DataFrame or numpy array).
            y: Target values.
            
        Returns:
            self
        """
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier
        from .rule_classifier import RuleClassifier
        
        params = getattr(self.classifier, 'model_parameters', self._get_default_params(self.classifier.algorithm_type))
        algorithm_type = self.classifier.algorithm_type
        
        # Train a new Scikit-Learn model
        if algorithm_type == 'Random Forest':
            model = RandomForestClassifier(**params)
        elif algorithm_type == 'Decision Tree':
            model = DecisionTreeClassifier(**params)
        elif algorithm_type == 'Gradient Boosting Decision Trees':
            model = GradientBoostingClassifier(**params)
        else:
            raise ValueError(f"Unsupported algorithm type: {algorithm_type}")
            
        model.fit(X, y)
        
        # Determine feature names and class names
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
        # Get class names from y
        self.class_names = [str(c) for c in sorted(np.unique(np.asarray(y)))]
        
        # Extract rules
        if algorithm_type == 'Gradient Boosting Decision Trees':
            rules, init_scores, is_binary, gbdt_classes = RuleClassifier.get_gbdt_rules(
                model, self.feature_names, self.class_names
            )
            # Create new classifier with the rules
            self.classifier = RuleClassifier(rules, algorithm_type=algorithm_type)
            self.classifier._gbdt_init_scores = init_scores
            self.classifier._gbdt_is_binary = is_binary
            self.classifier._gbdt_classes = gbdt_classes
        else:
            rules = RuleClassifier.get_tree_rules(
                model, self.feature_names, self.class_names, algorithm_type=algorithm_type
            )
            # Create new classifier with the rules
            self.classifier = RuleClassifier(rules, algorithm_type=algorithm_type)
            
        # Update class properties
        self.classifier.class_labels = self.class_names
        self.classifier.num_classes = len(self.class_names)
        self.classifier._array_feature_names = self.feature_names
        
        return self
    
    @staticmethod
    def load(path: str) -> "PyRuleAnalyzer":
        """
        Load a PyRuleAnalyzer from a saved file.
        
        Args:
            path: Path to saved .pkl file.
        
        Returns:
            PyRuleAnalyzer: The loaded PyRuleAnalyzer instance.
        
        Example:
            >>> analyzer = PyRuleAnalyzer.load("files/my_analyzer.pkl")
        """
        classifier = RuleClassifier.load(path)
        feature_names = classifier._array_feature_names if hasattr(classifier, '_array_feature_names') else []
        class_names = classifier.class_labels
        return PyRuleAnalyzer(classifier, feature_names, class_names)
    
    @staticmethod
    def _get_default_params(model: str) -> Dict:
        """Return sensible default parameters for each model type."""
        defaults = {
            "Decision Tree": {"max_depth": None, "random_state": 42},
            "Random Forest": {"n_estimators": 100, "max_depth": None, "random_state": 42},
            "Gradient Boosting Decision Trees": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "random_state": 42}
        }
        return defaults.get(model, {"random_state": 42})
    
    # ==========================================================================
    # REFINEMENT
    # ==========================================================================
    
    def execute_rule_refinement(
        self,
        test_path: str = None,
        X=None,
        y=None,
        remove_below_n_classifications: int = -1,
        save_final_model: bool = False,
        save_report: bool = False
    ) -> Dict[str, Any]:
        """
        Refine the classifier by removing redundant and low-usage rules.
        
        Args:
            test_path: Path to CSV file for evaluating rule usage.
            X: Input data for evaluating rule usage (DataFrame or ndarray).
            y: Target labels for evaluating rule usage.
            remove_below_n_classifications: Minimum usage threshold for rules.
                            Rules used fewer times than this value will be removed.
                            Use -1 to disable. Default is -1.
            save_final_model: If True, save refined model to files/final_model.pkl.
                            Default is False.
            save_report: If True, save refinement report to files/.
                       Default is False.
        
        Returns:
            Dictionary with refinement statistics:
            - rules_before: Number of rules before refinement
            - rules_after: Number of rules after refinement
            - rules_removed: Number of rules removed
            - reduction_percent: Percentage reduction
        
        Example:
            >>> stats = analyzer.execute_rule_refinement(
            ...     X=X_test, y=y_test,
            ...     remove_below_n_classifications=5
            ... )
            >>> print(f"Removed {stats['rules_removed']} rules ({stats['reduction_percent']:.1f}%)")
        """
        rules_before = len(self.classifier.initial_rules)
        
        # Execute refinement using the appropriate analyzer
        # The analyzer handles save_final_model and save_report internally
        if self.classifier.algorithm_type == 'Decision Tree':
            from .dt_analyzer import DTAnalyzer
            analyzer = DTAnalyzer(self.classifier)
            analyzer.execute_rule_refinement(
                file_path=test_path,
                X=X, y=y,
                remove_below_n_classifications=remove_below_n_classifications,
                save_final_model=save_final_model,
                save_report=save_report
            )
        elif self.classifier.algorithm_type == 'Random Forest':
            from .rf_analyzer import RFAnalyzer
            analyzer = RFAnalyzer(self.classifier)
            analyzer.execute_rule_refinement(
                file_path=test_path,
                X=X, y=y,
                remove_below_n_classifications=remove_below_n_classifications,
                save_final_model=save_final_model,
                save_report=save_report
            )
        elif self.classifier.algorithm_type == 'Gradient Boosting Decision Trees':
            from .gbdt_analyzer import GBDTAnalyzer
            analyzer = GBDTAnalyzer(self.classifier)
            analyzer.execute_rule_refinement(
                file_path=test_path,
                X=X, y=y,
                remove_below_n_classifications=remove_below_n_classifications,
                save_final_model=save_final_model,
                save_report=save_report
            )
        else:
            raise ValueError(f"Unsupported algorithm type: {self.classifier.algorithm_type}")
        
        rules_after = len(self.classifier.final_rules) if self.classifier.final_rules else len(self.classifier.initial_rules)
        
        return {
            "rules_before": rules_before,
            "rules_after": rules_after,
            "rules_removed": rules_before - rules_after,
            "reduction_percent": ((rules_before - rules_after) / rules_before * 100) if rules_before > 0 else 0
        }
    
    def compare_initial_final_results(self, test_path: str = None, X = None, y = None) -> None:
        """
        Compare the initial and final (refined) models on the test dataset.
        Prints out performance metrics and complexity scores.
        
        Args:
            test_path: Path to the CSV test file.
            X: Input data for test (DataFrame or ndarray).
            y: Target labels for test.
            
        Example:
            >>> analyzer.compare_initial_final_results(X=X_test, y=y_test)
        """
        return self.classifier.compare_initial_final_results(test_path, X=X, y=y)
    
    # ==========================================================================
    # PREDICTION
    # ==========================================================================
    
    def predict(
        self,
        X,
        use_refined: bool = True
    ):
        """
        Predict class labels for input data.
        
        Args:
            X: Input data (dict, series, list for single; DataFrame, ndarray for batch).
            use_refined: If True, use refined rules (if available). Default is True.
        
        Returns:
            Predicted class label(s).
        """
        import pandas as pd
        import numpy as np
        
        # Single sample prediction (dict)
        if isinstance(X, dict):
            predicted_class, _, _ = self.classifier.classify(X, final=use_refined)
            return predicted_class
            
        # Single sample prediction (Pandas Series)
        if isinstance(X, pd.Series):
            predicted_class, _, _ = self.classifier.classify(X.to_dict(), final=use_refined)
            return predicted_class
            
        # Batch prediction
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Compile arrays if not done yet
        if not getattr(self.classifier, '_arrays_compiled', False):
            self.classifier.compile_tree_arrays(feature_names=self.feature_names)
            
        return self.classifier.predict_batch(
            X,
            feature_names=self.feature_names,
            use_final=use_refined
        )
    
    def predict_proba(
        self,
        X,
        use_refined: bool = True
    ):
        """
        Predict class probabilities for input data.
        
        Args:
            X: Input data (dict, series, list for single; DataFrame, ndarray for batch).
            use_refined: If True, use refined rules (if available). Default is True.
        
        Returns:
            Class probabilities.
        """
        import pandas as pd
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        if not getattr(self.classifier, '_arrays_compiled', False):
            self.classifier.compile_tree_arrays(feature_names=self.feature_names)
            
        return self.classifier.predict_batch_proba(
            X,
            feature_names=self.feature_names
        )
    
    # ==========================================================================
    # EXPORT
    # ==========================================================================
    
    def export(
        self,
        base_name: str = "model",
        formats: Optional[List[str]] = None,
        use_refined: bool = True
    ) -> Dict[str, str]:
        """
        Export the classifier to one or more file formats.
        
        Args:
            base_name: Base name for exported files (without extension).
                      Files will be saved in the files/ directory.
            formats: List of formats to export to.
                   Options: "python", "binary", "c".
                   If None, exports to "python" and "binary".
            use_refined: If True, export refined rules (if available).
                       If False, export original rules.
                       Default is True.
        
        Returns:
            Dictionary mapping format to file path.
        
        Example:
            >>> files = analyzer.export("my_model", formats=["python", "binary"])
            >>> print(f"Exported to: {files}")
            # Output: {'python': 'files/my_model.py', 'binary': 'files/my_model.bin'}
        """
        # Switch to initial rules if not using refined
        if not use_refined and self.classifier.final_rules:
            # Temporarily swap rules
            temp_final = self.classifier.final_rules
            self.classifier.final_rules = []
        
        try:
            result = self.classifier.export(
                base_name=base_name,
                formats=formats,
                feature_names=self.feature_names
            )
        finally:
            # Restore refined rules
            if not use_refined and self.classifier.final_rules:
                self.classifier.final_rules = temp_final
        
        return result
    
    # ==========================================================================
    # SAVE/LOAD
    # ==========================================================================
    
    def save(self, path: str) -> None:
        """
        Save the PyRuleAnalyzer to a file.
        
        Args:
            path: Path to save the PyRuleAnalyzer (.pkl file).
        
        Example:
            >>> analyzer.save("files/my_analyzer.pkl")
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    # ==========================================================================
    # INSPECTION
    # ==========================================================================
    
    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the classifier.
        
        Returns:
            Dictionary with classifier information:
            - model_type: Type of model
            - n_features: Number of features
            - n_classes: Number of classes
            - n_rules_initial: Number of rules before refinement
            - n_rules_final: Number of rules after refinement
            - feature_names: List of feature names
            - class_names: List of class names
        """
        return {
            "model_type": self.classifier.algorithm_type,
            "n_features": len(self.feature_names),
            "n_classes": len(self.class_names),
            "n_rules_initial": len(self.classifier.initial_rules),
            "n_rules_final": len(self.classifier.final_rules) if self.classifier.final_rules else len(self.classifier.initial_rules),
            "feature_names": self.feature_names,
            "class_names": self.class_names
        }

    def summary_report(self):
        """Generate a summary report dictionary for interactive use."""
        rules = self.classifier.final_rules if self.classifier.final_rules else self.classifier.initial_rules
        rules_list = []
        for r in rules:
            rules_list.append({
                'name': r.name,
                'conditions': r.conditions,
                'class': r.class_
            })
            
        return {
            "model_type": self.classifier.algorithm_type,
            "total_rules": len(rules),
            "classes": self.class_names,
            "rules": rules_list
        }

    def to_python(self, file_path: str, **kwargs):
        """Proxy to RuleClassifier.to_python()"""
        return self.classifier.to_python(file_path, **kwargs)

    def to_c_header(self, file_path: str, **kwargs):
        """Proxy to RuleClassifier.to_c_header()"""
        return self.classifier.to_c_header(file_path, **kwargs)

    def to_binary(self, file_path: str, **kwargs):
        """Proxy to RuleClassifier.to_binary()"""
        return self.classifier.to_binary(file_path, **kwargs)

    @classmethod
    def load_binary(cls, file_path: str):
        """Proxy to RuleClassifier.load_binary() wrapped in PyRuleAnalyzer"""
        classifier = RuleClassifier.load_binary(file_path)
        feature_names = classifier._array_feature_names if hasattr(classifier, '_array_feature_names') else []
        class_names = classifier.class_labels
        return cls(classifier, feature_names, class_names)

    def edit_rules(self):
        """Proxy to RuleClassifier.edit_rules()"""
        return self.classifier.edit_rules()
    
    def __repr__(self) -> str:
        summary = self.summary()
        return (f"PyRuleAnalyzer(model={summary['model_type']}, "
                f"features={summary['n_features']}, "
                f"classes={summary['n_classes']}, "
                f"rules={summary['n_rules_final']})")
