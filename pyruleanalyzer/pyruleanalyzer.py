"""
Simple API for pyruleanalyzer

This module provides a high-level, easy-to-use interface for the entire
classifier lifecycle: create, refine, predict, and export.

Example:
    from pyruleanalyzer import PyRuleAnalyzer
    
    # Create and refine the classifier in one step
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
    
    # Method to init.
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
    
    # Method to create.
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
    
    # Method to new model.
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
        
    # Method to build an analyzer from an already-trained scikit-learn model.
    @staticmethod
    def from_sklearn(
        model,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
    ) -> "PyRuleAnalyzer":
        """
        Wrap an already-fitted scikit-learn estimator and extract its rules.

        Use this when the model was trained elsewhere -- a saved estimator, a
        grid search result, someone else's pipeline -- and only the rule
        extraction, refinement, verification and export steps are needed.

        Args:
            model: A fitted ``DecisionTreeClassifier``, ``RandomForestClassifier``
                or ``GradientBoostingClassifier``.
            feature_names: Ordered feature names. Defaults to the estimator's
                ``feature_names_in_``, then to ``feature_0 ... feature_n``.
            class_names: Class labels. Defaults to the estimator's ``classes_``.

        Returns:
            PyRuleAnalyzer: An analyzer holding the extracted rules.

        Raises:
            ValueError: If the estimator type is not supported.

        Example:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> clf = RandomForestClassifier(n_estimators=10).fit(X, y)
            >>> analyzer = PyRuleAnalyzer.from_sklearn(clf, list(X.columns))
        """
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        if isinstance(model, GradientBoostingClassifier):
            algorithm_type = 'Gradient Boosting Decision Trees'
        elif isinstance(model, RandomForestClassifier):
            algorithm_type = 'Random Forest'
        elif isinstance(model, DecisionTreeClassifier):
            algorithm_type = 'Decision Tree'
        else:
            raise ValueError(
                f"Unsupported estimator {type(model).__name__}: expected a "
                "DecisionTreeClassifier, RandomForestClassifier or "
                "GradientBoostingClassifier."
            )

        if feature_names is None:
            names = getattr(model, 'feature_names_in_', None)
            if names is not None:
                feature_names = [str(n) for n in names]
            else:
                n_features = int(getattr(model, 'n_features_in_', 0))
                feature_names = [f"feature_{i}" for i in range(n_features)]
        feature_names = list(feature_names)

        if class_names is None:
            class_names = [str(c) for c in getattr(model, 'classes_', [])]
        class_names = [str(c) for c in class_names]

        if algorithm_type == 'Gradient Boosting Decision Trees':
            rules, init_scores, is_binary, gbdt_classes = RuleClassifier.get_gbdt_rules(
                model, feature_names, class_names
            )
            classifier = RuleClassifier(rules, algorithm_type=algorithm_type)
            classifier._gbdt_init_scores = init_scores
            classifier._gbdt_is_binary = is_binary
            classifier._gbdt_classes = gbdt_classes
        else:
            rules = RuleClassifier.get_tree_rules(
                model, feature_names, class_names, algorithm_type=algorithm_type
            )
            classifier = RuleClassifier(rules, algorithm_type=algorithm_type)

        classifier.class_labels = class_names
        classifier.num_classes = len(class_names)
        classifier._array_feature_names = feature_names
        # scikit-learn casts X to float32 before comparing it with the
        # thresholds; comparing in float64 routes a sample lying within one
        # float32 ULP above a threshold to the other child.
        classifier.input_dtype = 'float32'

        return PyRuleAnalyzer(classifier, feature_names, class_names)

    # Method to fit.
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
        
        # Train the new Scikit-Learn model
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
        
        # Extract rules from the freshly trained estimator
        trained = PyRuleAnalyzer.from_sklearn(model, self.feature_names,
                                              self.class_names)
        self.classifier = trained.classifier
        
        return self
    
    # Method to load.
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
    
    # Method to get default params.
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
    
    # Method to execute rule refinement.
    def execute_rule_refinement(
        self,
        test_path: str = None,
        X=None,
        y=None,
        remove_below_n_classifications: int = -1,
        save_final_model: bool = False,
        save_report: bool = False,
        refine_between_trees: bool = False,
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
            refine_between_trees: Also merge rules with identical conditions
                across the trees of a Gradient Boosting model (semantic
                redundancy; the merged rule carries the sum of their values,
                so the score is unchanged). Refused for Random Forest, where
                it would change the soft vote; no effect on a Decision Tree.
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
                refine_between_trees=refine_between_trees,
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
                refine_between_trees=refine_between_trees,
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
    
    # Method to compare initial final results.
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
    
    # Method to predict.
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
            
        # predict_batch compiles (once) and uses the arrays of the requested
        # rule set.
        return self.classifier.predict_batch(
            X,
            feature_names=self.feature_names,
            use_final=use_refined
        )
    
    # Method to predict proba.
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
            
        return self.classifier.predict_batch_proba(
            X,
            feature_names=self.feature_names,
            use_final=use_refined,
        )
    
    # ==========================================================================
    # EXPORT
    # ==========================================================================
    
    # Method to export.
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
        return self.classifier.export(
            base_name=base_name,
            formats=formats,
            feature_names=self.feature_names,
            use_final=use_refined,
        )

    # Method to export hcpn.
    def export_hcpn(
        self,
        base_name: str = "model",
        which: str = "both",
        sample=None,
        feature_names: Optional[List[str]] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, str]:
        """Generate CPN Tools ``.cpn`` HCPN models for the initial and/or final model.

        Converts the extracted rules into a Hierarchical Coloured Petri Net in
        the native CPN Tools 4.x format so the model can be opened, visualised
        and simulated in `CPN Tools <https://cpntools.org>`_. The mapping
        follows the proofs of correctness (Theorems 1 and 2) for Decision Tree,
        Gradient Boosting Decision Trees (binary and multiclass) and Random
        Forest models.

        Args:
            base_name: Base name for the output files (saved under ``files/`` if
                no path separator is given). Produces ``<base>_initial.cpn``
                and/or ``<base>_final.cpn``.
            which: Which model(s) to export: ``"initial"``, ``"final"`` or
                ``"both"`` (default). ``"final"`` requires that refinement has
                been run.
            sample: Optional example input sample (dict or sequence) used as the
                initial marking of the top-level input place so the generated
                net is immediately simulatable.
            feature_names: Ordered feature names. Defaults to the analyzer's
                feature names, then inference from the rules.
            model_name: Display name for the model. Defaults to ``base_name``.

        Returns:
            Dictionary mapping ``"initial"``/``"final"`` to the written paths.

        Example:
            >>> analyzer = PyRuleAnalyzer.create(train, test,
            ...     model="Gradient Boosting Decision Trees", refine=True)
            >>> analyzer.export_hcpn("gbdt", which="both", sample=X_test.iloc[0])
            {'initial': 'files/gbdt_initial.cpn', 'final': 'files/gbdt_final.cpn'}
        """
        from .cpn_tools_exporter import export_cpn_tools

        if "/" not in base_name and "\\" not in base_name:
            base_path = f"files/{base_name}"
        else:
            base_path = base_name

        fn = feature_names or self.feature_names
        name = model_name or os.path.basename(base_name)
        has_final = bool(self.classifier.final_rules)

        which = which.lower()
        results: Dict[str, str] = {}

        if which in ("initial", "both"):
            results["initial"] = export_cpn_tools(
                self.classifier, f"{base_path}_initial.cpn",
                rules=self.classifier.initial_rules, feature_names=fn,
                sample=sample, use_final=False, model_name=f"{name}_initial",
            )
            print(f"  [OK] CPN Tools (initial model): {results['initial']}")

        if which in ("final", "both"):
            if not has_final:
                if which == "final":
                    raise ValueError(
                        "No final (refined) rules available. Run "
                        "execute_rule_refinement() before exporting the final HCPN."
                    )
            else:
                results["final"] = export_cpn_tools(
                    self.classifier, f"{base_path}_final.cpn",
                    rules=self.classifier.final_rules, feature_names=fn,
                    sample=sample, use_final=True, model_name=f"{name}_final",
                )
                print(f"  [OK] CPN Tools (final model):   {results['final']}")

        return results


    # Method to model check.
    def model_check(
        self,
        which: str = "final",
        samples=None,
        cpn_path: Optional[str] = None,
        base_name: str = "model",
        max_nodes: int = 200_000,
        check_consistency: bool = True,
        test_samples=None,
        askctl: bool = False,
        verbose: bool = True,
        reduction: Optional[str] = None,
        reduction_options: Optional[Dict[str, Any]] = None,
    ):
        """Verify the generated HCPN model.

        Builds (or reuses) the ``.cpn`` model and verifies it with the four
        techniques of :mod:`pyruleanalyzer.model_checker`: CTL model checking on
        the occurrence graph (termination, absence of spurious deadlock,
        inevitability and recoverability of the decision, a single valid label,
        safeness, leaf determinism and selection, boosting-stage precedence,
        complete voting), SCC analysis (home marking), structural analysis of
        the leaf guards (disjointness for every input) and conformance testing
        of the net against this classifier.

        Running it on ``which="both"`` is the acceptance test of a refinement: a
        property that held before refinement and fails after it means the
        refinement broke the model, which accuracy alone does not reveal.

        Args:
            which: ``"initial"``, ``"final"`` or ``"both"``.
            samples: Samples to fix in the net's initial marking (a DataFrame,
                a list of rows, or a single row). Structural properties hold for
                every input by construction; more samples widen the evidence
                for the per-sample ones (valid label, prediction consistency).
            cpn_path: Existing ``.cpn`` to check instead of exporting a new one.
                Only valid together with ``which="initial"`` or ``"final"``.
            base_name: Base name used when exporting the ``.cpn`` models.
            max_nodes: Occurrence-graph budget; beyond it the state-space
                properties get no verdict.
            check_consistency: Also compare the class the net computes with this
                classifier's own prediction.
            test_samples: Inputs for the conformance test (defaults to
                ``samples``); one occurrence sequence each, so a whole test set
                is affordable.
            askctl: Write the matching ASK-CTL (SML) script next to each
                ``.cpn``, ready to paste into CPN Tools.
            verbose: Print the report of each verified model.
            reduction: ``"stubborn"`` decides each property on a stubborn-set
                reduced occurrence graph (partial-order reduction in CPNCheck)
                instead of the full one. Same verdicts; independent trees and
                boosting channels are no longer interleaved, so a forest of
                ``n`` trees needs about ``n`` markings instead of ``2**n``.
                No CPN Tools oracle values are produced in that mode.
                ``"sweep"``, ``"equivalence"`` and ``"symmetry"`` are CPNCheck's
                other reductions; the tree nets are neither timed nor
                symmetric, so ``"stubborn"`` is the one that helps here.
            reduction_options: Settings of the reduction, passed to CPNCheck:
                ``{"progress": ...}`` for ``"sweep"``, ``{"equivalence":
                ...}`` or ``{"symmetry": ...}``.

        Returns:
            ModelCheckResult: For ``which="initial"`` or ``"final"``.
            Dict[str, ModelCheckResult]: For ``which="both"``, keyed by stage.

        Example:
            >>> analyzer.execute_rule_refinement(X=X_test, y=y_test)
            >>> results = analyzer.model_check(which="both", samples=X_test[:5])
            >>> results["final"].passed
            True
        """
        from .model_checker import CPNModelChecker, compare_results

        which = which.lower()
        stages = ["initial", "final"] if which == "both" else [which]

        paths: Dict[str, str] = {}
        if cpn_path is not None:
            if which == "both":
                raise ValueError("cpn_path cannot be combined with which='both'")
            paths[which] = cpn_path
        else:
            first = samples
            if hasattr(samples, "iloc"):
                first = samples.iloc[0]
            elif isinstance(samples, (list, tuple)) and samples and \
                    not isinstance(samples[0], (int, float)):
                first = samples[0]
            paths = self.export_hcpn(base_name=base_name, which=which,
                                     sample=first)

        results: Dict[str, Any] = {}
        for stage in stages:
            path = paths.get(stage)
            if path is None:
                continue
            try:
                labels = [int(c) for c in self.class_names]
            except (TypeError, ValueError):
                labels = None
            checker = CPNModelChecker(path, feature_names=self.feature_names,
                                      class_labels=labels, max_nodes=max_nodes)
            result = checker.check(
                samples=samples,
                classifier=self.classifier if check_consistency else None,
                use_final=(stage == "final"), test_samples=test_samples,
                verbose=verbose, reduction=reduction,
                **(reduction_options or {}),
            )
            if askctl:
                checker.export_askctl(os.path.splitext(path)[0] + ".sml")
            if verbose:
                print(result.report())
            results[stage] = result

        if which == "both":
            if verbose and len(results) == 2:
                print()
                print(compare_results(results["initial"], results["final"]))
            return results
        return results.get(stages[0])

    # ==========================================================================
    # SAVE/LOAD
    # ==========================================================================
    
    # Method to save.
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
    
    # Method to summary.
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

    # Method to summary report.
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

    # Method to to python.
    def to_python(self, file_path: str, **kwargs):
        """Proxy to RuleClassifier.to_python()"""
        return self.classifier.to_python(file_path, **kwargs)

    # Method to to c header.
    def to_c_header(self, file_path: str, **kwargs):
        """Proxy to RuleClassifier.to_c_header()"""
        return self.classifier.to_c_header(file_path, **kwargs)

    # Method to to binary.
    def to_binary(self, file_path: str, **kwargs):
        """Proxy to RuleClassifier.to_binary()"""
        return self.classifier.to_binary(file_path, **kwargs)

    # Method to load binary.
    @classmethod
    def load_binary(cls, file_path: str):
        """Proxy to RuleClassifier.load_binary() wrapped in PyRuleAnalyzer"""
        classifier = RuleClassifier.load_binary(file_path)
        feature_names = classifier._array_feature_names if hasattr(classifier, '_array_feature_names') else []
        class_names = classifier.class_labels
        return cls(classifier, feature_names, class_names)

    # Method to edit rules.
    def edit_rules(self):
        """Proxy to RuleClassifier.edit_rules()"""
        return self.classifier.edit_rules()
    
    # Method to repr.
    def __repr__(self) -> str:
        """
        Method to repr.
        
        Args:
            None
            
        Returns:
            Any: Result of the operation.
        """
        summary = self.summary()
        return (f"PyRuleAnalyzer(model={summary['model_type']}, "
                f"features={summary['n_features']}, "
                f"classes={summary['n_classes']}, "
                f"rules={summary['n_rules_final']})")
