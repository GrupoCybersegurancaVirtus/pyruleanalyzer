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

from sklearn.base import BaseEstimator, ClassifierMixin

from .rule_classifier import RuleClassifier


# Function to return the scikit-learn estimator class of a model type.
def _sklearn_estimator_class(algorithm_type: str, estimator: Optional[str] = None):
    """The scikit-learn classifier trained for ``algorithm_type``.

    Args:
        algorithm_type: "Decision Tree", "Random Forest" or
            "Gradient Boosting Decision Trees".
        estimator: Name of the scikit-learn class to train instead of the
            default one of the model type: "ExtraTreesClassifier" for a
            "Random Forest", "ExtraTreeClassifier" for a "Decision Tree".
            None (default) for the default class.

    Returns:
        type: The scikit-learn estimator class.

    Raises:
        ValueError: If the model type is not supported, or `estimator` is
            not a class of that type.
    """
    from sklearn.ensemble import (ExtraTreesClassifier,
                                  GradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

    classes = {
        'Decision Tree': (DecisionTreeClassifier, ExtraTreeClassifier),
        'Random Forest': (RandomForestClassifier, ExtraTreesClassifier),
        'Gradient Boosting Decision Trees': (GradientBoostingClassifier,),
    }
    if algorithm_type not in classes:
        raise ValueError(f"Unsupported algorithm type: {algorithm_type}")
    if estimator is None:
        return classes[algorithm_type][0]
    for cls in classes[algorithm_type]:
        if cls.__name__ == estimator:
            return cls
    raise ValueError(
        f"Unsupported estimator {estimator!r} for model {algorithm_type!r}: "
        f"expected one of {[c.__name__ for c in classes[algorithm_type]]}."
    )


# Function to name the estimator class of a fitted model, if not the default.
def _estimator_name(algorithm_type: str, model) -> Optional[str]:
    """The class name to remember for ``model``, None for the default class.

    Args:
        algorithm_type: The model type ``model`` was wrapped as.
        model: The fitted scikit-learn estimator.

    Returns:
        str | None: "ExtraTreesClassifier" / "ExtraTreeClassifier", or None
        when a refit with the default class trains the same kind of model.
    """
    default = _sklearn_estimator_class(algorithm_type)
    for cls in type(model).__mro__:
        if cls is default:
            return None
        try:
            _sklearn_estimator_class(algorithm_type, cls.__name__)
            return cls.__name__
        except ValueError:
            continue
    return None


class PyRuleAnalyzer(ClassifierMixin, BaseEstimator):
    """
    High-level interface for creating, refining, and deploying classifiers.

    This class simplifies the pyruleanalyzer workflow into a few intuitive
    methods, handling all the complexity internally.

    It is a scikit-learn classifier: ``get_params``/``set_params`` expose the
    model type (``model``) and the hyperparameters of the underlying
    scikit-learn estimator, so ``sklearn.base.clone``, ``Pipeline``,
    ``cross_val_score`` and ``GridSearchCV`` work with the
    ``new_model(...).fit(X, y)`` workflow. ``fit`` only trains the model and
    extracts its rules; refinement (``execute_rule_refinement``) is a separate
    step and is not part of a cross-validation or grid search.

    Attributes:
        classifier (RuleClassifier): The underlying RuleClassifier instance.
        feature_names (list): Names of features used by the model.
        class_names (list): Names of target classes.
        classes_ (np.ndarray): The class labels, as returned by ``predict``
            (fitted analyzers only). Labels that are not integers are
            encoded internally (``classifier.label_table``).

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
    def __init__(
        self,
        classifier: Optional[RuleClassifier] = None,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        *,
        model: Optional[str] = None,
        estimator: Optional[str] = None,
        **params,
    ):
        """
        Initialize a PyRuleAnalyzer with an existing RuleClassifier, or an
        untrained one from a model type and its hyperparameters.

        ``PyRuleAnalyzer(model=..., **params)`` is the form
        ``sklearn.base.clone`` uses (``get_params`` returns ``model`` and the
        hyperparameters); ``new_model`` is the documented way to build it.

        Args:
            classifier: A trained RuleClassifier instance. None builds an
                untrained analyzer from ``model`` and ``params``.
            feature_names: List of feature names.
            class_names: List of class names.
            model: Model type of an untrained analyzer - "Decision Tree"
                (default), "Random Forest" or "Gradient Boosting Decision Trees".
            estimator: scikit-learn class an untrained analyzer trains when
                it is not the default of ``model``: "ExtraTreesClassifier"
                (a "Random Forest") or "ExtraTreeClassifier" (a "Decision
                Tree"). ``get_params`` reports it for an analyzer wrapping
                such a model, so ``clone`` then ``fit`` trains it again.
            **params: Hyperparameters of the scikit-learn estimator of an
                untrained analyzer, passed as they are.

        Raises:
            TypeError: If ``model``, ``estimator`` or hyperparameters are
                given together with ``classifier``.
        """
        if classifier is None:
            algorithm_type = model or "Decision Tree"
            if estimator is not None:
                _sklearn_estimator_class(algorithm_type, estimator)  # validate
            classifier = RuleClassifier([], algorithm_type=algorithm_type)
            # fit() trains the scikit-learn estimator with these.
            classifier.model_parameters = params
            classifier.sklearn_estimator = estimator
        elif model is not None or estimator is not None or params:
            raise TypeError(
                "model, estimator and hyperparameters only apply to an "
                "untrained analyzer (classifier=None); use "
                "PyRuleAnalyzer.new_model()."
            )
        self.classifier = classifier
        self.feature_names = feature_names if feature_names is not None else []
        self.class_names = class_names if class_names is not None else []

    # ==========================================================================
    # SCIKIT-LEARN ESTIMATOR API
    # ==========================================================================

    # Method to get params.
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Parameters of this estimator, as scikit-learn expects them.

        Args:
            deep: Ignored (the analyzer holds no nested estimator).

        Returns:
            Dictionary with ``model`` (the model type), ``estimator`` when
            the wrapped scikit-learn class is not the default of the model
            type (e.g. "ExtraTreesClassifier"), and the hyperparameters of
            the scikit-learn estimator.
        """
        params = getattr(self.classifier, 'model_parameters', None) or {}
        out = {"model": self.classifier.algorithm_type}
        estimator = getattr(self.classifier, 'sklearn_estimator', None)
        if estimator is not None:
            out["estimator"] = estimator
        return {**out, **params}

    # Method to set params.
    def set_params(self, **params) -> "PyRuleAnalyzer":
        """
        Set the model type and/or hyperparameters; they apply at the next fit.

        Changing ``model`` replaces the classifier with an untrained one of
        that type (keeping the hyperparameters), which trains the default
        class of that type unless ``estimator`` is given too.

        Args:
            **params: ``model``, ``estimator`` and/or hyperparameters of the
                scikit-learn estimator.

        Returns:
            self

        Raises:
            ValueError: If a hyperparameter is not one of the estimator's,
                or ``estimator`` is not a class of the model type.
        """
        current = self.get_params()
        model = params.pop("model", current.pop("model"))
        estimator = current.pop("estimator", None)
        if model != self.classifier.algorithm_type:
            estimator = None   # a new model type trains its default class
        estimator = params.pop("estimator", estimator)
        valid = _sklearn_estimator_class(model, estimator)().get_params(deep=False)
        invalid = [k for k in params if k not in valid]
        if invalid:
            raise ValueError(
                f"Invalid parameter(s) {invalid} for estimator "
                f"{type(self).__name__} with model={model!r}. Valid "
                f"parameters are: {['model', 'estimator'] + sorted(valid)}."
            )
        current.update(params)
        if model != self.classifier.algorithm_type:
            self.classifier = RuleClassifier([], algorithm_type=model)
        self.classifier.model_parameters = current
        self.classifier.sklearn_estimator = estimator
        return self

    # Method to return the integer class labels.
    @property
    def classes_(self) -> np.ndarray:
        """The class labels of a fitted analyzer, as ``predict`` returns them.

        Integer labels as integers; other labels ('cat', 0.5) as the
        estimator had them (the rules work on their codes, see
        ``RuleClassifier.label_table``).
        """
        if not self.class_names:
            raise AttributeError(
                "This PyRuleAnalyzer is not fitted yet; it has no classes_."
            )
        table = getattr(self.classifier, 'label_table', None)
        if table is not None:
            return np.asarray(table)
        try:
            return np.asarray([int(c) for c in self.class_names])
        except (TypeError, ValueError):
            return np.asarray(self.class_names)

    # Method to tell scikit-learn whether the analyzer is fitted.
    def __sklearn_is_fitted__(self) -> bool:
        """True once the analyzer holds extracted rules."""
        return bool(self.classifier.initial_rules)

    # Method to score.
    def score(self, X, y, sample_weight=None) -> float:
        """
        Mean accuracy of ``predict(X)`` (refined rules, if any) on ``y``.

        Args:
            X: Test samples (DataFrame or 2-D array).
            y: True labels.
            sample_weight: Optional sample weights.

        Returns:
            float: The accuracy.
        """
        from sklearn.metrics import accuracy_score
        y_true, y_pred = np.asarray(y), np.asarray(self.predict(X))
        if getattr(self.classifier, 'label_table', None) is not None:
            # Compared as codes: labels such as 0.5 are not a classification
            # target type scikit-learn's metrics accept.
            y_true = self.classifier._encode_labels(y_true)
            y_pred = self.classifier._encode_labels(y_pred)
        return accuracy_score(y_true, y_pred, sample_weight=sample_weight)

    # ==========================================================================
    # FACTORY METHODS
    # ==========================================================================

    # Method to read the feature names of a CSV file.
    @staticmethod
    def _csv_feature_names(path: str) -> List[str]:
        """The feature columns of a CSV, named as RuleClassifier.process_data names them.

        A first line with letters is a header (the last column is the
        target); otherwise the features are ``v1 ... vn``.

        Args:
            path: Path to the CSV file.

        Returns:
            list: The feature names, in column order.
        """
        with open(path, 'r', encoding='latin-1') as f:
            first_line = f.readline().strip()
        if any(c.isalpha() for c in first_line):
            columns = pd.read_csv(path, nrows=0, encoding='latin-1').columns
            return [str(c).strip() for c in columns][:-1]
        n_columns = pd.read_csv(path, header=None, nrows=1, encoding='latin-1').shape[1]
        return [f'v{i+1}' for i in range(n_columns - 1)]

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
                         See PyRuleAnalyzer.execute_rule_refinement().
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
        
        # get_params() reports the hyperparameters the model was trained with.
        classifier.model_parameters = params

        # Extract feature names and class names. The rules name the features
        # after the training CSV's columns; their order is the CSV's, which
        # is the column order of an array passed to predict().
        feature_names = list(getattr(classifier, '_array_feature_names', None) or [])
        if not feature_names:
            feature_names = PyRuleAnalyzer._csv_feature_names(train_path)
            used = {var for rule in classifier.initial_rules
                    for var, _, _ in rule.parsed_conditions}
            if not used.issubset(feature_names):
                feature_names = []
        class_names = PyRuleAnalyzer._class_names_of(classifier)
        
        # Create PyRuleAnalyzer instance
        analyzer = PyRuleAnalyzer(classifier, feature_names, class_names)
        
        # Optionally refine
        if refine:
            if refine_params is None:
                refine_params = {}
            analyzer.execute_rule_refinement(test_path=test_path, **refine_params)
        
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

        # An empty RuleClassifier carrying the hyperparameters, so fit()
        # knows how to initialize the underlying sklearn model.
        return PyRuleAnalyzer(model=model, **params)
        
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

        An ``ExtraTreesClassifier`` is wrapped as a "Random Forest": once
        fitted it is the same thing (``estimators_`` of classification trees
        whose ``predict_proba`` is averaged); randomized splits and
        ``bootstrap`` only change how it is trained. The analyzer remembers
        the class (``get_params()["estimator"]``), so refitting it (``fit``,
        or ``clone`` then ``fit``) trains an ``ExtraTreesClassifier`` again;
        likewise for an ``ExtraTreeClassifier`` wrapped as a "Decision Tree".

        Labels that are not integers ('cat'/'dog', 0.5) are supported: the
        rules and every engine work on their codes (position in
        ``classes_``), and ``predict``, ``classes_``, ``score`` and the
        Python export answer the original labels; the C, Arduino and binary
        exports answer the codes and document the table.

        Missing values (NaN) are routed at every split as scikit-learn does
        (``tree_.missing_go_to_left``), by every engine and export, for
        decision trees and forests. ``GradientBoostingClassifier`` refuses
        NaN, and so does a model extracted from it.

        Args:
            model: A fitted ``DecisionTreeClassifier`` (or ``ExtraTreeClassifier``),
                ``RandomForestClassifier``, ``ExtraTreesClassifier`` or
                ``GradientBoostingClassifier``.
            feature_names: Ordered feature names. Defaults to the estimator's
                ``feature_names_in_``, then to ``feature_0 ... feature_n``.
            class_names: Class labels, in ``classes_`` order. Defaults to the
                estimator's ``classes_``.

        Returns:
            PyRuleAnalyzer: An analyzer holding the extracted rules.

        Raises:
            ValueError: If the estimator type is not supported.

        Example:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> clf = RandomForestClassifier(n_estimators=10).fit(X, y)
            >>> analyzer = PyRuleAnalyzer.from_sklearn(clf, list(X.columns))
        """
        from sklearn.ensemble import (ExtraTreesClassifier,
                                      GradientBoostingClassifier,
                                      RandomForestClassifier)
        from sklearn.tree import DecisionTreeClassifier

        if isinstance(model, GradientBoostingClassifier):
            algorithm_type = 'Gradient Boosting Decision Trees'
        elif isinstance(model, (RandomForestClassifier, ExtraTreesClassifier)):
            # Not a subclass of RandomForestClassifier, but fitted the same
            # way: a soft vote over DecisionTreeClassifier-like estimators_.
            algorithm_type = 'Random Forest'
        elif isinstance(model, DecisionTreeClassifier):
            algorithm_type = 'Decision Tree'
        else:
            raise ValueError(
                f"Unsupported estimator {type(model).__name__}: expected a "
                "DecisionTreeClassifier, RandomForestClassifier, "
                "ExtraTreesClassifier or GradientBoostingClassifier."
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
            class_names = list(getattr(model, 'classes_', []))
        # Integer labels are kept; any other labels are replaced by their
        # codes 0..K-1 (classes_ order) and remembered in a table.
        label_table = RuleClassifier._label_table_for(class_names)
        class_names = RuleClassifier._label_names(class_names)

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
        classifier.label_table = label_table
        classifier._array_feature_names = feature_names
        # get_params() (and so clone) reproduces the estimator's settings,
        # and its class (ExtraTrees is wrapped as a 'Random Forest').
        classifier.model_parameters = model.get_params(deep=False)
        classifier.sklearn_estimator = _estimator_name(algorithm_type, model)
        # scikit-learn casts X to float32 before comparing it with the
        # thresholds; comparing in float64 routes a sample lying within one
        # float32 ULP above a threshold to the other child.
        classifier.input_dtype = 'float32'
        # The constructor compiled the native function before the labels
        # above were known; a forest maps its argmax through them.
        if algorithm_type == 'Random Forest':
            classifier.update_native_model(classifier.initial_rules)

        return PyRuleAnalyzer(classifier, feature_names,
                              PyRuleAnalyzer._class_names_of(classifier))

    # Method to give the class names an analyzer shows for a classifier.
    @staticmethod
    def _class_names_of(classifier: RuleClassifier) -> list:
        """The labels ``predict`` answers: the label table, else the labels.

        Args:
            classifier: The RuleClassifier.

        Returns:
            list: Original labels when the classifier encodes them, else its
            ``class_labels``.
        """
        table = getattr(classifier, 'label_table', None)
        if table is not None:
            return list(table)
        return list(classifier.class_labels)

    # Method to fit.
    def fit(self, X, y) -> "PyRuleAnalyzer":
        """
        Fit the model according to the given training data.

        Trains the scikit-learn class the analyzer wraps (the default of the
        model type, or the one ``get_params()["estimator"]`` names, e.g. an
        ``ExtraTreesClassifier``) and extracts its rules. ``y`` may hold any
        labels scikit-learn accepts (strings too), and ``X`` missing values
        where the estimator accepts them (decision trees and forests).

        Args:
            X: Training vectors (e.g. pandas DataFrame or numpy array).
            y: Target values.

        Returns:
            self
        """
        params = getattr(self.classifier, 'model_parameters', None)
        if params is None:
            params = self._get_default_params(self.classifier.algorithm_type)
        algorithm_type = self.classifier.algorithm_type
        estimator = getattr(self.classifier, 'sklearn_estimator', None)

        # Train the new Scikit-Learn model
        model = _sklearn_estimator_class(algorithm_type, estimator)(**params)
        model.fit(X, y)
        
        # Determine feature names and class names
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
        # Extract rules from the freshly trained estimator; the class names
        # come from its classes_ (integer labels, see from_sklearn).
        trained = PyRuleAnalyzer.from_sklearn(model, self.feature_names)
        # Keep the parameters as they were set, for get_params().
        trained.classifier.model_parameters = params
        self.classifier = trained.classifier
        self.class_names = trained.class_names
        
        return self
    
    # Method to load.
    @staticmethod
    def load(path: str) -> "PyRuleAnalyzer":
        """
        Load a PyRuleAnalyzer from a saved file.

        A path ending in ``.json`` is read with :meth:`load_json`, which is
        safe on untrusted files. Any other path is unpickled: **loading a
        pickle executes code, so only load .pkl files you trust.**

        Args:
            path: Path to a saved .json or .pkl file.

        Returns:
            PyRuleAnalyzer: The loaded PyRuleAnalyzer instance.

        Example:
            >>> analyzer = PyRuleAnalyzer.load("files/my_analyzer.json")
        """
        if str(path).lower().endswith('.json'):
            return PyRuleAnalyzer.load_json(path)
        loaded = RuleClassifier.load(path)
        # save() pickles the analyzer itself; a bare RuleClassifier (written
        # by the analyzers' save_final_model, for instance) is wrapped.
        if isinstance(loaded, PyRuleAnalyzer):
            return loaded
        classifier = loaded
        feature_names = classifier._array_feature_names if hasattr(classifier, '_array_feature_names') else []
        class_names = PyRuleAnalyzer._class_names_of(classifier)
        return PyRuleAnalyzer(classifier, feature_names, class_names)

    # Identification of the analyzer's JSON format (see save_json)
    _JSON_FORMAT = 'pyruleanalyzer.PyRuleAnalyzer'
    _JSON_VERSION = 1

    # Method to load an analyzer saved with save_json.
    @staticmethod
    def load_json(path: str) -> "PyRuleAnalyzer":
        """
        Load a PyRuleAnalyzer saved with :meth:`save_json`.

        Safe on untrusted files: the file is parsed as JSON and every field is
        validated; nothing in it is imported or executed. A file written by
        ``RuleClassifier.to_json`` is accepted too and wrapped.

        Args:
            path: Path to the .json file.

        Returns:
            PyRuleAnalyzer: The loaded analyzer, ready to predict, refine,
            edit, model-check and export.

        Raises:
            ValueError: If the file is not a valid JSON model.

        Example:
            >>> analyzer = PyRuleAnalyzer.load_json("files/my_analyzer.json")
        """
        data = RuleClassifier._json_read_file(path)
        if isinstance(data, dict) and data.get('format') == RuleClassifier._JSON_FORMAT:
            classifier = RuleClassifier.from_dict(data)
            return PyRuleAnalyzer(classifier, list(classifier._array_feature_names),
                                  PyRuleAnalyzer._class_names_of(classifier))
        if not isinstance(data, dict) or data.get('format') != PyRuleAnalyzer._JSON_FORMAT:
            raise ValueError(
                'Invalid JSON model: expected an object with "format": '
                f'{PyRuleAnalyzer._JSON_FORMAT!r}.'
            )
        version = data.get('version')
        if isinstance(version, bool) or version != PyRuleAnalyzer._JSON_VERSION:
            raise ValueError(
                f'Unsupported JSON model version {version!r:.40} (this version of '
                f'pyruleanalyzer reads version {PyRuleAnalyzer._JSON_VERSION}).'
            )
        keys = {'format', 'version', 'feature_names', 'class_names', 'classifier'}
        if set(data) != keys:
            raise ValueError(
                f'Invalid JSON model: expected the fields {sorted(keys)}, got '
                f'{sorted(data)[:10]}.'
            )
        names = {}
        for key in ('feature_names', 'class_names'):
            if not isinstance(data[key], list):
                raise ValueError(f'Invalid JSON model: {key} must be a list.')
            names[key] = list(data[key])
            for v in names[key]:
                if isinstance(v, bool) or not isinstance(v, (str, int)):
                    raise ValueError(
                        f'Invalid JSON model: {key} must hold strings or integers.'
                    )
        classifier = RuleClassifier.from_dict(data['classifier'])
        class_names = names['class_names']
        if getattr(classifier, 'label_table', None) is not None:
            # The classifier's table keeps the labels' types (0.5 stays a
            # float; class_names is written as strings and integers).
            class_names = list(classifier.label_table)
        return PyRuleAnalyzer(classifier, names['feature_names'], class_names)

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
        max_error_increase: Optional[int] = None,
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
            max_error_increase: Error-aware refinement: remove a low-usage
                rule only while the refinement-set errors stay within this
                many samples of those before the stage (0 = never more
                errors). None (default) = usage only.

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
                max_error_increase=max_error_increase,
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
                max_error_increase=max_error_increase,
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
                max_error_increase=max_error_increase,
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
        
        # The engines answer integer labels (codes, for labels that are not
        # integers); translate them back to the estimator's labels.
        decode = self.classifier._decode_labels

        # Single sample prediction (dict)
        if isinstance(X, dict):
            predicted_class, _, _ = self.classifier.classify(X, final=use_refined)
            return decode(predicted_class)

        # Single sample prediction (Pandas Series)
        if isinstance(X, pd.Series):
            predicted_class, _, _ = self.classifier.classify(X.to_dict(), final=use_refined)
            return decode(predicted_class)

        # Batch prediction. predict_batch compiles (once) and uses the arrays
        # of the requested rule set.
        X, names = self._batch_input(X)
        return decode(self.classifier.predict_batch(
            X,
            feature_names=names,
            use_final=use_refined
        ))

    # Method to turn a batch into an array and the names of its columns.
    def _batch_input(self, X):
        """The batch as an array, with the feature name of each column.

        A DataFrame is matched to the model by column name, not position: its
        columns may come in another order, or include extra ones.

        Args:
            X: A DataFrame or a 2-D array (columns in training order).

        Returns:
            tuple: (array, feature names of its columns).

        Raises:
            ValueError: If a DataFrame lacks a feature the model uses.
        """
        if not isinstance(X, pd.DataFrame):
            # No names (an analyzer loaded from a binary model, say): columns
            # are taken in the order the arrays were compiled with.
            return X, self.feature_names or None
        names = list(X.columns)
        missing = [f for f in self.feature_names if f not in names]
        if missing:
            raise ValueError(f"X is missing the feature(s) {missing}")
        return X.values, names
    
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
        X, names = self._batch_input(X)
        return self.classifier.predict_batch_proba(
            X,
            feature_names=names,
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

        os.makedirs(os.path.dirname(base_path) or '.', exist_ok=True)

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
            # The net answers the integer labels the rules hold (codes, for
            # labels that are not integers), so the label domain is theirs.
            try:
                labels = [int(c) for c in self.classifier.class_labels]
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

        A path ending in ``.json`` is written with :meth:`save_json`, the
        safe format; any other path is pickled (kept for backward
        compatibility: loading a pickle executes code, so only share or load
        .pkl files you trust).

        Args:
            path: Path to save the PyRuleAnalyzer (.json or .pkl file).

        Example:
            >>> analyzer.save("files/my_analyzer.json")
        """
        if str(path).lower().endswith('.json'):
            return self.save_json(path)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    # Method to save as JSON.
    def save_json(self, path: str, indent: Optional[int] = None) -> None:
        """
        Save the PyRuleAnalyzer to a JSON file -- the safe format.

        Unlike a pickle, the file holds only data, so :meth:`load_json` can
        read one from an untrusted source without executing code. Unlike the
        ``.bin`` export it keeps the rules (initial, refined, their usage and
        error counts, leaf values and distributions), the GBDT metadata, the
        feature and class names and the hyperparameters, so the loaded
        analyzer can be refined again, edited, model-checked and exported.
        Thresholds read back bit for bit.

        The file is an object with ``"format": "pyruleanalyzer.PyRuleAnalyzer"``,
        ``"version": 1``, ``"feature_names"``, ``"class_names"`` and
        ``"classifier"`` (the ``RuleClassifier.to_dict`` format).

        Args:
            path: Path to the .json file.
            indent: Indentation for a human-readable file. Default None
                (compact).

        Raises:
            ValueError: If a value cannot be represented in JSON (an object
                passed as a hyperparameter, a classifier with no rules).

        Example:
            >>> analyzer.save_json("files/my_analyzer.json")
        """
        import json
        label = RuleClassifier._json_label_out
        data = {
            'format': PyRuleAnalyzer._JSON_FORMAT,
            'version': PyRuleAnalyzer._JSON_VERSION,
            'feature_names': [label(f) for f in self.feature_names],
            'class_names': [label(c) for c in self.class_names],
            'classifier': self.classifier.to_dict(),
        }
        text = json.dumps(data, allow_nan=False, indent=indent)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)

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
        """Proxy to RuleClassifier.to_python(), writing to ``file_path``."""
        kwargs.setdefault('feature_names', self.feature_names or None)
        return self.classifier.to_python(filename=file_path, **kwargs)

    # Method to run an array-based export on the refined rules, or the initial ones.
    def _export_arrays(self, writer, file_path: str, **kwargs):
        """Compile the default rule set for the duration of ``writer``.

        The binary and C exports serialise the arrays compiled last, which can
        be the other rule set; like the Python export, they should write the
        refined rules when there are any.

        Args:
            writer: The classifier's export method.
            file_path: Where to write.
            **kwargs: Passed to ``writer``.

        Returns:
            Whatever ``writer`` returns.
        """
        clf = self.classifier
        rules = clf.final_rules if clf.final_rules else clf.initial_rules
        saved = clf._current_array_state() if clf._arrays_compiled else None
        names = self.feature_names or None
        clf._set_array_state(clf._array_state_for_rules(rules, names))
        try:
            return writer(file_path, **kwargs)
        finally:
            if saved is not None:
                clf._set_array_state(saved)

    # Method to to c header.
    def to_c_header(self, file_path: str, **kwargs):
        """Write the C header of the refined (else initial) rules."""
        return self._export_arrays(self.classifier.to_c_header, file_path,
                                   **kwargs)

    # Method to to binary.
    def to_binary(self, file_path: str, **kwargs):
        """Write the binary model of the refined (else initial) rules."""
        return self._export_arrays(self.classifier.to_binary, file_path,
                                   **kwargs)

    # Method to load binary.
    @classmethod
    def load_binary(cls, file_path: str):
        """Proxy to RuleClassifier.load_binary() wrapped in PyRuleAnalyzer"""
        classifier = RuleClassifier.load_binary(file_path)
        feature_names = classifier._array_feature_names if hasattr(classifier, '_array_feature_names') else []
        class_names = cls._class_names_of(classifier)
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
