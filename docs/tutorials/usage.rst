Usage
=====


This tutorial demonstrates how to integrate the :ref:`RuleClassifier<rule_classifier>` into your own machine learning pipeline. We'll cover data preparation, model training, rule extraction, analysis, batch prediction, and export using Decision Tree, Random Forest, and Gradient Boosting Decision Trees classifiers.

.. _tutorials/usage#prerequisites:

Prerequisites
-------------

Ensure you have the following Python packages installed:

- `scikit-learn`
- `pandas`
- `numpy`

You can install them using pip:

.. code-block:: bash

    pip install scikit-learn pandas numpy

Or simply install the package (all dependencies are included):

.. code-block:: bash

    pip install pyruleanalyzer

You must also create the following folder structure in the directory you'll be executing your code:

.. code-block:: text

    your_project/
    ├──examples/
    │  └──files/
    └──your_python_code.py

Prepare Your Dataset
--------------------

Your dataset should be in CSV format with the following characteristics:

- Each row represents a single sample.
- The last column is the target class label.
- All other columns are feature values.
- All values and classes must be non-infinite numbers, so make sure to include an encoder in your pipeline if you have string data.

Example:

.. code-block:: text

    5,1,0,0,1
    1,1,2,0,0
    9,1,1,0,1

Split your dataset into training and testing sets. Here's how you can do it using pandas and scikit-learn:

.. code-block:: python

    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Load the dataset
    df = pd.read_csv("your_dataset.csv")

    # Split into features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y
    )

    # Save to CSV files without index
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)

Train a Model and Extract Rules
-------------------------------

Use the ``new_classifier`` method from :ref:`RuleClassifier<rule_classifier>` to train a model and extract rules. You can choose between a Decision Tree, Random Forest, or Gradient Boosting Decision Trees classifier.

Example with Decision Tree:

.. code-block:: python

    from pyruleanalyzer import RuleClassifier


    # Define model parameters for the sklearn model
    model_params = {"max_depth": 5}

    # Create a RuleClassifier instance
    classifier = RuleClassifier.new_classifier(
        train_path="train.csv",
        test_path="test.csv",
        model_parameters=model_params,
        algorithm_type="Decision Tree"
    )

Example with Random Forest:

.. code-block:: python

    model_params = {"n_estimators": 100, "max_depth": 5}

    classifier = RuleClassifier.new_classifier(
        train_path="train.csv",
        test_path="test.csv",
        model_parameters=model_params,
        algorithm_type="Random Forest"
    )

Example with Gradient Boosting Decision Trees:

.. code-block:: python

    model_params = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1}

    classifier = RuleClassifier.new_classifier(
        train_path="train.csv",
        test_path="test.csv",
        model_parameters=model_params,
        algorithm_type="Gradient Boosting Decision Trees"
    )

You can also load a previously trained sklearn model from a pickle file instead of training a new one, by providing the ``model_path`` parameter:

.. code-block:: python

    classifier = RuleClassifier.new_classifier(
        train_path="train.csv",
        test_path="test.csv",
        model_parameters={},
        model_path="path/to/your_model.pkl",
        algorithm_type="Decision Tree"
    )

This process will:

- Train (or load) the specified model.
- Extract decision rules from the trained model.
- Initialize a :ref:`RuleClassifier<rule_classifier>` instance with the extracted rules.

Analyze and Refine the Rules
----------------------------

After initializing the :ref:`RuleClassifier<rule_classifier>` instance, you can analyze and refine the extracted rules using the appropriate analyzer class (:ref:`DTAnalyzer<dt_analyzer>`, :ref:`RFAnalyzer<rf_analyzer>`, or :ref:`GBDTAnalyzer<gbdt_analyzer>`) based on the algorithm type.

.. code-block:: python

    from pyruleanalyzer import DTAnalyzer, RFAnalyzer, GBDTAnalyzer

    # For a Decision Tree classifier
    analyzer = DTAnalyzer(classifier)
    analyzer.execute_rule_analysis(
        file_path="test.csv",
        remove_low_usage=-1,
        save_final_model=True,
        save_report=True
    )

Parameters:

- ``file_path``: Path to the test dataset CSV file.
- ``remove_low_usage``: Remove rules used less than or equal to this number of times during classification. Use ``-1`` to disable this feature.
- ``save_final_model``: Whether to save the final refined model to ``files/final_model.pkl``.
- ``save_report``: Whether to save the analysis report to ``files/output_classifier_<type>.txt``.

This method will:

- Evaluate the rules on the test dataset.
- Iteratively remove duplicate rules until convergence.
- Optionally prune infrequently used rules (with sibling promotion to maintain coverage).
- Update the :ref:`RuleClassifier<rule_classifier>` instance with the refined rule set.

Parameters:

- ``file_path``: Path to the test dataset CSV file.
- ``remove_low_usage``: Remove rules used less than or equal to this number of times during classification. Use ``-1`` to disable this feature.
- ``save_final_model``: Whether to save the final refined model to ``files/final_model.pkl``.
- ``save_report``: Whether to save the analysis report to ``files/output_classifier_<type>.txt``.

This method will:

- Evaluate the rules on the test dataset.
- Iteratively remove duplicate rules until convergence.
- Optionally prune infrequently used rules (with sibling promotion to maintain coverage).
- Update the :ref:`RuleClassifier<rule_classifier>` instance with the refined rule set.

You can also use the analyzer classes directly for finer control:

.. code-block:: python

    from pyruleanalyzer import DTAnalyzer, RFAnalyzer, GBDTAnalyzer

    # For a Decision Tree classifier
    analyzer = DTAnalyzer(classifier)
    analyzer.execute_rule_analysis(
        file_path="test.csv",
        remove_below_n_classifications=1
    )

Make Predictions
----------------

Single-sample prediction
^^^^^^^^^^^^^^^^^^^^^^^^

Use the ``classify`` method to make predictions on individual samples. If your dataset didn't include a header row you must name your features as ``"v{column}"`` where ``column`` is the column index in the csv. If ``final`` is set to true the classifier will use the refined rule set to classify the sample.

.. code-block:: python

    sample = {"feature_1": 1, "feature_2": 23, "feature_4": 34, "feature_n": 654}
    predicted_class, votes, probabilities = classifier.classify(sample, final=True)

Returns:

- ``predicted_class``: The predicted class label.
- ``votes``: A list of votes from individual rules or trees (for Random Forest and GBDT).
- ``probabilities``: A list of class probabilities (for Random Forest; ``None`` for Decision Tree and GBDT).

Batch prediction (vectorized)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For high-performance prediction on large datasets, use the ``predict_batch`` method. This uses compiled numpy arrays and optional C acceleration for significantly faster inference compared to per-sample classification.

.. code-block:: python

    import numpy as np

    # X_test should be a numpy array of shape (n_samples, n_features)
    X_test = df_test.iloc[:, :-1].values.astype(np.float32)
    feature_names = list(df_test.columns[:-1])

    # Predict class labels for all samples at once
    predictions = classifier.predict_batch(X_test, feature_names=feature_names, use_final=True)

You can also obtain class probabilities:

.. code-block:: python

    # Returns array of shape (n_samples, n_classes)
    probabilities = classifier.predict_batch_proba(X_test, feature_names=feature_names)

The ``predict_batch`` method handles all three algorithm types transparently:

- **Decision Tree**: First-match rule traversal.
- **Random Forest**: Soft voting with averaged per-tree probability distributions.
- **GBDT**: Additive scoring with sigmoid (binary) or softmax (multiclass).

Compare Metrics
---------------

You can use the ``compare_initial_final_results`` method to generate useful metrics on both the original rule set and the final pruned one. This method logs accuracy, confusion matrices, divergent predictions, interpretability scores (SCS -- Structural Complexity Score), and other metrics.

.. code-block:: python

    classifier.compare_initial_final_results("test.csv")

Exporting the Classifier
-------------------------

The :ref:`RuleClassifier<rule_classifier>` supports three export formats for deploying the trained model outside of Python or for maximum inference speed:

Export to standalone Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generates a self-contained ``.py`` file with the decision logic as pure Python code:

.. code-block:: python

    classifier.export_to_native_python(
        feature_names=feature_names,
        filename="my_classifier.py"
    )

Export to binary
^^^^^^^^^^^^^^^^

Exports the compiled tree arrays to a compact binary file (PYRA format). This format can be loaded back efficiently:

.. code-block:: python

    # Export
    classifier.export_to_binary(filepath="model.bin")

    # Load back
    loaded = RuleClassifier.load_binary("model.bin")
    predictions = loaded.predict_batch(X_test, feature_names=feature_names)

Export to C header
^^^^^^^^^^^^^^^^^^

Exports the classifier as a standalone C header file (``.h``), suitable for embedded systems and microcontrollers (e.g. Arduino):

.. code-block:: python

    classifier.export_to_c_header(
        filepath="model.h",
        guard_name="MY_MODEL_H"
    )

The generated header includes all tree arrays as ``const`` data and a ``predict(const float *features)`` function.

Saving and Loading
------------------

The :ref:`RuleClassifier<rule_classifier>` can be serialized to a pickle file and loaded back later. The native compiled function is automatically rebuilt upon loading.

.. code-block:: python

    import pickle

    # Save
    with open("classifier.pkl", "wb") as f:
        pickle.dump(classifier, f)

    # Load
    loaded = RuleClassifier.load("classifier.pkl")

Editing
-------

You can also manually edit the final rules by calling the ``edit_rules()`` method in your classifier instance.

The program will spawn an interactive menu that allows you to edit the rules.

In the first screen you can select a rule for editing by typing its associated number or name. You may enter 'exit' to exit.

After selecting a rule, you'll be presented with its conditions. You can use 'a' to add a new condition, 'r' to remove and 'c' to change the predicted class. By entering 's' you'll save the changes and return to the previous menu.

New conditions are added as 'variable operator value', e.g.: "v5 > 10.5"

.. code-block:: python

    classifier.edit_rules()

During editing:

- Type the rule number or name to open it.
- Use:
    - :code:`a` -- add condition
    - :code:`r` -- remove condition
    - :code:`c` -- change class
    - :code:`s` -- save changes
- Type exit at the main prompt to finish editing.

This is useful if you want to refine the automatically extracted rules with domain knowledge.
