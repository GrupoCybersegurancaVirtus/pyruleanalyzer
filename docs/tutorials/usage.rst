Usage
=====


This tutorial demonstrates how to integrate the :ref:`RuleClassifier<rule_classifier>` into your own machine learning pipeline. We'll cover data preparation, model training, rule extraction, and analysis using both Decision Tree and Random Forest classifiers.

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
        X, y, test_size=0.25, stratify=True
    )

    # Save to CSV files without index
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)

Train a Model and Extract Rules
-------------------------------

Use the `new_classifier` method from :ref:`RuleClassifier<rule_classifier>` to train a model and extract rules. You can choose between a Decision Tree or Random Forest classifier.

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

This process will:

- Train the specified model on your training data.
- Extract decision rules from the trained model.
- Initialize a :ref:`RuleClassifier<rule_classifier>` instance with the extracted rules.

Analyze and Refine the Rules
----------------------------

After initializing the :ref:`RuleClassifier<rule_classifier>` instance, you can analyze and refine the extracted rules using the `execute_rule_analysis` method.

.. code-block:: python

    classifier.execute_rule_analysis(
        file_path="test.csv",
        remove_duplicates="soft",
        remove_below_n_classifications=1
    )

Parameters:

- `file_path`: Path to the test dataset CSV file.
- `remove_duplicates`: Strategy to remove duplicate rules. Options:

  - `"soft"`: Remove duplicates within the same tree.
  - `"hard"`: Remove duplicates across different trees.
  - `"custom"`: Use the custom function defined with `set_custom_rule_removal` for duplicate removal.
  - `"none"`: Do not remove duplicates.
- `remove_below_n_classifications`: Remove rules used less than or equal to this number of times during classification. Use -1 to disable this feature.

This method will:

- Evaluate the rules on the test dataset.
- Remove duplicate and infrequently used rules based on the specified parameters.
- Update the `RuleClassifier` instance with the refined rule set.

Make Predictions
----------------

Use the `classify` method to make predictions on new samples. If your dataset didn't include a header row you must name your features as “v{column}” where `column` is the column index in the csv. If `final` is set to true the classifier will use the refined rule set to classify the sample.

.. code-block:: python
    
    sample = {"feature_1": 1, "feature_2": 23, "feature_4": 34, ..., "feature_n": 654}
    predicted_class, votes, probabilities = classifier.classify(sample, final=True)

Returns:

- `predicted_class`: The predicted class label.
- `votes`: A list of votes from individual rules or trees (if using a random forest).
- `probabilities`: A dictionary of class probabilities (if using a random forest).

Compare Metrics
---------------

You can use the `compare_initial_final_results` method to generate useful metrics on both the original rule set and the final pruned one. This method logs accuracy, confusion matrices, divergent predictions, interpretability scores, and other metrics. The results are also saved on `examples/files/output_final_classifier_dt.txt` for the decision tree algorithm and `examples/files/output_final_classifier.txt` for the random forest.

.. code-block:: python
    
    classifier.compare_initial_final_results(test_path)

Editing
-------

You can also manually edit the final rules by calling the `edit_rules()` method in your classifier instance.

The program will spawn an interactive menu that allows you to edit the rules.

In the first screen you can select a rule for editing by typing its associated number or name. You may enter 'exit' to exit.

After selecting a rule, you'll be presented with its conditions. You can use 'a' to add a new condition, 'r' to remove and 'c' to change the predicted class. By entering 's' you'll save the changes and return to the previous menu.

New conditions are added as 'variable operator value', e.g.: "v5 > 10.5"