COVID-19
========

This tutorial will guide you as we train and refine a decision tree model using the `Brazilian dataset of symptomatic patients for screening the risk of COVID-19 <https://data.mendeley.com/datasets/b7zcgmmwx4/5>`_ repository.

Prerequisites
-------------

For this example we will be using the ``rapid_balanced.csv`` dataset, so make sure to download it from the repository above before continuing.

Next, install the required packages.

.. code-block:: bash

    pip install pyruleanalyzer pandas scikit-learn

Prepare Your Dataset
--------------------

As specified in the :doc:`usage guide<../usage>`, the :ref:`RuleClassifier<rule_classifier>` integrates directly with Scikit-Learn pipelines.

We can use the following script to load and split the data. Be sure to adapt it to your current pipeline as needed:

.. code-block:: python

    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Load the dataset
    df = pd.read_csv("rapid_balanced.csv")

    # Split into features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y
    )

Training the Tree and Extracting Its Rules
---------------------------------------------------

The :ref:`RuleClassifier<rule_classifier>` behaves as a standard scikit-learn model, but with built-in method overloading that lets you use either standard DataFrames/NumPy arrays or strings representing CSV file paths.

.. code-block:: python

    from pyruleanalyzer import RuleClassifier

    # Create a RuleClassifier instance
    classifier = RuleClassifier(algorithm_type="Decision Tree")
    
    # Method 1: Train the internal model with in-memory DataFrames
    classifier.fit(X_train, y_train)

    # Method 2: Train using the CSV paths directly
    classifier.fit("train.csv", "test.csv")

Refinement
-------

With the :ref:`RuleClassifier<rule_classifier>` instance in hands, we can now execute a rule analysis using the :ref:`DTAnalyzer<dt_analyzer>` class, which will refine the tree by removing duplicate rules.

.. code-block:: python

    from pyruleanalyzer import DTAnalyzer

    analyzer = DTAnalyzer(classifier)
    
    # We can pass the validation dataframe directly or pass a test.csv path
    # To use our split data: 
    # pd.concat([X_test, y_test], axis=1).to_csv("test.csv", index=False)
    
    analyzer.execute_rule_analysis(
        file_path="test.csv", # Or use validation dataframe if adapted
        remove_below_n_classifications=-1,
        save_final_model=True,
        save_report=True
    )

Parameters:

- ``file_path``: Path to the test dataset CSV file.
- ``remove_below_n_classifications``: Remove rules used less than or equal to this number of times during classification. Use ``-1`` to disable this feature.
- ``save_final_model``: Whether to save the final refined model to ``files/final_model.pkl``.
- ``save_report``: Whether to save the analysis report to ``files/output_classifier_<type>.txt``.

Editing
-------

After refining, you may also want to manually inspect and adjust the final rules before deploying the model.
The :ref:`RuleClassifier<rule_classifier>` class provides an ``edit_rules()`` method that starts an interactive terminal session to perform these edits.

You'll be able to:

- List all current final rules with their names, predicted classes, and conditions.
- Select a rule by number or name.
- Add new conditions (e.g. v5 > 10.5).
- Remove existing conditions by index.
- Change the predicted class of a rule.
- Save your edits, which will:
    - Re-parse the conditions to keep them consistent,
    - Append an _edited suffix to the rule's name,
    - Persist the entire classifier to files/edited_model.pkl.

.. code-block:: python

    # Enter manual editing mode
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

Using the model
---------------

Single-sample prediction
^^^^^^^^^^^^^^^^^^^^^^^^^

To use the refined model to classify new entries we can use the ``predict`` method. It accepts dictionaries, DataFrames or NumPy arrays directly:

.. code-block:: python

    # Replace with actual values of your dataset
    sample = {"Symptom- Throat Pain": 0, "Symptom- Dyspnea": 1, "Symptom- Fever": 0, "Are you a health professional?": 0}
    predicted_class = classifier.predict(sample)
    print(f"Predicted class: {predicted_class}")

Batch prediction
^^^^^^^^^^^^^^^^^

For high-performance inference on the entire test set, use ``predict`` with a DataFrame or ndarray:

.. code-block:: python

    import numpy as np

    predictions = classifier.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Batch Accuracy: {accuracy:.4f}")

Comparing Metrics
^^^^^^^^^^^^^^^^^

Use the ``compare_initial_final_results`` method to evaluate both the original and refined rule sets, including accuracy, confusion matrices, divergent predictions, and interpretability scores:

.. code-block:: python

    analyzer.compare_initial_final_results("test.csv")

Exporting
^^^^^^^^^

You can export the trained classifier to different formats for deployment:

.. code-block:: python

    # Standalone Python file
    classifier.to_python("covid_classifier.py")

    # Binary format (PYRA)
    classifier.to_binary("covid_model.bin")

    # C header for embedded systems
    classifier.to_c_header("covid_model.h")
