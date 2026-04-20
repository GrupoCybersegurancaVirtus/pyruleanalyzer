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

Prepare Your Dataset
--------------------

Your dataset can be any standard format accepted by Scikit-Learn (NumPy arrays or Pandas DataFrames).

Example:

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

Train a Model and Extract Rules
-------------------------------

The :ref:`RuleClassifier<rule_classifier>` behaves exactly like a Scikit-Learn estimator. You instantiate it with the desired ``algorithm_type`` and call ``fit(X, y)``.

It supports two forms of input (overloading):
1. **NumPy Arrays or Pandas DataFrames (Standard Scikit-Learn approach)**
2. **Paths to CSV files (Legacy approach, using X as `train_path` and y as `test_path`)**

Example with Decision Tree (Arrays):

.. code-block:: python

    from pyruleanalyzer import PyRuleAnalyzer

    # Create a RuleClassifier instance
    model = PyRuleAnalyzer.new_model(model="Decision Tree")
    
    # Train the model and extract rules automatically
    model.fit(X_train, y_train)

Example with Decision Tree (CSV paths):

.. code-block:: python

    model = PyRuleAnalyzer.new_model(model="Decision Tree")
    model.fit("train.csv", "test.csv")

Example with Random Forest:

.. code-block:: python

    model = PyRuleAnalyzer.new_model(model="Random Forest")
    model.fit(X_train, y_train)

Example with Gradient Boosting Decision Trees:

.. code-block:: python

    model = PyRuleAnalyzer.new_model(model="Gradient Boosting Decision Trees")
    model.fit(X_train, y_train)

Analyze and Refine the Rules
----------------------------

After fitting the model, you can analyze and refine the extracted rules using the :class:`PyRuleAnalyzer<pyruleanalyzer.PyRuleAnalyzer>` directly, which automatically handles the logic for Decision Trees, Random Forests, and GBDT.

.. code-block:: python

    from pyruleanalyzer import PyRuleAnalyzer

    # For a Decision Tree classifier
    model = PyRuleAnalyzer.new_model(model="Decision Tree")
    model.fit(X_train, y_train)
    
    # Remove redundant rules by evaluating on a test set (can be memory arrays or CSV path)
    model.execute_rule_refinement(
        X=X_test, y=y_test,
        remove_below_n_classifications=-1,
        refine_between_trees=False,
        save_final_model=False,
        save_report=False
    )

Interactive Reporting
---------------------

If you are using Jupyter Notebooks, you can easily view a summary report of your model without needing to parse text files:

.. code-block:: python

    report = model.summary_report()
    
    print(f"Total Active Rules: {report['total_rules']}")
    print(f"Classes: {report['classes']}")
    
    # You can even load the rules into a DataFrame for visualization!
    rules_df = pd.DataFrame(report['rules'])
    print(rules_df.head())

Make Predictions
----------------

You can make predictions on individual samples or batches using the standard Scikit-Learn API:

.. code-block:: python

    # Batch prediction on a DataFrame or NumPy array
    predictions = model.predict(X_test)
    
    # Or get class probabilities (for Random Forest / GBDT)
    probabilities = model.predict_proba(X_test)

    # Single-sample prediction using a dictionary
    sample = {"feature_1": 1.5, "feature_2": 2.3, "feature_3": 0.5}
    predicted_class = model.predict(sample)

Exporting the Classifier
-------------------------

The :ref:`RuleClassifier<rule_classifier>` supports fluid export formats for deploying the trained model outside of Python or for maximum inference speed:

Export to standalone Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generates a self-contained ``.py`` file with the decision logic as pure Python code:

.. code-block:: python

    model.to_python("my_classifier.py")

Export to binary
^^^^^^^^^^^^^^^^

Exports the compiled tree arrays to a compact binary file (PYRA format). This format can be loaded back efficiently:

.. code-block:: python

    # Export
    model.to_binary("model.bin")

    # Load back
    loaded = RuleClassifier.load_binary("model.bin")
    predictions = loaded.predict(X_test)

Export to C header
^^^^^^^^^^^^^^^^^^

Exports the classifier as a standalone C header file (``.h``), suitable for embedded systems and microcontrollers (e.g. Arduino):

.. code-block:: python

    model.to_c_header("model.h")

The generated header includes all tree arrays as ``const`` data and a ``predict(const float *features)`` function.

Saving and Loading
------------------

The :ref:`RuleClassifier<rule_classifier>` can be serialized to a pickle file and loaded back later. The native compiled function is automatically rebuilt upon loading.

.. code-block:: python

    import pickle

    # Save
    with open("classifier.pkl", "wb") as f:
        pickle.dump(model, f)

    # Load
    with open("classifier.pkl", "rb") as f:
        loaded_model = pickle.load(f)

Editing
-------

You can also manually edit the final rules by calling the ``edit_rules()`` method in your classifier instance.

.. code-block:: python

    model.edit_rules()

During editing:

- Type the rule number or name to open it.
- Use:
    - :code:`a` -- add condition
    - :code:`r` -- remove condition
    - :code:`c` -- change class
    - :code:`s` -- save changes
- Type exit at the main prompt to finish editing.
