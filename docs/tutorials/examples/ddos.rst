DDOS
====

This guide will go through the process of refining a random forest model using the `DDoS evaluation dataset (CIC-DDoS2019) <https://www.unb.ca/cic/datasets/ddos-2019.html>`_ repository.

Prerequisites
-------------

For this example we will be using the ``UDPLag.csv`` dataset, it is a large dataset that contains realistic traffic data with multiclass targeting and it's included in the ``CSV-03-11.zip``, downloadable from the repository above.

To start off, install the required packages.

.. code-block:: bash

    pip install pyruleanalyzer scikit-learn pandas numpy

Prepare Your Dataset
--------------------

As specified in the :doc:`usage guide<../usage>`, the :ref:`RuleClassifier<rule_classifier>` integrates directly with standard data pipelines.

We can use the following script to apply an encoder to the string columns, remove infinites and split the data. Be sure to adapt it to your current pipeline as needed:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

    # Load the dataset
    df = pd.read_csv("UDPLag.csv")

    # Encoding string columns into integers
    string_cols = ['Flow ID', ' Source IP', ' Destination IP', ' Timestamp', 'SimillarHTTP']

    # Dropping entries with infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    feature_encoder = OrdinalEncoder().fit(df[string_cols].astype('str'))
    df[string_cols] = feature_encoder.transform(df[string_cols].astype('str'))

    label_encoder = LabelEncoder().fit(df[' Label'])
    df[' Label'] = label_encoder.transform(df[' Label'])

    # Split into features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25
    )

Training the Forest and Extracting Its Rules
---------------------------------------------------

The :ref:`RuleClassifier<rule_classifier>` works exactly like a Scikit-Learn estimator.

.. code-block:: python

    from pyruleanalyzer import PyRuleAnalyzer, RuleClassifier

    # Create a RuleClassifier instance
    classifier = PyRuleAnalyzer.new_model(model="Random Forest")
    
    # Train the model
    classifier.fit(X_train, y_train)

Refinement
-------

With the :ref:`RuleClassifier<rule_classifier>` instance in hands, we can now execute a rule analysis using the :ref:`RFAnalyzer<rf_analyzer>` class, which will refine the forest by removing duplicate rules.

.. code-block:: python

    from pyruleanalyzer import PyRuleAnalyzer, RFAnalyzer

    analyzer = RFAnalyzer(classifier)
    
    # To evaluate rules, we need the validation set data
    # (Let's save it temporarily for the analyzer to use)
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv("test.csv", index=False)
    
    analyzer.execute_rule_analysis(
        file_path="test.csv",
        remove_below_n_classifications=-1,
        save_final_model=True,
        save_report=True
    )

Parameters:

- ``file_path``: Path to the test dataset CSV file.
- ``remove_below_n_classifications``: Remove rules used less than or equal to this number of times during classification. Use ``-1`` to disable this feature.
- ``save_final_model``: Whether to save the final refined model to ``files/final_model.pkl``.
- ``save_report``: Whether to save the analysis report to ``files/output_classifier_<type>.txt``.

Since this is a large dataset and the algorithm goes through many iterative steps to ensure no new duplicate rules are accidentally created during refinement, it may take a longer time to fully complete the analysis.

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
    sample = {"Flow ID": "172.16.0.5-192.168.50.4-35468-49856-17", " Source IP": "172.16.0.5", " Inbound": 1}

    # Don't forget to encode if necessary
    # encoded_sample = ... 

    predicted_class = classifier.predict(sample)

    actual_class = label_encoder.inverse_transform([predicted_class])
    print(f"Predicted: {actual_class[0]}")

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

Use the ``compare_initial_final_results`` method to evaluate both the original and refined rule sets:

.. code-block:: python

    analyzer.compare_initial_final_results("test.csv")

Exporting
^^^^^^^^^

You can export the trained classifier to different formats for deployment:

.. code-block:: python

    # Standalone Python file
    classifier.to_python("ddos_classifier.py")

    # Binary format (PYRA)
    classifier.to_binary("ddos_model.bin")

    # C header for embedded systems
    classifier.to_c_header("ddos_model.h")
