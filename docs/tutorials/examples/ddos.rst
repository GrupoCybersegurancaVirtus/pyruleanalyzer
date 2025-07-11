DDOS
====

This guild will go through the process of pruning a random forest model using the `DDoS evaluation dataset (CIC-DDoS2019) <https://www.unb.ca/cic/datasets/ddos-2019.html>`_ repository.

Prerequisites
-------------

For this example we will be using the `UDPLag.csv` dataset, it is a large dataset that contains realistic traffic data with multiclass targeting and it's included in the `CSV-03-11.zip`, downloadable from the repository above.

To start off, follow the instructions specified in the :ref:`prerequisites section<tutorials/usage#prerequisites>` of the :doc:`usage guide<../usage>`.

Prepare Your Dataset
--------------------

As specified in the :doc:`usage guide<../usage>`, the :ref:`RuleClassifier<rule_classifier>` `new_classifier` method expects a dataset split into two files: `train.csv` and `test.csv`. The dataset must also be formatted with the following characteristics:

- Each row represents a single sample.
- The last column is the target class label.
- All other columns are feature values.
- All values and classes must be non-infinite numbers, so make sure to include an encoder in your pipeline if you have string data.

We can use the following script to apply an encoder to the string columns, remove infinites and split the data, be sure to adapt it to your current pipeline as needed:

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

    label_encoder = LabelEncoder().fit(y)
    y = label_encoder.transform(y)

    # Split into features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25
    )

    # Save to CSV files without index
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)

Training the Forest and Extracting Its Rules
---------------------------------------------------

The `new_classifier` method from :ref:`RuleClassifier<rule_classifier>` will train a `scikit-learn` model, extract its rules and create a new :ref:`RuleClassifier<rule_classifier>` instance. It expects the path to the newly created CSV files, an algorithm type (can be either "Decision Tree" or "Random Forest") and the model parameters to be used in the respective `scikit-learn` model.

.. code-block:: python

    from pyruleanalyzer import RuleClassifier

    # Define the model parameters
    model_params = {"max_depth": 5, "n_estimators": 100}

    # Create a RuleClassifier instance
    classifier = RuleClassifier.new_classifier(
        train_path="train.csv",
        test_path="test.csv",
        model_parameters=model_params,
        algorithm_type="Random Forest"
    )

Pruning
-------

With the :ref:`RuleClassifier<rule_classifier>` instance in hands, we can now execute a rule analysis with the `execute_rule_analysis` method, which will refine the forest by removing duplicate rules. This method expects the `test.csv` file, a duplicate removal method (which can be either "soft", removing duplicate rules in a single tree, "hard", deleting duplicate rules in distinct trees, only applicable to random forest models, "custom", that will use a custom function previously defined with the `set_custom_rule_removal` method, or "none", that will not remove any rules). You may also optionally specify rule removal based on classification count, which will remove rules that classify `n` or fewer entries with the `remove_below_n_classifications` parameter (disabled by default).

.. code-block:: python

    classifier.execute_rule_analysis(
        file_path="test.csv",
        remove_duplicates="soft"
    )

Since this is a large dataset and the algorithm goes through many iterative steps to ensure no new duplicate rules are accidentally created during pruning, it may take a longer time to fully complete the analysis, specially if you use the "hard" removal method.

Using the model
---------------

To use the refined model to classify new entries we can use the `classify` method with the `final` parameter set to `True`, this will force the :ref:`RuleClassifier<rule_classifier>` instance we just trained to use the rule set generated after pruning. If your dataset didn't include a header row you must name your features as “v{column}” where `column` is the column index in the csv.

.. code-block:: python
    
    # Replace with actual values of your dataset
    sample = {"Flow ID": "172.16.0.5-192.168.50.4-35468-49856-17", " Source IP": "172.16.0.5", ..., " Inbound": 1}

    encoded_sample = feature_encoder.transform(sample)

    predicted_class, votes, probabilities = classifier.classify(encoded_sample, final=True)
    
    actual_class = label_encoder.inverse_transform(predicted_class)
