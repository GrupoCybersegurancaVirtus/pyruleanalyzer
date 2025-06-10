COVID-19
========

This tutorial will guide you as we train and prune a decision tree model using the `Brazilian dataset of symptomatic patients for screening the risk of COVID-19 <https://data.mendeley.com/datasets/b7zcgmmwx4/5>`_ repository.

Prerequisites
-------------

For this example we will be using the `rapid_balanced.csv` dataset, so make sure to download it from the repository above before continuing.

Next, install the required packages and create the folder structure, as described in the :ref:`prerequisites section<tutorials/usage#prerequisites>` of the :doc:`usage guide<../usage>`.

Prepare Your Dataset
--------------------

As specified in the :doc:`usage guide<../usage>`, the :ref:`RuleClassifier<rule_classifier>` `new_classifier` method expects a dataset split into two files: `train.csv` and `test.csv`. The dataset must also be formatted with the following characteristics:

- No header row.
- Each row represents a single sample.
- The last column is the target class label.
- All other columns are feature values.
- All values and classes must be non-infinite numbers, so make sure to include an encoder in your pipeline if you have string data.

We can use the following script to remove the header row and split the data, be sure to adapt it to your current pipeline as needed:

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
        X, y, test_size=0.25, stratify=True
    )

    # Save to CSV files without headers and index
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv("train.csv", index=False, header=False)
    test_df.to_csv("test.csv", index=False, header=False)

Training the Tree and Extracting Its Rules
---------------------------------------------------

The `new_classifier` method from :ref:`RuleClassifier<rule_classifier>` will train a `scikit-learn` model, extract its rules and create a new :ref:`RuleClassifier<rule_classifier>` instance. It expects the path to the newly created CSV files, an algorithm type (can be either "Decision Tree" or "Random Forest") and the model parameters to be used in the respective `scikit-learn` model.

.. code-block:: python

    from pyruleanalyzer import RuleClassifier

    # Define the model parameters
    model_params = {"max_depth": 5}

    # Create a RuleClassifier instance
    classifier = RuleClassifier.new_classifier(
        train_path="train.csv",
        test_path="test.csv",
        model_parameters=model_params,
        algorithm_type="Decision Tree"
    )

Pruning
-------

With the :ref:`RuleClassifier<rule_classifier>` instance in hands, we can now execute a rule analysis with the `execute_rule_analysis` method, which will refine the tree by removing duplicate rules. This method expects the `test.csv` file, a duplicate removal method (which can be either "soft", removing duplicate rules in a single tree; "hard", deleting duplicate rules in distinct trees and only applicable to random forest models; "custom", that will use a custom function previously defined with the `set_custom_rule_removal` method; or "none", that will not remove any rules). You may also optionally specify rule removal based on classification count, which will remove rules that classify `n` or fewer entries with the `remove_below_n_classifications` parameter (disabled by default).

.. code-block:: python

    classifier.execute_rule_analysis(
        file_path="test.csv",
        remove_duplicates="soft"
    )

Using the model
---------------

To use the refined model to classify new entries we can use the `classify` method with the `final` parameter set to `True`, this will force the :ref:`RuleClassifier<rule_classifier>` instance we just trained to use the rule set generated after pruning. You must name your features as “v{column}” where `column` is the column index in the csv.

.. code-block:: python
    
    # Replace with actual values of your dataset
    sample = {"v1": 1, "v2": 23, "v3": 34, ..., "vn": 654}
    predicted_class, votes, probabilities = classifier.classify(sample, final=True)
