import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyruleanalyzer.rule_classifier import RuleClassifier

# train_path = "examples/data/covid_train.csv"
# test_path = "examples/data/covid_test.csv"

# train_path = "examples/data/mushrooms_encoded_train.csv"
# test_path = "examples/data/mushrooms_encoded_test.csv"

train_path = "examples/data/iris_train.csv"
test_path = "examples/data/iris_test.csv"

# train_path = "examples/data/CICIDS2017-Wed2_train.csv"
# test_path = "examples/data/CICIDS2017-Wed2_test.csv"

# train_path = "examples/data/train-set1.csv"
# test_path = "examples/data/test-set1.csv"

# Model parameters
model_parameters = {
    'random_state': 42
}

# Generating the initial rule based model
classifier = RuleClassifier.new_classifier(train_path, test_path, model_parameters, algorithm_type='Decision Tree')

# Executing the rule analysis method
# remove_duplicates = "soft" (in the same tree, probably does not affect the final metrics), "hard" (between trees, may affect the final metrics), "custom" (custom function to remove duplicates) or "none" (no removal)
# remove_below_n_classifications = -1 (no removal), 0 (removal of rules with 0 classifications), or any other integer (removal of rules with equal or less than n classifications)
classifier.execute_rule_analysis(test_path, remove_duplicates="soft", remove_below_n_classifications=-1)

# Comparing initial and final results
classifier.compare_initial_final_results(test_path)