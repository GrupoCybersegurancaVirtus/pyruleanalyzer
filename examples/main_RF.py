from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from pyruleanalyzer.rule_classifier import RuleClassifier

train_path = "examples/data/rapid_balanceado_treinamento.csv"
test_path = "examples/data/rapid_balanceado_teste.csv"

# train_path = "examples/data/mushrooms_encoded_train.csv"
# test_path = "examples/data/mushrooms_encoded_test.csv"

# train_path = "examples/data/iris_train.csv"
# test_path = "examples/data/iris_test.csv"

# Model parameters
model_parameters = {
    'criterion': 'gini',
    'bootstrap': True,
    'oob_score': False,
    'n_jobs': None,
    'verbose': 0,
    'warm_start': False,
    'ccp_alpha': 0.0,
    'max_samples': None,
    'n_estimators': 100,
    'random_state': 42
}

# Generating the initial rule based model
classifier = RuleClassifier.new_classifier(train_path, test_path, model_parameters, algorithm_type='Random Forest')

# Executing the rule analysis method
# remove_duplicates = "soft" (in the same tree, probably does not affect the final metrics), "hard" (between trees, may affect the final metrics), "custom" (custom function to remove duplicates) or "none" (no removal)
# remove_below_n_classifications = -1 (no removal), 0 (removal of rules with 0 classifications), or any other integer (removal of rules with equal or less than n classifications)
classifier.execute_rule_analysis(test_path, remove_duplicates="soft", remove_below_n_classifications=-1)

# Comparing initial and final results
classifier.compare_initial_final_results(test_path)
