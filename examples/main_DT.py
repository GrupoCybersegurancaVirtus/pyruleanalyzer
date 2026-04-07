import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyruleanalyzer.rule_classifier import RuleClassifier
from pyruleanalyzer._accel import HAS_C_EXTENSION

train_path = "examples/data/covid_train.csv"
test_path = "examples/data/covid_test.csv"

# train_path = "examples/data/CICIDS2017-Wed2_train.csv"
# test_path = "examples/data/CICIDS2017-Wed2_test.csv"

# train_path = "examples/data/train-set1.csv"
# test_path = "examples/data/test-set1.csv"

# train_path = "examples/data/A Machine Learning-Based Classification and Prediction Technique for DDoS Attacks/train.csv"
# test_path = "examples/data/A Machine Learning-Based Classification and Prediction Technique for DDoS Attacks/test.csv"

# train_path = "examples/data/DDoS Attack Classification Leveraging Data Balancing and Hyperparameter Tuning Approach Using Ensemble Machine Learning with XAI/train.csv"
# test_path = "examples/data/DDoS Attack Classification Leveraging Data Balancing and Hyperparameter Tuning Approach Using Ensemble Machine Learning with XAI/test.csv"

# Model parameters
model_parameters = {
    'random_state': 42
}

# ==============================================================================
# 1. RULE EXTRACTION AND ANALYSIS
# ==============================================================================
# Generating the initial rule based model
classifier = RuleClassifier.new_classifier(train_path, test_path, model_parameters, algorithm_type='Decision Tree')

# Executing the rule analysis method
# remove_duplicates = "soft" (in the same tree, probably does not affect the final metrics), "hard" (between trees, may affect the final metrics), "custom" (custom function to remove duplicates) or "none" (no removal)
# remove_below_n_classifications = -1 (no removal), 0 (removal of rules with 0 classifications), or any other integer (removal of rules with equal or less than n classifications)
classifier.execute_rule_analysis(test_path, remove_duplicates="soft", remove_below_n_classifications=-1)

# Comparing initial and final results
classifier.compare_initial_final_results(test_path)

# ==============================================================================
# 2. BATCH PREDICTION (Vectorized, with optional C acceleration)
# ==============================================================================
print("\n" + "=" * 80)
print("BATCH PREDICTION DEMO (Decision Tree)")
print("=" * 80)
print(f"C extension available: {HAS_C_EXTENSION}")

# Load test data as numpy array
X_train, _, X_test, y_test, _, _, feature_names = RuleClassifier.process_data(train_path, test_path)

# Compile tree arrays for vectorized prediction
classifier.compile_tree_arrays(feature_names=feature_names)

# Batch predict
start = time.time()
y_batch = classifier.predict_batch(X_test, feature_names=feature_names)
t_batch = time.time() - start

acc_batch = np.mean(y_batch == y_test)
print(f"  Samples:   {len(y_test)}")
print(f"  Accuracy:  {acc_batch:.5f}")
print(f"  Time:      {t_batch:.4f}s")
print(f"  Speed:     {len(y_test) / max(t_batch, 1e-9):.0f} samples/s")

# Batch predict probabilities
y_proba = classifier.predict_batch_proba(X_test, feature_names=feature_names)
print(f"  Proba shape: {y_proba.shape}")

# ==============================================================================
# 3. EXPORT FORMATS (Using export_all for selective export)
# ==============================================================================
print("\n" + "=" * 80)
print("EXPORT FORMATS")
print("=" * 80)

# Export to all formats (Python, Binary, C)
classifier.export_all(
    base_name="files/dt_model",
    feature_names=feature_names,
    export_python=True,   # Export to .py
    export_binary=True,   # Export to .bin
    export_c=True         # Export to .h
)

# Alternative: Export only specific formats
# classifier.export_all("files/dt_model", export_python=False, export_binary=True, export_c=False)

# Size comparison
export_files = [
    ("Python (.py)", "files/dt_model.py"),
    ("Binary (.bin)", "files/dt_model.bin"),
    ("C Header (.h)", "files/dt_model.h")
]

print(f"{'FORMAT':<30} | {'SIZE':>12}")
print("-" * 48)
for label, path in export_files:
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"{label:<30} | {size / 1024:>10.2f} KB")

# Verify binary round-trip
clf_loaded = RuleClassifier.load_binary("files/dt_model.bin")
y_loaded = clf_loaded.predict_batch(X_test, feature_names=feature_names)
match = np.mean(y_loaded == y_batch) == 1.0
print(f"\nBinary round-trip match: {'OK' if match else 'MISMATCH'}")
