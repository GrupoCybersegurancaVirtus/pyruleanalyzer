"""
Simple Example: Using the PyRuleAnalyzer API

This script demonstrates the simplified workflow for using pyruleanalyzer.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from pyruleanalyzer import PyRuleAnalyzer

# ==============================================================================
# 1. PREPARE DATA (using Iris dataset for demonstration)
# ==============================================================================

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split into train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save as CSV (simulating real-world scenario)
pd.DataFrame(X_train, columns=iris.feature_names).to_csv("files/train.csv", index=False)
pd.DataFrame(X_test, columns=iris.feature_names).to_csv("files/test.csv", index=False)

# Save labels separately
pd.DataFrame(y_train, columns=["target"]).to_csv("files/train_labels.csv", index=False)
pd.DataFrame(y_test, columns=["target"]).to_csv("files/test_labels.csv", index=False)

print("=" * 60)
print("SIMPLE PYRULEANALYZER EXAMPLE")
print("=" * 60)

# ==============================================================================
# 2. CREATE CLASSIFIER (One line!)
# ==============================================================================

analyzer = PyRuleAnalyzer.create(
    train_path="files/train.csv",
    test_path="files/test.csv",
    model="Decision Tree",
    params={"max_depth": 5, "random_state": 42},
    refine=True  # Automatically refine!
)

print(f"\nCreated: {analyzer}")

# ==============================================================================
# 3. GET SUMMARY
# ==============================================================================

summary = analyzer.summary()
print(f"\nSummary:")
print(f"  Model: {summary['model_type']}")
print(f"  Features: {summary['n_features']}")
print(f"  Classes: {summary['n_classes']}")
print(f"  Rules (before): {summary['n_rules_initial']}")
print(f"  Rules (after): {summary['n_rules_final']}")

# ==============================================================================
# 4. PREDICT (One line!)
# ==============================================================================

predictions = analyzer.predict(X_test)
print(f"\nPredictions: {predictions[:10]}...")  # Show first 10

# Calculate accuracy
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.2%}")

# ==============================================================================
# 5. EXPORT (One line!)
# ==============================================================================

files = analyzer.export("iris_model", formats=["python", "binary"])
print(f"\nExported to: {files}")

# ==============================================================================
# 6. SAVE ANALYZER (Optional)
# ==============================================================================

analyzer.save("files/analyzer.pkl")
print(f"\nAnalyzer saved to: files/analyzer.pkl")

# ==============================================================================
# 7. LOAD ANALYZER (Later)
# ==============================================================================

loaded_analyzer = PyRuleAnalyzer.load("files/analyzer.pkl")
print(f"\nLoaded: {loaded_analyzer}")

# Predict with loaded analyzer
loaded_predictions = loaded_analyzer.predict(X_test)
print(f"Accuracy (loaded): {np.mean(loaded_predictions == y_test):.2%}")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
