# PyRuleAnalyzer - Simple Interface for pyruleanalyzer

## Overview

The **PyRuleAnalyzer** class provides a simplified, high-level interface for using pyruleanalyzer. Instead of dealing with multiple classes and complex workflows, you can create, refine, predict, and export classifiers with just a few simple method calls.

## Quick Start

```python
from pyruleanalyzer import PyRuleAnalyzer

# 1. CREATE - Train and refine in one step
analyzer = PyRuleAnalyzer.create(
    train_path="data/train.csv",
    test_path="data/test.csv",
    model="Decision Tree",
    params={"max_depth": 5},
    refine=True
)

# 2. PREDICT - Get predictions
predictions = analyzer.predict(X_test)

# 3. EXPORT - Export to multiple formats
analyzer.export("my_model", formats=["python", "binary"])
```

## Installation

```bash
pip install -e .
```

## API Reference

### PyRuleAnalyzer.create()

Create a new classifier from CSV data files.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `train_path` | str | Path to training CSV file |
| `test_path` | str | Path to test CSV file |
| `model` | str | Model type: "Decision Tree", "Random Forest", or "Gradient Boosting Decision Trees" (default: "Decision Tree") |
| `params` | dict | Model hyperparameters (default: sensible defaults) |
| `refine` | bool | Automatically refine rules (default: False) |
| `refine_params` | dict | Parameters for refinement (see PyRuleAnalyzer.refine()) |
| `save_models` | bool | Save intermediate models to files/ (default: False) |

**Returns:** `PyRuleAnalyzer` instance

**Example:**

```python
# Decision Tree with defaults
analyzer = PyRuleAnalyzer.create(
    train_path="data/train.csv",
    test_path="data/test.csv",
    model="Decision Tree"
)

# Random Forest with custom params
analyzer = PyRuleAnalyzer.create(
    train_path="data/train.csv",
    test_path="data/test.csv",
    model="Random Forest",
    params={"n_estimators": 200, "max_depth": 10},
    refine=True
)

# GBDT with refinement
analyzer = PyRuleAnalyzer.create(
    train_path="data/train.csv",
    test_path="data/test.csv",
    model="Gradient Boosting Decision Trees",
    params={"n_estimators": 100, "learning_rate": 0.1},
    refine=True,
    refine_params={"remove_low_usage": 5}
)
```

### PyRuleAnalyzer.refine()

Refine the classifier by removing low-usage rules.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `test_path` | str | Path to CSV file for evaluating rule usage |
| `remove_low_usage` | int | Minimum usage threshold for rules. Use -1 to disable (default: -1) |
| `save_final_model` | bool | Save refined model to files/final_model.pkl (default: False) |
| `save_report` | bool | Save refinement report (default: False) |

**Returns:** Dictionary with refinement statistics

**Example:**

```python
stats = analyzer.refine(
    test_path="data/test.csv",
    remove_low_usage=5
)

print(f"Removed {stats['rules_removed']} rules ({stats['reduction_percent']:.1f}%)")
```

### PyRuleAnalyzer.predict()

Predict class labels for input data.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | np.ndarray or pd.DataFrame | Input data, shape (n_samples, n_features) |
| `use_refined` | bool | Use refined rules if available (default: True) |

**Returns:** `np.ndarray` of predicted class labels

**Example:**

```python
# Predict with numpy array
predictions = analyzer.predict(X_test)

# Predict with pandas DataFrame
import pandas as pd
X_df = pd.DataFrame(X_test, columns=feature_names)
predictions = analyzer.predict(X_df)

# Use original (unrefined) rules
predictions = analyzer.predict(X_test, use_refined=False)
```

### PyRuleAnalyzer.predict_proba()

Predict class probabilities (Random Forest only).

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | np.ndarray or pd.DataFrame | Input data, shape (n_samples, n_features) |
| `use_refined` | bool | Use refined rules if available (default: True) |

**Returns:** `np.ndarray` of class probabilities, shape (n_samples, n_classes)

**Example:**

```python
probabilities = analyzer.predict_proba(X_test)
print(f"Probabilities for first sample: {probabilities[0]}")
```

### PyRuleAnalyzer.export()

Export the classifier to one or more file formats.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_name` | str | Base name for exported files (default: "model") |
| `formats` | list | Formats to export: "python", "binary", "c" (default: ["python", "binary"]) |
| `use_refined` | bool | Export refined rules if available (default: True) |

**Returns:** Dictionary mapping format to file path

**Example:**

```python
# Export to Python and binary
files = analyzer.export("my_model")
# Output: {'python': 'files/my_model.py', 'binary': 'files/my_model.bin'}

# Export to all formats
files = analyzer.export("my_model", formats=["python", "binary", "c"])

# Export only to C header
files = analyzer.export("my_model", formats=["c"])
```

### PyRuleAnalyzer.save() / PyRuleAnalyzer.load()

Save and load PyRuleAnalyzer instances.

**Example:**

```python
# Save
analyzer.save("files/my_analyzer.pkl")

# Load
analyzer = PyRuleAnalyzer.load("files/my_analyzer.pkl")
```

### PyRuleAnalyzer.summary()

Get a summary of the classifier.

**Returns:** Dictionary with classifier information

**Example:**

```python
summary = analyzer.summary()
print(f"Model: {summary['model_type']}")
print(f"Features: {summary['n_features']}")
print(f"Classes: {summary['n_classes']}")
print(f"Rules: {summary['n_rules_initial']} -> {summary['n_rules_final']}")
```

## Complete Examples

### Example 1: Decision Tree (Simplest)

```python
from pyruleanalyzer import PyRuleAnalyzer

# Create, refine, and export in 3 lines
analyzer = PyRuleAnalyzer.create(
    train_path="data/train.csv",
    test_path="data/test.csv",
    model="Decision Tree",
    params={"max_depth": 5},
    refine=True
)

predictions = analyzer.predict(X_test)
analyzer.export("dt_model")
```

### Example 2: Random Forest with Probabilities

```python
from pyruleanalyzer import PyRuleAnalyzer

# Create Random Forest
analyzer = PyRuleAnalyzer.create(
    train_path="data/train.csv",
    test_path="data/test.csv",
    model="Random Forest",
    params={"n_estimators": 100, "max_depth": 10},
    refine=True
)

# Get predictions and probabilities
predictions = analyzer.predict(X_test)
probabilities = analyzer.predict_proba(X_test)

# Export
analyzer.export("rf_model", formats=["python", "binary"])
```

### Example 3: GBDT with Custom Refinement

```python
from pyruleanalyzer import PyRuleAnalyzer

# Create GBDT
analyzer = PyRuleAnalyzer.create(
    train_path="data/train.csv",
    test_path="data/test.csv",
    model="Gradient Boosting Decision Trees",
    params={"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
    refine=True,
    refine_params={
        "remove_low_usage": 5             # Remove rules used < 5 times
    }
)

# Get refinement stats
stats = analyzer.refine("data/test.csv")
print(f"Rules reduced by {stats['reduction_percent']:.1f}%")

# Predict and export
predictions = analyzer.predict(X_test)
analyzer.export("gbdt_model")
```

### Example 4: Save and Load PyRuleAnalyzer

```python
from pyruleanalyzer import PyRuleAnalyzer

# Create and save
analyzer = PyRuleAnalyzer.create(
    train_path="data/train.csv",
    test_path="data/test.csv",
    model="Decision Tree",
    refine=True
)
analyzer.save("files/my_analyzer.pkl")

# Later... load and use
analyzer = PyRuleAnalyzer.load("files/my_analyzer.pkl")
predictions = analyzer.predict(X_test)
```

## Comparison: Old API vs New PyRuleAnalyzer API

### Old API (Complex)

```python
from pyruleanalyzer import PyRuleAnalyzer

# Step 1: Create classifier
classifier = RuleClassifier.new_classifier(
    train_path="data/train.csv",
    test_path="data/test.csv",
    model_parameters={"max_depth": 5},
    algorithm_type="Decision Tree",
    save_initial_model=True,
    save_sklearn_model=True
)

# Step 2: Analyze and refine
# analyzer = DTAnalyzer(classifier) -> we now just call it from classifier
classifier.execute_rule_refinement(
    file_path="data/test.csv",
    remove_low_usage=-1,
    save_final_model=True,
    save_report=True
)

# Step 3: Compile arrays for prediction
classifier.compile_tree_arrays()

# Step 4: Predict
predictions = classifier.predict_batch(X_test)

# Step 5: Export
classifier.export_to_native_python(feature_names=feature_names)
classifier.export_to_binary()
```

### New PyRuleAnalyzer API (Simple)

```python
from pyruleanalyzer import PyRuleAnalyzer

# All in one step!
analyzer = PyRuleAnalyzer.create(
    train_path="data/train.csv",
    test_path="data/test.csv",
    model="Decision Tree",
    params={"max_depth": 5},
    refine=True
)

# Predict
predictions = analyzer.predict(X_test)

# Export
analyzer.export("my_model")
```

## Default Parameters

The PyRuleAnalyzer API uses sensible defaults for each model type:

| Model | Default Parameters |
|-------|-------------------|
| Decision Tree | `max_depth=None`, `random_state=42` |
| Random Forest | `n_estimators=100`, `max_depth=None`, `random_state=42` |
| GBDT | `n_estimators=100`, `max_depth=3`, `learning_rate=0.1`, `random_state=42` |

## File Structure

The PyRuleAnalyzer API saves files to the `files/` directory:

```
files/
├── my_model.py          # Python export
├── my_model.bin         # Binary export
├── my_model.h           # C header export (if requested)
├── my_analyzer.pkl      # Saved PyRuleAnalyzer (if using save())
├── initial_model.pkl    # Initial model (if save_models=True)
├── final_model.pkl      # Refined model (if save_final_model=True)
└── sklearn_model.pkl    # Sklearn model (if save_models=True)
```

## Migration Guide

### From RuleClassifier to PyRuleAnalyzer

**Old code:**
```python
classifier = RuleClassifier.new_classifier(...)
classifier.execute_rule_refinement(...)
classifier.compile_tree_arrays()
predictions = classifier.predict_batch(X_test)
classifier.export_to_native_python()
```

**New code:**
```python
analyzer = PyRuleAnalyzer.create(..., refine=True)
predictions = analyzer.predict(X_test)
analyzer.export()
```

### From Analyzers to PyRuleAnalyzer

**Old code:**
```python
classifier = RuleClassifier.new_classifier(...)
analyzer = DTAnalyzer(classifier)  # or RFAnalyzer, GBDTAnalyzer
analyzer.execute_rule_refinement(...)
```

**New code:**
```python
model = PyRuleAnalyzer.new_model(model='Decision Tree')
model.fit(X_train, y_train)
model.execute_rule_refinement(X=X_test, y=y_test)
```

## Troubleshooting

### Issue: "CSV file has no target column"

**Solution:** Ensure your CSV has a target column. The last column is used as the target by default.

### Issue: "Feature names mismatch"

**Solution:** Make sure training and test data have the same features in the same order.

### Issue: "Refinement removes too many rules"

**Solution:** Increase `remove_low_usage` threshold or set to -1 to disable.

## Advanced Usage

### Accessing the Underlying RuleClassifier

You can still access the full power of RuleClassifier through the PyRuleAnalyzer:

```python
analyzer = PyRuleAnalyzer.create(...)

# Access the underlying RuleClassifier
classifier = analyzer.classifier

# Use any RuleClassifier method
classifier.display_metrics(...)
classifier.find_duplicated_rules(...)
```

### Custom Refinement Workflow

```python
# Create without refinement
analyzer = PyRuleAnalyzer.create(..., refine=False)

# Refine with different parameters
stats = analyzer.refine(test_path, remove_low_usage=5)

# Compare results
print(f"Rules reduced by {stats['reduction_percent']:.1f}%")
```

## License

MIT License - See LICENSE file in the project root.

## Support

For issues and questions, please open an issue on GitHub.
