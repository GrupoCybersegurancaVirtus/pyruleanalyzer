# Pipeline API - Simple Interface for pyruleanalyzer

## Overview

The **Pipeline API** provides a simplified, high-level interface for using pyruleanalyzer. Instead of dealing with multiple classes and complex workflows, you can create, optimize, predict, and export classifiers with just a few simple method calls.

## Quick Start

```python
from pyruleanalyzer import Pipeline

# 1. CREATE - Train and optimize in one step
pipeline = Pipeline.create(
    train_path="data/train.csv",
    test_path="data/test.csv",
    model="Decision Tree",
    params={"max_depth": 5},
    optimize=True
)

# 2. PREDICT - Get predictions
predictions = pipeline.predict(X_test)

# 3. EXPORT - Export to multiple formats
pipeline.export("my_model", formats=["python", "binary"])
```

## Installation

```bash
pip install -e .
```

## API Reference

### Pipeline.create()

Create a new classifier from CSV data files.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `train_path` | str | Path to training CSV file |
| `test_path` | str | Path to test CSV file |
| `model` | str | Model type: "Decision Tree", "Random Forest", or "Gradient Boosting Decision Trees" (default: "Decision Tree") |
| `params` | dict | Model hyperparameters (default: sensible defaults) |
| `optimize` | bool | Automatically optimize rules (default: False) |
| `optimize_params` | dict | Parameters for optimization (see Pipeline.optimize()) |
| `save_models` | bool | Save intermediate models to files/ (default: False) |

**Returns:** `Pipeline` instance

**Example:**

```python
# Decision Tree with defaults
pipeline = Pipeline.create(
    train_path="data/train.csv",
    test_path="data/test.csv",
    model="Decision Tree"
)

# Random Forest with custom params
pipeline = Pipeline.create(
    train_path="data/train.csv",
    test_path="data/test.csv",
    model="Random Forest",
    params={"n_estimators": 200, "max_depth": 10},
    optimize=True
)

# GBDT with optimization
pipeline = Pipeline.create(
    train_path="data/train.csv",
    test_path="data/test.csv",
    model="Gradient Boosting Decision Trees",
    params={"n_estimators": 100, "learning_rate": 0.1},
    optimize=True,
    optimize_params={"remove_duplicates": "soft", "remove_low_usage": 5}
)
```

### Pipeline.optimize()

Optimize the classifier by removing redundant and low-usage rules.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `test_path` | str | Path to CSV file for evaluating rule usage |
| `remove_duplicates` | str | Duplicate removal: "soft" (aggressive), "hard" (conservative), or "none" (default: "soft") |
| `remove_low_usage` | int | Minimum usage threshold for rules. Use -1 to disable (default: -1) |
| `save_final_model` | bool | Save optimized model to files/final_model.pkl (default: False) |
| `save_report` | bool | Save optimization report (default: False) |

**Returns:** Dictionary with optimization statistics

**Example:**

```python
stats = pipeline.optimize(
    test_path="data/test.csv",
    remove_duplicates="soft",
    remove_low_usage=5
)

print(f"Removed {stats['rules_removed']} rules ({stats['reduction_percent']:.1f}%)")
```

### Pipeline.predict()

Predict class labels for input data.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | np.ndarray or pd.DataFrame | Input data, shape (n_samples, n_features) |
| `use_optimized` | bool | Use optimized rules if available (default: True) |

**Returns:** `np.ndarray` of predicted class labels

**Example:**

```python
# Predict with numpy array
predictions = pipeline.predict(X_test)

# Predict with pandas DataFrame
import pandas as pd
X_df = pd.DataFrame(X_test, columns=feature_names)
predictions = pipeline.predict(X_df)

# Use original (unoptimized) rules
predictions = pipeline.predict(X_test, use_optimized=False)
```

### Pipeline.predict_proba()

Predict class probabilities (Random Forest only).

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | np.ndarray or pd.DataFrame | Input data, shape (n_samples, n_features) |
| `use_optimized` | bool | Use optimized rules if available (default: True) |

**Returns:** `np.ndarray` of class probabilities, shape (n_samples, n_classes)

**Example:**

```python
probabilities = pipeline.predict_proba(X_test)
print(f"Probabilities for first sample: {probabilities[0]}")
```

### Pipeline.export()

Export the classifier to one or more file formats.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_name` | str | Base name for exported files (default: "model") |
| `formats` | list | Formats to export: "python", "binary", "c" (default: ["python", "binary"]) |
| `use_optimized` | bool | Export optimized rules if available (default: True) |

**Returns:** Dictionary mapping format to file path

**Example:**

```python
# Export to Python and binary
files = pipeline.export("my_model")
# Output: {'python': 'files/my_model.py', 'binary': 'files/my_model.bin'}

# Export to all formats
files = pipeline.export("my_model", formats=["python", "binary", "c"])

# Export only to C header
files = pipeline.export("my_model", formats=["c"])
```

### Pipeline.save() / Pipeline.load()

Save and load Pipeline instances.

**Example:**

```python
# Save
pipeline.save("files/my_pipeline.pkl")

# Load
pipeline = Pipeline.load("files/my_pipeline.pkl")
```

### Pipeline.summary()

Get a summary of the classifier.

**Returns:** Dictionary with classifier information

**Example:**

```python
summary = pipeline.summary()
print(f"Model: {summary['model_type']}")
print(f"Features: {summary['n_features']}")
print(f"Classes: {summary['n_classes']}")
print(f"Rules: {summary['n_rules_initial']} -> {summary['n_rules_final']}")
```

## Complete Examples

### Example 1: Decision Tree (Simplest)

```python
from pyruleanalyzer import Pipeline

# Create, optimize, and export in 3 lines
pipeline = Pipeline.create(
    train_path="data/train.csv",
    test_path="data/test.csv",
    model="Decision Tree",
    params={"max_depth": 5},
    optimize=True
)

predictions = pipeline.predict(X_test)
pipeline.export("dt_model")
```

### Example 2: Random Forest with Probabilities

```python
from pyruleanalyzer import Pipeline

# Create Random Forest
pipeline = Pipeline.create(
    train_path="data/train.csv",
    test_path="data/test.csv",
    model="Random Forest",
    params={"n_estimators": 100, "max_depth": 10},
    optimize=True,
    optimize_params={"remove_duplicates": "soft"}
)

# Get predictions and probabilities
predictions = pipeline.predict(X_test)
probabilities = pipeline.predict_proba(X_test)

# Export
pipeline.export("rf_model", formats=["python", "binary"])
```

### Example 3: GBDT with Custom Optimization

```python
from pyruleanalyzer import Pipeline

# Create GBDT
pipeline = Pipeline.create(
    train_path="data/train.csv",
    test_path="data/test.csv",
    model="Gradient Boosting Decision Trees",
    params={"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
    optimize=True,
    optimize_params={
        "remove_duplicates": "hard",      # Conservative
        "remove_low_usage": 5             # Remove rules used < 5 times
    }
)

# Get optimization stats
stats = pipeline.optimize("data/test.csv")
print(f"Rules reduced by {stats['reduction_percent']:.1f}%")

# Predict and export
predictions = pipeline.predict(X_test)
pipeline.export("gbdt_model")
```

### Example 4: Save and Load Pipeline

```python
from pyruleanalyzer import Pipeline

# Create and save
pipeline = Pipeline.create(
    train_path="data/train.csv",
    test_path="data/test.csv",
    model="Decision Tree",
    optimize=True
)
pipeline.save("files/my_pipeline.pkl")

# Later... load and use
pipeline = Pipeline.load("files/my_pipeline.pkl")
predictions = pipeline.predict(X_test)
```

## Comparison: Old API vs New Pipeline API

### Old API (Complex)

```python
from pyruleanalyzer import RuleClassifier, DTAnalyzer

# Step 1: Create classifier
classifier = RuleClassifier.new_classifier(
    train_path="data/train.csv",
    test_path="data/test.csv",
    model_parameters={"max_depth": 5},
    algorithm_type="Decision Tree",
    save_initial_model=True,
    save_sklearn_model=True
)

# Step 2: Analyze and optimize
analyzer = DTAnalyzer(classifier)
analyzer.execute_rule_analysis(
    file_path="data/test.csv",
    remove_duplicates="soft",
    remove_below_n_classifications=-1,
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

### New Pipeline API (Simple)

```python
from pyruleanalyzer import Pipeline

# All in one step!
pipeline = Pipeline.create(
    train_path="data/train.csv",
    test_path="data/test.csv",
    model="Decision Tree",
    params={"max_depth": 5},
    optimize=True
)

# Predict
predictions = pipeline.predict(X_test)

# Export
pipeline.export("my_model")
```

## Default Parameters

The Pipeline API uses sensible defaults for each model type:

| Model | Default Parameters |
|-------|-------------------|
| Decision Tree | `max_depth=None`, `random_state=42` |
| Random Forest | `n_estimators=100`, `max_depth=None`, `random_state=42` |
| GBDT | `n_estimators=100`, `max_depth=3`, `learning_rate=0.1`, `random_state=42` |

## File Structure

The Pipeline API saves files to the `files/` directory:

```
files/
├── my_model.py          # Python export
├── my_model.bin         # Binary export
├── my_model.h           # C header export (if requested)
├── my_pipeline.pkl      # Saved Pipeline (if using save())
├── initial_model.pkl    # Initial model (if save_models=True)
├── final_model.pkl      # Optimized model (if save_final_model=True)
└── sklearn_model.pkl    # Sklearn model (if save_models=True)
```

## Migration Guide

### From RuleClassifier to Pipeline

**Old code:**
```python
classifier = RuleClassifier.new_classifier(...)
classifier.execute_rule_analysis(...)
classifier.compile_tree_arrays()
predictions = classifier.predict_batch(X_test)
classifier.export_to_native_python()
```

**New code:**
```python
pipeline = Pipeline.create(..., optimize=True)
predictions = pipeline.predict(X_test)
pipeline.export()
```

### From Analyzers to Pipeline

**Old code:**
```python
classifier = RuleClassifier.new_classifier(...)
analyzer = DTAnalyzer(classifier)  # or RFAnalyzer, GBDTAnalyzer
analyzer.execute_rule_analysis(...)
```

**New code:**
```python
pipeline = Pipeline.create(..., optimize=True)
# Or manually:
pipeline = Pipeline.create(...)
pipeline.optimize(...)
```

## Troubleshooting

### Issue: "CSV file has no target column"

**Solution:** Ensure your CSV has a target column. The last column is used as the target by default.

### Issue: "Feature names mismatch"

**Solution:** Make sure training and test data have the same features in the same order.

### Issue: "Optimization removes too many rules"

**Solution:** Use `remove_duplicates="hard"` for conservative optimization, or increase `remove_low_usage` threshold.

## Advanced Usage

### Accessing the Underlying RuleClassifier

You can still access the full power of RuleClassifier through the Pipeline:

```python
pipeline = Pipeline.create(...)

# Access the underlying RuleClassifier
classifier = pipeline.classifier

# Use any RuleClassifier method
classifier.display_metrics(...)
classifier.find_duplicated_rules(...)
```

### Custom Optimization Workflow

```python
# Create without optimization
pipeline = Pipeline.create(..., optimize=False)

# Optimize multiple times with different parameters
stats1 = pipeline.optimize(test_path, remove_duplicates="soft")
stats2 = pipeline.optimize(test_path, remove_duplicates="hard")

# Compare results
print(f"Soft: {stats1['reduction_percent']:.1f}% reduction")
print(f"Hard: {stats2['reduction_percent']:.1f}% reduction")
```

## License

MIT License - See LICENSE file in the project root.

## Support

For issues and questions, please open an issue on GitHub.
