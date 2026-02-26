# pyRuleAnalyzer

[![Python 3.7+](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/pyruleanalyzer.svg)](https://pypi.org/project/pyruleanalyzer/)

**pyRuleAnalyzer** is a Python tool for **rule extraction, analysis, optimization, and simplification** from scikit-learn tree-based models. It converts black-box models into human-readable rule sets, removes redundancies, evaluates interpretability, and exports standalone Python classifiers for high-performance inference.

**Supported models:** Decision Tree, Random Forest, and Gradient Boosting Decision Trees (GBDT).

---

## Table of Contents

- [Key Features](#key-features)
- [How It Works](#how-it-works)
  - [Rule Extraction](#1-rule-extraction)
  - [Rule Optimization](#2-rule-optimization)
  - [Classification Strategies](#3-classification-strategies)
  - [Native Python Export](#4-native-python-export)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
  - [Decision Tree](#decision-tree)
  - [Random Forest](#random-forest)
  - [Gradient Boosting (GBDT)](#gradient-boosting-gbdt)
  - [Interactive Rule Editing](#interactive-rule-editing)
  - [Export Standalone Classifier](#export-standalone-classifier)
  - [Custom Rule Removal](#custom-rule-removal)
- [API Reference](#api-reference)
  - [RuleClassifier](#ruleclassifier)
  - [Rule](#rule)
  - [Analyzer Classes](#analyzer-classes)
- [Project Structure](#project-structure)
- [Output Formats](#output-formats)
- [Documentation](#documentation)
- [License](#license)

---

## Key Features

| Feature | Description |
|---|---|
| **Rule Extraction** | Traverses sklearn tree structures to extract every decision path as a human-readable rule |
| **Boundary Redundancy Removal** | Merges sibling rules that split on the same threshold but lead to the same class |
| **Semantic Redundancy Removal** | Eliminates identical rules across different trees in ensemble models (RF) |
| **Low-Usage Pruning** | Removes rarely-triggered rules with automatic sibling promotion |
| **Custom Pruning** | Inject your own rule removal logic via callback functions |
| **Native Python Compilation** | Compiles rules into optimized `if/else` Python code (in-memory via `exec()` and file export) |
| **Interpretability Metrics** | Depth balance, attribute usage, complexity score, feature coverage |
| **Interactive Editing** | Terminal-based rule editor: add/remove conditions, change class labels |
| **Pickle Serialization** | Save/load models with automatic native model recompilation |
| **Benchmark Suite** | Compare sklearn vs. pyRuleAnalyzer (accuracy, speed, file size) |

---

## How It Works

### 1. Rule Extraction

A decision tree is a series of if/else splits. Each path from root to leaf becomes a **Rule** -- a list of conditions plus a predicted class:

```
         Decision Tree                     Extracted Rules
         ─────────────                     ───────────────

           [v1 <= 3.5]                  Rule1 (Class 0):
           /          \                   v1 <= 3.5
          /            \                  v2 <= 1.2
    [v2 <= 1.2]    [v2 <= 4.8]
     /      \       /      \          Rule2 (Class 1):
    /        \     /        \           v1 <= 3.5
 Class 0  Class 1  Class 1  Class 2     v2 > 1.2

                                      Rule3 (Class 1):
                                        v1 > 3.5
                                        v2 <= 4.8

                                      Rule4 (Class 2):
                                        v1 > 3.5
                                        v2 > 4.8
```

For **Random Forests**, rules are extracted from every tree in the ensemble. For **GBDT**, each tree's leaves carry residual values (leaf_value) and a learning rate that contribute to additive scoring.

### 2. Rule Optimization

pyRuleAnalyzer applies multiple optimization passes to simplify the rule set while preserving (or minimally impacting) accuracy:

#### a) Boundary Redundancy Removal (Sibling Merging)

When two sibling rules share the same class and differ only in a complementary last condition (`<= T` vs `> T`), they are merged into their parent:

```
    BEFORE (2 rules)                    AFTER (1 rule)
    ────────────────                    ───────────────

    Rule1 (Class 0):                  Rule1_merged (Class 0):
      v1 <= 3.5          ──merge──>     v1 <= 3.5
      v2 <= 1.2                         (last condition removed)

    Rule2 (Class 0):
      v1 <= 3.5
      v2 > 1.2

    Both children predict the same       The split on v2 was
    class, so the v2 split is            unnecessary -- the parent
    redundant.                           alone is sufficient.
```

This process runs **iteratively until convergence** -- merging at one level may expose new mergeable siblings at the level above:

```
    Iteration 1                   Iteration 2                   Iteration 3
    ───────────                   ───────────                   ───────────

       [v1]                          [v1]                         Class 0
      /    \                        /    \                     (single rule,
    [v2]   [v2]      merge       [v2]   Class 0   merge        no conditions)
   / \     / \      ──────>      / \              ──────>
  C0  C0  C0  C0               C0  C0

  4 rules                       2 rules                        1 rule
```

#### b) Semantic Redundancy Removal (Inter-Tree, Random Forest)

In Random Forests, different trees may produce **identical rules** (same conditions, same class). These are deduplicated into a single representative:

```
    Tree 1:                     Tree 2:
    Rule "RF1_Rule3":           Rule "RF2_Rule7":
      v1 <= 3.5                  v1 <= 3.5          ──> SAME! Keep only one
      v2 > 1.2                   v2 > 1.2               representative rule
      Class: 1                   Class: 1
```

#### c) Low-Usage Pruning with Sibling Promotion

Rules that match very few (or zero) test samples are candidates for removal. When a rule is pruned, its **sibling is promoted** by stripping its last condition (since the distinguishing split no longer exists):

```
    BEFORE                           AFTER
    ──────                           ─────

       [v1 <= 3.5]                    [v1 <= 3.5]
       /          \                  (promoted: last condition stripped)
    Rule_A      Rule_B               Rule_A_promoted
    (used 500x) (used 0x)            Class 1
    Class 1      Class 2              v1 <= 3.5
                    ^
                    │
              removed (0 usage)     Rule_B removed, Rule_A promoted
                                    to parent level
```

Promotion is processed **deepest-first** to handle cascading correctly -- promoting a deep rule may make its parent eligible for further promotion.

#### d) Custom Pruning

Inject domain-specific logic via `set_custom_rule_removal()`:

```python
def my_pruner(rules):
    filtered = [r for r in rules if some_domain_condition(r)]
    removed = [(r, r) for r in rules if r not in filtered]
    return filtered, removed

classifier.set_custom_rule_removal(my_pruner)
classifier.execute_rule_analysis(test_path, remove_duplicates="custom")
```

### 3. Classification Strategies

Each algorithm type uses a different strategy to classify new samples:

```
    Decision Tree                Random Forest               GBDT
    ─────────────                ─────────────               ────

    First-match:                 Majority voting:            Additive scoring:

    for rule in rules:           votes = {}                  for class_group:
      if all conditions          for rule in rules:            score = init_score
         match:                    if match:                   for tree:
        return rule.class          votes[class] += 1             if match:
                                 return max(votes)                 score += contribution
    return default_class                                     binary:  sigmoid(score)
                                                             multi:   argmax(scores)
```

### 4. Native Python Export

Rules are compiled into optimized, standalone Python code. The tree structure is **reconstructed** from rules and emitted as nested `if/else` blocks:

```python
# Generated file: dt_classifier.py (no dependencies required)

def predict(sample):
    if sample['v1'] <= 3.5:
        if sample['v2'] <= 1.2:
            return 0  # Class 0
        else:
            return 1  # Class 1
    else:
        if sample['v2'] <= 4.8:
            return 1  # Class 1
        else:
            return 2  # Class 2
```

**Performance:** Native export typically achieves **10-20x faster inference** than the rule-matching engine, with file sizes often **90%+ smaller** than sklearn pickle files.

---

## Installation

```bash
pip install pyruleanalyzer
```

Or install from source:

```bash
git clone https://github.com/GrupoCybersegurancaVirtus/pyruleanalyzer.git
cd pyruleanalyzer
pip install -e .
```

**Dependencies:** `numpy`, `pandas`, `scikit-learn`, `matplotlib`

---

## Quick Start

```python
from pyruleanalyzer import RuleClassifier

# 1. Create classifier (trains sklearn model + extracts rules)
classifier = RuleClassifier.new_classifier(
    "train.csv", "test.csv",
    model_parameters={'random_state': 42},
    algorithm_type='Decision Tree'
)

# 2. Optimize rules (remove redundancies + prune low-usage)
classifier.execute_rule_analysis(
    "test.csv",
    remove_duplicates="soft",
    remove_below_n_classifications=1
)

# 3. Compare initial vs optimized model
classifier.compare_initial_final_results("test.csv")
```

**Output includes:**
- Number of rules before/after optimization
- Accuracy, precision, recall, F1-score (macro)
- Confusion matrices
- Interpretability metrics (depth, complexity, feature coverage)
- Divergence report (cases where initial and final models disagree)

---

## Usage Examples

### Decision Tree

```python
from pyruleanalyzer import RuleClassifier

classifier = RuleClassifier.new_classifier(
    "train.csv", "test.csv",
    model_parameters={'random_state': 42},
    algorithm_type='Decision Tree'
)

# "soft" = merge siblings within the same tree (safe, no accuracy loss)
classifier.execute_rule_analysis("test.csv", remove_duplicates="soft", remove_below_n_classifications=-1)
classifier.compare_initial_final_results("test.csv")
```

### Random Forest

```python
from pyruleanalyzer import RuleClassifier

model_params = {
    'n_estimators': 100,
    'max_features': 'sqrt',
    'random_state': 42
}

classifier = RuleClassifier.new_classifier(
    "train.csv", "test.csv",
    model_parameters=model_params,
    algorithm_type='Random Forest'
)

# "hard" = also remove identical rules across different trees
classifier.execute_rule_analysis("test.csv", remove_duplicates="hard", remove_below_n_classifications=1)
classifier.compare_initial_final_results("test.csv")
```

### Gradient Boosting (GBDT)

```python
from pyruleanalyzer import RuleClassifier

classifier = RuleClassifier.new_classifier(
    "train.csv", "test.csv",
    model_parameters={},
    algorithm_type='Gradient Boosting Decision Trees'
)

classifier.execute_rule_analysis("test.csv", remove_duplicates="hard", remove_below_n_classifications=1)
classifier.compare_initial_final_results("test.csv")
```

### Interactive Rule Editing

After analysis, you can manually edit rules through an interactive terminal interface:

```python
# Load a previously saved model
classifier = RuleClassifier.load("examples/files/final_model.pkl")

# Open the interactive editor
classifier.edit_rules()

# Options available in the editor:
#   - Add/remove conditions from a rule
#   - Change a rule's predicted class
#   - Delete entire rules
#   - View current rule set
```

### Export Standalone Classifier

Export rules as a standalone Python file with zero dependencies:

```python
# Get feature names from training data
X_train, _, X_test, y_test, _, _, feature_names = RuleClassifier.process_data("train.csv", "test.csv")

# Export
classifier.export_to_native_python(feature_names, filename="my_classifier.py")
```

The exported file can be used independently:

```python
import my_classifier

prediction = my_classifier.predict({
    'feature1': 0.5,
    'feature2': 1.2,
    'feature3': 3.0
})
```

### Custom Rule Removal

```python
def remove_short_rules(rules):
    """Remove rules with fewer than 2 conditions."""
    kept = [r for r in rules if len(r.conditions) >= 2]
    removed = [(r, r) for r in rules if len(r.conditions) < 2]
    return kept, removed

classifier.set_custom_rule_removal(remove_short_rules)
classifier.execute_rule_analysis("test.csv", remove_duplicates="custom")
```

---

## API Reference

### RuleClassifier

The main class that handles the entire pipeline.

#### Factory & I/O

| Method | Description |
|---|---|
| `RuleClassifier.new_classifier(train_path, test_path, model_parameters, model_path=None, algorithm_type='Random Forest')` | Train a sklearn model, extract rules, and build a RuleClassifier |
| `RuleClassifier.load(path)` | Load a pickled RuleClassifier (auto-recompiles native model) |
| `RuleClassifier.process_data(train_path, test_path)` | Load CSV data, apply LabelEncoding, return arrays and feature names |

#### Analysis Pipeline

| Method | Description |
|---|---|
| `execute_rule_analysis(file_path, remove_duplicates="none", remove_below_n_classifications=-1)` | Run the full optimization pipeline |
| `compare_initial_final_results(file_path)` | Compare initial vs. final model with metrics and divergence analysis |

**`remove_duplicates` options:**
- `"none"` -- No redundancy removal
- `"soft"` -- Intra-tree boundary merging only (safe for all algorithms)
- `"medium"` -- Broader boundary definitions
- `"hard"` -- Intra-tree + inter-tree semantic merging (RF/GBDT)
- `"custom"` -- Use the function set via `set_custom_rule_removal()`

**`remove_below_n_classifications` options:**
- `-1` -- Disabled (no low-usage pruning)
- `0` -- Remove rules with zero matches
- `N` -- Remove rules matching N or fewer samples

#### Classification

| Method | Description |
|---|---|
| `classify(sample, final=False)` | Classify a sample dict; returns `(class, votes, probabilities)` |
| `classify_dt(data, rules)` | Static: first-match classification for Decision Trees |
| `classify_rf(data, rules)` | Static: majority-voting classification for Random Forests |
| `classify_gbdt(data, rules, init_scores, is_binary, classes)` | Static: additive scoring for GBDT |

#### Export & Editing

| Method | Description |
|---|---|
| `export_to_native_python(feature_names, filename)` | Write a standalone `.py` classifier file |
| `update_native_model(rules)` | Compile rules into in-memory Python function via `exec()` |
| `edit_rules()` | Open interactive terminal rule editor |
| `set_custom_rule_removal(func)` | Set a custom pruning callback |

#### Metrics

| Method | Description |
|---|---|
| `calculate_structural_complexity(rules, n_features_total)` | Compute interpretability metrics for a rule set |
| `display_metrics(y_true, y_pred, correct, total, file, class_names)` | Print accuracy, precision, recall, F1, specificity, confusion matrix |

### Rule

Represents a single decision path (root to leaf). Uses `__slots__` for memory efficiency.

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` | Unique identifier (e.g., `"DT1_Rule36_Class0"`, `"RF5_Rule12_Class1"`) |
| `class_` | `str` | Predicted class label |
| `conditions` | `List[str]` | Human-readable conditions (e.g., `["v1 <= 3.5", "v2 > 1.2"]`) |
| `parsed_conditions` | `List[Tuple]` | Pre-parsed `(variable, operator, threshold)` tuples |
| `usage_count` | `int` | Number of test samples matched |
| `error_count` | `int` | Wrong predictions count |
| `leaf_value` | `float` | Raw residual at the leaf (GBDT only) |
| `learning_rate` | `float` | Learning rate (GBDT only) |
| `contribution` | `float` | `learning_rate * leaf_value` (GBDT only) |
| `class_group` | `str` | Class group this tree contributes to (GBDT only) |

**Name conventions after optimization:**
- `Rule1_&_Rule2` -- Merged sibling rules
- `Rule1_promoted` -- Sibling promoted after partner was pruned
- `Rule1_edited` -- Manually edited via interactive editor

### Analyzer Classes

Specialized wrappers for algorithm-specific analysis:

| Class | Algorithm | Extra Tracking |
|---|---|---|
| `DTAnalyzer` | Decision Tree | Intra-tree redundancy, low-usage |
| `RFAnalyzer` | Random Forest | Intra-tree, inter-tree, low-usage |
| `GBDTAnalyzer` | GBDT | Intra-tree, low-impact, low-usage |

Each provides `execute_rule_analysis()` and `compare_initial_final_results()` with algorithm-specific progress tracking and redundancy breakdowns.

---

## Project Structure

```
pyruleanalyzer/
├── pyruleanalyzer/
│   ├── __init__.py              # Exports: RuleClassifier, Rule, DTAnalyzer, RFAnalyzer, GBDTAnalyzer
│   ├── rule_classifier.py       # Core engine: Rule class, RuleClassifier class
│   ├── dt_analyzer.py           # Decision Tree analyzer (wraps RuleClassifier)
│   ├── rf_analyzer.py           # Random Forest analyzer (wraps RuleClassifier)
│   └── gbdt_analyzer.py         # GBDT analyzer (wraps RuleClassifier)
├── examples/
│   ├── data/                    # CSV datasets for testing
│   ├── files/                   # Generated outputs (pkl, txt, py)
│   ├── main_DT.py               # End-to-end Decision Tree pipeline
│   ├── main_RF.py               # End-to-end Random Forest pipeline
│   ├── main_GBDT.py             # End-to-end GBDT pipeline
│   ├── edited_DT.py             # Interactive rule editing (DT)
│   ├── edited_RF.py             # Interactive rule editing (RF)
│   ├── sklearn_vs_ruleclassifier_DT.py   # Benchmark: sklearn vs native export (DT)
│   ├── sklearn_vs_ruleclassifier_RF.py   # Benchmark: sklearn vs native export (RF)
│   └── sklearn_vs_ruleclassifier_GBDT.py # Benchmark: sklearn vs native export (GBDT)
├── docs/                        # Sphinx documentation source
├── pyproject.toml
├── setup.py
└── README.md
```

---

## Output Formats

pyRuleAnalyzer produces three types of output files:

| Format | File | Description |
|---|---|---|
| **Pickle** (`.pkl`) | `sklearn_model.pkl` | The original trained sklearn model |
| | `initial_model.pkl` | RuleClassifier before optimization |
| | `final_model.pkl` | RuleClassifier after optimization |
| | `edited_model.pkl` | RuleClassifier after manual editing |
| **Text Report** (`.txt`) | `output_classifier_*.txt` | Initial analysis: rules, accuracy, per-rule stats |
| | `output_final_classifier_*.txt` | Comparison report: before/after metrics, divergences, interpretability |
| **Standalone Python** (`.py`) | `*_classifier.py` | Zero-dependency predict function (DT/RF: no imports; GBDT: `math` only) |

---

## Documentation

Full documentation is available at:

**https://grupocybersegurancavirtus.github.io/pyruleanalyzer/**

To build the docs locally:

```bash
pip install -e .[docs]
cd docs
make html
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

**Developed by [GrupoCybersegurancaVirtus](https://github.com/GrupoCybersegurancaVirtus)**
