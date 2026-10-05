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
  - [High-Performance Batch Prediction](#5-high-performance-batch-prediction)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
  - [Decision Tree](#decision-tree)
  - [Random Forest](#random-forest)
  - [Gradient Boosting (GBDT)](#gradient-boosting-gbdt)
- [Batch Prediction & Probabilities](#batch-prediction--probabilities)
- [Binary Export & Loading](#binary-export--loading)
- [Full Arduino/ESP32 Sketch Export](#full-arduinoesp32-sketch-export)
- [Modeling for Arduino / ESP32](#modeling-for-arduino--esp32)
- [CPN Tools HCPN Export (Formal Verification)](#cpn-tools-hcpn-export-formal-verification)
- [Automatic Model Checking](#automatic-model-checking)
- [Verified Pipeline (train -> verify -> refine -> verify -> deploy)](#verified-pipeline-train---verify---refine---verify---deploy)
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
| **Low-Usage Refinement** | Removes rarely-triggered rules with automatic sibling promotion |
| **Custom Refinement** | Inject your own rule removal logic via callback functions |
| **Native Python Compilation** | Compiles rules into optimized `if/else` Python code (in-memory via `exec()` and file export) |
| **Batch Prediction** | Vectorized `predict_batch()` using compiled tree arrays -- with optional C extension for **1.45x faster inference than sklearn** |
| **Binary Export** | Compact `.bin` format for instant model loading via `load_binary()` |
| **C Header Export** | Standalone `.h` file with a `predict()` function for Arduino and embedded targets |
| **C Extension Acceleration** | Optional compiled C extension for tree traversal; falls back gracefully to numpy if unavailable |
| **HCPN Export** | Converts the model into a CPN Tools `.cpn` Hierarchical Coloured Petri Net |
| **Automatic Model Checking** | Builds the reachability graph of the generated net and answers 18 CTL properties without leaving Python |
| **Verified Pipeline** | One call: train, verify, refine, verify again, export, deploy -- gated on the verification |
| **ASK-CTL Generation** | Writes the CPN Tools SML query script matching the exported model |
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

#### b) Semantic Redundancy Removal (Inter-Tree, Gradient Boosting)

Different boosting stages of the same class channel may contain **rules with identical regions** (same set of conditions). With `refine_between_trees=True` they are folded into one rule placed in one of the trees, whose contribution is the **sum** of theirs; the other trees abstain (add 0) on that region, so the score -- and every prediction -- is unchanged:

```
    Stage 3:                    Stage 7:                     Merged (stage 3):
    Rule "GBDT1T3_Rule2":       Rule "GBDT1T7_Rule5":        v1 <= 3.5
      v1 <= 3.5                   v1 <= 3.5          ──>      v2 > 1.2
      v2 > 1.2                    v2 > 1.2                    contribution = c3 + c7
      contribution c3             contribution c7             (stage 7: no rule there)
```

Random Forest refuses this stage: its prediction is the *average* of one distribution per tree, and removing voters changes the average.

#### c) Low-Usage Refinement with Sibling Promotion

Rules that match very few (or zero) test samples are candidates for removal. When a rule is refined (removed), its **sibling is promoted** by stripping its last condition (since the distinguishing split no longer exists):

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

Usage counts follow the rules through the refinement: a merged parent carries the sum of its two leaves' counts, and a promoted sibling adds the count of the rule it absorbed, so every final rule's `usage_count` is the number of refinement samples in its (possibly enlarged) region. The counts are measured on the data passed to `execute_rule_refinement`; evaluate the refined model on data that was **not** used there, or the reported accuracy is optimistic.

#### d) Custom Refinement

You can inject custom logic to remove specific rules based on domain knowledge. Provide a callback function that takes the list of rules and returns the rules to be kept:

```python
def my_custom_refiner(rules):
    # Keep only rules that predict Class 1
    return [r for r in rules if r.class_ == 1]

classifier.set_custom_rule_removal(my_custom_refiner)
```

### 3. Classification Strategies

Each algorithm type uses a different strategy to classify new samples:

```
    Decision Tree                Random Forest               GBDT
    ─────────────                ─────────────               ────

    First-match:                 Soft voting:                Additive scoring:

    for rule in rules:           p = zeros(n_classes)        for class_group:
      if all conditions          for tree:                     score = init_score
         match:                    if a rule matches:          for tree:
        return rule.class            p += its distribution       if match:
                                        (normalised)               score += contribution
    return default_class         return argmax(p)            binary:  score >= 0
                                  (default_class if no       multi:   argmax(scores)
                                   tree matched)
```

Every engine -- `classify`, `predict_batch`, the native function, the Python / binary / C / Arduino exports and the CPN -- computes exactly this function, for the initial and for the refined rule set (`tests/test_rule_fidelity.py` checks all of them against a reference implementation, on samples placed on and one ULP around every threshold). Models built from scikit-learn compare `float32(x)` with the thresholds, as scikit-learn does (`input_dtype='float32'`); the exports carry that setting.

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

### 5. High-Performance Batch Prediction

For maximum inference speed, pyRuleAnalyzer provides a **vectorized batch prediction** pipeline that compiles the tree structure into flat arrays and traverses them using C-level operations:

```
rule_classifier.py
  └── predict_batch() / predict_batch_proba()
       └── _accel.traverse_tree_batch() / traverse_tree_batch_multi()
            ├── [FAST] _tree_traversal.c  (compiled C extension, ~1.45x faster than sklearn)
            └── [FALLBACK] numpy vectorized traversal (no compiler required)
```

**How it works:**

1. **`compile_tree_arrays()`** converts the rule set into flat numpy arrays (`feature_indices`, `thresholds`, `children_left`, `children_right`, `leaf_classes`) with self-loop sentinel nodes at each leaf. This eliminates active-masking overhead during traversal.

2. **`predict_batch(X)`** feeds the entire dataset through the compiled arrays in one vectorized pass -- no Python-level per-sample loops. For ensemble models (RF/GBDT), all trees are batched into a single C call via `traverse_tree_batch_multi()`.

3. **C extension** (`_tree_traversal.c`) is compiled automatically during `pip install` if a C compiler is available. It iterates all samples over `max_depth` steps, performing branchless left/right child selection. If the C extension is not available, the same algorithm runs in pure numpy.

**Performance results (Iris dataset, 100 estimators):**

| Engine | Relative Speed |
|---|---|
| `predict_batch` (C extension) | **1.45x faster** than sklearn |
| `predict_batch` (numpy fallback) | ~0.35x sklearn speed |
| Native Python export (`.py`) | ~0.05-0.1x sklearn speed |
| `classify()` per-sample | ~0.001x sklearn speed |

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

**C Extension (optional):** During installation, the build system attempts to compile the C extension (`_tree_traversal.c`) for accelerated batch prediction. This requires a C compiler (e.g., MSVC on Windows, gcc on Linux/macOS). If no compiler is found, installation succeeds normally and batch prediction falls back to numpy. You can check whether the C extension is active:

```python
from pyruleanalyzer._accel import HAS_C_EXTENSION
print(f'C extension available: {HAS_C_EXTENSION}')
```

---

## Quick Start

The API is fully compatible with Scikit-Learn.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from pyruleanalyzer import PyRuleAnalyzer

# 1. Load data
df = pd.read_csv("dataset.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Rule usage for the refinement is measured on a split of its own, so the
# test set stays unseen by every step that changes the model.
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# 2. Create and Train (extracts rules automatically)
model = PyRuleAnalyzer.new_model(model='Decision Tree')
model.fit(X_train, y_train)

# 3. Optimize rules (removes redundancies + refines low-usage)
model.execute_rule_refinement(
    X=X_val, y=y_val, # Used for evaluating rule usage
    remove_below_n_classifications=-1
)

# 4. Predict using the Scikit-Learn API
y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"Accuracy: {accuracy:.4f}")

# 5. Interactive Report
report = model.summary_report()
print(f"Total Active Rules: {report['total_rules']}")
```

---

## Usage Examples

### Decision Tree

```python
from pyruleanalyzer import PyRuleAnalyzer

model = PyRuleAnalyzer.new_model(model='Decision Tree')
model.fit(X_train, y_train)

# Analyze and compare
model.execute_rule_refinement(X=X_test, y=y_test, remove_below_n_classifications=-1)
model.compare_initial_final_results(X=X_test, y=y_test)
```

### Random Forest

```python
from pyruleanalyzer import PyRuleAnalyzer

model = PyRuleAnalyzer.new_model(model='Random Forest')
model.fit(X_train, y_train)

# Low-usage rules can be removed; inter-tree merging is refused for RF (soft vote)
model.execute_rule_refinement(X=X_val, y=y_val, remove_below_n_classifications=1)
model.compare_initial_final_results(X=X_test, y=y_test)
```

### Gradient Boosting (GBDT)

```python
from pyruleanalyzer import PyRuleAnalyzer

model = PyRuleAnalyzer.new_model(model='Gradient Boosting Decision Trees')
model.fit(X_train, y_train)

# Usage is measured on a validation split; stages with identical regions are
# folded together (sum of contributions, predictions unchanged)
model.execute_rule_refinement(X=X_val, y=y_val, remove_below_n_classifications=1,
                              refine_between_trees=True)
model.compare_initial_final_results(X=X_test, y=y_test)
```

### Batch Prediction & Probabilities

Batch predictions run automatically using our vectorized high-performance C extension (if available) or NumPy when you call `predict()` or `predict_proba()` with multiple samples (DataFrames or 2D arrays):

```python
# Predict all samples at once
predictions = model.predict(X_test)

# Get class probabilities
probabilities = model.predict_proba(X_test)
print(f'Shape: {probabilities.shape}')  # (n_samples, n_classes)
```

### Export to Standalone Python

Generates a self-contained `.py` file with the decision logic as pure Python code:

```python
model.to_python("my_classifier.py")
```

### Binary Export & Loading

Export a compiled model as a compact binary file for fast loading, without needing the original training data or sklearn:

```python
# Export to binary
model.to_binary('model.bin')

# Later, load the binary model (no sklearn needed)
loaded = RuleClassifier.load_binary('model.bin')
preds = loaded.predict(X_test)
```

The binary format uses a compact encoding (magic `b'PYRA'`, version 1) that stores only the tree arrays -- typically **much smaller** than pickle files.

### C Header Export (Embedded/Arduino)

Export a standalone C header file for use on microcontrollers or embedded systems:

```python
# Export to C header
model.to_c_header('model.h')
```

The generated `.h` file contains:
- Const arrays for the compiled tree structure
- A self-contained `predict(const float *features)` function
- No external dependencies -- ready for Arduino, STM32, ESP32, etc.

```c
// Usage in Arduino/C:
#include "model.h"

float sample[] = {5.1, 3.5, 1.4, 0.2};
int predicted_class = predict(sample);
```

### Full Arduino/ESP32 Sketch Export

For a **complete ready-to-upload sketch** (with `setup()`, `loop()`, sensor placeholders and Serial output), use the `full_pipeline` with `generate_arduino_sketch=True`:

```python
from pyruleanalyzer import full_pipeline

results = full_pipeline(
    train_csv='train.csv',
    target_feature='Target',
    model_type='Decision Tree',
    max_depth=10,
    generate_arduino_sketch=True,  # ← generates .ino file
    board_model='uno',              # or 'auto' for auto-detect
    serial_baud=115200,
)

# Generated files:
print(results['generated_files']['arduino'])   # path to model.ino
print(results['memory_check'])                  # flash/ram compatibility check
```

The generated `.ino` sketch is **fully self-contained**:
- Tree data as inline C arrays (no external files needed)
- `pyra_traverse_tree()` + `pyra_predict()` in C for fast inference
- `setup()` prints model metadata to Serial
- `loop()` reads features, runs prediction, outputs JSON via Serial
- `read_features()` with TODO placeholders for your sensors

```cpp
// Generated sketch (model.ino):
#define SERIAL_BAUD 115200
float features[4];

void read_features(void) {
    features[0] = 0.0; // TODO: read sensor 1
    features[1] = 0.0; // TODO: read sensor 2
    // ... edit with real sensor readings
}

void loop(void) {
    read_features();
    int32_t result = pyra_predict((const double*)features);
    Serial.print(F("{\"class\":"));
    Serial.print(result);
    Serial.println("}");
    delay(1000);
}
```

**Board compatibility:**

| Board | Flash | SRAM | Suitable for |
|-------|-------|------|--------------|
| Uno / Nano | ~140 KB | 2 KB | Simple models (~<5K nodes) |
| Mega | ~1 MB | 8 KB | Medium models |
| ESP32 | ~4 MB | 512 KB | Complex models (RF, GBDT) |

The pipeline auto-estimates memory usage and checks compatibility before generating. If the model is too large for the target board, it warns with "OVER!".

**Demo script:**

```bash
# Train on Iris + generate Uno sketch automatically:
python examples/arduino_example.py --dataset iris

# Try Wine dataset (larger model):
python examples/arduino_example.py --dataset wine --board mega

# Force ESP32 target for a larger Random Forest:
python examples/arduino_example.py \
    --dataset wine \
    --model "Random Forest" \
    --n-estimators 50 \
    --board esp32
```

For full details, see [docs/tutorials/arduino.rst](docs/tutorials/arduino.rst).

### Modeling for Arduino / ESP32

For in-depth guidance on training and optimizing models specifically for Arduino/ESP32 deployment — including depth strategies, memory validation, feature selection, and board-specific recommendations — see the dedicated modeling guide:

**[docs/tutorials/modeling_for_arduino.rst](docs/tutorials/modeling_for_arduino.rst)**

This guide covers:
- Depth vs. accuracy trade-offs per board type
- Memory estimation formulas (Flash + SRAM)
- Feature selection strategies for embedded systems
- GBDT optimization for microcontrollers
- Optimization checklist and troubleshooting

### CPN Tools HCPN Export (Formal Verification)

Convert a trained model into a **Hierarchical Coloured Petri Net (HCPN)** in the
native [CPN Tools](https://cpntools.org) `.cpn` format, so it can be opened,
visualised, simulated and formally verified (reachability, liveness,
boundedness) directly in CPN Tools. You can generate the HCPN for **both** the
initial model and the final, post-refinement model:

```python
analyzer = PyRuleAnalyzer.create(
    train_path="train.csv", test_path="test.csv",
    model="Gradient Boosting Decision Trees", refine=False)

analyzer.execute_rule_refinement(test_path="test.csv",
                                 remove_below_n_classifications=1)

# Generates files/gbdt_initial.cpn and files/gbdt_final.cpn
analyzer.export_hcpn("gbdt", which="both", sample=X_test.iloc[0])
```

Or via the generic export / the low-level classifier method:

```python
analyzer.classifier.export("gbdt", formats=["cpn"])          # uses refined rules if present
analyzer.classifier.to_cpn_tools("model.cpn", use_final=False)  # explicit rule set
```

The conversion implements the proofs of correctness from the accompanying
article (*Coloured Petri Nets-Based Modeling and Validation of Gradient Boosting
Decision Trees*):

- **Theorem 1 (Decision Tree → CPN).** Each root-to-leaf path becomes one
  transition whose guard is the Boolean translation of the path conditions;
  exactly one transition is enabled per sample. Each tree is emitted as its own
  CPN subpage with `P_in`/`P_out` ports.
- **Theorem 2 (GBDT → HCPN).** Each class channel evaluates the `M` boosting
  stages sequentially and accumulates `s + η·vₘ` from the initial estimator
  `s₀`. Every stage is a *substitution transition* bound to its tree subpage.
- **Decision module.** Binary GBDT decides on the sign of the score (`s_M ≥ 0`, scikit-learn's `raw_predictions >= 0`);
  multiclass GBDT uses NumPy-compatible `argmax` tie-breaking (lowest index wins
  ties). Decision Tree models export as a single CPN page, and Random Forest as
  a top page with one tree subpage per estimator — each emitting its leaf's
  class-probability vector — plus a **soft-voting** decision transition that
  sums the vectors and takes the `argmax`, reproducing scikit-learn's `predict`.

The generated model uses a `SAMPLE` record colour set (one `REAL` field per
feature) and a real input sample as the initial marking, so it is immediately
simulatable. Supported for **Decision Tree, Random Forest and GBDT** (binary and
multiclass), with no extra dependencies.

```bash
python examples/generate_hcpn_example.py                       # binary GBDT
python examples/generate_hcpn_example.py --classes 3           # multiclass GBDT
python examples/generate_hcpn_example.py --model "Random Forest"
```

### Automatic Model Checking

Exporting the net is half the job; the other half is checking that it still
behaves like a classifier. pyRuleAnalyzer builds the **occurrence graph** of the
generated `.cpn` under the Coloured Petri Net firing rule -- real token values,
no abstraction, read from the file CPN Tools would open -- and verifies it:

```python
from pyruleanalyzer import check_cpn

result = check_cpn("files/gbdt_final.cpn", class_labels=[0, 1, 2],
                   samples=X_test.iloc[:3],                  # model checked
                   classifier=analyzer.classifier, test_samples=X_test)
print(result.report())
```

```
Model checking: hcpn_mc3_final.cpn  [Gradient Boosting Decision Trees]
============================================================================
  occurrence graph : 19685 nodes, 56864 arcs
  SCC graph        : 19685 nodes, 56864 arcs, 1 dead marking(s), 1 home marking(s)
  semantics        : CTL over the occurrence graph, dead markings with a self-loop

  CTL model checking on the occurrence graph
  ------------------------------------------------------------------------------
  A1   Termination                  AF dead                                  PASS
  A2   No spurious deadlock         !EF(dead & !pred)                        PASS
  A3   Inevitable decision          AF pred                                  PASS
  A4   Prediction recoverability    AG EF pred                               PASS
  A5   Unique output                AG |Prediction| <= 1                     PASS
  A6   Valid label                  AG Prediction subset L                   PASS
  A7   Safeness                     AG forall p: |p| <= 1                    PASS
  B1   Leaf determinism             forall T: AG |EN(leaves_T)| <= 1         PASS
  B2   Inevitable leaf selection    forall T: AF EN(leaves_T)                PASS
  B3   Single tree output           forall T: AG |out_T| <= 1                PASS
  C1   Stage precedence             forall k,m: !E[!acc(k,m-1) U acc(k,m)]   PASS
  C2   No premature score           forall k: !E[!acc(k,M) U score(k)]       PASS
  C3a  Inevitable decision firing   AF EN(decide)                            PASS
  C3b  Decision determinism         AG |EN(decide)| <= 1                     PASS

  SCC analysis of the occurrence graph
  ------------------------------------------------------------------------------
  A8   Home marking                 one terminal SCC                         PASS

  Structural guard analysis (every input)
  ------------------------------------------------------------------------------
  B4   Guard disjointness           forall T, i!=j: box_i & box_j = {}       PASS

  16 passed, 0 failed, 0 skipped -- VERIFIED
```

Four techniques, reported separately because they establish different things:

| Technique | Properties | Establishes |
|---|---|---|
| **CTL model checking** on the occurrence graph | A1-A7, B1-B3, C1-C3b, D1-D2b | temporal properties of every execution, for the input in the initial marking |
| **SCC analysis** | A8 home marking | a marking reachable from every marking (not a CTL formula) |
| **Structural guard analysis** | B4 guard disjointness | the leaves partition the feature space, for **every** input |
| **Conformance testing** | PC | the net computes the classifier's class, on the tested inputs |

Readings worth keeping straight: A3 is *inevitability* (`AF`), not
reachability; A4 (`AG EF pred`) is *not* a home-marking property -- that is A8;
B1 is per input while B4 covers every input, so a refinement overlap the tested
input never reaches passes B1 and fails B4; and the temporal properties say the
net *behaves like a classifier*, not that it computes the right class -- a net
predicting a wrong but valid class passes every CTL property and fails only PC.

**Dead markings.** The checker uses textbook CTL with a self-loop on dead
markings, so a run that stops before `phi` refutes `AF phi`. CPN Tools' ASK-CTL
does not: its `EV` is vacuously true at a dead marking (read from
`cpnsim/statespacefiles/ASKCTL/ASKCTL.sml`, and confirmed by running it), so a
net that deadlocks before predicting satisfies `EV(PRED)`. The generated
ASK-CTL scripts encode `AF phi` as `EV(phi) AND NOT E[not phi U (dead and not phi)]`.

**Validated against CPN Tools.** `pyruleanalyzer.cpntools_oracle` runs the
CPN Tools 4.0.1 simulator headlessly -- compiling the net, entering the
state-space tool exactly as the GUI does, and evaluating the generated ASK-CTL
queries -- and `pyruleanalyzer.cpn_mutants` derives one mutant per property, so
agreement is also measured on nets where the properties fail:

```bash
python examples/cpn_tools_crosscheck.py        # -> files/crosscheck/report.md
```

**State-space explosion.** A `K`-class GBDT with `M` stages per class has
`(3M+3)^K + 2` markings (the test suite checks the formula). Beyond `max_nodes`
the state-space properties get no verdict; the structural analysis and the
conformance test still run.

From the analyzer, verifying both stages reports what the refinement changed:

```python
results = analyzer.model_check(which="both", samples=X_test.iloc[:3],
                               test_samples=X_test, askctl=True)
```

`askctl=True` also writes, next to each `.cpn`, the CPN Tools script asking the
same questions (evaluate `use "<file>.sml";` after *Enter State Space*).

---

### Verified Pipeline (train -> verify -> refine -> verify -> deploy)

`verified_pipeline()` chains the whole flow and makes the verification a gate
rather than a report:

```python
from pyruleanalyzer import verified_pipeline

result = verified_pipeline(
    train_csv="data/train.csv", test_csv="data/test.csv",
    target_feature="Target",
    model_type="Gradient Boosting Decision Trees",
    params={"n_estimators": 8, "max_depth": 3, "random_state": 42},
    remove_below_n_classifications=1,
    verify_samples=3,
    export_formats=("python", "binary", "c", "arduino"),
)

result["passed"]                       # every property held on both models
result["verification"]["final"].report()
result["exports"]["arduino"]           # files/model.ino
```

```
1. train a scikit-learn model, or take one that is already trained
2. extract the rules and export the initial HCPN
3. model check the initial model                      (baseline)
4. refine the rules
5. model check the refined model                      (acceptance test)
6. export: Python, binary, C header, Arduino sketch
7. deploy to the edge device                          (optional)
```

With `fail_on_violation=True` (the default) step 5 raises `VerificationError`
instead of exporting a model whose net no longer satisfies what the initial one
did; the failing `ModelCheckResult` is attached to the exception.

An already-trained estimator skips step 1:

```python
clf = RandomForestClassifier(n_estimators=50).fit(X_train, y_train)

result = verified_pipeline(X=X, y=y, sklearn_model=clf,
                           export_formats=("c", "arduino"))
```

`PyRuleAnalyzer.from_sklearn(clf)` does the same wrapping on its own, for use
outside the pipeline.

To also flash the board, give the pipeline the board's FQBN and port; it drives
`arduino-cli` and reports the exact commands when the CLI is not installed:

```python
verified_pipeline(..., export_formats=("arduino",),
                  deploy_fqbn="arduino:avr:nano", deploy_port="COM3",
                  upload=True)
```

```bash
python examples/verified_pipeline_example.py
python examples/verified_pipeline_example.py --classes 3 --askctl
python examples/verified_pipeline_example.py --arduino --fqbn arduino:avr:nano --port COM3 --upload
```

For a full hardware-in-the-loop run (compile, upload, drive the board over
serial and compare its predictions against the host), see
`examples/arduino_hardware_test.py`.

### Interactive Rule Editing

After analysis, you can manually edit rules through an interactive terminal interface:

```python
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

### Selective Multi-Format Export

Use `export_all()` to export to multiple formats with fine-grained control:

```python
# Export to Python and Binary (skip C)
classifier.export_all(
    base_name="files/my_model",
    feature_names=feature_names,
    export_python=True,
    export_binary=True,
    export_c=False
)

# Alternative: Export only C header for embedded systems
classifier.export_all(
    base_name="embedded/model",
    feature_names=feature_names,
    export_python=False,
    export_binary=False,
    export_c=True
)
```

**Generated files:**
- `my_model.py` - Standalone Python classifier
- `my_model.bin` - Compact binary format (fast loading)
- `my_model.h` - C header for embedded systems

### File Output Control

Control which files are automatically saved during the pipeline:

```python
# Create classifier without saving intermediate files
classifier = RuleClassifier.new_classifier(
    train_path, test_path, model_parameters,
    algorithm_type='Decision Tree',
    save_initial_model=False,   # Don't save initial_model.pkl
    save_sklearn_model=False    # Don't save sklearn_model.pkl
)

# Execute analysis without saving final model and reports
classifier.execute_rule_refinement(
    test_path,
    remove_low_usage=-1,
    save_final_model=False,  # Don't save final_model.pkl
    save_report=False        # Don't save output_classifier_*.txt
)
```

**Default output location:** `files/` (project root)
- Models: `initial_model.pkl`, `sklearn_model.pkl`, `final_model.pkl`
- Reports: `output_classifier_*.txt`, `output_final_classifier_*.txt`
- Exports: `*.py`, `*.bin`, `*.h`

### Custom Rule Removal

```python
def remove_short_rules(rules):
    """Remove rules with fewer than 2 conditions."""
    kept = [r for r in rules if len(r.conditions) >= 2]
    removed = [(r, r) for r in rules if len(r.conditions) < 2]
    return kept, removed

classifier.set_custom_rule_removal(remove_short_rules)
new_rules, removed = classifier.adjust_and_remove_rules("custom")
```

---

## API Reference

### RuleClassifier

The main class that handles the entire pipeline.

#### Factory & I/O

| Method | Description |
|---|---|
| `RuleClassifier.new_classifier(train_path, test_path, model_parameters, model_path=None, algorithm_type='Random Forest', save_initial_model=True, save_sklearn_model=True)` | Train a sklearn model, extract rules, and build a RuleClassifier |
| `RuleClassifier.load(path)` | Load a pickled RuleClassifier (auto-recompiles native model) |
| `RuleClassifier.load_binary(filepath)` | Load a RuleClassifier from a compact `.bin` file (arrays only, no Rule objects) |
| `RuleClassifier.process_data(train_path, test_path)` | Load CSV data, apply LabelEncoding, return arrays and feature names |

#### Analysis Pipeline

| Method | Description |
|---|---|
| `execute_rule_refinement(X, y, file_path, remove_below_n_classifications=-1)` | Run the full optimization pipeline |
| `compare_initial_final_results(X, y, file_path)` | Compare initial vs. final model with metrics and divergence analysis |

`execute_rule_refinement` always merges boundary redundancy: sibling leaves of
the same tree that split one variable at one threshold with complementary
operators and give the same output are merged into their parent, until no such
pair is left. Predictions are unchanged.

**`adjust_and_remove_rules(method)` options** (one merge round, called directly):
- `"boundary"` -- Intra-tree boundary merging (default; safe for all algorithms)
- `"custom"` -- Use the function set via `set_custom_rule_removal()`

**`remove_below_n_classifications` options:**
- `-1` -- Disabled (no low-usage refinement)
- `0` -- Remove rules with zero matches
- `N` -- Remove rules matching N or fewer samples

#### Classification

| Method | Description |
|---|---|
| `classify(sample, final=False)` | Classify a sample dict with `initial_rules` (`final=False`) or `final_rules` (`final=True`); returns `(class, votes, probabilities)` |
| `predict_batch(X, feature_names=None, use_final=None)` | Vectorized batch prediction; `use_final=True/False` selects the refined/initial rule set (compiled on first use), `None` uses the arrays compiled last; returns int32 class labels |
| `predict_batch_proba(X, feature_names=None, use_final=None)` | Vectorized batch probability prediction; returns float64 array of shape `(n_samples, n_classes)` |
| `compile_tree_arrays(rules=None, feature_names=None)` | Compile rules into flat numpy arrays for `predict_batch()` / `predict_batch_proba()`; raises if the rules of a tree overlap (not a partition) |
| `classify_dt(data, rules)` | Static: first-match classification for Decision Trees |
| `classify_rf(data, rules)` | Static: majority-voting classification for Random Forests |
| `classify_gbdt(data, rules, init_scores, is_binary, classes)` | Static: additive scoring for GBDT |

#### Export & Editing

| Method | Description |
|---|---|
| `export(base_name, formats, feature_names)` | Export to multiple formats via list (e.g., `['python', 'binary']`) |
| `export_all(base_name, feature_names, export_python=True, export_binary=True, export_c=False)` | Export to multiple formats with selective boolean flags |
| `export_to_native_python(feature_names, filename)` | Write a standalone `.py` classifier file |
| `export_to_binary(filepath='model.bin')` | Export compiled tree arrays to a compact binary file |
| `export_to_c_header(filepath='model.h', guard_name='PYRULEANALYZER_MODEL_H')` | Export a standalone C header for embedded targets |
| `to_cpn_tools(filepath, use_final=None, sample=None, feature_names=None)` | Export the model as a CPN Tools `.cpn` HCPN (DT/RF/GBDT) for opening, visualising and simulating in CPN Tools |
| `update_native_model(rules)` | Compile rules into in-memory Python function via `exec()` |
| `edit_rules()` | Open interactive terminal rule editor |
| `set_custom_rule_removal(func)` | Set a custom refinement callback |

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

Each provides `execute_rule_refinement()` and `compare_initial_final_results()` with algorithm-specific progress tracking and redundancy breakdowns.

### Verification API

| Object | Purpose |
|---|---|
| `PyRuleAnalyzer.from_sklearn(model, feature_names=None)` | Wrap an already-fitted sklearn estimator and extract its rules |
| `PyRuleAnalyzer.export_hcpn(base_name, which="both", sample=None)` | Write the `.cpn` model(s) for CPN Tools |
| `PyRuleAnalyzer.model_check(which="final", samples=None, ...)` | Export if needed and verify; `which="both"` also reports the regressions |
| `check_cpn(path, samples=None, classifier=None, class_labels=None)` | Verify an existing `.cpn` in one call |
| `CPNToolsOracle()` / `compare_with_oracle(path)` | Run CPN Tools 4.0.1 headlessly and compare every value |
| `make_mutants(path, out_dir)` | One mutant per property, for sensitivity checks |
| `CPNModelChecker(path)` | The checker itself: `.check()`, `.ml_program()`, `.export_askctl()` |
| `ModelCheckResult` | `.passed`, `.failures`, `.properties`, `.report()`, `.to_dict()`, `.latex_rows()`, `.latex_table()` |
| `compare_results(initial, final)` | Side-by-side verdicts, flagging every regression |
| `PROPERTY_CATALOG` | The property descriptors: id, name, CTL formula, kind, families |
| `verified_pipeline(...)` | The whole flow, gated on the verification |
| `VerificationError` | Raised when the pipeline refuses to export; carries the failing result |

---

## Project Structure

```
pyruleanalyzer/
├── pyruleanalyzer/
│   ├── __init__.py              # Exports: RuleClassifier, Rule, DTAnalyzer, RFAnalyzer, GBDTAnalyzer
│   ├── rule_classifier.py       # Core engine: Rule class, RuleClassifier class
│   ├── _accel.py                # Acceleration layer (C extension / numpy fallback)
│   ├── _tree_traversal.c        # C extension for vectorized tree traversal
│   ├── dt_analyzer.py           # Decision Tree analyzer (wraps RuleClassifier)
│   ├── rf_analyzer.py           # Random Forest analyzer (wraps RuleClassifier)
│   ├── gbdt_analyzer.py         # GBDT analyzer (wraps RuleClassifier)
│   ├── cpn_tools_exporter.py    # HCPN export in the CPN Tools .cpn format
│   ├── cpn_semantics.py         # Reads a .cpn back and evaluates what it denotes
│   ├── cpn_statespace.py        # Occurrence graph under the CPN firing rule + SCCs
│   ├── model_checker.py         # CTL / SCC / structural / conformance verification
│   ├── cpntools_oracle.py       # Runs CPN Tools 4.0.1 headlessly for cross-checks
│   ├── cpn_mutants.py           # One mutant per property (sensitivity)
│   ├── full_pipeline.py         # Data -> Arduino sketch shortcut
│   └── verified_pipeline.py     # Train -> verify -> refine -> verify -> deploy
├── tests/
│   ├── test_dt_invariants.py    # Decision Tree invariant test suite (25 configs)
│   ├── test_rf_invariants.py    # Random Forest invariant test suite (25 configs)
│   ├── test_gbdt_invariants.py  # GBDT invariant test suite (25 configs)
│   ├── test_cpn_export.py       # .cpn structure, DTD validity, prediction consistency
│   └── test_model_checker.py    # Properties, mutants, CPN Tools agreement, gating
├── examples/
│   ├── data/                    # CSV datasets for testing
│   ├── files/                   # Generated outputs (pkl, txt, py, bin, h)
│   ├── main_DT.py               # End-to-end Decision Tree pipeline
│   ├── main_RF.py               # End-to-end Random Forest pipeline
│   ├── main_GBDT.py             # End-to-end GBDT pipeline
│   ├── edited_DT.py             # Interactive rule editing (DT)
│   ├── edited_RF.py             # Interactive rule editing (RF)
│   ├── sklearn_vs_ruleclassifier_DT.py   # Benchmark: sklearn vs pyRuleAnalyzer (DT)
│   ├── sklearn_vs_ruleclassifier_RF.py   # Benchmark: sklearn vs pyRuleAnalyzer (RF)
│   └── sklearn_vs_ruleclassifier_GBDT.py # Benchmark: sklearn vs pyRuleAnalyzer (GBDT)
├── docs/                        # Sphinx documentation source
├── pyproject.toml
├── setup.py
└── README.md
```

---

## Output Formats

pyRuleAnalyzer produces seven types of output files:

| Format | File | Description |
|---|---|---|
| **Pickle** (`.pkl`) | `sklearn_model.pkl` | The original trained sklearn model |
| | `initial_model.pkl` | RuleClassifier before optimization |
| | `final_model.pkl` | RuleClassifier after optimization |
| | `edited_model.pkl` | RuleClassifier after manual editing |
| **Text Report** (`.txt`) | `output_classifier_*.txt` | Initial analysis: rules, accuracy, per-rule stats |
| | `output_final_classifier_*.txt` | Comparison report: before/after metrics, divergences, interpretability |
| **Standalone Python** (`.py`) | `*_classifier.py` | Zero-dependency predict function (DT/RF: no imports; GBDT: `math` only) |
| **Binary** (`.bin`) | `model.bin` | Compact compiled tree arrays for `predict_batch()` / `load_binary()` |
| **C Header** (`.h`) | `model.h` | Standalone `predict()` function for embedded targets (Arduino, STM32, ESP32) |
| **Arduino Sketch** (`.ino`) | `model.ino` | Self-contained sketch with the model inlined, ready for `arduino-cli` |
| **HCPN** (`.cpn`) | `model_initial.cpn` | Hierarchical Coloured Petri Net of the unrefined model, for CPN Tools |
| | `model_final.cpn` | HCPN of the refined model |
| **ASK-CTL** (`.sml`) | `model_final.sml` | CPN Tools query script generated for that specific net |

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
