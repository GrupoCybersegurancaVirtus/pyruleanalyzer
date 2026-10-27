# pyRuleAnalyzer

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/pyruleanalyzer.svg)](https://pypi.org/project/pyruleanalyzer/)

**Turn a scikit-learn tree model into rules you can read, shrink, verify and
ship to a microcontroller -- with the same predictions all the way.**

A Decision Tree, Random Forest or Gradient Boosting model is a black box until
someone writes down what it does. pyRuleAnalyzer writes it down: every
root-to-leaf path becomes a rule. It then removes the rules that add nothing,
exports the model as a standalone Python, binary or C file (or a complete
Arduino/ESP32 sketch), and can prove -- with a Petri net and a model checker --
that the simplified model still behaves like a classifier.

![From a scikit-learn model to rules, a refined model, standalone exports, and an optional verification and deployment](https://raw.githubusercontent.com/GrupoCybersegurancaVirtus/pyruleanalyzer/main/docs/figures/pipeline.png)

* **Rules, not a black box.** Each decision path is a rule with its conditions,
  its class and how often it fired.
* **Refinement that keeps its promises.** Merge redundant sibling leaves, fold
  identical boosting stages, drop rarely used rules -- and keep the rule set a
  clean partition of the feature space.
* **One answer everywhere.** The in-memory engines, the Python, binary and C
  exports and the Petri net all compute the same function as scikit-learn,
  checked sample by sample at every threshold.
* **Fast and small.** A vectorised C extension predicts roughly on par with
  scikit-learn on a 100-tree forest; the binary and C exports are half its
  pickle size.
* **Verified, not just tested.** The model becomes a Hierarchical Coloured
  Petri Net that [CPNCheck](https://github.com/GrupoCybersegurancaVirtus/cpncheck)
  explores. A refinement that breaks the model is stopped before anything is
  exported.
* **To the edge.** Arduino and ESP32 sketches, board memory estimates,
  hardware-in-the-loop tests and a sensor simulator that runs the sketch on
  your PC.

**Contents:**
[Install](#install) ·
[A five-minute tour](#a-five-minute-tour) ·
[The ideas in one page](#the-ideas-in-one-page) ·
[How each model decides](#how-each-model-decides) ·
[Refinement](#refinement) ·
[Predict](#predict-fast) ·
[Export](#export-standalone-classifiers) ·
[Arduino and ESP32](#arduino-and-esp32) ·
[Formal verification](#formal-verification) ·
[Verified pipeline](#the-verified-pipeline) ·
[Edit and customise](#edit-and-customise) ·
[API reference](#api-reference) ·
[Project layout](#project-layout) ·
[Status](#status)

## Install

```bash
pip install pyruleanalyzer
```

Or from source, for development:

```bash
git clone https://github.com/GrupoCybersegurancaVirtus/pyruleanalyzer.git
cd pyruleanalyzer
pip install -e .
```

Python 3.9 or later. Dependencies: `numpy`, `pandas`, `scikit-learn`,
`matplotlib` and [`cpncheck`](https://pypi.org/project/cpncheck/) (pure Python,
the model checker for the exported nets).

**The C extension.** Batch prediction uses a compiled extension. The wheels on
PyPI ship it for Linux (x86_64, aarch64), macOS (Intel and Apple silicon) and
Windows (64-bit), Python 3.9 to 3.13, so `pip install` needs no compiler.
Elsewhere pip builds from source when a C compiler exists, and otherwise falls
back to numpy -- slower, same answers:

```python
from pyruleanalyzer._accel import HAS_C_EXTENSION
print(HAS_C_EXTENSION)   # True when the compiled extension is active
```

## A five-minute tour

### 1. A tree and its rules

Wine again, two features to keep the picture small. Train an ordinary
decision tree through pyRuleAnalyzer and ask for its rules:

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from pyruleanalyzer import PyRuleAnalyzer

X, y = load_wine(return_X_y=True, as_frame=True)
X = X[["proline", "flavanoids"]]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y)
# Rule usage is measured on a split of its own, so the test set stays unseen
# by every step that changes the model.
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.3, random_state=0, stratify=y_train)

model = PyRuleAnalyzer.new_model("Decision Tree", {"random_state": 0})
model.fit(X_train, y_train)
print(model)   # PyRuleAnalyzer(model=Decision Tree, features=2, classes=3, rules=8)
```

A tree is a series of if/else splits; each path from the root to a leaf is one
rule -- a list of conditions and the class the leaf predicts. On the iris
flowers, a depth-3 tree gives these five rules:

![An iris decision tree on the left; the five rules extracted from it on the right, one per leaf](https://raw.githubusercontent.com/GrupoCybersegurancaVirtus/pyruleanalyzer/main/docs/figures/rules.png)

For a **Random Forest** the rules of every tree are extracted. For a
**Gradient Boosting** model each leaf also carries the value that its boosting
stage adds to the score.

### 2. Refine

Eight rules is more than this problem needs. Refinement merges the leaves that
split for no reason and drops the rules that almost no sample reaches:

```python
stats = model.execute_rule_refinement(
    X=X_val, y=y_val, remove_below_n_classifications=1)
print(stats)
# {'rules_before': 8, 'rules_after': 3, 'rules_removed': 5, 'reduction_percent': 62.5}

print((model.predict(X_test, use_refined=False) == y_test).mean())   # 0.833  initial
print((model.predict(X_test) == y_test).mean())                      # 0.852  refined
```

![The decision regions of the wine tree: eight boxes before refinement, three after; the boxes never overlap](https://raw.githubusercontent.com/GrupoCybersegurancaVirtus/pyruleanalyzer/main/docs/figures/refinement.png)

Each rule is a box in the feature space, coloured by its class. Refinement
makes the boxes fewer and larger; the boxes of one tree still cover the space
without overlapping, which is the property that keeps every engine in
agreement. (Test accuracy moved by one sample out of 54 here. Refinement trades
a little fidelity for a much simpler model; always measure it on data the
refinement did not see.)

### 3. Read what is left

```python
report = model.summary_report()
for rule in report["rules"]:
    print(rule["name"], "->", rule["class"], rule["conditions"])
# Rule0_Class2_promoted -> 2 ['flavanoids <= 1.425000011920929']
# Rule3_Class1 -> 1 ['flavanoids > 1.425000011920929', 'proline <= 765.0']
# Rule7_Class0_promoted -> 0 ['flavanoids > 1.425000011920929', 'proline > 765.0']
```

Three rules a person can read: low flavanoids means class 2; otherwise
proline above 765 means class 0 and the rest class 1. A name ending in
`_promoted` is a rule that absorbed the region of a sibling that was removed;
`_&_` marks merged siblings.

### 4. Ask the model checker

```python
results = model.model_check(which="both", samples=X_test.iloc[:1],
                            test_samples=X_test)
```

The rules become a Petri net, [CPNCheck](https://github.com/GrupoCybersegurancaVirtus/cpncheck)
explores it, and 13 properties are checked on the initial model and on the
refined one. The comparison is the acceptance test of the refinement:

![The comparison table printed by model_check(which='both'): every property PASS before and after refinement](https://raw.githubusercontent.com/GrupoCybersegurancaVirtus/pyruleanalyzer/main/docs/figures/comparison.png)

*(The figure shows a small boosting model, which has more properties than a
single tree.)* See [Formal verification](#formal-verification) for what each
one says.

### 5. Ship it

```python
model.to_python("wine_classifier.py")   # no dependencies
model.to_binary("wine.bin")             # compact, loads without scikit-learn
model.to_c_header("wine.h")             # predict() in plain C
```

For a board, `verified_pipeline()` also writes the sketch and, with
`arduino-cli`, uploads it -- see [Arduino and ESP32](#arduino-and-esp32).

## The ideas in one page

| Term | Meaning |
|---|---|
| **Rule** | One root-to-leaf path: conditions on features, plus the leaf's output. |
| **Region (box)** | The set of inputs a rule's conditions select. |
| **Partition** | The rules of a tree cover the feature space and their regions never overlap, so exactly one rule fires per input. |
| **Boundary redundancy** | Sibling leaves that split on the same threshold and give the same output; the split is useless. |
| **Sibling promotion** | When a rule is removed, its sibling takes over the parent's region. |
| **Usage count** | How many samples of the refinement set reach a rule; carried through merges and promotions. |
| **Semantic redundancy** | Boosting stages with identical regions: they are folded into one rule that adds both contributions. |
| **HCPN** | Hierarchical Coloured Petri Net: the model written as places, transitions and guards, in the file format CPN Tools opens. |
| **Occurrence graph** | Every marking the net can reach from the input; what the model checker explores. |
| **Property (A1-D2, PC)** | A question put to the net: does it always decide, never decide twice, never overlap, agree with the model? |
| **Gate** | A refinement that breaks a property that held before is refused, and nothing is exported. |

## How each model decides

Every engine -- the in-memory classifier, the vectorised batch predictor, the
exported Python, binary and C files and the Petri net -- computes the same
function, for the initial and for the refined rule set:

![One sample through a decision tree (the one rule that matches), a random forest (summed per-tree distributions, argmax) and a boosting model (initial score plus each stage's contribution, sign of the score)](https://raw.githubusercontent.com/GrupoCybersegurancaVirtus/pyruleanalyzer/main/docs/figures/strategies.png)

* **Decision Tree**: the sample goes to the one rule whose region contains it;
  its class is the answer. With no matching rule (a hole left by a refinement)
  the model falls back to its default class.
* **Random Forest**: each tree answers with a class distribution; the
  distributions are normalised and summed, and the largest wins (lowest class
  index on a tie). A tree with no matching rule abstains.
* **Gradient Boosting**: the initial score plus the contribution
  (`learning_rate * leaf_value`) of the matching leaf of every stage. Binary
  models decide on `score >= 0`, exactly as scikit-learn does; multiclass
  models take the argmax over the class channels.

**Precision matters.** scikit-learn casts inputs to `float32` before comparing
them with the thresholds. Models built with `fit()` or `from_sklearn()` do the
same (`input_dtype='float32'`), and so do the exports, so a value one ULP above
a threshold goes to the same side as in scikit-learn.
[`tests/test_rule_fidelity.py`](https://github.com/GrupoCybersegurancaVirtus/pyruleanalyzer/blob/main/tests/test_rule_fidelity.py)
compares every engine, including the compiled C header, against a reference
implementation on samples placed on and one ULP around every threshold.

## Refinement

`execute_rule_refinement()` runs the passes below, in this order. The same call
works for all three model types.

| Pass | What it does | Predictions |
|---|---|---|
| **Usage count** | Runs the refinement samples through the rules and counts how often each fires. | unchanged |
| **Boundary merge** | Sibling leaves with the same output and complementary conditions (`<= T` and `> T`) are merged into their parent, repeatedly, until no pair is left. The merged rule carries the sum of the two usage counts. | unchanged |
| **Low-usage removal** | Rules used `remove_below_n_classifications` times or fewer are removed, deepest first; the sibling of a removed rule is promoted. `-1` disables it, `0` removes only never-used rules. | may change |
| **Inter-tree fusion** (GBDT, `refine_between_trees=True`) | Stages of one class with an identical region are folded into one rule whose contribution is the sum. The other stages abstain there. | unchanged |

```python
# Random Forest: low-usage removal only (see below for why not inter-tree).
model.execute_rule_refinement(X=X_val, y=y_val, remove_below_n_classifications=1)

# Gradient Boosting: also fold identical stages.
model.execute_rule_refinement(X=X_val, y=y_val, remove_below_n_classifications=1,
                              refine_between_trees=True)
```

Merging siblings is a promotion upwards:

```
    BEFORE                              AFTER
    ------                              -----
    v1 <= 3.5                           v1 <= 3.5
     |-- v2 <= 1.2  -> class 0     =>    -> class 0        (the v2 split
     `-- v2 >  1.2  -> class 0                               told nothing)
```

Removing a rarely used leaf promotes its sibling the same way when the sibling
is a leaf, so the removed region is absorbed. When the sibling is a subtree
there is no single rule to promote: the removed region becomes a hole, answered
by the model's `default_class` and counted in `classifier.fallback_activations`.
On an unpruned decision tree this reached about 3% of the test samples, so
check that counter after an aggressive removal.

* **Random Forest refuses inter-tree fusion** (with a `RuntimeWarning`): its
  prediction is the average of one distribution per tree, so removing a voter
  changes the average.
* **Measure on fresh data.** Usage counts come from the data passed to the
  refinement. Evaluate the refined model on a different split, or the accuracy
  you see is optimistic.
* `compare_initial_final_results(X=..., y=...)` prints accuracy, precision,
  recall, F1, a confusion matrix, the structural complexity score and the
  samples on which the two models disagree.

### Custom refinement

Supply your own rule to remove, from domain knowledge:

```python
def drop_short_rules(rules):
    # Keep only rules with at least two conditions; report the others as removed.
    kept = [r for r in rules if len(r.conditions) >= 2]
    removed = [(r, r) for r in rules if len(r.conditions) < 2]
    return kept, removed

model.classifier.set_custom_rule_removal(drop_short_rules)
new_rules, removed = model.classifier.adjust_and_remove_rules("custom")
```

## Predict fast

`predict()` and `predict_proba()` take a DataFrame or a 2-D array and run on
compiled tree arrays: the rules are turned into flat numpy arrays once, and
every sample is pushed through them in a single vectorised pass -- in C when
the extension is available, in numpy otherwise. `use_refined=False` answers
with the initial rules.

```python
labels = model.predict(X_test)
proba  = model.predict_proba(X_test)            # shape (n_samples, n_classes)
initial = model.predict(X_test, use_refined=False)
```

![Throughput of scikit-learn, pyRuleAnalyzer with the C extension, with the numpy fallback and the exported Python file; and the size on disk of the scikit-learn pickle, the exported .py, .bin and .h](https://raw.githubusercontent.com/GrupoCybersegurancaVirtus/pyruleanalyzer/main/docs/figures/benchmark.png)

The figure is measured by
[`docs/figures/make_figures.py`](https://github.com/GrupoCybersegurancaVirtus/pyruleanalyzer/blob/main/docs/figures/make_figures.py)
on a 100-tree Random Forest (depth <= 10, 20 features, 10 000 test samples,
one thread). Read it as an order of magnitude, not a promise; run your own
model with the scripts in `examples/`:

* the C extension is **roughly on par with scikit-learn** `predict` (1.04x on
  the machine the figure was last measured on; the ratio depends on the
  machine, the compiler and the model);
* the numpy fallback is about a third of scikit-learn's speed;
* the exported `.py` is meant to run **without** this package or scikit-learn,
  one sample at a time -- not to compete with vectorised code;
* the `.bin` and `.h` files are about half the size of the scikit-learn pickle;
  the generated `.py` is *larger* than the pickle for a forest this size.

Single samples go through `model.classifier.classify(sample_dict, final=True)`
and return `(class, votes, probabilities)`.

## Export standalone classifiers

| Method | Writes | Needs at run time |
|---|---|---|
| `to_python(path)` | nested `if/else` Python, one function `predict(sample)` | nothing (GBDT: `math`) |
| `to_binary(path)` | compact tree arrays (magic `PYRA`, versioned) | this package, to load; not scikit-learn |
| `to_c_header(path)` | const arrays and a `predict(const double *features)` function in C | a C compiler |
| `export(base_name, formats=[...])` | any of the above, plus `"cpn"`, under `files/` | -- |

```python
from pyruleanalyzer import PyRuleAnalyzer

model.to_binary("wine.bin")
loaded = PyRuleAnalyzer.load_binary("wine.bin")    # arrays only, no scikit-learn
labels = loaded.classifier.predict_batch(X_test.values, list(X_test.columns))
```

```python
# wine_classifier.py, as generated for the refined wine tree
def _predict(sample):
    if sample.get('flavanoids', 0) <= 1.425000011920929:
        return 2
    else:
        if sample.get('proline', 0) <= 765.0:
            return 1
        else:
            return 0


def predict(sample):
    return _predict({k: _f32(v) for k, v in sample.items()})   # float32, as scikit-learn
```

A loaded `.bin` supports batch prediction but has no `Rule` objects, so it
cannot be refined or edited. `export(..., use_refined=False)` exports the
initial rules instead of the refined ones.

## Arduino and ESP32

A complete, ready-to-upload sketch comes from the same model:

```python
model.classifier.export_to_arduino_ino("model.ino", board_model="uno")
```

or in one call from a CSV, with a memory check:

```python
from pyruleanalyzer import full_pipeline

results = full_pipeline(
    train_csv="train.csv", target_feature="Target",
    model_type="Decision Tree", max_depth=10,
    generate_arduino_sketch=True, board_model="uno",   # or 'auto'
    serial_baud=115200,
)
print(results["generated_files"]["arduino"])   # path to model.ino
print(results["memory_check"])                 # flash / RAM against the board
```

`full_pipeline()` refines on a validation split held out from the training
data (`validation_size=0.2`), never on the test CSV. A test CSV is matched to
the training columns by name; one without the target column is ignored with a
warning. String labels are encoded with one `LabelEncoder` for both files
(`results["label_classes"]` maps the codes back), and missing values are
filled with 0 with a warning.

The sketch is self-contained: the tree arrays are inline, `pyra_predict()` is
plain C, `setup()` prints the model's metadata and `loop()` reads the features,
predicts and prints JSON on the serial port. `read_features()` is the one place
you edit -- it ships with `TODO` placeholders for your sensors:

```cpp
void read_features(void) {
    features[0] = 0.0; // TODO: read sensor 1
    features[1] = 0.0; // TODO: read sensor 2
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

The tool estimates flash and SRAM and compares them with the target board
(`uno`, `nano`, `mega`, `leonardo`, `esp32`), warning with `OVER!` when the
model will not fit. For depth strategies, feature selection and memory formulas
see the
[modelling guide](https://github.com/GrupoCybersegurancaVirtus/pyruleanalyzer/blob/main/docs/tutorials/modeling_for_arduino.rst).

**Test it without a board, or with one:**

| Script | What it checks |
|---|---|
| `examples/arduino_example.py` | trains on Iris/Wine and writes the sketch for a chosen board |
| `examples/arduino_hardware_test.py` | hardware in the loop: compiles, uploads, sends vectors over serial and compares the board's answers (and latency) with the PC's |
| `examples/arduino_sensor_simulation.py` | runs the **production sketch** (`setup()`/`loop()` as generated) on the PC with g++ and an Arduino shim, over time-varying signals -- synthetic regimes or a replay of your test set -- through the ADC or the serial port |

The simulator found two things worth knowing before you flash: a 10-bit linear
ADC destroys long-tailed features, and a 64-byte serial RX buffer with a
`delay()` in `loop()` silently drops lines longer than the buffer.

## Formal verification

Accuracy tells you the model is usually right. It does not tell you the
*refinement* left a model that still behaves like a classifier: overlapping
rules tend to agree on the class, so a broken rule set can keep its accuracy.
pyRuleAnalyzer therefore turns the rules into a **Hierarchical Coloured Petri
Net** in the native format of [CPN Tools](https://cpntools.org) 4.x, and checks
the net.

```python
model.export_hcpn("wine", which="both", sample=X_test.iloc[0])
# files/wine_initial.cpn and files/wine_final.cpn
```

![The three nets: a decision tree is one page with a transition per rule; a random forest has a substitution transition per tree and a Vote; a boosting model has a channel per class with a stage per tree and a Decide](https://raw.githubusercontent.com/GrupoCybersegurancaVirtus/pyruleanalyzer/main/docs/figures/hcpn.png)

The names are the ones in the files. A *place* holds tokens (here the input
sample, per-tree scores, the prediction); a *transition* fires when its guard
holds on the tokens it consumes. A decision-tree leaf is a transition whose
guard is the path's condition; the double-bordered boxes are *substitution
transitions*, each standing for a page of its own. The nets implement the
constructions proved in the accompanying article (*Coloured Petri Nets-Based
Modeling and Validation of Gradient Boosting Decision Trees*): one transition
per root-to-leaf path with exactly one enabled per sample (Theorem 1), and the
sequential accumulation `s + eta * v` along the stages of each class channel
(Theorem 2). The files open directly in CPN Tools for visualisation and
simulation.

### What gets checked

```python
from pyruleanalyzer import check_cpn

result = check_cpn("files/wine_final.cpn", class_labels=[0, 1, 2],
                   samples=X_test.iloc[:1],
                   classifier=model.classifier, test_samples=X_test)
print(result.report())
```

![The model checker's report for a refined boosting model: occurrence graph size, then A1-A8, B1-B4, C1-C3b and PC, all PASS, 17 passed, VERIFIED](https://raw.githubusercontent.com/GrupoCybersegurancaVirtus/pyruleanalyzer/main/docs/figures/terminal.png)

Four techniques, reported separately because they establish different things:

| Technique | Properties | What a verdict means |
|---|---|---|
| **CTL on the occurrence graph** | A1-A7, B1-B3, C1-C3b, D1-D2b | Temporal properties of every execution, **for the input in the initial marking** |
| **SCC analysis** | A8 home marking | A marking reachable from every marking |
| **Structural guard analysis** | B4 guard disjointness | The leaves partition the feature space, for **every input** |
| **Conformance testing** | PC | The net computes the classifier's class on the tested inputs |

The main properties (catalogue in `PROPERTY_CATALOG`):

| Id | Says |
|---|---|
| A1 / A2 | The net terminates, and never gets stuck without an answer |
| A3 / A4 | A decision is inevitable (`AF`), and always still reachable (`AG EF`) |
| A5 / A6 | At most one output, and it is a valid label |
| A7 / A8 | Safeness (at most one token per place); one home marking |
| B1 / B2 / B3 | Per tree: at most one leaf enabled, one is always selected, one output |
| **B4** | **No two leaves of a tree overlap, for every input** |
| C1-C3 | (GBDT) stages in order, no score before the last stage, one decision |
| D1-D2 | (RF) every tree votes, and the vote is unique |
| PC | The net's prediction equals the model's |

**B4 is the acceptance criterion of a refinement.** B1 is per input: it can
only see overlaps the tested input reaches. B4 reasons about every input. Here
a rule loses its last condition (a bug of the kind a refinement could have):

![model_check on a rule set with an overlap: A1-B3 PASS, B4 FAIL with the pair of rules and an input where both hold, PC skipped, VIOLATION FOUND](https://raw.githubusercontent.com/GrupoCybersegurancaVirtus/pyruleanalyzer/main/docs/figures/violation.png)

Everything the temporal properties say is about behaviour -- the net decides
once, with a valid label -- not about being *right*. A net that predicts a
wrong but valid class passes every CTL property; only PC (net against model)
catches it.

### Dead markings

The checker uses textbook CTL with a self-loop on dead markings, so a run that
stops before `phi` refutes `AF phi`. CPN Tools' ASK-CTL reads `EV` differently:
it is vacuously true at a dead marking, so a net that deadlocks before
predicting satisfies `EV(PRED)`. `model_check(..., askctl=True)` writes the
CPN Tools query script that asks the same questions, encoding `AF phi` as
`EV(phi) AND NOT E[not phi U (dead and not phi)]` to remove that gap.

### Big models: state-space reductions

The graph of an ensemble is large because its trees fire independently: a
Random Forest of `n` trees has `2^n + 2` markings, and a `K`-class boosting
model with `M` stages per class has `(3M+3)^K + 2` (the tests check the
second formula). Beyond `max_nodes` the state-space properties get no verdict,
rather than a guess -- they are reported as *not decided*, and an ensemble
beyond a handful of trees gets there quickly: on a 20-tree Random Forest, 14 of
the 16 properties were not decided at `max_nodes=50_000`, and a 50-tree forest
took about 12 minutes. For ensembles, use partial-order reduction (stubborn
sets), which decides most of them on much smaller graphs:

```python
model.model_check(which="final", reduction="stubborn")
```

Keep in mind what remains when the graph is out of reach: B4 (structural, no
state space) and PC (conformance, one run per input) are still decided, but
they largely hold by construction -- the leaves of a scikit-learn tree are
disjoint and the net is generated from the same rules as the classifier -- so
on an unmodified model they confirm the export rather than add evidence.

A 12-tree forest has 4 098 markings. With the reduction, CPNCheck decides its
properties on 69 smaller graphs: 68 of them hold 1 020 markings between them,
and one -- for D2a, the property that observes every leaf -- is still the full
4 098. The sweep-line, symmetry and equivalence reductions of CPNCheck pass
through `reduction_options={...}`.

### Is the checker right?

* **Against CPN Tools.** `pyruleanalyzer.cpntools_oracle` runs the CPN Tools
  4.0.1 simulator headlessly, enters the state-space tool as the GUI does and
  evaluates the ASK-CTL queries. CPNCheck compares its statistics, verdicts
  and ML library with CPN Tools on its own samples and on 48 nets other people
  wrote (details in the CPNCheck README).
* **Sensitivity.** `pyruleanalyzer.cpn_mutants.make_mutants()` derives one
  mutant per property -- a small edit of a guard, an inscription or an arc that
  makes exactly that property false -- and the tests require every mutant to be
  caught. A checker that always says "true" would agree with CPN Tools on
  correct nets; it would not survive this.

```bash
pytest tests/test_model_checker.py                     # properties, mutants, gate
PYRA_CPNTOOLS=1 pytest tests/test_model_checker.py     # + agreement with CPN Tools
python examples/cpn_tools_crosscheck.py                # the full experiment
python examples/cpncheck_walkthrough.py                # an 11-step guided tour
```

The tree-specific part (the property catalogue, the mutation operators) lives
here as a plugin, `ml_trees`; CPNCheck itself knows nothing about trees and
checks any `.cpn` file.

## The verified pipeline

`verified_pipeline()` chains the whole flow and makes the verification a
**gate** instead of a report:

![Train, extract and export the initial net, check, refine, check again; the gate compares the verdicts and either exports and deploys or raises VerificationError](https://raw.githubusercontent.com/GrupoCybersegurancaVirtus/pyruleanalyzer/main/docs/figures/gate.png)

```python
from pyruleanalyzer import verified_pipeline

result = verified_pipeline(
    X=X, y=y,                                   # or train_csv=..., test_csv=...
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

With `fail_on_violation=True` (the default) the second check raises
`VerificationError`, with the failing result attached, instead of exporting a
model whose net no longer satisfies what the initial one did.

A property that could not be decided -- typically because the occurrence graph
went over `max_nodes` -- is not a pass either. With `fail_on_inconclusive=True`
(the default) such a stage raises `VerificationError` too; with `False` the
pipeline goes on, but `result["passed"]` is still `False` and
`result["inconclusive"]` lists the undecided property ids per stage
(`{"initial": [...], "final": [...]}`). For ensembles, pass
`reduction="stubborn"` (and a larger `max_nodes` if needed) so the properties
get a verdict.

The refinement counts rule usage on a validation split held out from the
training data (`validation_size=0.25` by default; its size is reported as
`result["metrics"]["refinement_rows"]`). The test split is used only for the
accuracy and the verification, so the reported accuracy is not optimistic.

An already-trained estimator skips the training step, and
`PyRuleAnalyzer.from_sklearn()` does the same wrapping on its own:

```python
from sklearn.ensemble import RandomForestClassifier

# A forest's state space grows as 2^n_trees: keep it small and use the
# stubborn-set reduction, or the gate stops on undecided properties.
clf = RandomForestClassifier(n_estimators=10, max_depth=5).fit(X_train, y_train)
result = verified_pipeline(X=X, y=y, sklearn_model=clf, reduction="stubborn",
                           export_formats=("c", "arduino"))

analyzer = PyRuleAnalyzer.from_sklearn(clf)       # outside the pipeline
```

To flash the board too, give the pipeline its FQBN and port; it drives
`arduino-cli`, and reports the exact commands if the CLI is not installed:

```python
verified_pipeline(..., export_formats=("arduino",),
                  deploy_fqbn="arduino:avr:nano", deploy_port="COM3", upload=True)
```

```bash
python examples/verified_pipeline_example.py
python examples/verified_pipeline_example.py --classes 3 --askctl
python examples/verified_pipeline_example.py --arduino --fqbn arduino:avr:nano --port COM3 --upload
```

## Edit and customise

**Edit rules by hand** in a terminal editor -- add or remove conditions, change
a class, delete a rule:

```python
model.edit_rules()
```

**Control what is written to disk.** Everything goes to `files/` (created if it
does not exist) unless you give a path:

| Call | Flag | File | Default |
|---|---|---|---|
| `new_classifier()` | `save_initial_model` | `initial_model.pkl` | `True` |
| `new_classifier()` | `save_sklearn_model` | `sklearn_model.pkl` | `True` |
| `execute_rule_refinement()` (analyzers) | `save_final_model` | `final_model.pkl` | `True` |
| `execute_rule_refinement()` (analyzers) | `save_report` | `output_classifier_*.txt` | `True` |

`PyRuleAnalyzer.execute_rule_refinement()` defaults both to `False`.
`model.save(path)` and `PyRuleAnalyzer.load(path)` pickle a model, recompiling
the native function on load.

**Security.** `PyRuleAnalyzer.load()`, `RuleClassifier.load()` and the
`model_pkl` argument use `pickle`, and checking a `.cpn` file evaluates the
inscriptions written in it: both can run arbitrary code. Only load models and
nets you trust.

## API reference

### PyRuleAnalyzer

The high-level entry point; it wraps a `RuleClassifier` and is a scikit-learn
classifier (`fit`, `predict`, `predict_proba`, `score`, `classes_`,
`get_params`/`set_params`), so `clone`, `Pipeline`, `cross_val_score` and
`GridSearchCV` work with `new_model(...)`:

```python
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(PyRuleAnalyzer.new_model("Decision Tree", {"random_state": 0}),
                      param_grid={"max_depth": [2, 3, 4]}, cv=5).fit(X_train, y_train)
```

The parameters are `model` and the hyperparameters of the scikit-learn
estimator. `fit` trains and extracts the rules; refinement is a separate step
(`execute_rule_refinement`), not part of a search or a cross-validation.

**Labels.** Only integer class labels are supported (`0, 1, 2, ...` or any
integers); string or other non-integer labels raise `ValueError`. Encode them
first, e.g. with `sklearn.preprocessing.LabelEncoder`.

| Method | Description |
|---|---|
| `PyRuleAnalyzer.new_model(model, params=None)` | An untrained analyzer: `"Decision Tree"`, `"Random Forest"` or `"Gradient Boosting Decision Trees"` |
| `.fit(X, y)` | Train the scikit-learn model and extract its rules |
| `PyRuleAnalyzer.from_sklearn(model, feature_names=None, class_names=None)` | Wrap an already-fitted estimator |
| `PyRuleAnalyzer.create(train_path, test_path, model, params, refine)` | Same, from CSV files |
| `.execute_rule_refinement(X, y, remove_below_n_classifications=-1, refine_between_trees=False, ...)` | Refine; returns `rules_before`, `rules_after`, `rules_removed`, `reduction_percent` |
| `.compare_initial_final_results(X, y)` | Metrics and disagreements, initial against refined |
| `.predict(X, use_refined=True)`, `.predict_proba(X, ...)` | Vectorised prediction |
| `.summary_report()` | A dict: model type, total rules, classes and every rule |
| `.to_python(path)`, `.to_binary(path)`, `.to_c_header(path)` | Standalone exports |
| `.export(base_name, formats, use_refined)` | Several formats at once, under `files/` |
| `.export_hcpn(base_name, which, sample)` | Write `<base>_initial.cpn` / `<base>_final.cpn` |
| `.model_check(which, samples, test_samples, reduction, askctl, ...)` | Export if needed and verify; `"both"` also compares |
| `.edit_rules()`, `.save(path)`, `PyRuleAnalyzer.load(path)`, `PyRuleAnalyzer.load_binary(path)` | Editing and persistence |

### RuleClassifier (`.classifier`)

| Method | Description |
|---|---|
| `classify(sample, final=False)` | One sample (dict); returns `(class, votes, probabilities)` |
| `predict_batch(X, feature_names=None, use_final=None)` | Vectorised classes; `use_final` picks the refined or initial set |
| `predict_batch_proba(X, feature_names=None, use_final=None)` | Vectorised probabilities |
| `compile_tree_arrays(rules=None, feature_names=None)` | Compile the rules into flat arrays; raises if the rules of a tree are not a partition |
| `export_to_arduino_ino(filepath, board_model='auto', serial_baud=115200)` | A full sketch, with a memory check |
| `to_cpn_tools(filepath, use_final=None, sample=None)` | The HCPN of the classifier |
| `find_duplicated_rules()`, `merge_boundary_redundancy()` | The refinement building blocks |
| `calculate_structural_complexity(rules, n_features_total)` | The Structural Complexity Score of a rule set |
| `set_custom_rule_removal(func)`, `adjust_and_remove_rules(method)` | Custom refinement |
| `RuleClassifier.load(path)`, `RuleClassifier.load_binary(path)` | Load a pickle or a `.bin` |

### Rule

One decision path. Uses `__slots__`.

| Attribute | Description |
|---|---|
| `name` | Unique id, e.g. `Rule3_Class1`, `DT0_Rule2_Class1` (RF), `GBDT1T2_Rule0_Class1` (GBDT) |
| `class_` | Predicted class |
| `conditions`, `parsed_conditions` | The conditions as text and as `(feature, op, threshold)` tuples |
| `usage_count`, `error_count` | Refinement samples that reached the rule, and how many of them it got wrong |
| `class_distribution` | (RF) the leaf's class counts |
| `leaf_value`, `learning_rate`, `contribution`, `class_group` | (GBDT) raw leaf value, rate, `learning_rate * leaf_value`, and the class channel |

### Analyzers

`DTAnalyzer`, `RFAnalyzer` and `GBDTAnalyzer` are the per-algorithm wrappers
`execute_rule_refinement()` dispatches to. Each exposes
`execute_rule_refinement()` and `compare_initial_final_results()`, with a
breakdown of the redundancy found (intra-tree, inter-tree, low impact, low
usage).

### Verification

| Object | Purpose |
|---|---|
| `check_cpn(path, samples=None, classifier=None, class_labels=None, ...)` | Verify an existing `.cpn` in one call |
| `CPNModelChecker(path)` | The checker: `.check()`, `.export_askctl()` |
| `ModelCheckResult` | `.passed`, `.failures`, `.properties`, `.report()`, `.to_dict()`, `.latex_table()` |
| `compare_results(initial, final)` | Side by side, flagging every regression |
| `PROPERTY_CATALOG` | The property descriptors: id, name, formula, kind, families |
| `CPNToolsOracle()` / `compare_with_oracle(path)` | Run CPN Tools 4.0.1 headlessly and compare |
| `make_mutants(path, out_dir)` | One mutant per property, for sensitivity checks |
| `verified_pipeline(...)`, `VerificationError` | The whole flow, and its gate |

## Project layout

```
pyruleanalyzer/
├── pyruleanalyzer/
│   ├── pyruleanalyzer.py     # PyRuleAnalyzer: the high-level entry point
│   ├── rule_classifier.py    # Rule and RuleClassifier: the engine
│   ├── exporters.py          # Python / binary / C export
│   ├── dt_analyzer.py  rf_analyzer.py  gbdt_analyzer.py
│   ├── _accel.py  _tree_traversal.c   # C extension and numpy fallback
│   ├── cpn_tools_exporter.py # HCPN export (.cpn)
│   ├── cpn_profile.py        # the tree profile, a CPNCheck plugin
│   ├── model_checker.py  cpn_semantics.py  cpn_statespace.py
│   ├── cpntools_oracle.py  cpn_mutants.py
│   ├── full_pipeline.py      # data -> Arduino sketch
│   └── verified_pipeline.py  # train -> verify -> refine -> verify -> deploy
├── tests/                    # invariants (DT/RF/GBDT), rule fidelity, CPN export, model checker
├── examples/                 # end-to-end scripts, benchmarks, Arduino tests, CPNCheck walkthrough
├── docs/                     # Sphinx site and docs/figures (the figures of this page)
├── pyproject.toml  setup.py
└── CHANGELOG.md
```

**Files produced.** `.pkl` (scikit-learn and rule-classifier models),
`output_*.txt` reports, `.py`, `.bin`, `.h`, `.ino`, `.cpn` (one per stage) and
`.sml` (ASK-CTL query scripts for CPN Tools).

## Tests

```bash
pytest tests/                                # everything
pytest tests/test_rule_fidelity.py           # every engine against the reference (~20 s; uses gcc if present)
pytest tests/test_cpn_export.py tests/test_model_checker.py   # after touching the exporter or refinement
```

Three invariants are checked on 25 configurations of each model type: the
initial rules reproduce scikit-learn; boundary merging alone never changes a
prediction; and degradation is monotonic as the low-usage threshold grows.

## Documentation

The full documentation, with tutorials, is at
**https://grupocybersegurancavirtus.github.io/pyruleanalyzer/**. To build it
locally:

```bash
pip install -e .[docs]
cd docs && make html
```

## Status

Version 2.0.0; [CHANGELOG.md](https://github.com/GrupoCybersegurancaVirtus/pyruleanalyzer/blob/main/CHANGELOG.md)
says what changed. The figures on this page are drawn from real runs by
[`docs/figures/make_figures.py`](https://github.com/GrupoCybersegurancaVirtus/pyruleanalyzer/blob/main/docs/figures/make_figures.py).

MIT licensed. Developed by
[GrupoCybersegurancaVirtus](https://github.com/GrupoCybersegurancaVirtus).
