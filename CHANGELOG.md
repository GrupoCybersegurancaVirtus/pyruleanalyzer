# Changelog

## Unreleased

**Breaking changes**
* `verified_pipeline()` refuses an inconclusive check: a property left
  undecided (usually an occurrence graph over `max_nodes`) sets
  `passed=False`, is listed in `result["inconclusive"]` and raises
  `VerificationError` unless `fail_on_inconclusive=False`. Ensembles should
  use `reduction="compositional"`.
* Low-usage removal no longer leaves holes in a Decision Tree or Random
  Forest: when the removed rule's sibling is a sub-branch, the sub-branch is
  raised into the parent's place (rules renamed `..._grafted`) instead of the
  region falling back to `default_class`. Refined rule sets, and so refined
  predictions, can differ from earlier releases; usage and error counts are
  recounted after the stage.
* Generated C headers and Arduino sketches always include `<math.h>` and
  write infinite thresholds as `HUGE_VAL` (`1e500` did not compile with MSVC).
* `.bin` files are written as version 4 for models that record NaN routing,
  column labels or a label table (every freshly extracted scikit-learn
  model); versions 1-3 still load. A `.bin` whose arrays do not form a tree
  (cycles, shared or unreachable nodes, a `max_depth` below the real depth)
  is rejected.
* `verified_pipeline()` (`validation_size=0.25`) and `full_pipeline()`
  (`validation_size=0.2`) refine on a validation split held out from the
  training data; the test data is used only for the reported metrics. The
  trees are therefore fit on fewer rows; `validation_size=0` restores the old
  behaviour.
* A single-sample prediction (`classify`, `predict(dict)`, the generated
  `.py`) raises `KeyError` when a feature the rules test is missing, instead
  of reading it as 0.
* `full_pipeline()`: a test CSV without the target is ignored with a warning
  instead of being replaced by training rows; string targets are encoded with
  one `LabelEncoder` (`results["label_classes"]`); test columns are matched by
  name; `fillna(0)` warns.
* A Gradient Boosting model with a non-constant `init` estimator raises
  `ValueError`: its initial score depends on the input and the rules cannot
  express it.
* `pyruleanalyzer.parsers`, an unused copy of the rule extractors, is removed.

**New**
* Compositional model checking, `reduction="compositional"`: each tree (or
  boosting channel) is explored on its own and the vote once per combination
  of outputs, with a per-property soundness argument and a fallback to the
  stubborn reduction. A 20-tree forest gets all 16 verdicts in 1.4 s (60
  markings instead of 2^20 + 2), a 50-tree forest in 7 s; on small ensembles
  it matched the full graph on 2 128 checks, mutants included.
* `PyRuleAnalyzer` is a scikit-learn classifier: `get_params`/`set_params`,
  `classes_`, `score`, and it works with `clone`, `Pipeline`,
  `cross_val_score` and `GridSearchCV`.
* Any class labels: strings and other non-integer labels are encoded as
  codes 0..K-1 internally and answered as the original labels by
  `predict`, `classes_`, `score` and the generated Python; C, Arduino and
  `.bin` answer codes and carry the code-to-label table.
* Missing values: trees fitted on data with NaN route a missing value as
  scikit-learn does (the side chosen per split), in every engine and export
  (Python, C, Arduino, `.bin`, JSON). GradientBoostingClassifier itself
  rejects NaN; the Petri net is exact for real-valued inputs and refuses NaN
  samples.
* Safe, versioned JSON model format (`save_json`/`load_json`, or `save`/
  `load` with a `.json` path) that keeps the rules, so a loaded model can be
  refined, edited, verified and exported again; loading never executes code.
* Error-aware refinement: `max_error_increase=k` removes a low-usage rule only
  while the refinement-set errors stay within `k` of those before the stage.
* `from_sklearn()` accepts `ExtraTreesClassifier`; refitting a wrapped
  ExtraTrees trains ExtraTrees again.
* Wheels for Python 3.14 (cibuildwheel 3.4.1); a CI job on the minimum
  declared dependency versions.

**Security**
* `load_binary()` validates the file (header, sizes, every node and child
  index) before the arrays reach the C traversal, and the C wrappers check
  their inputs; a corrupt or crafted `.bin` used to read out of bounds.
* Guards and inscriptions read from `.cpn` files are evaluated by a
  whitelisting AST interpreter instead of `eval` with emptied builtins, which
  could be escaped.

**Fixed**
* Random Forest predictions answer the class label, not the index of its
  column: labels other than 0..K-1 (`1..3`, `-1/1`, ...) used to come back as
  `0..K-1` from every engine, the Python, C, Arduino and binary exports and
  the Petri net. The binary format gains version 3, written only for a forest
  whose labels are not 0..K-1.
* Labels written as floats (`0.0`, `1.0`) work; a label of `-1` is no longer
  taken for an uncovered region by the Decision Tree code generators.
  Non-integer labels used to be replaced by indices (now supported, see New).
* `predict_proba` of a Decision Tree whose labels are not 0..K-1 used the
  label as a column index (an `IndexError` for labels 1..3).
* Infinite split thresholds were written as 0.0 in the Petri net.
* `predict()` and `predict_proba()` match a DataFrame's columns by name, in
  any order and with extra columns; a missing feature raises `ValueError`.
* `PyRuleAnalyzer.load()` reads what `save()` writes.
* `PyRuleAnalyzer.create(..., refine=True)` refines instead of failing.
* Feature names are escaped in generated Python (which is also executed in
  memory) and in C/Arduino comments; a crafted CSV header could run code.
* `full_pipeline()` passes `max_depth`, `n_estimators` and
  `min_samples_split` to the estimator; they used to be dropped silently.
* Forest rules without a class distribution put their vote in the column of
  their label (compiled arrays and Petri net), not in the column numbered
  like the label.
* `PyRuleAnalyzer.create()` fills `feature_names` from the training CSV, so
  an array passed to `predict()` is read in CSV column order.
* `examples/full_pipeline_example.py` uses `generate_arduino_sketch`.
* README: speed is on par with scikit-learn (not 1.4x faster); refinement can
  leave holes answered by `default_class`; large ensembles exceed the model
  checker's budget; only integer labels are supported; pickle and `.cpn`
  files should come from trusted sources.
* Packaging: dependency lower bounds, status Beta, Python 3.14 in CI.

## 2.0.0

The first release since 1.1.1 (September 2025).

**Breaking changes**
* Python 3.9 or later.
* One entry point, `PyRuleAnalyzer`, compatible with scikit-learn
  (`fit`, `predict`, `predict_proba`), with in-memory data; the analyzers
  (`DTAnalyzer`, `RFAnalyzer`, `GBDTAnalyzer`) share one
  `execute_rule_refinement` API.
* Refinement methods: `"refine_duplicates"` (the default) and `"custom"`;
  `"soft"`, `"medium"` and `"hard"` were removed, and so was the
  `remove_duplicates` parameter.
* Generated files go to `files/` with standard names; what is saved is chosen
  with `save_*` / `export_*` flags.

**New**
* A C extension for vectorised tree traversal, shipped compiled in the wheels
  for Linux, macOS and Windows (numpy fallback elsewhere).
* Export of every model as a hierarchical Coloured Petri Net for CPN Tools,
  and model checking of the exported nets with
  [CPNCheck](https://pypi.org/project/cpncheck/): the tree profile is
  published to CPNCheck as a plugin, with state-space reductions.
* `verified_pipeline()`: train, verify, refine, verify again, export and
  deploy to Arduino/ESP32; it refuses to export a model whose net lost a
  property.
* Standalone exports to Python, binary and C header, and full Arduino/ESP32
  sketches, with hardware-in-the-loop and simulated-sensor test scripts.

**Fixed**
* Predictions follow the rule set the caller asked for (`use_refined`), and
  inputs are compared in float32 as scikit-learn does.
* Refinement no longer deletes merged rules as unused; GBDT binary decisions
  use `score >= 0`, as scikit-learn does.
* Structural Complexity Score formula.
