# Changelog

## Unreleased

**Fixed**
* Random Forest predictions answer the class label, not the index of its
  column: labels other than 0..K-1 (`1..3`, `-1/1`, ...) used to come back as
  `0..K-1` from every engine, the Python, C, Arduino and binary exports and
  the Petri net. The binary format gains version 3, written only for a forest
  whose labels are not 0..K-1.
* Labels written as floats (`0.0`, `1.0`) work; a label of `-1` is no longer
  taken for an uncovered region by the Decision Tree code generators.
  Non-integer labels (strings, `0.5`) raise a `ValueError` instead of being
  replaced by indices.
* `predict()` and `predict_proba()` match a DataFrame's columns by name, in
  any order and with extra columns; a missing feature raises `ValueError`.
* `PyRuleAnalyzer.load()` reads what `save()` writes.
* `PyRuleAnalyzer.create(..., refine=True)` refines instead of failing.
* Feature names are escaped in generated Python (which is also executed in
  memory) and in C/Arduino comments; a crafted CSV header could run code.
* `full_pipeline()` passes `max_depth`, `n_estimators` and
  `min_samples_split` to the estimator; they used to be dropped silently.

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
