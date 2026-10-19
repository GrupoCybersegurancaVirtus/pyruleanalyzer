# Changelog

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
