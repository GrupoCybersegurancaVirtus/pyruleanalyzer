# Changelog

## Unreleased

**Added**
* Board presets `pico` (Raspberry Pi Pico, RP2040), `rp2040_16mb` (an
  RP2040 with 16 MB of flash), `esp32s3`, `esp32s3_16mb`, `esp32s3_32mb` and
  `esp32s3_16mb_custom` (one 15.9 MB app; the Arduino export writes its
  `partitions.csv`), for `fit_to_board`, `suggest_complexity` and
  `export_to_arduino_ino`. Their limits are those of the arduino-pico and
  arduino-esp32 cores, and their firmware size is measured.
* Preset `nrf52840` (nRF52840 boards with the Adafruit / nice!nano bootloader,
  Adafruit nRF52 core: 815,104 bytes of sketch flash). The Arduino sketch
  includes `Adafruit_TinyUSB.h` when the core uses TinyUSB, without which
  `Serial` does not link on that core.
* `examples/board_ladder_hil.py`: sizes one model per board of a ladder
  (Mega, nRF52840, ESP32, RP2040 16 MB by default), refines and compiles it
  (`prepare`, no hardware), then uploads it to the connected boards and
  compares each answer with the PC, timing the inference (`run`).

**Changed**
* On AVR (Uno, Nano, Mega, Leonardo) the C header and the Arduino sketch keep
  the tree arrays in flash (`PROGMEM`; far reads above 64 KB on the Mega).
  They used to be copied to SRAM, which capped a model at about 1.8 KB on the
  Uno and 7.8 KB on the Mega; it is now bounded by the 32 KB and 248 KB of
  flash. `Board.max_array_bytes` (32,767 on AVR) caps the nodes of one tree,
  and the AVR firmware sizes are the measured ones (1,732 / 2,072 bytes). The
  code for the other boards is unchanged.
* On AVR, the Arduino sketch of a forest or a boosting model walks a flash
  table of tree addresses in a loop instead of one inline call per tree,
  which cost about 176 bytes of code per tree (62 KB of a 375-tree model on
  the Mega). The sums are made in the same order, so the predictions do not
  change.
* The tree traversal of the C header and the Arduino sketch is no longer
  inlined into the call of every tree (`PYRA_NOINLINE`, GCC and Clang): GCC
  inlined it into each of the 750 calls of a 50-stage, 15-class GBDT, about
  420 bytes per tree, and the sketch for the nRF52840 overflowed its flash by
  223 KB; it now takes 669 KB of the 815 KB.
* On AVR the split thresholds are written as the largest float32 not above
  the threshold (`double` is a float32 there, and the compiler rounded to the
  nearest float32, which could lie above it): a float32 input now takes the
  same branch on the board as in Python.

**Fixed**
* The memory check of `export_to_arduino_ino` (and the estimates of
  `full_pipeline`) used wrong board limits: about 140 KB of flash for the Uno
  (it has 32 KB), 1 MB for the Mega (256 KB) and 4 MB / 512 KB for the ESP32
  (the default sketch partition is 1.25 MB and 320 KB of RAM). They now use
  the limits of `footprint.BOARDS`. An unknown `board_model` raises
  `ValueError` instead of being checked against the Uno.
* The Arduino sketch and the C header did not compile for AVR boards (Uno,
  Mega, Leonardo): avr-libc has no `HUGE_VAL`, which the tree arrays use.
  Both now define it when `math.h` does not.
* `export_to_c_header` wrote the file in the system encoding, so its
  non-ASCII comment came out garbled on Windows; it is now written in UTF-8,
  like the Arduino sketch.
* `fit_to_board` and the other size estimates undercounted Random Forests,
  so a forest sized to fill the ESP32 overflowed it by 20 to 92 KB (20 of
  21 sketches). Two costs were missing. The exports always write the
  missing-value side of each scikit-learn DT/RF node (one byte), so
  `missing_routing` now defaults to that (`None`: on for DT and RF, off for
  GBDT). A forest also adds each tree's class distribution to the vote, so a
  forest tree costs 174 bytes of code, not 70 (`RF_PER_TREE_BYTES`,
  `per_tree_bytes()`). Measured on 42 compiled ESP32 sketches, the estimate
  is now at most 2 KB below the compiled size.

## 2.2.2

**Fixed**
* `fit_to_board`, `probe_leaves` and `suggest_complexity` with
  `algorithm='xgboost'`, `'lightgbm'` or `'catboost'` (or the class names)
  fitted a scikit-learn `GradientBoostingClassifier` instead. They now fit
  that library's model, translating `max_depth` and `max_leaf_nodes` to its
  own settings (XGBoost `max_leaves` with `grow_policy='lossguide'`, LightGBM
  `num_leaves`, CatBoost `depth`: a symmetric tree of depth `d` has `2 ** d`
  leaves), count its trees as the exports store them, accept any labels, and
  `fit_to_board` returns the parameters in the library's names.

## 2.2.1

**Fixed**
* The coloured net of a Random Forest decides on scikit-learn's own floats.
  Each leaf now emits the row of the tree's `predict_proba` as scikit-learn
  stores it (`Rule.leaf_proba`, kept by JSON and pickle), and the `Vote`
  transition divides the sum by the number of trees (`vdiv`) before the
  argmax, as scikit-learn does. Before, the leaf rows were normalised again
  from the weighted counts and the sum was not divided: the scores could
  differ in the last bits, and a tie that the division creates was won by
  another class than scikit-learn's. Refined models, whose merged leaves have
  no scikit-learn row, still normalise the counts.

## 2.2.0

**Added**
* XGBoost, LightGBM and CatBoost classifiers. `from_sklearn` reads a fitted
  `XGBClassifier`, `LGBMClassifier` or `CatBoostClassifier` as a "Gradient
  Boosting Decision Trees" model, and `new_model("XGBoost" | "LightGBM" |
  "CatBoost")` (or `estimator="XGBClassifier"`, ...) trains one, with any
  labels; `get_params`, `set_params` and `clone` work as for scikit-learn.
  Refinement, every engine, the Python/C/Arduino/binary/JSON exports and the
  HCPN export work on them unchanged. Optional extras: `[xgboost]`,
  `[lightgbm]`, `[catboost]`, `[boosting]`.
* The raw scores are each library's bit for bit, and binary predictions are
  exactly the library's `predict`: XGBoost's `x < t` splits and float32
  sums, LightGBM's float64 comparisons and its reading of `|x| <= 1e-35` as
  0, CatBoost's symmetric trees and `x > border` splits, NaN routed where
  each library sends it, XGBoost DART weights. Multiclass models predict the
  argmax of the raw scores (XGBoost and LightGBM: of the softmax, which
  differs only on a rounding tie).
* `RuleClassifier.score_dtype` ('float64' or 'float32') and
  `RuleClassifier.binary_threshold` (0.0 for scikit-learn), honoured by every
  engine and export.
* `RuleClassifier.model_name`: "XGBoost", "LightGBM" or "CatBoost" for a
  model of those libraries (else `algorithm_type`), shown by `repr`,
  `summary()`, `summary_report()` and the Arduino sketch header, and kept by
  the JSON and binary round trips.
* `footprint` accepts the three libraries' class names.

**Changed**
* JSON models are written as version 3, and `.bin` files as version 5, only
  for models read from these libraries (they store the score arithmetic);
  scikit-learn models are written exactly as before, and every earlier
  version still loads.

## 2.1.2

**Fixed**
* The `bad_label` mutation operator writes a label one past the largest class
  of the net instead of a fixed `7`, which is a valid class of any model with
  eight or more classes; such mutants now violate the label-domain property
  (A6) instead of only changing the predicted class.

## 2.1.1

**Changed**
* `execute_rule_refinement` merges identical regions across the stages of a
  Gradient Boosting model by default (`refine_between_trees=None` means on for
  GBDT, off for Random Forest and Decision Tree). The merged rule carries the
  sum of the values, so predictions are unchanged; on a 50-stage, 8-class
  model it removed a further 25% of the rules and 10% of the exported size.
  Pass `refine_between_trees=False` for the previous behaviour.

## 2.1.0

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
* Embedded footprint, `pyruleanalyzer.footprint`: presets for Arduino Uno,
  Nano and Mega 2560 and for the ESP32 (default and 3 MB `huge_app`
  partitions), or any flash/RAM budget; `estimate_footprint()` bounds the
  size of an exported DT, RF or GBDT before training, `suggest_complexity()`
  previews the size over a grid of depths, leaves and trees, and
  `fit_to_board()` picks the number of trees and the leaves per tree that
  fill the board and score best on held-out data. The constants are
  calibrated on the ESP32 Arduino core 3.3.12.
* `execute_rule_refinement(usage_engine=...)` for Gradient Boosting:
  `'batch'` counts rule usage and errors through the compiled tree arrays,
  `'loop'` one sample at a time, and `'auto'` (default) uses `'batch'`
  whenever it applies. Both give identical refined rule sets.
* `examples/arduino_stream_test.py`, a streaming anomaly-detection test on
  Arduino/ESP32 boards (or a simulator through a serial URL) with any CSV
  dataset, and a configurable `examples/datasets_benchmark_example.py`.
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

**Performance**
* Gradient Boosting refinement counts usage and recounts it after the
  low-usage stage on compiled tree arrays: two to three orders of magnitude
  faster on large refinement sets (288 000 samples and 51 000 rules: 2.5 h
  to 25 s), with the same counts and predictions.
* The recount after low-usage removal accumulates the tree outputs instead of
  keeping one (samples x classes) array per tree, which for large multiclass
  ensembles reserved tens of GB.
* Requires CPNCheck 0.2.2, which loads large nets about five times faster
  (a net of 33 000 transitions: 67 s to 13 s), so verifying and
  fault-injecting large ensembles spends much less time reading the `.cpn`.

**Internal**
* `rule_classifier.py` (7 500 lines) is split into one module per
  responsibility (`_rules`, `_labels`, `_engine`, `_native_python`, `_arrays`,
  `_binary_format`, `_c_export`, `_refinement`, `_serialization`,
  `_reporting`, `_extraction`), combined into `RuleClassifier` as mixins; the
  Python and C/Arduino code generators share one generator each instead of
  near-identical copies. Import paths are unchanged, pickles from earlier
  versions still load, and every generated file is byte-identical to the
  previous code's on a reference corpus.

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
