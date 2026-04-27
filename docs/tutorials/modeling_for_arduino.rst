Modeling for Arduino / ESP32
=============================

This page covers strategies for training and configuring machine learning models optimized for deployment on resource-constrained microcontrollers like Arduino and ESP32. The pyruleanalyzer library exports trained scikit-learn models directly to self-contained C++ sketches (``.ino``) that run entirely on-device with zero external dependencies.

When deploying to Arduino/ESP32, memory is the primary constraint: every tree node consumes Flash ROM for constants and SRAM at runtime. This guide explains how to choose model hyperparameters, reduce footprint, and validate compatibility before uploading your sketch.


.. _modeling-for-arduino#why-pyruleanalyzer:

Why pyruleanalyzer for Arduino?
-------------------------------

The ``full_pipeline`` function bridges Python ML training and embedded deployment in a single step:

1. **Train in Python** — Use scikit-learn's full API (Decision Tree, Random Forest, GBDT).
2. **Compile to C arrays** — Trees are serialized as static ``const`` arrays (feature indices, thresholds, children pointers, leaf values).
3. **Generate self-contained sketch** — A single ``.ino`` file includes the model data, inference engine, and Arduino skeleton code.
4. **Memory validation** — Built-in Flash/SRAM estimation with board-specific thresholds alerts you before upload fails.


.. _modeling-for-arduino#constraints:

Resource Constraints by Board
-----------------------------

Different boards have dramatically different memory budgets. The table below summarizes the key limits:

.. list-table::
   :header-rows: 1
   :widths: 25 30 30

   * - Board
     - Flash (program storage)
     - SRAM (data)
   * - Arduino Uno / Nano
     - ~30 KB usable (~140 KB total, bootloader takes space)
     - 2 KB
   * - Arduino Mega 2560
     - ~256 KB usable (~1 MB total)
     - 8 KB
   * - Arduino Leonardo
     - ~28 KB usable (32 KB total, 2 KB boot loader)
     - 2.5 KB
   * - ESP32 (DevKit)
     - ~4 MB Flash
     - 512 KB SRAM


.. _modeling-for-arduino#decision-tree-strategies:

Decision Tree Strategies
-------------------------

A single Decision Tree is the most Arduino-friendly model type because it has no aggregation overhead.

**Recommended settings for Uno/Nano:**

.. code-block:: python

   from pyruleanalyzer import full_pipeline

   results = full_pipeline(
       train_csv='dataset.csv',
       target_feature='Target',
       model_type='Decision Tree',
       max_depth=8,          # Keep depth <= 10 for Uno/Nano
       generate_arduino_sketch=True,
       board_model='uno',
   )

**Node count formula:** A binary tree with depth *d* has at most ``2^d - 1`` nodes. Each node stores:

- `feature_idx`: int32 (4 bytes)
- `threshold`: double (8 bytes)  
- `children_left`: int32 (4 bytes)
- `children_right`: int32 (4 bytes)
- `value`: int32 × n_classes (4n bytes)

Total per node ≈ **20 + 4n** bytes. For *d=8* with 3 classes: ``255 nodes × 32 bytes = ~8 KB Flash`` — well within Uno limits.

For larger trees, use pruning or reduce ``max_depth`` until the memory check passes.


.. _modeling-for-arduino#random-forest-strategies:

Random Forest Strategies
-------------------------

Random Forests aggregate predictions from many trees via majority voting. Each tree adds its own node arrays to Flash, so footprint grows linearly with ``n_estimators``.

**Rule of thumb:** On Uno/Nano, aim for total nodes across all trees < 2000 (roughly ~64 KB Flash).

.. code-block:: python

   results = full_pipeline(
       train_csv='dataset.csv',
       target_feature='Target',
       model_type='Random Forest',
       n_estimators=10,      # Start small: 5-15 trees
       max_depth=6,          # Shallower trees to compensate for ensemble size
       generate_arduino_sketch=True,
       board_model='uno',
   )

**Optimization tips:**

- Use ``max_depth <= 8`` — deeper trees multiply memory by 2× per extra level.
- Start with ``n_estimators=5`` and incrementally add trees while monitoring the Flash percentage in ``results['memory_check']``.
- Consider ``min_samples_leaf=5`` or higher to prune leaf-level noise without adding depth.


.. _modeling-for-arduino#gbdt-strategies:

Gradient Boosting Decision Trees (GBDT) Strategies
----------------------------------------------------

GBDT models are **the least Arduino-friendly** because predictions require summing outputs from all trees rather than simple voting. Each tree's ``value`` arrays must remain in SRAM simultaneously during inference, doubling the baseline footprint compared to Random Forest with the same number of estimators.

.. code-block:: python

   results = full_pipeline(
       train_csv='dataset.csv',
       target_feature='Target',
       model_type='Gradient Boosting Decision Trees',
       n_estimators=20,          # Keep low: 10-30 trees
       max_depth=3,              # GBDT works well with shallow "stumps"
       learning_rate=0.1,        # Lower learning rate compensates for fewer estimators
       generate_arduino_sketch=True,
       board_model='mega',       # Recommend Mega or ESP32 for GBDT
   )

**When to use GBDT on Arduino:**

- Board has >= 8 KB SRAM (Mega) or >= 512 KB SRAM (ESP32).
- You need higher accuracy than Random Forest provides at the same depth.
- Tree depth is limited to ``<= 4`` and total nodes < 3000.

For Uno/Nano, prefer Decision Trees or small Random Forests over GBDT.


.. _modeling-for-arduino#feature-selection:

Feature Selection Strategies
-----------------------------

Each feature adds a column to the sensor-reading code (the ``read_features()`` function) and increases the size of the ``features[]`` array passed to the inference engine. While the feature arrays themselves are small, more features mean:

1. **More sensor hardware** — Each feature typically maps to one analog/digital pin or I²C sensor reading.
2. **Slower inference** — Feature reads add latency in every loop iteration.
3. **Overfitting risk** — Extra irrelevant features can degrade model generalization.

**Recommendations:**

- Use ``max_features`` in Random Forest to limit per-tree feature usage:

  .. code-block:: python

     results = full_pipeline(
         train_csv='dataset.csv',
         target_feature='Target',
         model_type='Random Forest',
         max_features=4,      # Only use 4 features per tree
         generate_arduino_sketch=True,
         board_model='uno',
     )

- Apply feature importance from the trained Decision Tree to select only top-K features before training:

  .. code-block:: python

     model = PyRuleAnalyzer.new_model(model='Decision Tree')
     model.fit(X_train, y_train)
     
     importances = model.feature_importance_
     top_features = [f for _, f in sorted(zip(importances, feature_names), reverse=True)[:5]]
     
     X_reduced = X_train[top_features]  # Train on reduced set


.. _modeling-for-arduino#memory-validation:

Memory Validation
------------------

The ``full_pipeline`` function automatically estimates Flash and SRAM usage after training. The ``memory_check`` result contains these key fields:

.. list-table::
   :header-rows: 1
   :widths: 30 40

   * - Field
     - Description
   * - ``flash_bytes``
     - Estimated Flash (program storage) used by tree arrays
   * - ``flash_percent``
     - Percentage of board's Flash capacity consumed
   * - ``ram_bytes``
     - Estimated SRAM (data) needed for inference buffers
   * - ``ram_percent``
     - Percentage of board's SRAM capacity consumed
   * - ``board_model``
     - Target board name (e.g., "uno", "mega", "esp32")
   * - ``fits_flash``
     - ``True`` if flash_bytes <= max_flash_bytes
   * - ``fits_ram``
     - ``True`` if ram_bytes <= max_ram_bytes
   * - ``flash_max_bytes``
     - Maximum Flash for the target board (bytes)
   * - ``ram_max_bytes``
     - Maximum SRAM for the target board (bytes)


Checking memory compatibility manually:

.. code-block:: python

    mc = results['memory_check']
    
    if not mc['fits_flash']:
        print(f"WARNING: Model exceeds Flash by {mc['flash_percent']:.1f}%")
        
    if not mc['fits_ram']:
        print(f"WARNING: Model exceeds SRAM by {mc['ram_percent']:.1f}%")


.. _modeling-for-arduino#depth-strategies:

Depth vs. Accuracy Trade-off
------------------------------

Tree depth is the single biggest lever for controlling Arduino footprint. Here's a practical guide:

.. list-table::
   :header-rows: 1
   :widths: 15 20 30 30

   * - Max Depth
     - Max Nodes (per tree)
     - Flash per tree (~3 classes)
     - Recommended for
   * - 4
     - 15
     - ~480 bytes
     - Uno, Nano (large ensembles OK)
   * - 6
     - 63
     - ~2 KB
     - Uno, Nano (small ensembles)
   * - 8
     - 255
     - ~8 KB
     - Uno (single tree), Mega
   * - 10
     - 1023
     - ~32 KB
     - Mega only
   * - 14+
     - 16K+
     - >500 KB
     - ESP32 only


.. _modeling-for-arduino#board-selection:

Choosing a Board
----------------

Use the ``board_model`` parameter to target your board. Set it to ``'auto'`` (default) to let pyruleanalyzer auto-detect and pick the best-fit board:

.. code-block:: python

   results = full_pipeline(
       train_csv='dataset.csv',
       target_feature='Target',
       model_type='Decision Tree',
       generate_arduino_sketch=True,
       board_model='auto',  # Auto-select best board
   )

Board presets used internally:

.. list-table::
   :header-rows: 1
   :widths: 25 30 30 20

   * - Board Model
     - Flash Max (bytes)
     - SRAM Max (bytes)
     - Default Baud
   * - ``uno``
     - 143360
     - 2048
     - 115200
   * - ``nano``
     - 143360
     - 2048
     - 115200
   * - ``mega``
     - 1048576
     - 8192
     - 115200
   * - ``leonardo``
     - 32768
     - 2560
     - 115200
   * - ``esp32``
     - 4194304
     - 524288
     - 115200


.. _modeling-for-arduino#example-workflows:

Example Workflows
------------------

Workflow 1: Small model for Arduino Uno (Iris dataset)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyruleanalyzer import full_pipeline

   results = full_pipeline(
       train_csv='files/dataset_iris.csv',
       target_feature='Target',
       model_type='Decision Tree',
       max_depth=8,
       generate_arduino_sketch=True,
       board_model='uno',
   )

   mc = results['memory_check']
   print(f"Flash: {mc['flash_percent']:.1f}% | SRAM: {mc['ram_percent']:.1f}%")
   # Output example: Flash: 5.2% | SRAM: 8.1%


Workflow 2: Medium model for Arduino Mega (Wine dataset)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   results = full_pipeline(
       train_csv='files/dataset_wine.csv',
       target_feature='Target',
       model_type='Random Forest',
       n_estimators=10,
       max_depth=8,
       generate_arduino_sketch=True,
       board_model='mega',
   )


Workflow 3: Large model for ESP32 (Breast Cancer dataset)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   results = full_pipeline(
       train_csv='files/dataset_heart.csv',
       target_feature='Target',
       model_type='Gradient Boosting Decision Trees',
       n_estimators=30,
       max_depth=4,
       learning_rate=0.1,
       generate_arduino_sketch=True,
       board_model='esp32',
   )


Workflow 4: Auto-detect best board
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   results = full_pipeline(
       train_csv='dataset.csv',
       target_feature='Target',
       model_type='Random Forest',
       n_estimators=15,
       max_depth=6,
       generate_arduino_sketch=True,
       board_model='auto',  # Auto-select
   )

   print(f"Auto-selected: {results['memory_check']['board_model'].upper()}")


.. _modeling-for-arduino#optimization-checklist:

Optimization Checklist
-----------------------

Before uploading your sketch to a physical board, verify the following:

- [ ] ``fits_flash`` is ``True`` in ``memory_check``.
- [ ] ``fits_ram`` is ``True`` in ``memory_check``.
- [ ] Total nodes across all trees < 2000 for Uno/Nano (< 10K for Mega).
- [ ] ``max_depth`` is minimized while maintaining acceptable accuracy.
- [ ] ``n_estimators`` is the minimum needed for your target accuracy.
- [ ] ``read_features()`` placeholders are filled with real sensor code.
- [ ] Serial baud rate matches your monitor configuration (default 115200).


.. _modeling-for-arduino#troubleshooting:

Troubleshooting
----------------

Sketch exceeds Flash on Uno/Nano
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reduce ``max_depth`` by 2 and retrain. If still too large, reduce ``n_estimators`` or switch to a single Decision Tree.

.. code-block:: python

   results = full_pipeline(
       train_csv='dataset.csv',
       target_feature='Target',
       model_type='Decision Tree',
       max_depth=6,              # Reduced from 8
       generate_arduino_sketch=True,
       board_model='uno',
   )

SRAM exceeded on Mega
~~~~~~~~~~~~~~~~~~~~~

GBDT models with many estimators consume significant SRAM for intermediate values. Reduce ``n_estimators`` or switch to Random Forest:

.. code-block:: python

   results = full_pipeline(
       train_csv='dataset.csv',
       target_feature='Target',
       model_type='Random Forest',     # Switch from GBDT
       n_estimators=15,                # Reduced from 30
       max_depth=4,
       generate_arduino_sketch=True,
       board_model='mega',
   )

ESP32 recommended for large models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your model needs significant depth and many trees, deploy to ESP32 instead:

.. code-block:: python

   results = full_pipeline(
       train_csv='dataset.csv',
       target_feature='Target',
       model_type='Gradient Boosting Decision Trees',
       n_estimators=50,
       max_depth=6,
       learning_rate=0.1,
       generate_arduino_sketch=True,
       board_model='esp32',  # ESP32 has plenty of memory
   )


API Reference
-------------

``full_pipeline()`` parameters relevant to Arduino export:

.. list-table::
   :header-rows: 1
   :widths: 35 40 15

   * - Parameter
     - Description
     - Default
   * - ``generate_arduino_sketch``
     - Generate self-contained .ino sketch
     - ``False``
   * - ``board_model``
     - Target board: ``auto``, ``uno``, ``nano``, ``mega``, ``leonardo``, ``esp32``
     - ``"auto"``
   * - ``serial_baud``
     - Serial port baud rate
     - ``115200``
   * - ``include_sensor_placeholders``
     - Add TODO comments in ``read_features()``
     - ``True``
   * - ``max_flash_percent``
     - Max Flash usage for compatibility check (%)
     - ``90.0``
   * - ``max_ram_percent``
     - Max SRAM usage for compatibility check (%)
     - ``85.0``

``RuleClassifier.export_to_arduino_ino()`` method:

.. code-block:: python

    result = model.classifier.export_to_arduino_ino(
        filepath='model.ino',      # Output file path
        board_model='uno',         # Target board
        serial_baud=115200,        # Serial baud rate
        include_sensor_placeholders=True,  # TODO placeholders
    )

    result = {
        'ino_path': '/path/to/model.ino',
        'memory_check': {...},     # Flash/SRAM estimation dict
        'metadata': {...},          # Model metadata embedded in sketch
    }


See Also
--------

- :doc:`arduino` — Full Arduino/ESP32 Sketch Export tutorial (setup, upload, sensor integration)
- :doc:`usage` — General RuleClassifier usage and model training
- `Arduino CLI <https://docs.arduino.cc/software/ide-v2/#cli-reference>`_ — Command-line upload reference
- `PlatformIO <https://platformio.org/>`_ — Cross-platform embedded build system