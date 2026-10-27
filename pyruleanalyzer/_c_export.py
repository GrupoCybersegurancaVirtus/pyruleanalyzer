"""C header and Arduino sketch generation.

Provides ``CExportMixin``: the compiled tree arrays written as a
self-contained C header (``export_to_c_header``) or a complete Arduino /
ESP32 sketch (``export_to_arduino_ino``), with the shared C helpers and the
memory estimate used to pick a board.
"""

import math
import os
import re
from typing import List


# Helper function to write a float as a C literal
def _format_c_float(v) -> str:
    """Format a float value as a C-compatible literal.

    Special values (inf, -inf, nan) are mapped to the C89 ``HUGE_VAL`` macro
    of <math.h> (+infinity on IEEE targets, avr-libc included):
      * +inf  → "HUGE_VAL"   (the split isolating missing values)
      * -inf  → "-HUGE_VAL"  (leaves; never compared)
      * nan   → "-HUGE_VAL"  (fallback; treated as -inf)

    A literal such as ``1e500`` is out of range: GCC turns it into infinity
    with a warning, MSVC refuses to compile it.

    Args:
        v: The value.

    Returns:
        str: The C literal (``%.17g`` for finite values).
    """
    if math.isinf(v):
        return "HUGE_VAL" if v > 0 else "-HUGE_VAL"
    if math.isnan(v):
        return "-HUGE_VAL"  # safe fallback
    return f'{v:.17g}'


# Helper function for memory estimation (used by export_to_arduino_ino)
def _compute_memory_for_rule_classifier(rf_model) -> dict:
    """Compute memory estimates for a RuleClassifier instance.

    Args:
        rf_model: A RuleClassifier with compiled tree arrays.

    Returns:
        Dict with flash and RAM byte estimates.
    """
    avr_max_flash = 143360   # Arduino Uno: ~140KB available
    avr_max_sram = 2048      # Arduino Uno: 2KB SRAM
    esp32_max_flash = 4 * 1024 * 1024  # ESP32: 4MB Flash
    esp32_max_sram = 524288              # ESP32: 512KB SRAM

    estimated_flash_bytes = 0
    total_ram_bytes = 0

    if not hasattr(rf_model, '_tree_arrays') or not rf_model._tree_arrays:
        return {
            'estimated_flash_bytes': 0,
            'total_ram_bytes': 0,
            'avr_max_flash': avr_max_flash,
            'avr_max_sram': avr_max_sram,
            'esp32_max_flash': esp32_max_flash,
            'esp32_max_sram': esp32_max_sram,
        }

    for tree in rf_model._tree_arrays:
        if 'init_score' in tree:
            continue  # GBDT init entry — no tree structure
        n_nodes = len(tree['feature_idx'])
        estimated_flash_bytes += n_nodes * 36 + 512  # ~36 bytes per node + header

    total_ram_bytes = estimated_flash_bytes * 0.1  # ~10% overhead

    return {
        'estimated_flash_bytes': estimated_flash_bytes,
        'total_ram_bytes': total_ram_bytes,
        'avr_max_flash': avr_max_flash,
        'avr_max_sram': avr_max_sram,
        'esp32_max_flash': esp32_max_flash,
        'esp32_max_sram': esp32_max_sram,
    }


class CExportMixin:
    """C header and Arduino (.ino) export."""

    # Method to check whether the C exports route missing values
    def _c_routes_missing(self) -> bool:
        """Whether the C/Arduino exports carry the missing-value routing.

        Gradient Boosting never does: scikit-learn's GradientBoostingClassifier
        refuses NaN, so its exports are written as before.

        Returns:
            bool: True when every compiled tree has its ``missing_left``
            array and there is at least one tree to traverse.
        """
        if self.algorithm_type == 'Gradient Boosting Decision Trees':
            return False
        trees = self._tree_arrays
        return (self._trees_route_missing(trees)
                and any('init_score' not in t for t in trees))

    # Method to generate the C function that walks one tree
    def _c_traverse_function(self, name: str, routes_missing: bool,
                             leaf_comment: str = '') -> List[str]:
        """The C source of the tree traversal of the C/Arduino exports.

        Without missing-value routing it is the function the exports always
        had (``x <= t`` goes left, so a NaN goes right). With it, a NaN goes
        to the side the split recorded (``missing_left``), as in
        scikit-learn.

        Args:
            name (str): Function name.
            routes_missing (bool): Emit the ``missing_left`` parameter and
                the ``isnan`` test.
            leaf_comment (str): Comment appended to the leaf test.

        Returns:
            List[str]: The lines, followed by an empty line.
        """
        f32 = getattr(self, 'input_dtype', None) == 'float32'
        # scikit-learn casts X to float32 before comparing.
        value = ('(double)(float)features[feat_idx[node]]' if f32
                 else 'features[feat_idx[node]]')
        out = [
            f'static inline int32_t {name}(',
            '    const double *features,',
            '    const int32_t *feat_idx,',
            '    const double *thresh,',
            '    const int32_t *left,',
            '    const int32_t *right,',
        ]
        if routes_missing:
            out.append('    const uint8_t *missing_left,')
        out += [
            '    int max_depth',
            ') {',
            '    int32_t node = 0;',
            '    int d;',
            '    for (d = 0; d < max_depth; d++) {',
            f'        if (left[node] == node) break;{leaf_comment}',
        ]
        if routes_missing:
            out += [
                f'        double x = {value};',
                '        if (isnan(x))  /* missing value: the side the split recorded */',
                '            node = missing_left[node] ? left[node] : right[node];',
                '        else if (x <= thresh[node])',
            ]
        else:
            out.append(f'        if ({value} <= thresh[node])')
        out += [
            '            node = left[node];',
            '        else',
            '            node = right[node];',
            '    }',
            '    return node;',
            '}',
            '',
        ]
        return out

    # Method to pass each tree's missing-value routing to the traversal
    @staticmethod
    def _c_with_missing_args(source: str, routes_missing: bool) -> str:
        """Add ``treeN_missing_left`` to every traversal call of `source`.

        Args:
            source (str): Generated C source.
            routes_missing (bool): Whether the traversal takes the argument.

        Returns:
            str: The source, unchanged when `routes_missing` is False.
        """
        if not routes_missing:
            return source
        return re.sub(r'tree(\d+)_right, TREE\1_DEPTH\)',
                      r'tree\1_right, tree\1_missing_left, TREE\1_DEPTH)', source)

    # Method to document the code -> label table in generated C
    def _c_label_table_comment(self, line_comment: bool = False) -> List[str]:
        """Comment lines naming the original label of each integer code.

        The C, Arduino and binary targets answer integer codes; for labels
        that are not integers the code is the position in ``classes_``.

        Args:
            line_comment (bool): ``//`` comments (Arduino) instead of a block.

        Returns:
            List[str]: The lines (none when the labels are integers).
        """
        table = getattr(self, 'label_table', None)
        if table is None:
            return []

        def show(lab):
            """A label on one ASCII comment line."""
            text = repr(lab) if not isinstance(lab, str) else lab
            return self._c_comment(text.encode('ascii', 'backslashreplace').decode('ascii'))

        if line_comment:
            out = ['// Class labels: pyra_predict() returns the code; original labels:']
            out += [f'//   {i} = {show(lab)}' for i, lab in enumerate(table)]
        else:
            out = ['/* Class labels: predict() returns the code; original labels:']
            out += [f'   {i} = {show(lab)}' for i, lab in enumerate(table)]
            out.append('*/')
        out.append('')
        return out

    # Method to generate the model #defines and the feature-order comment
    def _c_model_defines(self, default_cls: int,
                         line_comment: bool = False) -> List[str]:
        """The size/default ``#define`` block, the feature order and labels.

        Args:
            default_cls (int): Value of ``DEFAULT_CLASS``.
            line_comment (bool): ``//`` comments (Arduino) instead of blocks.

        Returns:
            List[str]: The lines, ending with the label table comment.
        """
        out = [
            f'#define N_FEATURES {len(self._array_feature_names)}',
            f'#define N_CLASSES  {self.num_classes}',
            f'#define N_TREES    {len(self._tree_arrays)}',
            f'#define DEFAULT_CLASS {default_cls}',
            '',
        ]
        names = self._array_feature_names
        if line_comment:
            out.append('// Feature order:')
            out += [f'//   [{i}] {self._c_comment(n)}' for i, n in enumerate(names)]
        else:
            out.append('/* Feature order:')
            out += [f'   [{i}] {self._c_comment(n)}' for i, n in enumerate(names)]
            out.append('*/')
        out.append('')
        out += self._c_label_table_comment(line_comment=line_comment)
        return out

    # Method to generate the const arrays of every tree
    def _c_tree_arrays(self, routes_missing: bool,
                       line_comment: bool = False) -> List[str]:
        """The ``TREEn_DEPTH`` macro and const arrays of each compiled tree.

        Args:
            routes_missing (bool): Also emit ``treeN_missing_left``.
            line_comment (bool): ``//`` comments (Arduino) instead of blocks.

        Returns:
            List[str]: The lines, an empty line after each tree.
        """
        out: List[str] = []
        a = out.append
        for t_idx, tree in enumerate(self._tree_arrays):
            if 'init_score' in tree:
                continue  # GBDT init handled separately

            n_nodes = len(tree['feature_idx'])
            tree_depth = tree.get('max_depth', 50)
            title = f'Tree {t_idx}: {n_nodes} nodes, depth {tree_depth}'
            a(f'// {title}' if line_comment else f'/* {title} */')
            a(f'#define TREE{t_idx}_DEPTH {tree_depth}')

            vals = ', '.join(str(v) for v in tree['feature_idx'])
            a(f'static const int32_t tree{t_idx}_feature[{n_nodes}] = {{{vals}}};')

            # threshold — special float values as C-compatible literals
            vals = ', '.join(_format_c_float(v) for v in tree['threshold'])
            a(f'static const double tree{t_idx}_threshold[{n_nodes}] = {{{vals}}};')

            vals = ', '.join(str(v) for v in tree['children_left'])
            a(f'static const int32_t tree{t_idx}_left[{n_nodes}] = {{{vals}}};')

            vals = ', '.join(str(v) for v in tree['children_right'])
            a(f'static const int32_t tree{t_idx}_right[{n_nodes}] = {{{vals}}};')

            # missing_left: 1 where a split sends a missing value (NaN) left
            if routes_missing:
                vals = ', '.join(str(int(v)) for v in tree['missing_left'])
                a(f'static const uint8_t tree{t_idx}_missing_left[{n_nodes}] = {{{vals}}};')

            # value
            if self.algorithm_type == 'Decision Tree':
                vals = ', '.join(str(v) for v in tree['value'])
                a(f'static const int32_t tree{t_idx}_value[{n_nodes}] = {{{vals}}};')
            elif self.algorithm_type == 'Random Forest':
                flat = tree['value'].flatten()
                vals = ', '.join(f'{v:.17g}' for v in flat)
                a(f'static const double tree{t_idx}_value[{n_nodes * self.num_classes}] = {{{vals}}};')
            elif self.algorithm_type == 'Gradient Boosting Decision Trees':
                vals = ', '.join(f'{v:.17g}' for v in tree['value'])
                a(f'static const double tree{t_idx}_value[{n_nodes}] = {{{vals}}};')

            a('')
        return out

    # Method to generate the C predict function of the model
    def _c_predict_function(self, name: str, traverse: str,
                            rf_math_include: bool = False) -> List[str]:
        """The C function answering the class code of one feature vector.

        Calls to `traverse` are written without the ``missing_left``
        argument; :meth:`_c_with_missing_args` adds it afterwards.

        Args:
            name (str): Name of the predict function.
            traverse (str): Name of the tree traversal function.
            rf_math_include (bool): Precede the Random Forest function with
                ``#include <math.h>`` (the Arduino sketch always did).

        Returns:
            List[str]: The lines (none for an unknown algorithm type).
        """
        out: List[str] = []
        a = out.append

        def call(t_idx):
            """The traversal call of tree `t_idx`."""
            return (f'{traverse}(features, tree{t_idx}_feature, tree{t_idx}_threshold, '
                    f'tree{t_idx}_left, tree{t_idx}_right, TREE{t_idx}_DEPTH)')

        signature = f'static inline int32_t {name}(const double *features) {{'
        if self.algorithm_type == 'Decision Tree':
            a(signature)
            a(f'    int32_t leaf = {call(0)};')
            a('    return tree0_value[leaf];')
            a('}')
        elif self.algorithm_type == 'Random Forest':
            if rf_math_include:
                a('#include <math.h>')
                a('')
            a(signature)
            a('    double prob_sum[N_CLASSES] = {0};')
            a('    int voted = 0;')
            for t_idx in range(len(self._tree_arrays)):
                a('    {')
                a(f'        int32_t leaf = {call(t_idx)};')
                a('        double total = 0.0;')
                a('        int k;')
                a(f'        for (k = 0; k < N_CLASSES; k++) total += tree{t_idx}_value[leaf * N_CLASSES + k];')
                a('        if (total > 0.0) {')
                a('            voted = 1;')
                a(f'            for (k = 0; k < N_CLASSES; k++) prob_sum[k] += tree{t_idx}_value[leaf * N_CLASSES + k] / total;')
                a('        }')
                a('    }')
            # Every tree abstained (refinement left the sample uncovered in
            # all of them): default class, as predict_batch and the CPN do.
            a('    if (!voted) return DEFAULT_CLASS;')
            a('    int32_t best = 0;')
            a('    int k;')
            a('    for (k = 1; k < N_CLASSES; k++) {')
            a('        if (prob_sum[k] > prob_sum[best]) best = k;')
            a('    }')
            # best is a column of the distribution; answer its label
            rf_labels = ', '.join(str(v) for v in self._rf_labels(self.num_classes))
            a(f'    static const int32_t class_map[N_CLASSES] = {{ {rf_labels} }};')
            a('    return class_map[best];')
            a('}')
        elif self.algorithm_type == 'Gradient Boosting Decision Trees':
            classes = self._gbdt_classes or []
            init_scores = self._gbdt_init_scores or {}
            tree_class_groups_c = self._tree_class_groups
            assert tree_class_groups_c is not None

            a('#include <math.h>')
            a('')
            a(signature)
            if self._gbdt_is_binary and len(classes) >= 2:
                pos_class = classes[1]
                init_s = init_scores.get(pos_class, 0.0)
                a(f'    double score = {init_s:.17g};')
                for t_idx, tree in enumerate(self._tree_arrays):
                    if 'init_score' in tree:
                        continue
                    if tree_class_groups_c[t_idx] != pos_class:
                        continue
                    a(f'    score += tree{t_idx}_value[{call(t_idx)}];')
                # Decide on the raw score, as scikit-learn does (raw >= 0).
                a(f'    return (score >= 0.0) ? {int(classes[1])} : {int(classes[0])};')
                a('}')
            else:
                a(f'    double scores[{len(classes)}] = {{')
                inits = ', '.join(f'{init_scores.get(cl, 0.0):.17g}' for cl in classes)
                a(f'        {inits}')
                a('    };')
                for t_idx, tree in enumerate(self._tree_arrays):
                    if 'init_score' in tree:
                        continue
                    cg = tree_class_groups_c[t_idx]
                    col = list(classes).index(cg) if cg in classes else 0
                    a(f'    scores[{col}] += tree{t_idx}_value[{call(t_idx)}];')
                a('    int32_t best = 0;')
                a('    int k;')
                a(f'    for (k = 1; k < {len(classes)}; k++) {{')
                a('        if (scores[k] > scores[best]) best = k;')
                a('    }')
                # Map col index back to class label
                a(f'    static const int32_t class_map[{len(classes)}] = {{')
                cls_vals = ', '.join(str(int(cl)) for cl in classes)
                a(f'        {cls_vals}')
                a('    };')
                a('    return class_map[best];')
                a('}')
        return out

    # Method to execute export_to_c_header
    def export_to_c_header(
        self,
        filepath: str = 'model.h',
        guard_name: str = 'PYRULEANALYZER_MODEL_H',
    ) -> None:
        """
        Export the compiled tree arrays as a standalone C header file.

        The generated header contains const arrays suitable for Arduino /
        embedded targets. It includes a ``predict(const float *features)``
        function that traverses the trees and returns the predicted class.

        Args:
            filepath (str): Output file path.
            guard_name (str): Include-guard macro name.
        """
        if not hasattr(self, '_arrays_compiled') or not self._arrays_compiled:
            raise RuntimeError(
                'Tree arrays not compiled. Call compile_tree_arrays() first.'
            )

        default_cls = self._default_class_int()

        lines: List[str] = []
        a = lines.append
        routes_missing = self._c_routes_missing()

        a(f'#ifndef {guard_name}')
        a(f'#define {guard_name}')
        a('')
        a('/* Auto-generated by pyruleanalyzer — do not edit */')
        a('')
        a('#include <stdint.h>')
        if routes_missing:
            a('#include <math.h>  /* HUGE_VAL; isnan: missing values (NaN) */')
        else:
            a('#include <math.h>  /* HUGE_VAL */')
        a('')
        lines.extend(self._c_model_defines(default_cls))
        lines.extend(self._c_tree_arrays(routes_missing))

        # Predict function
        lines.extend(self._c_traverse_function(
            'traverse_tree', routes_missing, '  /* leaf: self-loop */'))
        lines.extend(self._c_predict_function('predict', 'traverse_tree'))

        a('')
        a(f'#endif /* {guard_name} */')
        a('')

        with open(filepath, 'w') as f:
            f.write(self._c_with_missing_args('\n'.join(lines), routes_missing))

        print(f'[EXPORT] C header saved to {filepath} '
              f'({os.path.getsize(filepath)} bytes)')

    # Method to export Arduino .ino sketch
    def export_to_arduino_ino(
        self,
        filepath: str = 'model.ino',
        board_model: str = 'auto',
        serial_baud: int = 115200,
        include_sensor_placeholders: bool = True,
        include_memory_check: bool = True,
        max_flash_percent: float = 90.0,
        max_ram_percent: float = 85.0,
    ) -> dict:
        """
        Export the compiled model to a complete Arduino/ESP32 .ino sketch file.

        The generated sketch is self-contained and ready for upload via
        Arduino IDE or PlatformIO. It includes:
          - The C header with tree data (inline as #define constants)
          - The predict() function inline in the sketch
          - A setup() that prints model metadata
          - A loop() that reads features, runs prediction, and outputs results

        Parameters
        ----------
        filepath : str
            Output .ino file path. Default is 'model.ino'.
        board_model : str
            Target board to determine memory limits and architecture hints.
            Options: 'auto', 'uno', 'nano', 'mega', 'esp32', 'leonardo'.
            Default is 'auto' (detects from model size).
        serial_baud : int
            Serial baud rate for output. Default is 115200.
        include_sensor_placeholders : bool
            If True, includes a read_features() function with TODO placeholders
            that the user should replace with real sensor readings.
            Default is True.
        include_memory_check : bool
            If True, includes compile-time checks for Flash/RAM limits in
            comments at the top of the file. Also prints memory report to stdout.
            Default is True.
        max_flash_percent : float
            Maximum percentage of Flash memory to allow (for compatibility check).
            Default is 90.0.
        max_ram_percent : float
            Maximum percentage of SRAM to allow (for compatibility check).
            Default is 85.0.

        Returns
        -------
        dict
            Dictionary with keys:
              - 'ino_path': Path to the generated .ino file
              - 'memory_check': Dict with memory estimation results
              - 'metadata': Model metadata embedded in the sketch
        """
        if not hasattr(self, '_arrays_compiled') or not self._arrays_compiled:
            raise RuntimeError(
                'Tree arrays not compiled. Call compile_tree_arrays() first.'
            )

        # Memory limits per board type (Flash bytes, SRAM bytes)
        BOARD_MEMORY = {
            'uno':      (143360, 2048),     # Arduino Uno: ~140KB flash, 2KB SRAM
            'nano':     (143360, 2048),     # Arduino Nano: same as Uno
            'mega':     (1048576, 8192),    # Arduino Mega: 1MB flash, 8KB SRAM
            'leonardo': (32768, 2560),      # Leonardo: ~32KB for sketch, 2.5KB SRAM
            'esp32':    (4194304, 524288),  # ESP32: 4MB flash, 512KB SRAM
        }

        # Estimate Flash and RAM usage
        estimates = _compute_memory_for_rule_classifier(self)
        est_flash = estimates['estimated_flash_bytes']
        est_ram = estimates['total_ram_bytes']

        # Determine target board
        if board_model == 'auto':
            # Auto-detect: pick the smallest board that fits
            if est_flash <= 143360 and est_ram <= 2048:
                board_model = 'uno'
            elif est_flash <= 1048576 and est_ram <= 8192:
                board_model = 'mega'
            else:
                board_model = 'esp32'

        max_flash, max_sram = BOARD_MEMORY.get(board_model, (143360, 2048))
        
        flash_pct = (est_flash / max_flash) * 100 if max_flash > 0 else 0
        ram_pct = (est_ram / max_sram) * 100 if max_sram > 0 else 0
        
        fits_flash = est_flash <= (max_flash * max_flash_percent / 100.0)
        fits_ram = est_ram <= (max_sram * max_ram_percent / 100.0)

        # Default class
        default_cls = self._default_class_int()

        # Build the sketch content
        lines = []
        a = lines.append

        # Header comment block with metadata and memory check
        a('// ============================================================')
        a(f'// Auto-generated Arduino/ESP32 Sketch by pyruleanalyzer')
        a(f'// Model: {self.algorithm_type}')
        a(f'// Features: {len(self._array_feature_names)} | Classes: {self.num_classes} | Trees: {len(self._tree_arrays)}')
        a(f'// Target Board: {board_model.upper()}')
        a('// ============================================================')
        a('')

        if include_memory_check:
            flash_status = 'OK' if fits_flash else 'OVER!'
            ram_status = 'OK' if fits_ram else 'OVER!'
            a('// --- Memory Check (Target: ' + board_model.upper() + ') ---')
            a(f'// Flash: {est_flash:,} / {max_flash:,} bytes ({flash_pct:.1f}%) [{flash_status}]')
            a(f'// SRAM:  {est_ram:,} / {max_sram:,} bytes ({ram_pct:.1f}%) [{ram_status}]')
            if not fits_flash or not fits_ram:
                a('// WARNING: Model may NOT fit on this board! Reduce tree depth or number of trees.')
            a('')

        # Include the inline C header content
        a('// ============================================================')
        a(f'// MODEL DATA ({len(self._tree_arrays)} trees, {len(self._array_feature_names)} features)')
        a('// ============================================================')
        
        # Inline header: the C header's content without the #ifndef guard
        routes_missing = self._c_routes_missing()

        if routes_missing:
            a('#include <math.h>  // HUGE_VAL; isnan: missing values (NaN)')
        else:
            a('#include <math.h>  // HUGE_VAL')
        lines.extend(self._c_model_defines(default_cls, line_comment=True))
        lines.extend(self._c_tree_arrays(routes_missing, line_comment=True))

        # Inline traverse_tree function
        a('// ============================================================')
        a('// PREDICTION ENGINE (C inline — no library needed)')
        a('// ============================================================')
        a('')
        lines.extend(self._c_traverse_function('pyra_traverse_tree', routes_missing))
        lines.extend(self._c_predict_function(
            'pyra_predict', 'pyra_traverse_tree', rf_math_include=True))

        a('')

        # Arduino sketch section
        a('// ============================================================')
        a('// ARDUINO SKETCH SECTION')
        a('// ============================================================')
        a('')
        a(f'#define SERIAL_BAUD {serial_baud}')
        a('#define SAMPLE_INTERVAL_MS 1000')
        a('')

        # Global feature variables
        a('// Feature buffer (modify read_features() to populate these)')
        # double, matching pyra_predict(const double *): on AVR double is the
        # same 4-byte type as float, and on ESP32 a float buffer read through
        # a double pointer would be garbage.
        a(f'double features[N_FEATURES];')
        a('')

        if include_sensor_placeholders:
            a('// ============================================================')
            a('// TODO: Replace with your sensor reading logic')
            a('// ============================================================')
            a('void read_features(void) {')
            for i in range(min(len(self._array_feature_names), 10)):
                feat_name = self._array_feature_names[i] if i < len(self._array_feature_names) else f'feature_{i}'
                a(f'    features[{i}] = 0.0; // TODO: read {self._c_comment(feat_name)}')
            remaining = max(0, len(self._array_feature_names) - 10)
            if remaining > 0:
                a(f'    // ... and {remaining} more features')
            a('}')
            a('')

        # Setup function
        a('void setup(void) {')
        a('    Serial.begin(SERIAL_BAUD);')
        a('    while (!Serial) delay(10);')
        a('    Serial.println(F("pyruleanalyzer model loaded"));')
        a(f'    Serial.print(F("Algorithm: ")); Serial.println(F("{self.algorithm_type}"));')
        a(f'    Serial.print(F("Features: ")); Serial.print(N_FEATURES);')
        a(f'    Serial.print(F(" | Classes: ")); Serial.print(N_CLASSES);')
        a(f'    Serial.print(F(" | Trees: ")); Serial.println(N_TREES);')
        a('}')
        a('')

        # Loop function
        a('void loop(void) {')
        a('    unsigned long start_us = micros();')
        a('')
        if include_sensor_placeholders:
            a('    read_features();  // Populate features[] from sensors')
        else:
            a('    // TODO: populate features[] from your sensors')
        a('')
        a('    int32_t result = pyra_predict(features);')
        a('')
        a('    Serial.print(F("{\\"class\\":"));')
        a('    Serial.print(result);')
        a('    Serial.print(F(", \\"trees\\":"));')
        a(f'    Serial.print(N_TREES);')
        a('    Serial.println("}");')
        a('')
        a('    unsigned long elapsed_us = micros() - start_us;')
        a('    Serial.print(F("("));')
        a('    Serial.print(elapsed_us);')
        a('    Serial.println(F(" us)"));')
        a('')
        a('    delay(SAMPLE_INTERVAL_MS);')
        a('}')

        # Write the file
        ino_dir = os.path.dirname(filepath)
        if ino_dir:
            os.makedirs(ino_dir, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self._c_with_missing_args('\n'.join(lines), routes_missing))

        # Print memory report to stdout
        avr_pct_flash = (est_flash / max_flash) * 100 if max_flash > 0 else 0
        avr_pct_ram = (est_ram / max_sram) * 100 if max_sram > 0 else 0
        
        print(f'  ✅ Arduino sketch generated: {filepath}')
        print(f'     Flash: {est_flash:,} bytes ({avr_pct_flash:.1f}% of {max_flash:,})')
        print(f'     SRAM:  {est_ram:,} bytes ({avr_pct_ram:.1f}% of {max_sram:,})')
        
        if not fits_flash or not fits_ram:
            print('     ⚠️  WARNING: Model may NOT fit on target board!')

        file_size = os.path.getsize(filepath)
        
        return {
            'ino_path': filepath,
            'memory_check': {
                'board_model': board_model,
                'flash_bytes': est_flash,
                'ram_bytes': est_ram,
                'max_flash': max_flash,
                'max_sram': max_sram,
                'flash_percent': avr_pct_flash,
                'ram_percent': avr_pct_ram,
                'fits_flash': fits_flash,
                'fits_ram': fits_ram,
            },
            'metadata': {
                'algorithm_type': self.algorithm_type,
                'n_features': len(self._array_feature_names),
                'n_classes': self.num_classes,
                'n_trees': len(self._tree_arrays),
                'default_class': default_cls,
                'file_size_bytes': file_size,
            },
        }
