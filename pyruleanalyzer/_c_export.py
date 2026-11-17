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

# avr-libc (Arduino Uno, Mega, Leonardo) defines INFINITY but not the C89
# HUGE_VAL the tree arrays use, so the sketch would not compile on AVR.
_HUGE_VAL_FALLBACK = [
    '#ifndef HUGE_VAL',
    '#define HUGE_VAL (__builtin_huge_val())',
    '#endif',
]

# Partition tables written beside the sketch for the presets that need one.
# One factory app of 0xFF0000 bytes fills a 16 MB flash after the bootloader,
# the partition table and NVS (no OTA, no file system).
_CUSTOM_PARTITIONS = {
    'esp32s3_16mb_custom': ('# Name,   Type, SubType, Offset,  Size\n'
                            'nvs,      data, nvs,     0x9000,  0x5000\n'
                            'app0,     app,  factory, 0x10000, 0xFF0000\n'),
}

# Where the tree arrays live and how they are read. On AVR (Uno, Mega,
# Leonardo) a `static const` array is copied to RAM at start-up, which caps the
# model at the 2 KB (Uno) or 8 KB (Mega) of SRAM; with PROGMEM it stays in
# flash and is read with pgm_read_*. Above 64 KB of flash (Mega 2560) the
# arrays can sit past the reach of a 16-bit pointer, so the reads take 32-bit
# far addresses. Elsewhere (ESP32, RP2040, a PC) the macros are plain array
# accesses and the generated code is the one the exports always had.
_ROM_PRELUDE = [
    '#ifndef PYRA_ROM',
    '/* One traversal function for every tree: inlining it into each call',
    '   costs hundreds of bytes per tree of an ensemble (316 KB for 750 trees',
    '   on a Cortex-M4). */',
    '#if defined(__GNUC__)',
    '#define PYRA_NOINLINE __attribute__((noinline))',
    '#else',
    '#define PYRA_NOINLINE',
    '#endif',
    '#if defined(__AVR__)',
    '#include <avr/pgmspace.h>',
    '#if __SIZEOF_DOUBLE__ != 4',
    '#error "pyruleanalyzer: the AVR export expects a 4-byte double"',
    '#endif',
    '#define PYRA_ROM PROGMEM',
    '#if defined(__AVR_HAVE_ELPM__)  /* more than 64 KB of flash (Mega 2560) */',
    'typedef uint_farptr_t pyra_addr_t;',
    '#define PYRA_ADDR(v) pgm_get_far_address(v)',
    '#define PYRA_I32(a, i) ((int32_t)pgm_read_dword_far((a) + 4UL * (uint32_t)(i)))',
    '#define PYRA_DBL(a, i) ((double)pgm_read_float_far((a) + 4UL * (uint32_t)(i)))',
    '#define PYRA_U8(a, i) ((uint8_t)pgm_read_byte_far((a) + (uint32_t)(i)))',
    '#else',
    'typedef uint16_t pyra_addr_t;',
    '#define PYRA_ADDR(v) ((pyra_addr_t)(v))',
    '#define PYRA_I32(a, i) ((int32_t)pgm_read_dword((a) + 4U * (uint16_t)(i)))',
    '#define PYRA_DBL(a, i) ((double)pgm_read_float((a) + 4U * (uint16_t)(i)))',
    '#define PYRA_U8(a, i) ((uint8_t)pgm_read_byte((a) + (uint16_t)(i)))',
    '#endif',
    '#define PYRA_ARR(T) pyra_addr_t',
    '#define PYRA_SYM(n) __asm__(#n) __attribute__((used))',
    '#define PYRA_T(d, f) (f)  /* threshold: largest float32 <= t */',
    '#else',
    '#define PYRA_ROM',
    '#define PYRA_ARR(T) const T *',
    '#define PYRA_ADDR(v) (v)',
    '#define PYRA_I32(a, i) ((a)[i])',
    '#define PYRA_DBL(a, i) ((a)[i])',
    '#define PYRA_U8(a, i) ((a)[i])',
    '#define PYRA_SYM(n)',
    '#define PYRA_T(d, f) (d)',
    '#endif',
    '#endif',
]

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


def _format_c_threshold(v) -> str:
    """Format a split threshold as a C literal that is exact on every target.

    A split sends ``x`` left when ``x <= t``. Where ``double`` is a double the
    literal is ``t`` itself (``%.17g``). On AVR ``double`` is a float32, and the
    compiler would round ``t`` to the *nearest* float32, which can lie above
    ``t``: a float32 input equal to that value would then go left on the board
    and right in Python. ``PYRA_T`` picks, on AVR, the largest float32 not
    above ``t`` instead; for a float32 input ``x``, ``x <= t`` holds exactly
    when ``x`` is at most that value, so the split is decided as in Python.

    Args:
        v: The threshold.

    Returns:
        str: ``PYRA_T(<double>, <float32>f)``, or the literal of
        :func:`_format_c_float` for an infinite or NaN threshold.
    """
    if not math.isfinite(v):
        return _format_c_float(v)
    import numpy as np
    f = np.float32(v)
    if float(f) > v:
        f = np.nextafter(f, np.float32(-np.inf))
    single = f'{float(f):.9g}'
    if '.' not in single and 'e' not in single:
        single += '.0'  # "3f" is not a C literal; "3.0f" is
    return f'PYRA_T({v:.17g}, {single}f)'


# Helper function for memory estimation (used by export_to_arduino_ino)
def _compute_memory_for_rule_classifier(rf_model) -> dict:
    """Compute memory estimates for a RuleClassifier instance.

    Args:
        rf_model: A RuleClassifier with compiled tree arrays.

    Returns:
        Dict with flash and RAM byte estimates.
    """
    from .footprint import BOARDS
    # Sketch limits of the Uno and the ESP32 (footprint.BOARDS).
    avr_max_flash = BOARDS['uno'].flash_bytes
    avr_max_sram = BOARDS['uno'].ram_bytes
    esp32_max_flash = BOARDS['esp32'].flash_bytes
    esp32_max_sram = BOARDS['esp32'].ram_bytes

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

        A scikit-learn Gradient Boosting never does: GradientBoostingClassifier
        refuses NaN, so its exports are written as before. One read from
        XGBoost, LightGBM or CatBoost does, as those libraries route NaN.

        Returns:
            bool: True when every compiled tree has its ``missing_left``
            array and there is at least one tree to traverse.
        """
        if (self.algorithm_type == 'Gradient Boosting Decision Trees'
                and not self._gbdt_routes_nan()):
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
        value = ('(double)(float)features[PYRA_I32(feat_idx, node)]' if f32
                 else 'features[PYRA_I32(feat_idx, node)]')
        out = [
            f'static inline PYRA_NOINLINE int32_t {name}(',
            '    const double *features,',
            '    PYRA_ARR(int32_t) feat_idx,',
            '    PYRA_ARR(double) thresh,',
            '    PYRA_ARR(int32_t) left,',
            '    PYRA_ARR(int32_t) right,',
        ]
        if routes_missing:
            out.append('    PYRA_ARR(uint8_t) missing_left,')
        out += [
            '    int max_depth',
            ') {',
            '    int32_t node = 0;',
            '    int d;',
            '    for (d = 0; d < max_depth; d++) {',
            '        int32_t l = PYRA_I32(left, node);',
            f'        if (l == node) break;{leaf_comment}',
        ]
        if routes_missing:
            out += [
                f'        double x = {value};',
                '        if (isnan(x))  /* missing value: the side the split recorded */',
                '            node = PYRA_U8(missing_left, node) ? l : PYRA_I32(right, node);',
                '        else if (x <= PYRA_DBL(thresh, node))',
            ]
        else:
            out.append(f'        if ({value} <= PYRA_DBL(thresh, node))')
        out += [
            '            node = l;',
            '        else',
            '            node = PYRA_I32(right, node);',
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
        return re.sub(r'PYRA_ADDR\(tree(\d+)_right\), TREE\1_DEPTH\)',
                      r'PYRA_ADDR(tree\1_right), PYRA_ADDR(tree\1_missing_left), '
                      r'TREE\1_DEPTH)', source)

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
                       line_comment: bool = False,
                       avr_table: bool = False) -> List[str]:
        """The ``TREEn_DEPTH`` macro and const arrays of each compiled tree.

        Args:
            routes_missing (bool): Also emit ``treeN_missing_left``.
            line_comment (bool): ``//`` comments (Arduino) instead of blocks.
            avr_table (bool): Give every array an assembler name
                (``PYRA_SYM``), for the AVR tree table of
                :meth:`_c_avr_tree_table`.

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

            def sym(key, t_idx=t_idx):
                """The assembler name of one array (AVR tree table only)."""
                return f' PYRA_SYM(pyra_t{t_idx}_{key})' if avr_table else ''
            title = f'Tree {t_idx}: {n_nodes} nodes, depth {tree_depth}'
            a(f'// {title}' if line_comment else f'/* {title} */')
            a(f'#define TREE{t_idx}_DEPTH {tree_depth}')

            vals = ', '.join(str(v) for v in tree['feature_idx'])
            a(f'static const int32_t tree{t_idx}_feature[{n_nodes}]{sym('f')} PYRA_ROM = {{{vals}}};')

            # threshold — special float values as C-compatible literals
            vals = ', '.join(_format_c_threshold(v) for v in tree['threshold'])
            a(f'static const double tree{t_idx}_threshold[{n_nodes}]{sym('t')} PYRA_ROM = {{{vals}}};')

            vals = ', '.join(str(v) for v in tree['children_left'])
            a(f'static const int32_t tree{t_idx}_left[{n_nodes}]{sym('l')} PYRA_ROM = {{{vals}}};')

            vals = ', '.join(str(v) for v in tree['children_right'])
            a(f'static const int32_t tree{t_idx}_right[{n_nodes}]{sym('r')} PYRA_ROM = {{{vals}}};')

            # missing_left: 1 where a split sends a missing value (NaN) left
            if routes_missing:
                vals = ', '.join(str(int(v)) for v in tree['missing_left'])
                a(f'static const uint8_t tree{t_idx}_missing_left[{n_nodes}]{sym('m')} PYRA_ROM = {{{vals}}};')

            # value
            if self.algorithm_type == 'Decision Tree':
                vals = ', '.join(str(v) for v in tree['value'])
                a(f'static const int32_t tree{t_idx}_value[{n_nodes}]{sym('v')} PYRA_ROM = {{{vals}}};')
            elif self.algorithm_type == 'Random Forest':
                flat = tree['value'].flatten()
                vals = ', '.join(f'{v:.17g}' for v in flat)
                a(f'static const double tree{t_idx}_value[{n_nodes * self.num_classes}]{sym('v')} PYRA_ROM = {{{vals}}};')
            elif self.algorithm_type == 'Gradient Boosting Decision Trees':
                vals = ', '.join(f'{v:.17g}' for v in tree['value'])
                a(f'static const double tree{t_idx}_value[{n_nodes}]{sym('v')} PYRA_ROM = {{{vals}}};')

            a('')
        return out

    # Method to list the trees the AVR tree table walks, in sum order
    def _c_table_trees(self) -> List[tuple]:
        """``(t_idx, column)`` of every tree the predict function adds.

        A Random Forest adds every tree (column 0, unused); a multiclass GBDT
        adds every tree to the score of its class column; a binary GBDT only
        the trees of the positive class. The order is the order of the
        unrolled predict function, so the sums are the same.
        """
        trees = [(t, tr) for t, tr in enumerate(self._tree_arrays) if 'init_score' not in tr]
        if self.algorithm_type == 'Random Forest':
            return [(t, 0) for t, _ in trees]
        classes = self._gbdt_classes or []
        groups = self._tree_class_groups
        if self._gbdt_is_binary and len(classes) >= 2:
            return [(t, 0) for t, _ in trees if groups[t] == classes[1]]
        return [(t, list(classes).index(groups[t]) if groups[t] in classes else 0)
                for t, _ in trees]

    # Method to generate the AVR tree table of an ensemble sketch
    def _c_avr_tree_table(self, routes_missing: bool) -> List[str]:
        """The flash table of tree addresses that replaces one call per tree.

        On AVR every unrolled traversal call loads four or five 32-bit far
        addresses and adds a value read from flash: about 176 bytes of code
        per tree, a quarter of a Mega filled by a 375-tree model. The table
        holds, per tree, the 24-bit addresses of its arrays (``.long`` in an
        assembler block, since a C initializer cannot hold an address above
        64 KB on AVR) and, in a second array, its depth and class column; the
        predict function walks it in a loop. Only compiled on AVR.
        """
        entries = self._c_table_trees()
        keys = ['f', 't', 'l', 'r'] + (['m'] if routes_missing else []) + ['v']
        depths = [int(self._tree_arrays[t].get('max_depth', 50)) for t, _ in entries]
        out = ['#if defined(__AVR__)',
               '// Tree table: the flash addresses of the arrays of every tree, walked',
               '// in a loop instead of one inline call per tree (smaller code on AVR).',
               '#define PYRA_TREE_TABLE 1',
               f'#define PYRA_TABLE_N {len(entries)}',
               f'#define PYRA_TABLE_W {len(keys)}',
               '__asm__(',
               r'    "  .pushsection .progmem.data,\"a\",@progbits\n"',
               r'    "  .global pyra_tree_table\n"',
               r'    "pyra_tree_table:\n"']
        for t, _ in entries:
            names = ', '.join(f'pyra_t{t}_{k}' for k in keys)
            out.append(f'    "  .long {names}' + r'\n"')
        out += [r'    "  .popsection\n");',
                'extern const uint8_t pyra_tree_table[] PROGMEM;',
                f'static const uint16_t pyra_tree_meta[{2 * len(entries)}] PROGMEM = {{'
                + ', '.join(f'{d}, {c}' for d, (_, c) in zip(depths, entries)) + '};',
                '#if defined(__AVR_HAVE_ELPM__)',
                '#define PYRA_U16(a, i) ((uint16_t)pgm_read_word_far((a) + 2UL * (uint32_t)(i)))',
                '#else',
                '#define PYRA_U16(a, i) ((uint16_t)pgm_read_word((a) + 2U * (uint16_t)(i)))',
                '#endif',
                '#endif',
                '']
        return out

    # Method to generate the table-driven predict function (AVR)
    def _c_predict_function_table(self, name: str, traverse: str,
                                  routes_missing: bool) -> List[str]:
        """The predict function of :meth:`_c_predict_function`, with the trees
        walked through the AVR tree table instead of unrolled.

        It performs the same additions in the same order, so it returns the
        same class; only the code size changes.
        """
        out: List[str] = []
        a = out.append
        args = ', '.join(f'(pyra_addr_t)PYRA_I32(pyra_d, {i})'
                         for i in range(5 if routes_missing else 4))
        loop = ['    uint16_t pyra_t;',
                '    for (pyra_t = 0; pyra_t < PYRA_TABLE_N; pyra_t++) {',
                '        pyra_addr_t pyra_d = PYRA_ADDR(pyra_tree_table)'
                ' + (pyra_addr_t)(4UL * PYRA_TABLE_W * pyra_t);',
                f'        int32_t leaf = {traverse}(features, {args}, '
                '(int)PYRA_U16(PYRA_ADDR(pyra_tree_meta), 2 * pyra_t));',
                '        pyra_addr_t pyra_v = (pyra_addr_t)PYRA_I32(pyra_d, PYRA_TABLE_W - 1);']
        signature = f'static inline int32_t {name}(const double *features) {{'
        if self.algorithm_type == 'Random Forest':
            a(signature)
            a('    double prob_sum[N_CLASSES] = {0};')
            a('    int voted = 0;')
            out.extend(loop)
            a('        double total = 0.0;')
            a('        int k;')
            a('        for (k = 0; k < N_CLASSES; k++) total += PYRA_DBL(pyra_v, leaf * N_CLASSES + k);')
            a('        if (total > 0.0) {')
            a('            voted = 1;')
            a('            for (k = 0; k < N_CLASSES; k++) prob_sum[k] += PYRA_DBL(pyra_v, leaf * N_CLASSES + k) / total;')
            a('        }')
            a('    }')
            a('    if (!voted) return DEFAULT_CLASS;')
            a('    int32_t best = 0;')
            a('    int k;')
            a('    for (k = 1; k < N_CLASSES; k++) {')
            a('        if (prob_sum[k] > prob_sum[best]) best = k;')
            a('    }')
            rf_labels = ', '.join(str(v) for v in self._rf_labels(self.num_classes))
            a(f'    static const int32_t class_map[N_CLASSES] = {{ {rf_labels} }};')
            a('    return class_map[best];')
            a('}')
            return out
        classes = self._gbdt_classes or []
        init_scores = self._gbdt_init_scores or {}
        stype = 'float' if getattr(self, 'score_dtype', 'float64') == 'float32' else 'double'
        cut = float(getattr(self, 'binary_threshold', 0.0))
        a(signature)
        if self._gbdt_is_binary and len(classes) >= 2:
            a(f'    {stype} score = {init_scores.get(classes[1], 0.0):.17g};')
            out.extend(loop)
            a('        score += PYRA_DBL(pyra_v, leaf);')
            a('    }')
            from ._boosting_import import binary_test
            op, c = binary_test(cut)
            a(f'    return (score {op} {_format_c_float(c)}) ? {int(classes[1])} : {int(classes[0])};')
            a('}')
            return out
        a(f'    {stype} scores[{len(classes)}] = {{')
        a('        ' + ', '.join(f'{init_scores.get(cl, 0.0):.17g}' for cl in classes))
        a('    };')
        out.extend(loop)
        a('        scores[PYRA_U16(PYRA_ADDR(pyra_tree_meta), 2 * pyra_t + 1)] += PYRA_DBL(pyra_v, leaf);')
        a('    }')
        a('    int32_t best = 0;')
        a('    int k;')
        a(f'    for (k = 1; k < {len(classes)}; k++) {{')
        a('        if (scores[k] > scores[best]) best = k;')
        a('    }')
        a(f'    static const int32_t class_map[{len(classes)}] = {{')
        a('        ' + ', '.join(str(int(cl)) for cl in classes))
        a('    };')
        a('    return class_map[best];')
        a('}')
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
            return (f'{traverse}(features, PYRA_ADDR(tree{t_idx}_feature), '
                    f'PYRA_ADDR(tree{t_idx}_threshold), PYRA_ADDR(tree{t_idx}_left), '
                    f'PYRA_ADDR(tree{t_idx}_right), TREE{t_idx}_DEPTH)')

        signature = f'static inline int32_t {name}(const double *features) {{'
        if self.algorithm_type == 'Decision Tree':
            a(signature)
            a(f'    int32_t leaf = {call(0)};')
            a('    return PYRA_I32(PYRA_ADDR(tree0_value), leaf);')
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
                a(f'        for (k = 0; k < N_CLASSES; k++) total += PYRA_DBL(PYRA_ADDR(tree{t_idx}_value), leaf * N_CLASSES + k);')
                a('        if (total > 0.0) {')
                a('            voted = 1;')
                a(f'            for (k = 0; k < N_CLASSES; k++) prob_sum[k] += PYRA_DBL(PYRA_ADDR(tree{t_idx}_value), leaf * N_CLASSES + k) / total;')
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

            # XGBoost adds its trees in float32: a float accumulator rounds
            # after every stage (score + value is computed in double, then
            # rounded once to float, which is the float32 sum).
            stype = 'float' if getattr(self, 'score_dtype', 'float64') == 'float32' else 'double'
            cut = float(getattr(self, 'binary_threshold', 0.0))
            a('#include <math.h>')
            a('')
            a(signature)
            if self._gbdt_is_binary and len(classes) >= 2:
                pos_class = classes[1]
                init_s = init_scores.get(pos_class, 0.0)
                a(f'    {stype} score = {init_s:.17g};')
                for t_idx, tree in enumerate(self._tree_arrays):
                    if 'init_score' in tree:
                        continue
                    if tree_class_groups_c[t_idx] != pos_class:
                        continue
                    a(f'    score += PYRA_DBL(PYRA_ADDR(tree{t_idx}_value), {call(t_idx)});')
                # Decide on the raw score, as scikit-learn does (raw >= 0);
                # other libraries move the cutoff (binary_threshold).
                from ._boosting_import import binary_test
                op, c = binary_test(cut)
                a(f'    return (score {op} {_format_c_float(c)}) ? {int(classes[1])} : {int(classes[0])};')
                a('}')
            else:
                a(f'    {stype} scores[{len(classes)}] = {{')
                inits = ', '.join(f'{init_scores.get(cl, 0.0):.17g}' for cl in classes)
                a(f'        {inits}')
                a('    };')
                for t_idx, tree in enumerate(self._tree_arrays):
                    if 'init_score' in tree:
                        continue
                    cg = tree_class_groups_c[t_idx]
                    col = list(classes).index(cg) if cg in classes else 0
                    a(f'    scores[{col}] += PYRA_DBL(PYRA_ADDR(tree{t_idx}_value), {call(t_idx)});')
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
        lines.extend(_HUGE_VAL_FALLBACK)
        lines.extend(_ROM_PRELUDE)
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

        with open(filepath, 'w', encoding='utf-8') as f:
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
            Options: 'auto', a preset of ``footprint.BOARDS`` ('uno', 'nano',
            'mega2560' or 'mega', 'nrf52840', 'pico', 'rp2040_16mb', 'esp32', 'esp32_huge_app',
            'esp32s3', 'esp32s3_16mb', 'esp32s3_16mb_custom', 'esp32s3_32mb') or
            'leonardo'. 'esp32s3_16mb_custom' also writes a ``partitions.csv``
            beside the sketch (one 15.9 MB app); select the Custom partition
            scheme when compiling.
            Default is 'auto' (the smallest of Uno, Mega and ESP32 that fits).
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

        # Sketch limits per board (flash bytes, SRAM bytes): the presets of
        # footprint.BOARDS (the limits of each board's Arduino core), plus the
        # Leonardo (ArduinoCore-avr boards.txt) and 'mega' for 'mega2560'.
        from .footprint import BOARDS
        BOARD_MEMORY = {name: (b.flash_bytes, b.ram_bytes) for name, b in BOARDS.items()}
        BOARD_MEMORY['mega'] = BOARD_MEMORY['mega2560']
        BOARD_MEMORY['nano_old'] = BOARD_MEMORY['nano']  # old bootloader, same limits
        BOARD_MEMORY['leonardo'] = (28672, 2560)

        # Estimate Flash and RAM usage
        estimates = _compute_memory_for_rule_classifier(self)
        est_flash = estimates['estimated_flash_bytes']
        est_ram = estimates['total_ram_bytes']

        # Determine target board
        if board_model == 'auto':
            # Auto-detect: pick the smallest board that fits
            board_model = 'esp32'
            for name in ('uno', 'mega'):
                flash_lim, ram_lim = BOARD_MEMORY[name]
                if est_flash <= flash_lim and est_ram <= ram_lim:
                    board_model = name
                    break

        if board_model not in BOARD_MEMORY:
            raise ValueError(f'unknown board_model {board_model!r}; known: '
                             f"'auto', {', '.join(repr(k) for k in sorted(BOARD_MEMORY))}")
        max_flash, max_sram = BOARD_MEMORY[board_model]
        
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
        a(f'// Model: {self.model_name}')
        a(f'// Features: {len(self._array_feature_names)} | Classes: {self.num_classes} | Trees: {len(self._tree_arrays)}')
        a(f'// Target Board: {board_model.upper()}')
        if board_model in _CUSTOM_PARTITIONS:
            a('// Partition scheme: Custom (the partitions.csv beside this sketch;')
            a('//   arduino-cli: --fqbn esp32:esp32:esp32s3:FlashSize=16M,PartitionScheme=custom)')
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
        lines.extend(_HUGE_VAL_FALLBACK)
        lines.extend(_ROM_PRELUDE)
        # Adafruit nRF52 (and other TinyUSB) cores: Serial is the TinyUSB CDC
        # class, which only links when the sketch includes its header.
        lines.extend(['#if defined(USE_TINYUSB)', '#include <Adafruit_TinyUSB.h>', '#endif'])
        lines.extend(self._c_model_defines(default_cls, line_comment=True))
        avr_table = (self.algorithm_type in ('Random Forest', 'Gradient Boosting Decision Trees')
                     and len(self._c_table_trees()) > 1)
        lines.extend(self._c_tree_arrays(routes_missing, line_comment=True,
                                         avr_table=avr_table))
        if avr_table:
            lines.extend(self._c_avr_tree_table(routes_missing))

        # Inline traverse_tree function
        a('// ============================================================')
        a('// PREDICTION ENGINE (C inline — no library needed)')
        a('// ============================================================')
        a('')
        lines.extend(self._c_traverse_function('pyra_traverse_tree', routes_missing))
        if avr_table:
            a('#if defined(PYRA_TREE_TABLE)')
            lines.extend(self._c_predict_function_table(
                'pyra_predict', 'pyra_traverse_tree', routes_missing))
            a('#else')
        lines.extend(self._c_predict_function(
            'pyra_predict', 'pyra_traverse_tree', rf_math_include=True))
        if avr_table:
            a('#endif')

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
        a(f'    Serial.print(F("Algorithm: ")); Serial.println(F("{self.model_name}"));')
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
        if board_model in _CUSTOM_PARTITIONS:
            # The core copies a partitions.csv found beside the sketch.
            part = os.path.join(os.path.dirname(os.path.abspath(filepath)), 'partitions.csv')
            with open(part, 'w', encoding='utf-8') as f:
                f.write(_CUSTOM_PARTITIONS[board_model])

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
