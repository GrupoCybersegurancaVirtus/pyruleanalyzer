"""The ``.bin`` model format.

Provides ``BinaryFormatMixin``: writing the compiled tree arrays to the
compact binary format (``export_to_binary``), reading it back into a
classifier ready for ``predict_batch`` (``load_binary``) and validating the
arrays read from an untrusted file.
"""

import os
import struct
import numpy as np


class BinaryFormatMixin:
    """Binary (.bin) export and loading."""

    # =========================================================================
    # BINARY / C HEADER EXPORT
    # =========================================================================

    # Method to serialise the code -> label table for the binary format
    @staticmethod
    def _pack_label_table(table: list) -> bytes:
        """The label table as the bytes of the binary format (version 4).

        Args:
            table (list): Original labels (str, float, int, bool), by code.

        Returns:
            bytes: ``n_labels`` (uint16) then one typed entry per label.

        Raises:
            ValueError: If a label cannot be written.
        """
        out = [struct.pack('<H', len(table))]
        for lab in table:
            if isinstance(lab, bool):
                out.append(struct.pack('<BB', 3, 1 if lab else 0))
            elif isinstance(lab, str):
                encoded = lab.encode('utf-8')
                if len(encoded) > 0xFFFF:
                    raise ValueError('A class label is too long for the binary format.')
                out.append(struct.pack('<BH', 0, len(encoded)) + encoded)
            elif isinstance(lab, int):
                out.append(struct.pack('<Bq', 2, lab))
            elif isinstance(lab, float):
                out.append(struct.pack('<Bd', 1, lab))
            else:
                raise ValueError(f'Cannot write class label {lab!r:.60} to a binary model.')
        return b''.join(out)

    # Method to execute export_to_binary
    def export_to_binary(self, filepath: str = 'model.bin') -> None:
        """
        Export the compiled tree arrays to a compact binary file.

        File format (all little-endian):
            Header:
                4 bytes  magic: b'PYRA'
                1 byte   version: 1, or 2 when the input precision is set,
                         3 with column labels, 4 with missing-value routing
                         or a label table (see the flags)
                1 byte   algorithm_type: 0=DT, 1=RF, 2=GBDT
                2 bytes  n_features (uint16)
                2 bytes  n_classes (uint16)
                2 bytes  n_trees (uint16)
                4 bytes  default_class (int32)
            Version 2 and later:
                1 byte   flags: bit 0 = compare inputs rounded to float32
                         (``input_dtype='float32'``, as scikit-learn does).
                         Version 4 also: bit 1 = per-node missing-value
                         routing stored with each tree; bit 2 = column
                         labels follow; bit 3 = label table follows.
            Version 3 (a Random Forest whose labels are not 0..K-1), or
            version 4 with flag bit 2:
                n_classes * 4 bytes  label of each distribution column (int32)
            Version 4 with flag bit 3 (labels that are not integers; the
            model answers their codes 0..K-1, the table names them):
                2 bytes  n_labels (uint16)
                For each label, in code order:
                    1 byte  type: 0 = text, 1 = float64, 2 = int64, 3 = bool
                    text:   2 bytes length (uint16) + utf-8 bytes
                    float64 / int64: 8 bytes; bool: 1 byte
            For GBDT additionally:
                1 byte   is_binary (bool)
                2 bytes  n_gbdt_classes (uint16)
                For each gbdt_class:
                    4 bytes  class_label (int32)
                For each gbdt_class:
                    8 bytes  init_score (float64)
            Feature names:
                For each feature:
                    2 bytes  name_len (uint16)
                    N bytes  name (utf-8)
            For each tree:
                4 bytes  n_nodes (int32)
                For GBDT:
                    1 byte   is_init (bool)
                    4 bytes  class_group (int32, only if not init)
                    If is_init:
                        8 bytes  init_score (float64)
                        continue to next tree
                feature_idx:   n_nodes * 4 bytes (int32)
                threshold:     n_nodes * 8 bytes (float64)
                children_left: n_nodes * 4 bytes (int32)
                children_right:n_nodes * 4 bytes (int32)
                missing_left:  n_nodes * 1 byte (uint8, 1 = a NaN goes
                               left; version 4 with flag bit 1 only)
                value:
                    DT:   n_nodes * 4 bytes (int32)
                    RF:   n_nodes * n_classes * 8 bytes (float64)
                    GBDT: n_nodes * 8 bytes (float64)
                max_depth:     4 bytes (int32)

        Models that record no missing-value routing and whose labels are
        integers are written exactly as before (versions 1 to 3).

        Args:
            filepath (str): Output file path.
        """
        import struct

        if not hasattr(self, '_arrays_compiled') or not self._arrays_compiled:
            raise RuntimeError(
                'Tree arrays not compiled. Call compile_tree_arrays() first.'
            )

        algo_map = {'Decision Tree': 0, 'Random Forest': 1,
                     'Gradient Boosting Decision Trees': 2}
        algo_byte = algo_map[self.algorithm_type]

        try:
            default_cls = int(str(self.default_class).replace('Class', '').strip())
        except (ValueError, AttributeError):
            default_cls = 0

        # Count real trees (skip GBDT init entries for tree count)
        n_trees = len(self._tree_arrays)

        with open(filepath, 'wb') as f:
            # --- Header ---
            # Version 1 had no room for the input precision; a model that
            # rounds its inputs to float32 is written as version 2, so a
            # reloaded model routes samples exactly like the one exported.
            f32 = getattr(self, 'input_dtype', None) == 'float32'
            # Version 3 adds the label of each column of a forest's
            # distributions, written only when they are not 0..K-1.
            rf_labels = (self._rf_labels(self.num_classes)
                         if self.algorithm_type == 'Random Forest' else None)
            has_labels = (rf_labels is not None
                          and rf_labels != list(range(self.num_classes)))
            # Version 4: the missing-value routing of every split, the code
            # -> label table, and the column labels of a Decision Tree too
            # (predict_proba of a reloaded tree needs them).
            routes_missing = self._c_routes_missing()
            table = getattr(self, 'label_table', None)
            col_labels = self._rf_labels(self.num_classes)
            dt_labels = (self.algorithm_type == 'Decision Tree'
                         and col_labels != list(range(self.num_classes)))
            if routes_missing or table is not None or dt_labels:
                version = 4
                has_labels = has_labels or dt_labels
                rf_labels = col_labels
            else:
                version = 3 if has_labels else (2 if f32 else 1)
            flags = ((1 if f32 else 0)
                     | (2 if version == 4 and routes_missing else 0)
                     | (4 if version == 4 and has_labels else 0)
                     | (8 if version == 4 and table is not None else 0))
            f.write(b'PYRA')                                      # magic
            f.write(struct.pack('<B', version))                    # version
            f.write(struct.pack('<B', algo_byte))                  # algorithm_type
            f.write(struct.pack('<H', len(self._array_feature_names)))  # n_features
            f.write(struct.pack('<H', self.num_classes))           # n_classes
            f.write(struct.pack('<H', n_trees))                    # n_trees
            f.write(struct.pack('<i', default_cls))                # default_class
            if version >= 2:
                f.write(struct.pack('<B', flags))                  # flags
            if has_labels:
                for lab in rf_labels:
                    f.write(struct.pack('<i', lab))                # class labels
            if flags & 8:
                f.write(self._pack_label_table(table))             # label table

            # --- GBDT metadata ---
            if self.algorithm_type == 'Gradient Boosting Decision Trees':
                classes = self._gbdt_classes or []
                f.write(struct.pack('<B', 1 if self._gbdt_is_binary else 0))
                f.write(struct.pack('<H', len(classes)))
                for cl in classes:
                    f.write(struct.pack('<i', int(cl)))
                init_scores = self._gbdt_init_scores or {}
                for cl in classes:
                    f.write(struct.pack('<d', init_scores.get(cl, 0.0)))

            # --- Feature names ---
            for name in self._array_feature_names:
                encoded = name.encode('utf-8')
                f.write(struct.pack('<H', len(encoded)))
                f.write(encoded)

            # --- Trees ---
            tree_class_groups = self._tree_class_groups
            for i, tree in enumerate(self._tree_arrays):
                if 'init_score' in tree:
                    # GBDT init entry
                    f.write(struct.pack('<i', 0))       # n_nodes = 0 (sentinel)
                    f.write(struct.pack('<B', 1))        # is_init = True
                    f.write(struct.pack('<d', tree['init_score']))
                    assert tree_class_groups is not None
                    f.write(struct.pack('<i', int(tree_class_groups[i])))
                    continue

                n_nodes = len(tree['feature_idx'])
                f.write(struct.pack('<i', n_nodes))

                if self.algorithm_type == 'Gradient Boosting Decision Trees':
                    f.write(struct.pack('<B', 0))        # is_init = False
                    assert tree_class_groups is not None
                    f.write(struct.pack('<i', int(tree_class_groups[i])))

                f.write(tree['feature_idx'].tobytes())
                f.write(tree['threshold'].tobytes())
                f.write(tree['children_left'].tobytes())
                f.write(tree['children_right'].tobytes())
                if flags & 2:
                    f.write(np.ascontiguousarray(
                        tree['missing_left'], dtype=np.uint8).tobytes())
                f.write(tree['value'].tobytes())
                f.write(struct.pack('<i', tree.get('max_depth', 50)))

        print(f'[EXPORT] Binary model saved to {filepath} '
              f'({os.path.getsize(filepath)} bytes)')

    # Method to execute load_binary
    @classmethod
    def load_binary(cls, filepath: str) -> 'RuleClassifier':
        """
        Load a RuleClassifier from a binary file created by export_to_binary().

        The returned object supports predict_batch() but NOT rule-level
        operations (no Rule objects are reconstructed).

        Args:
            filepath (str): Path to the .bin file.

        Returns:
            RuleClassifier: A classifier ready for predict_batch().

        Raises:
            ValueError: If the file is not a valid model: unknown magic,
                version or algorithm, truncated or trailing data, or tree
                arrays the traversal could not walk safely (see
                `_validate_tree_arrays`). The traversal does no bounds
                checks, so the file is never trusted.
        """
        import struct

        algo_names = {0: 'Decision Tree', 1: 'Random Forest',
                      2: 'Gradient Boosting Decision Trees'}

        with open(filepath, 'rb') as f:
            data = f.read()
        pos = 0

        # Read exactly `n` bytes, raising on a short read (not struct.error)
        def _read(n):
            nonlocal pos
            if n < 0 or pos + n > len(data):
                raise ValueError(
                    f'Invalid binary model file: truncated at byte {pos} '
                    f'(needed {n} more bytes, {len(data) - pos} left).'
                )
            chunk = data[pos:pos + n]
            pos += n
            return chunk

        # Unpack one struct format from the next bytes
        def _unpack(fmt):
            return struct.unpack(fmt, _read(struct.calcsize(fmt)))[0]

        magic = _read(4) if len(data) >= 4 else data
        if magic != b'PYRA':
            raise ValueError(f'Invalid binary model file (magic={magic!r})')
        version = _unpack('<B')
        if version not in (1, 2, 3, 4):
            raise ValueError(f'Unsupported binary model version: {version}')
        algo_byte = _unpack('<B')
        if algo_byte not in algo_names:
            raise ValueError(f'Invalid binary model file: unknown algorithm '
                             f'type {algo_byte}.')
        n_features = _unpack('<H')
        n_classes = _unpack('<H')
        n_trees = _unpack('<H')
        default_cls = _unpack('<i')
        flags = _unpack('<B') if version >= 2 else 0
        known_flags = 0x0F if version >= 4 else 0x01
        if flags & ~known_flags:
            raise ValueError(f'Invalid binary model file: unknown flags {flags:#04x}.')
        class_labels = list(range(n_classes))
        if version == 3 or flags & 4:
            class_labels = [_unpack('<i') for _ in range(n_classes)]
        label_table = None
        if flags & 8:
            n_labels = _unpack('<H')
            if n_labels != n_classes:
                raise ValueError('Invalid binary model file: the label table '
                                 f'has {n_labels} labels for {n_classes} classes.')
            label_table = []
            for _ in range(n_labels):
                kind = _unpack('<B')
                if kind == 0:
                    try:
                        label_table.append(_read(_unpack('<H')).decode('utf-8'))
                    except UnicodeDecodeError as e:
                        raise ValueError(f'Invalid binary model file: label '
                                         f'is not utf-8 ({e}).') from None
                elif kind == 1:
                    label_table.append(_unpack('<d'))
                elif kind == 2:
                    label_table.append(_unpack('<q'))
                elif kind == 3:
                    label_table.append(bool(_unpack('<B')))
                else:
                    raise ValueError(f'Invalid binary model file: unknown '
                                     f'label type {kind}.')
            if len(set(label_table)) != len(label_table):
                raise ValueError('Invalid binary model file: repeated class labels.')
        has_missing = bool(flags & 2)

        algorithm_type = algo_names[algo_byte]
        if algorithm_type == 'Random Forest' and n_classes < 1:
            raise ValueError('Invalid binary model file: a Random Forest '
                             'needs at least one class.')

        # GBDT metadata
        gbdt_is_binary = False
        gbdt_classes = []
        gbdt_init_scores = {}
        if algorithm_type == 'Gradient Boosting Decision Trees':
            gbdt_is_binary = bool(_unpack('<B'))
            n_gbdt_classes = _unpack('<H')
            for _ in range(n_gbdt_classes):
                gbdt_classes.append(_unpack('<i'))
            for cl in gbdt_classes:
                gbdt_init_scores[cl] = _unpack('<d')

        # Feature names
        feature_names = []
        for _ in range(n_features):
            name_len = _unpack('<H')
            try:
                feature_names.append(_read(name_len).decode('utf-8'))
            except UnicodeDecodeError as e:
                raise ValueError(f'Invalid binary model file: feature name '
                                 f'is not utf-8 ({e}).') from None

        # Bytes per node: feature_idx, threshold, both children, value.
        value_size = {'Decision Tree': 4, 'Random Forest': n_classes * 8,
                      'Gradient Boosting Decision Trees': 8}[algorithm_type]
        node_size = 4 + 8 + 4 + 4 + value_size + (1 if has_missing else 0)

        # Trees
        tree_arrays = []
        tree_class_groups = []
        tree_is_init = []

        for t in range(n_trees):
            n_nodes = _unpack('<i')

            if algorithm_type == 'Gradient Boosting Decision Trees':
                is_init = bool(_unpack('<B'))
                if is_init:
                    init_score = _unpack('<d')
                    cg = _unpack('<i')
                    tree_arrays.append({'init_score': init_score})
                    tree_class_groups.append(cg)
                    tree_is_init.append(True)
                    continue
                else:
                    cg = _unpack('<i')
                    tree_class_groups.append(cg)
                    tree_is_init.append(False)

            # Bounded by the bytes left, so a forged count cannot make us
            # allocate (or traverse) more than the file holds.
            if n_nodes < 1 or n_nodes * node_size > len(data) - pos:
                raise ValueError(
                    f'Invalid binary model file: tree {t} has n_nodes='
                    f'{n_nodes} ({len(data) - pos} bytes left).'
                )

            feature_idx = np.frombuffer(_read(n_nodes * 4), dtype='<i4').astype(np.int32)
            threshold_arr = np.frombuffer(_read(n_nodes * 8), dtype='<f8').astype(np.float64)
            children_left = np.frombuffer(_read(n_nodes * 4), dtype='<i4').astype(np.int32)
            children_right = np.frombuffer(_read(n_nodes * 4), dtype='<i4').astype(np.int32)
            missing_left = None
            if has_missing:
                missing_left = np.frombuffer(_read(n_nodes), dtype=np.uint8).copy()

            if algorithm_type == 'Decision Tree':
                value_arr = np.frombuffer(_read(n_nodes * 4), dtype='<i4').astype(np.int32)
            elif algorithm_type == 'Random Forest':
                value_arr = np.frombuffer(
                    _read(n_nodes * n_classes * 8), dtype='<f8'
                ).astype(np.float64).reshape(n_nodes, n_classes)
            else:  # GBDT
                value_arr = np.frombuffer(_read(n_nodes * 8), dtype='<f8').astype(np.float64)

            tree = {
                'feature_idx': feature_idx,
                'threshold': threshold_arr,
                'children_left': children_left,
                'children_right': children_right,
                'value': value_arr,
                'max_depth': _unpack('<i'),
            }
            if missing_left is not None:
                tree['missing_left'] = missing_left
            cls._validate_tree_arrays(tree, n_features, t)
            tree_arrays.append(tree)

        if pos != len(data):
            raise ValueError(
                f'Invalid binary model file: {len(data) - pos} unexpected '
                'trailing bytes.'
            )

        # Build a minimal RuleClassifier instance
        # We use __new__ to bypass __init__ (no Rule objects needed)
        obj = cls.__new__(cls)
        obj.algorithm_type = algorithm_type
        obj.num_classes = n_classes
        obj.class_labels = class_labels
        obj.default_class = default_cls
        obj.initial_rules = []
        obj.final_rules = []
        obj.duplicated_rules = []
        obj.specific_rules = []
        obj.native_fn = None
        obj._gbdt_init_scores = gbdt_init_scores if gbdt_init_scores else None
        obj._gbdt_is_binary = gbdt_is_binary
        obj._gbdt_classes = gbdt_classes if gbdt_classes else None
        obj._array_feature_names = feature_names
        obj._tree_arrays = tree_arrays
        obj._tree_class_groups = tree_class_groups if tree_class_groups else None
        obj._tree_is_init = tree_is_init if tree_is_init else None
        obj._arrays_compiled = True
        # No rules: predict_batch always answers with these arrays.
        obj._arrays_source = None
        obj._arrays_source_len = None
        obj._array_states = []
        obj._native_source = None
        obj.fallback_activations = 0
        obj.input_dtype = 'float32' if flags & 1 else None
        obj.label_table = label_table
        obj.sklearn_estimator = None

        return obj

    # Method to check that tree arrays can be traversed safely
    @staticmethod
    def _validate_tree_arrays(tree: dict, n_features: int, tree_no: int = 0) -> None:
        """Check one tree's arrays against what the traversal assumes.

        The C traversal reads ``X[i, feature_idx[node]]`` and follows the
        children without bounds checks, so arrays from an untrusted source
        must hold:

        - a leaf is a self-loop (both children are the node itself) with
          ``feature_idx == 0`` and ``threshold == -inf``, as
          `_build_single_tree_arrays` writes it;
        - any other node has ``0 <= feature_idx < n_features`` and both
          children in ``[0, n_nodes)``, neither being the node itself;
        - ``0 <= max_depth <= n_nodes``;
        - the split nodes form a tree rooted at node 0: no node is reached
          twice (no cycle, no child shared by two splits), every node is
          reachable from the root, and no leaf is deeper than ``max_depth``
          (the traversal runs exactly ``max_depth`` steps, so a deeper leaf
          would leave samples on a split node).

        Args:
            tree (dict): 'feature_idx', 'threshold', 'children_left',
                'children_right', 'max_depth'.
            n_features (int): Number of input columns the model reads.
            tree_no (int): Position of the tree, for the error message.

        Raises:
            ValueError: Naming the first violated invariant.
        """
        feat = tree['feature_idx']
        thr = tree['threshold']
        left = tree['children_left']
        right = tree['children_right']
        n_nodes = len(feat)

        # Raise with the tree and the first offending node
        def _fail(what, mask):
            node = int(np.flatnonzero(mask)[0])
            raise ValueError(
                f'Invalid tree arrays (tree {tree_no}, node {node}): {what}.'
            )

        if n_nodes < 1 or not (len(thr) == len(left) == len(right) == n_nodes):
            raise ValueError(f'Invalid tree arrays (tree {tree_no}): '
                             'arrays are empty or of different lengths.')
        missing = tree.get('missing_left')
        if missing is not None:
            if len(missing) != n_nodes:
                raise ValueError(f'Invalid tree arrays (tree {tree_no}): '
                                 'missing_left has the wrong length.')
            bad = np.asarray(missing) > 1
            if bad.any():
                _fail('missing_left must be 0 or 1', bad)
        ids = np.arange(n_nodes)
        is_leaf = (left == ids) & (right == ids)
        bad = is_leaf & ((feat != 0) | ~np.isneginf(thr))
        if bad.any():
            _fail('a leaf must have feature_idx 0 and threshold -inf', bad)
        inner = ~is_leaf
        bad = inner & ((feat < 0) | (feat >= n_features))
        if bad.any():
            _fail(f'feature_idx out of range [0, {n_features})', bad)
        bad = inner & ((left < 0) | (left >= n_nodes)
                       | (right < 0) | (right >= n_nodes))
        if bad.any():
            _fail(f'child index out of range [0, {n_nodes})', bad)
        bad = inner & ((left == ids) | (right == ids))
        if bad.any():
            _fail('a split node cannot be its own child', bad)
        depth = int(tree['max_depth'])
        if not 0 <= depth <= n_nodes:
            raise ValueError(f'Invalid tree arrays (tree {tree_no}): '
                             f'max_depth={depth} not in [0, {n_nodes}].')

        # The split nodes must form a tree rooted at node 0: walk it level by
        # level, so a child pointing back to an ancestor (a cycle) or to a
        # node another split already reached shows up as a node seen twice.
        # Such arrays stay in bounds but route samples wrongly.
        seen = np.zeros(n_nodes, dtype=bool)
        seen[0] = True
        frontier = np.zeros(1, dtype=np.intp)
        level = 0
        while True:
            frontier = frontier[inner[frontier]]
            if frontier.size == 0:
                break
            level += 1
            if level > depth:
                # The traversal runs exactly max_depth steps, so a sample
                # deeper than that would stop on a split node and read its
                # (meaningless) value.
                raise ValueError(
                    f'Invalid tree arrays (tree {tree_no}): max_depth={depth} '
                    f'is smaller than the depth of the tree (> {depth}).')
            kids = np.concatenate((left[frontier], right[frontier]))
            if seen[kids].any() or np.unique(kids).size != kids.size:
                counts = np.bincount(kids, minlength=n_nodes)
                _fail('node reached twice (a cycle or a child shared by two '
                      'splits): the split nodes do not form a tree',
                      (counts > 1) | (seen & (counts > 0)))
            seen[kids] = True
            frontier = kids
        # `_build_single_tree_arrays` numbers only the nodes it creates while
        # walking from the root, so every node is reachable.
        if not seen.all():
            _fail('node not reachable from the root', ~seen)
