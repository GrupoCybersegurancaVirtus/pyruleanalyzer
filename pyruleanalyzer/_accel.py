"""
Tree traversal acceleration layer.

This module provides a unified interface to the tree traversal function,
automatically selecting the fastest available backend:

1. C extension (pyruleanalyzer._tree_traversal) — compiled native code
2. NumPy fallback — pure Python with numpy vectorized operations

Usage::

    from pyruleanalyzer._accel import traverse_tree_batch, traverse_tree_batch_multi
    from pyruleanalyzer._accel import HAS_C_EXTENSION
"""

import numpy as np

# --- Try to import the C extension ---
HAS_C_EXTENSION = False
_c_traverse_tree_batch = None
_c_traverse_tree_batch_multi = None

try:
    from pyruleanalyzer._tree_traversal import (
        traverse_tree_batch as _c_traverse_tree_batch,
        traverse_tree_batch_multi as _c_traverse_tree_batch_multi,
    )
    HAS_C_EXTENSION = True
except ImportError:
    pass


# --- NumPy fallback implementations ---

# Function to numpy traverse tree batch.
def _numpy_traverse_tree_batch(
    X: np.ndarray,
    feature_idx: np.ndarray,
    threshold: np.ndarray,
    children_left: np.ndarray,
    children_right: np.ndarray,
    max_depth: int = 50,
    missing_left=None,
) -> np.ndarray:
    """
    Traverse a single tree for all samples using numpy indexing (fallback).

    Args:
        X: Input data, shape (n_samples, n_features), float64.
        feature_idx: Per-node feature index, shape (n_nodes,), int32.
        threshold: Per-node threshold, shape (n_nodes,), float64.
        children_left: Per-node left child, shape (n_nodes,), int32.
        children_right: Per-node right child, shape (n_nodes,), int32.
        max_depth: Maximum depth of the tree.
        missing_left: Optional per-node flag, shape (n_nodes,): a NaN goes
            left where it is nonzero. Without it a NaN goes right
            (``NaN <= t`` is False), as in the C extension.

    Returns:
        Array of leaf node indices, shape (n_samples,), int32.
    """
    n_samples = X.shape[0]
    node_ids = np.zeros(n_samples, dtype=np.int32)
    sample_idx = np.arange(n_samples)
    nan_left = None if missing_left is None else np.asarray(missing_left) != 0

    for _ in range(max_depth):
        feat = feature_idx[node_ids]
        values = X[sample_idx, feat]
        go_left = values <= threshold[node_ids]
        if nan_left is not None:
            go_left |= np.isnan(values) & nan_left[node_ids]
        node_ids = np.where(go_left, children_left[node_ids],
                            children_right[node_ids])

    return node_ids


# Function to numpy traverse tree batch multi.
def _numpy_traverse_tree_batch_multi(
    X: np.ndarray,
    trees: list,
) -> np.ndarray:
    """
    Traverse multiple trees for all samples using numpy (fallback).

    Args:
        X: Input data, shape (n_samples, n_features), float64.
        trees: List of tuples (feature_idx, threshold, children_left,
               children_right, max_depth[, missing_left]).

    Returns:
        Array of leaf node indices, shape (n_trees, n_samples), int32.
    """
    n_samples = X.shape[0]
    n_trees = len(trees)
    result = np.empty((n_trees, n_samples), dtype=np.int32)

    for t, tree in enumerate(trees):
        feat_idx, thresh, left, right, depth = tree[:5]
        missing = tree[5] if len(tree) > 5 else None
        result[t] = _numpy_traverse_tree_batch(
            X, feat_idx, thresh, left, right, depth, missing
        )

    return result


# Function to check whether X holds a missing value.
def _has_nan(X) -> bool:
    """Whether the (float) input contains NaN.

    Args:
        X: Input data.

    Returns:
        bool: True if any value is NaN.
    """
    X = np.asarray(X)
    return bool(X.size) and X.dtype.kind == 'f' and bool(np.isnan(X).any())


# Function to check a missing-value routing array.
def _check_missing_left(missing_left, n_nodes: int) -> np.ndarray:
    """Validate a per-node missing-value routing array.

    Args:
        missing_left: Per-node flags.
        n_nodes: Number of nodes of the tree.

    Returns:
        The flags as a 1-D uint8 array.

    Raises:
        ValueError: If the length differs from the number of nodes.
    """
    arr = np.ascontiguousarray(missing_left, dtype=np.uint8)
    if arr.ndim != 1 or arr.shape[0] != n_nodes:
        raise ValueError('missing_left must be a 1-D array of length n_nodes')
    return arr


# --- Bounds checks for the C extension ---

# Function to check one tree's arrays before handing them to C.
def _check_c_inputs(
    X: np.ndarray,
    feature_idx: np.ndarray,
    threshold: np.ndarray,
    children_left: np.ndarray,
    children_right: np.ndarray,
    max_depth: int,
) -> np.ndarray:
    """
    Validate the inputs of one tree traversal before the C extension runs.

    The C code reads ``X[i, feature_idx[node]]`` and follows the children
    without any bounds check, so an out-of-range index would read outside
    the buffers (crash or memory disclosure). These checks are O(n_nodes),
    negligible next to the traversal itself.

    Args:
        X: Input data, 2-D float64 C-contiguous (already converted).
        feature_idx: Per-node feature index, 1-D int32 C-contiguous.
        threshold: Per-node threshold, 1-D float64 C-contiguous.
        children_left: Per-node left child, 1-D int32 C-contiguous.
        children_right: Per-node right child, 1-D int32 C-contiguous.
        max_depth: Number of traversal steps.

    Returns:
        X, or a single zero column when X has no columns and the tree is a
        lone leaf (the leaf reads column 0, whose value it ignores).

    Raises:
        ValueError: If an array has the wrong shape/dtype, the lengths
            differ, or a feature/child index is out of range.
    """
    if X.ndim != 2:
        raise ValueError(f'X must be 2-D, got {X.ndim}-D')
    for name, arr, dtype in (('feature_idx', feature_idx, np.int32),
                             ('threshold', threshold, np.float64),
                             ('children_left', children_left, np.int32),
                             ('children_right', children_right, np.int32)):
        if (arr.ndim != 1 or arr.dtype != dtype
                or not arr.flags['C_CONTIGUOUS']):
            raise ValueError(
                f'{name} must be a 1-D C-contiguous {np.dtype(dtype).name} array'
            )
    n_nodes = feature_idx.shape[0]
    if not (threshold.shape[0] == children_left.shape[0]
            == children_right.shape[0] == n_nodes):
        raise ValueError('Tree arrays must all have the same length (n_nodes)')

    # Nothing is read when there is no sample or no step.
    if X.shape[0] == 0 or int(max_depth) <= 0:
        return X
    if n_nodes == 0:
        raise ValueError('Tree arrays are empty: there is no root node')
    if (children_left.min() < 0 or children_left.max() >= n_nodes
            or children_right.min() < 0 or children_right.max() >= n_nodes):
        raise ValueError(f'Child index out of range [0, {n_nodes})')

    if X.shape[1] == 0 and n_nodes == 1:
        # A lone leaf: no column is tested, but the traversal still reads
        # one (checked below like any other).
        X = np.zeros((X.shape[0], 1), dtype=np.float64)
    n_cols = X.shape[1]
    if feature_idx.min() < 0:
        raise ValueError('feature_idx contains a negative index')
    needed = int(feature_idx.max()) + 1
    if n_cols < needed:
        raise ValueError(
            f'X has {n_cols} column(s) but the tree reads column '
            f'{needed - 1}: it needs at least {needed}'
        )
    return X


# --- Public API: auto-select best backend ---

# Function to traverse tree batch.
def traverse_tree_batch(
    X: np.ndarray,
    feature_idx: np.ndarray,
    threshold: np.ndarray,
    children_left: np.ndarray,
    children_right: np.ndarray,
    max_depth: int = 50,
    missing_left=None,
) -> np.ndarray:
    """
    Traverse a single tree for all samples simultaneously.

    Automatically uses the C extension if available, otherwise falls back
    to the numpy implementation. The C traversal only tests ``x <= t``
    (NaN goes right), so when `missing_left` is given and X holds a NaN the
    numpy traversal, which routes NaN per split, is used instead.

    Args:
        X: Input data, shape (n_samples, n_features), float64, C-contiguous.
        feature_idx: Per-node feature index, shape (n_nodes,), int32.
        threshold: Per-node threshold, shape (n_nodes,), float64.
        children_left: Per-node left child, shape (n_nodes,), int32.
        children_right: Per-node right child, shape (n_nodes,), int32.
        max_depth: Maximum depth of the tree.
        missing_left: Optional per-node flag, shape (n_nodes,): a NaN goes
            left where it is nonzero (scikit-learn's ``missing_go_to_left``).

    Returns:
        Array of leaf node indices, shape (n_samples,), int32.
    """
    if missing_left is not None and _has_nan(X):
        X = np.ascontiguousarray(X, dtype=np.float64)
        feature_idx = np.ascontiguousarray(feature_idx, dtype=np.int32)
        threshold = np.ascontiguousarray(threshold, dtype=np.float64)
        children_left = np.ascontiguousarray(children_left, dtype=np.int32)
        children_right = np.ascontiguousarray(children_right, dtype=np.int32)
        # Same bounds checks as before the C call: the arrays may come from
        # a binary file.
        X = _check_c_inputs(X, feature_idx, threshold, children_left,
                            children_right, int(max_depth))
        missing_left = _check_missing_left(missing_left, feature_idx.shape[0])
        return _numpy_traverse_tree_batch(
            X, feature_idx, threshold, children_left, children_right,
            int(max_depth), missing_left,
        )
    if HAS_C_EXTENSION:
        # Ensure arrays are C-contiguous with correct dtypes
        X = np.ascontiguousarray(X, dtype=np.float64)
        feature_idx = np.ascontiguousarray(feature_idx, dtype=np.int32)
        threshold = np.ascontiguousarray(threshold, dtype=np.float64)
        children_left = np.ascontiguousarray(children_left, dtype=np.int32)
        children_right = np.ascontiguousarray(children_right, dtype=np.int32)
        max_depth = int(max_depth)
        # The C code does no bounds checks: never hand it an unsafe index.
        X = _check_c_inputs(X, feature_idx, threshold, children_left,
                            children_right, max_depth)
        return _c_traverse_tree_batch(  # type: ignore[misc]
            X, feature_idx, threshold, children_left, children_right, max_depth
        )
    else:
        return _numpy_traverse_tree_batch(
            X, feature_idx, threshold, children_left, children_right, max_depth
        )


# Function to traverse tree batch multi.
def traverse_tree_batch_multi(
    X: np.ndarray,
    trees: list,
) -> np.ndarray:
    """
    Traverse multiple trees for all samples in a single call.

    Automatically uses the C extension if available, otherwise falls back
    to the numpy implementation. A tree tuple may carry a sixth element,
    its ``missing_left`` flags; when one does and X holds a NaN, every tree
    is traversed by numpy, which routes NaN per split (see
    `traverse_tree_batch`).

    Args:
        X: Input data, shape (n_samples, n_features), float64, C-contiguous.
        trees: List of tuples (feature_idx, threshold, children_left,
               children_right, max_depth[, missing_left]).

    Returns:
        Array of leaf node indices, shape (n_trees, n_samples), int32.
    """
    if (any(len(t) > 5 and t[5] is not None for t in trees)
            and _has_nan(X)):
        X = np.ascontiguousarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        result = np.empty((len(trees), n_samples), dtype=np.int32)
        for i, tree in enumerate(trees):
            missing = tree[5] if len(tree) > 5 else None
            result[i] = traverse_tree_batch(
                X, *tree[:5],
                missing_left=missing if missing is not None
                else np.zeros(len(tree[0]), dtype=np.uint8))
        return result
    trees = [t[:5] for t in trees]
    if HAS_C_EXTENSION:
        X = np.ascontiguousarray(X, dtype=np.float64)
        # Ensure each tree's arrays have correct dtype/contiguity, and are
        # in bounds: the multi-tree C entry point checks nothing at all.
        prepared = []
        X_c = X
        for feat_idx, thresh, left, right, depth in trees:
            tree = (
                np.ascontiguousarray(feat_idx, dtype=np.int32),
                np.ascontiguousarray(thresh, dtype=np.float64),
                np.ascontiguousarray(left, dtype=np.int32),
                np.ascontiguousarray(right, dtype=np.int32),
                int(depth),
            )
            X_t = _check_c_inputs(X, *tree)
            if X_t is not X:
                # A lone-leaf tree on a column-less X: one zero column is
                # safe for every tree (they all are lone leaves, or fail).
                X_c = X_t
            prepared.append(tree)
        if X_c is not X:
            for tree in prepared:
                _check_c_inputs(X_c, *tree)
        return _c_traverse_tree_batch_multi(X_c, prepared)  # type: ignore[misc]
    else:
        return _numpy_traverse_tree_batch_multi(X, trees)
