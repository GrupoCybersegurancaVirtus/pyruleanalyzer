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

def _numpy_traverse_tree_batch(
    X: np.ndarray,
    feature_idx: np.ndarray,
    threshold: np.ndarray,
    children_left: np.ndarray,
    children_right: np.ndarray,
    max_depth: int = 50,
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

    Returns:
        Array of leaf node indices, shape (n_samples,), int32.
    """
    n_samples = X.shape[0]
    node_ids = np.zeros(n_samples, dtype=np.int32)
    sample_idx = np.arange(n_samples)

    for _ in range(max_depth):
        feat = feature_idx[node_ids]
        go_left = X[sample_idx, feat] <= threshold[node_ids]
        node_ids = np.where(go_left, children_left[node_ids],
                            children_right[node_ids])

    return node_ids


def _numpy_traverse_tree_batch_multi(
    X: np.ndarray,
    trees: list,
) -> np.ndarray:
    """
    Traverse multiple trees for all samples using numpy (fallback).

    Args:
        X: Input data, shape (n_samples, n_features), float64.
        trees: List of tuples (feature_idx, threshold, children_left,
               children_right, max_depth).

    Returns:
        Array of leaf node indices, shape (n_trees, n_samples), int32.
    """
    n_samples = X.shape[0]
    n_trees = len(trees)
    result = np.empty((n_trees, n_samples), dtype=np.int32)

    for t, (feat_idx, thresh, left, right, depth) in enumerate(trees):
        result[t] = _numpy_traverse_tree_batch(
            X, feat_idx, thresh, left, right, depth
        )

    return result


# --- Public API: auto-select best backend ---

def traverse_tree_batch(
    X: np.ndarray,
    feature_idx: np.ndarray,
    threshold: np.ndarray,
    children_left: np.ndarray,
    children_right: np.ndarray,
    max_depth: int = 50,
) -> np.ndarray:
    """
    Traverse a single tree for all samples simultaneously.

    Automatically uses the C extension if available, otherwise falls back
    to the numpy implementation.

    Args:
        X: Input data, shape (n_samples, n_features), float64, C-contiguous.
        feature_idx: Per-node feature index, shape (n_nodes,), int32.
        threshold: Per-node threshold, shape (n_nodes,), float64.
        children_left: Per-node left child, shape (n_nodes,), int32.
        children_right: Per-node right child, shape (n_nodes,), int32.
        max_depth: Maximum depth of the tree.

    Returns:
        Array of leaf node indices, shape (n_samples,), int32.
    """
    if HAS_C_EXTENSION:
        # Ensure arrays are C-contiguous with correct dtypes
        X = np.ascontiguousarray(X, dtype=np.float64)
        feature_idx = np.ascontiguousarray(feature_idx, dtype=np.int32)
        threshold = np.ascontiguousarray(threshold, dtype=np.float64)
        children_left = np.ascontiguousarray(children_left, dtype=np.int32)
        children_right = np.ascontiguousarray(children_right, dtype=np.int32)
        return _c_traverse_tree_batch(  # type: ignore[misc]
            X, feature_idx, threshold, children_left, children_right, max_depth
        )
    else:
        return _numpy_traverse_tree_batch(
            X, feature_idx, threshold, children_left, children_right, max_depth
        )


def traverse_tree_batch_multi(
    X: np.ndarray,
    trees: list,
) -> np.ndarray:
    """
    Traverse multiple trees for all samples in a single call.

    Automatically uses the C extension if available, otherwise falls back
    to the numpy implementation.

    Args:
        X: Input data, shape (n_samples, n_features), float64, C-contiguous.
        trees: List of tuples (feature_idx, threshold, children_left,
               children_right, max_depth).

    Returns:
        Array of leaf node indices, shape (n_trees, n_samples), int32.
    """
    if HAS_C_EXTENSION:
        X = np.ascontiguousarray(X, dtype=np.float64)
        # Ensure each tree's arrays have correct dtype/contiguity
        prepared = []
        for feat_idx, thresh, left, right, depth in trees:
            prepared.append((
                np.ascontiguousarray(feat_idx, dtype=np.int32),
                np.ascontiguousarray(thresh, dtype=np.float64),
                np.ascontiguousarray(left, dtype=np.int32),
                np.ascontiguousarray(right, dtype=np.int32),
                int(depth),
            ))
        return _c_traverse_tree_batch_multi(X, prepared)  # type: ignore[misc]
    else:
        return _numpy_traverse_tree_batch_multi(X, trees)
