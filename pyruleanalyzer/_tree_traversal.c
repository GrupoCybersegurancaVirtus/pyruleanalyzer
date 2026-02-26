/*
 * _tree_traversal.c
 *
 * C extension for pyruleanalyzer: vectorized tree traversal.
 *
 * Replaces the Python/NumPy loop in RuleClassifier._traverse_tree_batch
 * with a compiled C loop, eliminating per-iteration Python overhead.
 *
 * The traversal uses the same self-loop leaf convention:
 *   - Leaf nodes: children_left[leaf] == leaf, children_right[leaf] == leaf
 *   - feature_idx[leaf] = 0 (valid column), threshold[leaf] = -inf
 *   - So (val <= -inf) is always False, picking children_right = leaf (no-op)
 *
 * Build:
 *   Compiled automatically via setup.py / pyproject.toml as a Python C extension.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdint.h>


/*
 * traverse_tree_batch(X, feature_idx, threshold, children_left, children_right, max_depth)
 *
 * Traverse a single decision tree for ALL samples simultaneously in C.
 *
 * Parameters
 * ----------
 * X : ndarray, shape (n_samples, n_features), dtype float64, C-contiguous
 * feature_idx : ndarray, shape (n_nodes,), dtype int32
 * threshold : ndarray, shape (n_nodes,), dtype float64
 * children_left : ndarray, shape (n_nodes,), dtype int32
 * children_right : ndarray, shape (n_nodes,), dtype int32
 * max_depth : int
 *
 * Returns
 * -------
 * ndarray, shape (n_samples,), dtype int32  -- leaf node indices
 */
static PyObject *
traverse_tree_batch(PyObject *self, PyObject *args)
{
    PyArrayObject *X_arr, *feat_arr, *thresh_arr, *left_arr, *right_arr;
    int max_depth;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!i",
                          &PyArray_Type, &X_arr,
                          &PyArray_Type, &feat_arr,
                          &PyArray_Type, &thresh_arr,
                          &PyArray_Type, &left_arr,
                          &PyArray_Type, &right_arr,
                          &max_depth))
        return NULL;

    /* Validate X: 2-D, float64, C-contiguous */
    if (PyArray_NDIM(X_arr) != 2 ||
        PyArray_TYPE(X_arr) != NPY_FLOAT64 ||
        !PyArray_IS_C_CONTIGUOUS(X_arr)) {
        PyErr_SetString(PyExc_ValueError,
                        "X must be a 2-D C-contiguous float64 array");
        return NULL;
    }

    /* Validate tree arrays: 1-D, int32 / float64, C-contiguous */
    if (PyArray_NDIM(feat_arr) != 1 || PyArray_TYPE(feat_arr) != NPY_INT32 ||
        !PyArray_IS_C_CONTIGUOUS(feat_arr)) {
        PyErr_SetString(PyExc_ValueError,
                        "feature_idx must be a 1-D C-contiguous int32 array");
        return NULL;
    }
    if (PyArray_NDIM(thresh_arr) != 1 || PyArray_TYPE(thresh_arr) != NPY_FLOAT64 ||
        !PyArray_IS_C_CONTIGUOUS(thresh_arr)) {
        PyErr_SetString(PyExc_ValueError,
                        "threshold must be a 1-D C-contiguous float64 array");
        return NULL;
    }
    if (PyArray_NDIM(left_arr) != 1 || PyArray_TYPE(left_arr) != NPY_INT32 ||
        !PyArray_IS_C_CONTIGUOUS(left_arr)) {
        PyErr_SetString(PyExc_ValueError,
                        "children_left must be a 1-D C-contiguous int32 array");
        return NULL;
    }
    if (PyArray_NDIM(right_arr) != 1 || PyArray_TYPE(right_arr) != NPY_INT32 ||
        !PyArray_IS_C_CONTIGUOUS(right_arr)) {
        PyErr_SetString(PyExc_ValueError,
                        "children_right must be a 1-D C-contiguous int32 array");
        return NULL;
    }

    npy_intp n_samples = PyArray_DIM(X_arr, 0);
    npy_intp n_features = PyArray_DIM(X_arr, 1);
    npy_intp n_nodes = PyArray_DIM(feat_arr, 0);

    /* Sanity checks */
    if (PyArray_DIM(thresh_arr, 0) != n_nodes ||
        PyArray_DIM(left_arr, 0) != n_nodes ||
        PyArray_DIM(right_arr, 0) != n_nodes) {
        PyErr_SetString(PyExc_ValueError,
                        "Tree arrays must all have the same length (n_nodes)");
        return NULL;
    }

    /* Get raw data pointers */
    const double *X = (const double *)PyArray_DATA(X_arr);
    const int32_t *feature_idx = (const int32_t *)PyArray_DATA(feat_arr);
    const double *threshold = (const double *)PyArray_DATA(thresh_arr);
    const int32_t *children_left = (const int32_t *)PyArray_DATA(left_arr);
    const int32_t *children_right = (const int32_t *)PyArray_DATA(right_arr);

    /* Create output array: node_ids, shape (n_samples,), dtype int32 */
    npy_intp out_dims[1] = {n_samples};
    PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNew(1, out_dims, NPY_INT32);
    if (result == NULL)
        return NULL;

    int32_t *node_ids = (int32_t *)PyArray_DATA(result);

    /* Initialize all samples at root (node 0) */
    npy_intp i;
    for (i = 0; i < n_samples; i++) {
        node_ids[i] = 0;
    }

    /*
     * Main traversal loop.
     *
     * For each depth level, iterate over ALL samples:
     *   1. Read the split feature index for the current node
     *   2. Read the corresponding feature value from X
     *   3. Compare against threshold
     *   4. Move to left or right child
     *
     * Leaf nodes use self-loops, so finished samples stay in place.
     * This is the same logic as the numpy version but without any
     * Python/numpy overhead per iteration.
     */
    int d;
    for (d = 0; d < max_depth; d++) {
        for (i = 0; i < n_samples; i++) {
            int32_t node = node_ids[i];
            int32_t feat = feature_idx[node];
            /* Read feature value: X[i, feat] */
            double val = X[i * n_features + feat];
            if (val <= threshold[node]) {
                node_ids[i] = children_left[node];
            } else {
                node_ids[i] = children_right[node];
            }
        }
    }

    return (PyObject *)result;
}


/*
 * traverse_tree_batch_multi(X, trees, n_trees)
 *
 * Traverse MULTIPLE trees for all samples in a single C call.
 * This avoids the Python loop over trees in predict_batch for RF/GBDT.
 *
 * Parameters
 * ----------
 * X : ndarray, shape (n_samples, n_features), dtype float64, C-contiguous
 * trees : list of tuples, each (feature_idx, threshold, children_left, children_right, max_depth)
 *         where each array is as in traverse_tree_batch.
 *
 * Returns
 * -------
 * ndarray, shape (n_trees, n_samples), dtype int32  -- leaf node indices per tree
 */
static PyObject *
traverse_tree_batch_multi(PyObject *self, PyObject *args)
{
    PyArrayObject *X_arr;
    PyObject *tree_list;

    if (!PyArg_ParseTuple(args, "O!O!",
                          &PyArray_Type, &X_arr,
                          &PyList_Type, &tree_list))
        return NULL;

    /* Validate X */
    if (PyArray_NDIM(X_arr) != 2 ||
        PyArray_TYPE(X_arr) != NPY_FLOAT64 ||
        !PyArray_IS_C_CONTIGUOUS(X_arr)) {
        PyErr_SetString(PyExc_ValueError,
                        "X must be a 2-D C-contiguous float64 array");
        return NULL;
    }

    npy_intp n_samples = PyArray_DIM(X_arr, 0);
    npy_intp n_features = PyArray_DIM(X_arr, 1);
    const double *X = (const double *)PyArray_DATA(X_arr);

    Py_ssize_t n_trees = PyList_GET_SIZE(tree_list);

    /* Create output: shape (n_trees, n_samples), dtype int32 */
    npy_intp out_dims[2] = {(npy_intp)n_trees, n_samples};
    PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNew(2, out_dims, NPY_INT32);
    if (result == NULL)
        return NULL;

    int32_t *all_leaf_ids = (int32_t *)PyArray_DATA(result);

    Py_ssize_t t;
    for (t = 0; t < n_trees; t++) {
        PyObject *tree_tuple = PyList_GET_ITEM(tree_list, t);

        if (!PyTuple_Check(tree_tuple) || PyTuple_GET_SIZE(tree_tuple) != 5) {
            PyErr_SetString(PyExc_ValueError,
                            "Each tree must be a tuple of (feature_idx, threshold, "
                            "children_left, children_right, max_depth)");
            Py_DECREF(result);
            return NULL;
        }

        PyArrayObject *feat_arr = (PyArrayObject *)PyTuple_GET_ITEM(tree_tuple, 0);
        PyArrayObject *thresh_arr = (PyArrayObject *)PyTuple_GET_ITEM(tree_tuple, 1);
        PyArrayObject *left_arr = (PyArrayObject *)PyTuple_GET_ITEM(tree_tuple, 2);
        PyArrayObject *right_arr = (PyArrayObject *)PyTuple_GET_ITEM(tree_tuple, 3);
        PyObject *depth_obj = PyTuple_GET_ITEM(tree_tuple, 4);

        int max_depth = (int)PyLong_AsLong(depth_obj);
        if (max_depth == -1 && PyErr_Occurred()) {
            Py_DECREF(result);
            return NULL;
        }

        const int32_t *feature_idx = (const int32_t *)PyArray_DATA(feat_arr);
        const double *threshold = (const double *)PyArray_DATA(thresh_arr);
        const int32_t *children_left = (const int32_t *)PyArray_DATA(left_arr);
        const int32_t *children_right = (const int32_t *)PyArray_DATA(right_arr);

        /* Pointer to this tree's row in the output */
        int32_t *node_ids = all_leaf_ids + t * n_samples;

        /* Initialize at root */
        npy_intp i;
        for (i = 0; i < n_samples; i++) {
            node_ids[i] = 0;
        }

        /* Traverse */
        int d;
        for (d = 0; d < max_depth; d++) {
            for (i = 0; i < n_samples; i++) {
                int32_t node = node_ids[i];
                int32_t feat = feature_idx[node];
                double val = X[i * n_features + feat];
                if (val <= threshold[node]) {
                    node_ids[i] = children_left[node];
                } else {
                    node_ids[i] = children_right[node];
                }
            }
        }
    }

    return (PyObject *)result;
}


/* Module method table */
static PyMethodDef TreeTraversalMethods[] = {
    {"traverse_tree_batch",
     traverse_tree_batch, METH_VARARGS,
     "Traverse a single tree for all samples (vectorized C implementation).\n\n"
     "Parameters:\n"
     "  X : ndarray (n_samples, n_features), float64, C-contiguous\n"
     "  feature_idx : ndarray (n_nodes,), int32\n"
     "  threshold : ndarray (n_nodes,), float64\n"
     "  children_left : ndarray (n_nodes,), int32\n"
     "  children_right : ndarray (n_nodes,), int32\n"
     "  max_depth : int\n\n"
     "Returns:\n"
     "  ndarray (n_samples,), int32 -- leaf node indices"},

    {"traverse_tree_batch_multi",
     traverse_tree_batch_multi, METH_VARARGS,
     "Traverse multiple trees for all samples in a single C call.\n\n"
     "Parameters:\n"
     "  X : ndarray (n_samples, n_features), float64, C-contiguous\n"
     "  trees : list of tuples (feature_idx, threshold, left, right, max_depth)\n\n"
     "Returns:\n"
     "  ndarray (n_trees, n_samples), int32 -- leaf node indices per tree"},

    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef tree_traversal_module = {
    PyModuleDef_HEAD_INIT,
    "_tree_traversal",
    "C-accelerated tree traversal for pyruleanalyzer.\n\n"
    "Provides compiled-C versions of the tree traversal loop,\n"
    "replacing the Python/NumPy loop with native code for ~3-5x speedup.",
    -1,
    TreeTraversalMethods
};

/* Module initialization */
PyMODINIT_FUNC
PyInit__tree_traversal(void)
{
    PyObject *m;
    m = PyModule_Create(&tree_traversal_module);
    if (m == NULL)
        return NULL;

    /* Initialize numpy C API */
    import_array();

    return m;
}
