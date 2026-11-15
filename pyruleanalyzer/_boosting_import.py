"""Read fitted XGBoost, LightGBM and CatBoost classifiers as GBDT rules.

Each library is read into one neutral form -- per class, an ordered list of
binary trees whose splits send ``x <= threshold`` to the left child -- and
then into the same :class:`Rule` objects that ``get_gbdt_rules`` builds for a
scikit-learn ``GradientBoostingClassifier``, so refinement, every prediction
engine, the exports and the HCPN/model checking work on them unchanged.

The libraries differ in three places, and each difference is resolved here so
that the rules reproduce the library's raw score bit for bit:

* **Split operator.** XGBoost sends ``x < t`` left (in float32). For float32
  ``x`` and ``t`` that is exactly ``x <= prev(t)``, the largest float32 below
  ``t``. LightGBM sends ``x <= t`` left (in float64). CatBoost sets a leaf
  bit when ``x > border`` (in float32), so ``x <= border`` is the other child.
* **Precision.** XGBoost and CatBoost compare float32 inputs
  (``input_dtype='float32'``); LightGBM compares float64 inputs
  (``input_dtype=None``). XGBoost accumulates the raw score in float32, tree
  by tree, from its base margin (``score_dtype='float32'``); LightGBM and
  CatBoost accumulate in float64.
* **Binary decision.** scikit-learn predicts the positive class when the raw
  score is ``>= 0``. XGBoost predicts it when ``sigmoid(score) > 0.5`` in
  float32, LightGBM when ``p > 1 - p`` in float64, CatBoost when the raw score
  is ``> 0``. Each is a monotone test of the score, hence ``score >= c`` for
  one constant ``c`` (``binary_threshold``), found here once.

Multiclass models predict the argmax of the raw scores, lowest index on a tie.
XGBoost (``multi:softprob``) and LightGBM take the argmax of the softmax
probabilities instead; the two agree except when two probabilities round to
the same value (scores within about one float32 ULP for XGBoost).

XGBoost's DART weights each tree at prediction; the weight is folded into the
leaves (the float32 product XGBoost computes), so DART models are read too.

What the rules cannot express is refused with a ``ValueError``: categorical
splits, linear boosters, LightGBM linear trees and zero-as-missing,
CatBoost categorical/text features and non-symmetric trees, and objectives
other than the binary and multiclass ones listed in each reader.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np

from ._rules import Rule, make_condition

# Library class names, as `_sklearn_estimator_class` and `from_sklearn` see them.
XGB = 'XGBClassifier'
LGBM = 'LGBMClassifier'
CATBOOST = 'CatBoostClassifier'
EXTERNAL_GBDT = (XGB, LGBM, CATBOOST)
# The model name shown to the user for each library (reports, repr, exports).
LIBRARY_NAMES = {XGB: 'XGBoost', LGBM: 'LightGBM', CATBOOST: 'CatBoost'}
# Library codes of the version 5 binary format (0 = scikit-learn).
LIBRARY_CODES = {None: 0, XGB: 1, LGBM: 2, CATBOOST: 3}


# ---------------------------------------------------------------------------
# Neutral tree form
# ---------------------------------------------------------------------------
# A tree is a dict of parallel lists indexed by node id (root = 0):
#   left[n], right[n]  child ids, -1 at a leaf
#   feature[n]         column index of the split (leaf: -1)
#   threshold[n]       float, x <= threshold goes left
#   nan_left[n]        True/False where NaN goes, None if NaN is refused
#   value[n]           leaf value (already scaled: the stage contribution)


def _new_tree() -> dict:
    return {'left': [], 'right': [], 'feature': [], 'threshold': [],
            'nan_left': [], 'value': []}


def _add_node(tree: dict, feature=-1, threshold=0.0, nan_left=None, value=0.0) -> int:
    tree['left'].append(-1)
    tree['right'].append(-1)
    tree['feature'].append(int(feature))
    tree['threshold'].append(float(threshold))
    tree['nan_left'].append(nan_left)
    tree['value'].append(float(value))
    return len(tree['left']) - 1


def _f32_prev(t: float) -> float:
    """The largest float32 strictly below float32 `t`, as a Python float."""
    return float(np.nextafter(np.float32(t), np.float32(-np.inf)))


# ---------------------------------------------------------------------------
# Binary cutoffs
# ---------------------------------------------------------------------------

def _first_true(pred, lo: float, hi: float, dtype) -> float:
    """Smallest value v of `dtype` in (lo, hi] with pred(v), by bisection on
    the ordered bit patterns of non-negative floats. pred must be monotone
    (False ... False True ... True) on that range and pred(hi) True."""
    itype = np.uint32 if dtype == np.float32 else np.uint64
    a = int(np.array(lo, dtype=dtype).view(itype))   # pred False here
    b = int(np.array(hi, dtype=dtype).view(itype))   # pred True here
    while b - a > 1:
        m = (a + b) // 2
        v = float(np.array(m, dtype=itype).view(dtype))
        if pred(v):
            b = m
        else:
            a = m
    return float(np.array(b, dtype=itype).view(dtype))


def xgboost_binary_threshold() -> float:
    """Smallest float32 margin that XGBClassifier predicts as the positive
    class (``sigmoid(margin) > 0.5``, computed by XGBoost itself in float32).

    A booster without trees returns sigmoid(base_margin), so XGBoost's own
    transform is queried directly; the result is the platform's.
    """
    import xgboost as xgb
    d = xgb.DMatrix(np.zeros((1, 1), dtype=np.float32), label=np.array([0.0]))
    bst = xgb.train({'objective': 'binary:logistic', 'base_score': 0.5},
                    d, num_boost_round=0)

    def positive(margin: float) -> bool:
        dm = xgb.DMatrix(np.zeros((1, 1), dtype=np.float32),
                         base_margin=np.array([margin], dtype=np.float32))
        return bool(bst.predict(dm)[0] > 0.5)

    if positive(0.0):
        return 0.0
    return _first_true(positive, 0.0, 1.0, np.float32)


def lightgbm_binary_threshold(sigmoid: float = 1.0) -> float:
    """Smallest float64 raw score that LGBMClassifier predicts as the positive
    class: ``argmax([1 - p, p])`` with ``p = 1 / (1 + exp(-sigmoid * raw))``
    (LightGBM's BinaryLogloss::ConvertOutput, in float64)."""
    def positive(raw: float) -> bool:
        p = 1.0 / (1.0 + math.exp(-sigmoid * raw))
        return p > 1.0 - p

    if positive(0.0):
        return 0.0
    return _first_true(positive, 0.0, 1.0, np.float64)


# CatBoost predicts the positive class when the raw score is > 0.
CATBOOST_BINARY_THRESHOLD = float(np.nextafter(0.0, 1.0))


def binary_test(cut: float) -> Tuple[str, float]:
    """The comparison generated code writes for ``score >= cut``.

    ``score >= smallest positive double`` is written ``score > 0.0`` (the
    same test on doubles), which reads as CatBoost's rule and needs no
    subnormal literal (CPN ML and float-only targets may not parse one).

    Returns:
        (operator, constant): ``('>=', cut)`` or ``('>', 0.0)``.
    """
    if cut == CATBOOST_BINARY_THRESHOLD:
        return '>', 0.0
    return '>=', float(cut)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def _xgb_parse_vector(text) -> List[float]:
    """'[5.0166667E-1]' or '5E-1' (XGBoost's base_score) -> list of floats."""
    s = str(text).strip()
    if s.startswith('['):
        s = s[1:-1]
    return [float(v) for v in s.split(',') if v.strip()]


def read_xgboost(model) -> dict:
    """Neutral form of a fitted ``xgboost.XGBClassifier``."""
    missing = model.get_params().get('missing', np.nan)
    if missing is not None and not (isinstance(missing, float) and math.isnan(missing)):
        raise ValueError(f'Unsupported XGBoost model: missing={missing!r} treats that '
                         'value as missing, which a threshold rule cannot route; '
                         'only missing=np.nan (the default) is supported.')
    booster = model.get_booster()
    J = json.loads(bytes(booster.save_raw(raw_format='json')))
    learner = J['learner']
    gb = learner['gradient_booster']
    if gb.get('name') not in ('gbtree', 'dart'):
        raise ValueError(f"Unsupported XGBoost booster {gb.get('name')!r}: only "
                         "tree boosters ('gbtree', 'dart') can be expressed as "
                         "rules ('gblinear' has no trees).")
    objective = learner['objective']['name']
    if objective not in ('binary:logistic', 'multi:softprob', 'multi:softmax'):
        raise ValueError(f'Unsupported XGBoost objective {objective!r}: expected '
                         "'binary:logistic', 'multi:softprob' or 'multi:softmax'.")
    mp = learner['learner_model_param']
    n_class = int(mp.get('num_class', '0'))
    if int(mp.get('num_target', '1')) != 1:
        raise ValueError('Unsupported XGBoost model: multi-target output.')
    n_groups = 1 if n_class <= 1 else n_class
    base = _xgb_parse_vector(mp['base_score'])

    # Initial margin of each output group, as XGBoost computes it in float32.
    if objective == 'binary:logistic':
        p = np.float32(base[0])
        init = [float(np.float32(-np.log(np.float32(1.0) / p - np.float32(1.0))))]
    else:
        init = [float(np.float32(base[k if len(base) > 1 else 0])) for k in range(n_groups)]

    gmodel = gb['model'] if 'model' in gb else gb['gbtree']['model']
    tree_info = [int(t) for t in gmodel['tree_info']]
    # DART multiplies each tree by its weight at prediction, in float32
    # (weight * leaf, then added); the product is folded into the leaves.
    weights = gmodel.get('weight_drop') or gb.get('weight_drop')
    trees_by_group: Dict[int, List[dict]] = {g: [] for g in range(n_groups)}
    for t_idx, (tj, gid) in enumerate(zip(gmodel['trees'], tree_info)):
        w = None if not weights else np.float32(weights[t_idx])
        if any(int(s) != 0 for s in tj.get('split_type', [])):
            raise ValueError('Unsupported XGBoost model: categorical splits '
                             'cannot be expressed as threshold rules.')
        lc, rc = tj['left_children'], tj['right_children']
        feat, cond = tj['split_indices'], tj['split_conditions']
        dleft = tj['default_left']
        tree = _new_tree()
        ids = {}

        def build(n):
            if lc[n] == -1:
                leaf = np.float32(cond[n])
                return _add_node(tree, value=float(leaf if w is None
                                                   else np.float32(w * leaf)))
            # x < t  ==  x <= prev_f32(t) for float32 x and t.
            nid = _add_node(tree, feature=feat[n], threshold=_f32_prev(cond[n]),
                            nan_left=bool(dleft[n]))
            tree['left'][nid] = build(lc[n])
            tree['right'][nid] = build(rc[n])
            return nid

        build(0)
        trees_by_group[gid].append(tree)

    return {
        'estimator': XGB,
        'n_groups': n_groups,
        'init': init,
        'trees': trees_by_group,
        'input_dtype': 'float32',
        'score_dtype': 'float32',
        'binary_threshold': xgboost_binary_threshold() if n_groups == 1 else 0.0,
        'routes_missing': True,
    }


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

# LightGBM reads an input whose magnitude is at most kZeroThreshold (1e-35f)
# as exactly 0.0 before any split compares it.
_LGBM_ZERO = float(np.float32(1e-35))


def _lgbm_effective_threshold(th: float) -> float:
    """Threshold t' with ``x <= t'`` equal to LightGBM's ``g(x) <= th``,
    where g maps every |x| <= kZeroThreshold to 0.0.

    th >= 0: g(x) <= th holds for x <= th and for the whole zero band, so
    t' = max(th, z). th < 0: it needs x <= th outside the band; for
    th < -z that is x <= th, otherwise x < -z, i.e. x <= prev(-z).
    """
    z = _LGBM_ZERO
    if th >= 0.0:
        return max(th, z)
    if th < -z:
        return th
    return float(np.nextafter(-z, -np.inf))


def read_lightgbm(model) -> dict:
    """Neutral form of a fitted ``lightgbm.LGBMClassifier``."""
    booster = model.booster_
    D = booster.dump_model()
    objective = str(D.get('objective', ''))
    parts = objective.split()
    name = parts[0] if parts else ''
    opts = dict(p.split(':', 1) for p in parts[1:] if ':' in p)
    if name not in ('binary', 'multiclass'):
        raise ValueError(f'Unsupported LightGBM objective {objective!r}: expected '
                         "'binary' or 'multiclass' (softmax).")
    if D.get('average_output'):
        raise ValueError("Unsupported LightGBM model: boosting_type='rf' averages "
                         'the trees instead of adding them.')
    n_groups = int(D.get('num_tree_per_iteration', 1))
    sigmoid = float(opts.get('sigmoid', 1.0))

    trees_by_group: Dict[int, List[dict]] = {g: [] for g in range(n_groups)}
    for i, info in enumerate(D['tree_info']):
        if info.get('is_linear'):
            raise ValueError('Unsupported LightGBM model: linear trees.')
        tree = _new_tree()

        def build(node):
            if 'leaf_value' in node or 'split_feature' not in node:
                if 'leaf_coeff' in node:
                    raise ValueError('Unsupported LightGBM model: linear trees.')
                return _add_node(tree, value=float(node.get('leaf_value', 0.0)))
            if node.get('decision_type') != '<=':
                raise ValueError('Unsupported LightGBM model: categorical splits '
                                 'cannot be expressed as threshold rules.')
            missing = node.get('missing_type', 'None')
            th_model = float(node['threshold'])
            th = _lgbm_effective_threshold(th_model)
            if missing == 'Zero':
                raise ValueError('Unsupported LightGBM model: zero_as_missing treats '
                                 '0.0 as missing, which a threshold rule cannot route.')
            if missing == 'NaN':
                nan_left = bool(node['default_left'])
            else:
                # missing_type None: LightGBM replaces NaN by 0.0.
                nan_left = bool(0.0 <= th_model)
            nid = _add_node(tree, feature=node['split_feature'], threshold=th,
                            nan_left=nan_left)
            tree['left'][nid] = build(node['left_child'])
            tree['right'][nid] = build(node['right_child'])
            return nid

        build(info['tree_structure'])
        trees_by_group[i % n_groups].append(tree)

    return {
        'estimator': LGBM,
        'n_groups': n_groups,
        # boost_from_average is folded into the first trees' leaves.
        'init': [0.0] * n_groups,
        'trees': trees_by_group,
        'input_dtype': None,
        'score_dtype': 'float64',
        'binary_threshold': lightgbm_binary_threshold(sigmoid) if n_groups == 1 else 0.0,
        'routes_missing': True,
    }


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------

def read_catboost(model) -> dict:
    """Neutral form of a fitted ``catboost.CatBoostClassifier``."""
    fd, path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    try:
        model.save_model(path, format='json')
        with open(path, encoding='utf-8') as fh:
            C = json.load(fh)
    finally:
        os.remove(path)

    if 'oblivious_trees' not in C:
        raise ValueError("Unsupported CatBoost model: only symmetric trees "
                         "(grow_policy='SymmetricTree', the default) are supported.")
    info = C.get('features_info', {})
    for kind in ('categorical_features', 'text_features', 'embedding_features'):
        if info.get(kind):
            raise ValueError(f'Unsupported CatBoost model: {kind.replace("_", " ")}.')
    params = model.get_all_params()
    loss = str(params.get('loss_function', ''))
    if loss not in ('Logloss', 'CrossEntropy', 'MultiClass'):
        raise ValueError(f'Unsupported CatBoost loss {loss!r}: expected '
                         "'Logloss', 'CrossEntropy' or 'MultiClass'.")
    scale, bias = C.get('scale_and_bias', [1.0, [0.0]])
    bias = bias if isinstance(bias, list) else [bias]
    if float(scale) != 1.0 or any(float(b) != 0.0 for b in bias):
        raise ValueError('Unsupported CatBoost model: a scale or bias other than '
                         '(1, 0) is applied after the trees are summed '
                         '(e.g. boost_from_average=True).')

    float_feats = info.get('float_features', [])
    col_of = {}
    nan_left_of = {}
    for ff in float_feats:
        col_of[int(ff['feature_index'])] = int(ff.get('flat_feature_index', ff['feature_index']))
        treat = ff.get('nan_value_treatment', 'AsIs')
        # A leaf bit is set when x > border; NaN compares false (AsIs) or as
        # the smallest value (Min) -> bit clear -> the "<=" child. Max -> the
        # ">" child. Forbidden: NaN is refused.
        nan_left_of[int(ff['feature_index'])] = (
            None if treat == 'Forbidden' else (treat != 'Max'))

    n_groups = len(bias) if loss == 'MultiClass' else 1
    trees_by_group: Dict[int, List[dict]] = {g: [] for g in range(n_groups)}
    for ot in C['oblivious_trees']:
        splits = ot['splits']
        for s in splits:
            if s.get('split_type') != 'FloatFeature':
                raise ValueError(f"Unsupported CatBoost split {s.get('split_type')!r}.")
        depth = len(splits)
        leaves = ot['leaf_values']
        if len(leaves) != (2 ** depth) * n_groups:
            raise ValueError('Unexpected CatBoost leaf layout.')
        for g in range(n_groups):
            tree = _new_tree()

            # Leaf index = sum over levels i of bit_i << i, bit_i = x > border_i.
            # Expand as a full binary tree whose level i tests split i.
            def build(level, index):
                if level == depth:
                    return _add_node(tree, value=float(leaves[index * n_groups + g]))
                s = splits[level]
                fi = int(s['float_feature_index'])
                nid = _add_node(tree, feature=col_of.get(fi, fi),
                                threshold=float(np.float32(s['border'])),
                                nan_left=nan_left_of.get(fi, True))
                tree['left'][nid] = build(level + 1, index)
                tree['right'][nid] = build(level + 1, index | (1 << level))
                return nid

            build(0, 0)
            trees_by_group[g].append(tree)

    return {
        'estimator': CATBOOST,
        'n_groups': n_groups,
        'init': [0.0] * n_groups,
        'trees': trees_by_group,
        'input_dtype': 'float32',
        'score_dtype': 'float64',
        'binary_threshold': CATBOOST_BINARY_THRESHOLD if n_groups == 1 else 0.0,
        'routes_missing': True,
    }


READERS = {XGB: read_xgboost, LGBM: read_lightgbm, CATBOOST: read_catboost}


def estimator_kind(model) -> Optional[str]:
    """The external GBDT class name of `model`, or None."""
    for cls in type(model).__mro__:
        if cls.__name__ in READERS and cls.__module__.split('.')[0] in (
                'xgboost', 'lightgbm', 'catboost'):
            return cls.__name__
    return None


# ---------------------------------------------------------------------------
# Neutral form -> Rules
# ---------------------------------------------------------------------------

def neutral_to_rules(form: dict, feature_names: List[str],
                     classes: List[str]) -> Tuple[List[Rule], Dict[str, float], bool, List[str]]:
    """Rules in the layout of ``get_gbdt_rules``.

    Returns:
        (rules, init_scores, is_binary, classes) as ``get_gbdt_rules``.
    """
    n_groups = form['n_groups']
    is_binary = n_groups == 1
    if is_binary and len(classes) != 2:
        raise ValueError(f'A binary model needs 2 classes, got {len(classes)}.')
    if not is_binary and len(classes) != n_groups:
        raise ValueError(f'The model has {n_groups} class outputs but '
                         f'{len(classes)} classes were given.')
    f32 = form['score_dtype'] == 'float32'

    def val(v):
        return float(np.float32(v)) if f32 else float(v)

    group_class = ([(0, classes[1])] if is_binary
                   else [(g, classes[g]) for g in range(n_groups)])
    rules: List[Rule] = []
    init_scores: Dict[str, float] = {}
    for g, label in group_class:
        init = val(form['init'][g])
        init_scores[label] = init
        r0 = Rule(f'GBDT{label}T0_Init_Class{label}', str(label), [],
                  leaf_value=init, learning_rate=None, class_group=str(label))
        r0.parsed_conditions = []
        rules.append(r0)
        for t_idx, tree in enumerate(form['trees'][g]):
            out: List[Rule] = []

            def recurse(n, cs, cp):
                if tree['left'][n] == -1:
                    r = Rule(f'GBDT{label}T{t_idx + 1}_Rule{len(out)}_Class{label}',
                             str(label), list(cs), leaf_value=val(tree['value'][n]),
                             learning_rate=1.0, class_group=str(label))
                    r.parsed_conditions = list(cp)
                    out.append(r)
                    return
                name = feature_names[tree['feature'][n]]
                th = tree['threshold'][n]
                nl = tree['nan_left'][n]
                recurse(tree['left'][n], cs + [f'{name} <= {th}'],
                        cp + [make_condition(name, '<=', th, nl)])
                recurse(tree['right'][n], cs + [f'{name} > {th}'],
                        cp + [make_condition(name, '>', th, None if nl is None else not nl)])

            recurse(0, [], [])
            rules.extend(out)
    return rules, init_scores, is_binary, list(classes)
