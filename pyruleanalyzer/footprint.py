"""
Size of a tree model on a microcontroller, before it is trained.

The C and Arduino exports store every tree as constant arrays with one entry
per node (``feature``, ``threshold``, ``left``, ``right``, ``value`` and, for
models that route missing values, ``missing_left``), and call each tree once
from the predict function. Their flash cost is therefore a linear function of
two counts:

    model bytes = PER_TREE_BYTES * trees + node_bytes(algorithm) * nodes

A scikit-learn tree with ``L`` leaves has ``2L - 1`` nodes and at most
``2 ** max_depth`` leaves, so the cost of a configuration is bounded before
training. :func:`suggest_complexity` turns that bound into the deepest trees
a flash budget allows, optionally refined by fitting a few trees on a sample
of the data to measure how full the trees actually get. The board is
optional: a preset of :data:`BOARDS`, or any flash and RAM sizes.
:func:`fit_to_board` goes further and picks the number of trees and the leaves
per tree that fill a board, comparing the candidates on validation data.

Rule refinement only removes rules, so the bound of the extracted model also
bounds the refined one.

The node layout is the one written by ``_c_export``; ``PER_TREE_BYTES`` and
the firmware size of the board presets were measured on 42 ESP32 sketches
compiled with arduino-esp32 3.3.12 (compiled flash = 276,469 + 69.6 * trees +
27.82 * nodes, maximum error 1.2 KB). Other boards, cores or sketches have
other constants: pass their flash and RAM sizes and the size of the sketch
without the model (``firmware_flash``, ``firmware_ram``).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

# Code per tree: the traversal call of the tree in the predict function.
PER_TREE_BYTES = 70

@dataclass(frozen=True)
class Board:
    """Memory of a target board as seen by an Arduino sketch.

    Attributes:
        flash_bytes: Flash of the application (the sketch partition).
        ram_bytes: RAM available to the sketch.
        firmware_flash: Flash of the sketch without the model (core, serial).
        firmware_ram: RAM of the sketch without the model.
        double_bytes: Size of ``double`` (4 on AVR, where it is a float).
        constants_in_ram: The exported ``static const`` arrays are copied to
            RAM (AVR, since the export does not use ``PROGMEM``).
    """
    flash_bytes: int
    ram_bytes: int
    firmware_flash: int = 0
    firmware_ram: int = 0
    double_bytes: int = 8
    constants_in_ram: bool = False


# ESP32 values were measured with arduino-esp32 3.3.12 (a serial streaming
# sketch); AVR firmware values are those of a minimal sketch using Serial.
BOARDS: Dict[str, Board] = {
    'uno': Board(32_256, 2_048, 1_500, 200, double_bytes=4, constants_in_ram=True),
    'nano': Board(30_720, 2_048, 1_500, 200, double_bytes=4, constants_in_ram=True),
    'mega2560': Board(253_952, 8_192, 1_600, 200, double_bytes=4, constants_in_ram=True),
    'esp32': Board(1_310_720, 327_680, 276_469, 23_380),
    'esp32_huge_app': Board(3_145_728, 327_680, 276_469, 23_380),
}


def make_board(board: Optional[Any] = None, **overrides) -> Optional[Board]:
    """A :class:`Board` from a preset name, a Board, or custom values.

    ``make_board('esp32', ram_bytes=200_000)`` changes one value of a preset;
    ``make_board(flash_bytes=..., ram_bytes=...)`` describes any board.
    Overrides set to None are ignored. Returns None when nothing is given.
    """
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if board is None and not overrides:
        return None
    if board is None:
        base = Board(flash_bytes=overrides.pop('flash_bytes', 0),
                     ram_bytes=overrides.pop('ram_bytes', 0))
    elif isinstance(board, Board):
        base = board
    else:
        if board not in BOARDS:
            raise ValueError(f'unknown board {board!r}; known: {sorted(BOARDS)}, '
                             'or pass flash_bytes and ram_bytes')
        base = BOARDS[board]
    from dataclasses import replace
    return replace(base, **overrides)


_ALIASES = {
    'gbdt': 'gbdt', 'gradient boosting decision trees': 'gbdt',
    'gradientboostingclassifier': 'gbdt',
    'rf': 'rf', 'random forest': 'rf', 'randomforestclassifier': 'rf',
    'extratreesclassifier': 'rf',
    'dt': 'dt', 'decision tree': 'dt', 'decisiontreeclassifier': 'dt',
}


def _algorithm(name: str) -> str:
    """'gbdt', 'rf' or 'dt' for any accepted spelling of an algorithm."""
    key = str(name).strip().lower()
    if key not in _ALIASES:
        raise ValueError(f"unknown algorithm {name!r}; use 'gbdt', 'rf' or 'dt'")
    return _ALIASES[key]


def node_bytes(algorithm: str = 'gbdt', n_classes: int = 2,
               missing_routing: bool = False, double_bytes: int = 8) -> int:
    """Flash bytes of one node in the C/Arduino export.

    Args:
        algorithm: 'gbdt', 'rf' or 'dt' (or the scikit-learn class name).
        n_classes: Number of classes (Random Forest stores one double per
            class and node).
        missing_routing: The model routes missing values (decision trees and
            forests trained on NaN), which adds one byte per node.
        double_bytes: Size of ``double`` on the target (4 on AVR).

    Returns:
        int: Bytes per node.
    """
    algo = _algorithm(algorithm)
    value = {'gbdt': double_bytes, 'dt': 4, 'rf': double_bytes * int(n_classes)}[algo]
    base = 4 + double_bytes + 4 + 4  # feature, threshold, left, right
    return base + value + (1 if missing_routing and algo != 'gbdt' else 0)


def n_trees(algorithm: str = 'gbdt', n_estimators: int = 100,
            n_classes: int = 2) -> int:
    """Trees of a fitted model: GBDT fits one per class and stage (one per
    stage for two classes), a forest one per estimator, a decision tree one."""
    algo = _algorithm(algorithm)
    if algo == 'gbdt':
        return int(n_estimators) * (1 if int(n_classes) <= 2 else int(n_classes))
    if algo == 'rf':
        return int(n_estimators)
    return 1


def model_bytes(trees: int, nodes: int, algorithm: str = 'gbdt',
                n_classes: int = 2, missing_routing: bool = False,
                double_bytes: int = 8) -> int:
    """Flash bytes of a model with `trees` trees and `nodes` nodes in total."""
    return int(PER_TREE_BYTES * trees
               + node_bytes(algorithm, n_classes, missing_routing, double_bytes) * nodes)


def runtime_ram_bytes(n_features: int, n_classes: int = 2,
                      double_bytes: int = 8) -> int:
    """RAM the predict call needs besides the firmware: the feature vector
    and one score per class."""
    return int(double_bytes * (int(n_features) + int(n_classes)))


def max_leaves_per_tree(max_depth: Optional[int] = None,
                        max_leaf_nodes: Optional[int] = None,
                        n_samples: Optional[int] = None) -> Optional[int]:
    """Most leaves a tree can have under these limits (None: unbounded)."""
    caps = []
    if max_depth is not None:
        caps.append(2 ** int(max_depth))
    if max_leaf_nodes is not None:
        caps.append(int(max_leaf_nodes))
    if n_samples is not None:
        caps.append(int(n_samples))
    return min(caps) if caps else None


def estimate_footprint(algorithm: str = 'gbdt', n_estimators: int = 100,
                       n_classes: int = 2, max_depth: Optional[int] = None,
                       max_leaf_nodes: Optional[int] = None,
                       n_samples: Optional[int] = None,
                       leaves_per_tree: Optional[float] = None,
                       missing_routing: bool = False,
                       double_bytes: int = 8) -> Dict[str, Any]:
    """Size of a tree model, as an upper bound or from a leaf count.

    Args:
        algorithm: 'gbdt', 'rf' or 'dt'.
        n_estimators: Boosting stages or forest trees (ignored for 'dt').
        n_classes: Number of classes.
        max_depth, max_leaf_nodes: The limits the model is trained with.
        n_samples: Training rows (a tree cannot have more leaves).
        leaves_per_tree: Mean leaves per tree, measured or expected; when
            omitted the trees are assumed full (the upper bound).
        missing_routing, double_bytes: See :func:`node_bytes`.

    Returns:
        dict: ``trees``, ``leaves`` (rules), ``nodes``, ``model_bytes`` and
        ``bound`` (True when the numbers are the upper bound).
    """
    trees = n_trees(algorithm, n_estimators, n_classes)
    cap = max_leaves_per_tree(max_depth, max_leaf_nodes, n_samples)
    bound = leaves_per_tree is None
    if bound:
        if cap is None:
            raise ValueError('an upper bound needs max_depth, max_leaf_nodes '
                             'or n_samples')
        per_tree = float(cap)
    else:
        per_tree = float(leaves_per_tree)
        if cap is not None:
            per_tree = min(per_tree, cap)
    leaves = per_tree * trees
    nodes = 2 * leaves - trees
    return {
        'trees': trees,
        'leaves': int(round(leaves)),
        'nodes': int(round(nodes)),
        'model_bytes': model_bytes(trees, int(round(nodes)), algorithm,
                                   n_classes, missing_routing, double_bytes),
        'bound': bound,
    }


def _make_estimator(algo: str, n_estimators: int, max_depth: Optional[int],
                    random_state: Any, params: Dict[str, Any]):
    """A scikit-learn estimator of `algo` with these settings."""
    if algo == 'gbdt':
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                          random_state=random_state, **params)
    if algo == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                      random_state=random_state, **params)
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(max_depth=max_depth, random_state=random_state,
                                  **params)


def _fitted_trees(model) -> List[Any]:
    """The fitted scikit-learn trees of a model."""
    if hasattr(model, 'tree_'):
        return [model]
    return [t for t in np.ravel(model.estimators_)]


def probe_leaves(X, y, algorithm: str = 'gbdt', max_depth: Optional[int] = None,
                 n_estimators: int = 10, max_rows: Optional[int] = 50_000,
                 random_state: Any = 0, **params) -> float:
    """Mean leaves per tree of a small model fitted on a sample of the data.

    Fits `n_estimators` stages (or forest trees) on at most `max_rows` rows
    drawn at random and returns the mean leaf count of the fitted trees. It
    tends to underestimate the trees of the full data, which have more rows
    to split, so treat it as the expected size, and the bound of
    :func:`estimate_footprint` as the guarantee.
    """
    algo = _algorithm(algorithm)
    X = np.asarray(X)
    y = np.asarray(y)
    if max_rows is not None and len(X) > max_rows:
        idx = np.random.default_rng(random_state).choice(len(X), max_rows, replace=False)
        X, y = X[idx], y[idx]
    model = _make_estimator(algo, n_estimators, max_depth, random_state, params)
    model.fit(X, y)
    return float(np.mean([t.tree_.n_leaves for t in _fitted_trees(model)]))


@dataclass
class ComplexityReport:
    """Size of each candidate depth and the deepest that fits a board.

    Attributes:
        rows: One dict per candidate: ``n_estimators``, ``max_leaf_nodes``,
            ``max_depth``, ``trees``, ``max_leaves``,
            ``max_bytes`` (upper bound of the model flash) and, when probed,
            ``expected_leaves`` and ``expected_bytes``; with a board,
            ``flash`` and ``ram`` (bound of the whole sketch), ``fits`` (the
            bound fits) and ``fits_expected``.
        board: The target (None: sizes only).
        flash_budget: Flash left for the model (None without a board).
        max_depth: Deepest depth whose bound fits in some candidate, so every
            model trained with it fits (None without a board or if none fits).
        max_depth_expected: Deepest depth whose probed size fits.
    """
    rows: List[Dict[str, Any]] = field(default_factory=list)
    board: Optional[Board] = None
    flash_budget: Optional[int] = None
    max_depth: Optional[int] = None
    max_depth_expected: Optional[int] = None

    def fitting(self, expected: bool = False) -> List[Dict[str, Any]]:
        """Candidates that fit the board, largest bound first."""
        key = 'fits_expected' if expected else 'fits'
        return sorted((r for r in self.rows if r.get(key)),
                      key=lambda r: r['max_bytes'], reverse=True)

    @property
    def largest(self) -> Optional[Dict[str, Any]]:
        """The fitting candidate with the largest guaranteed size, or None."""
        rows = self.fitting()
        return rows[0] if rows else None

    def to_frame(self):
        """The rows as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.rows)

    def __str__(self) -> str:
        cols = [c for c in ('n_estimators', 'max_leaf_nodes', 'max_depth', 'trees', 'max_leaves', 'max_bytes',
                            'expected_leaves', 'expected_bytes', 'flash', 'ram',
                            'fits', 'fits_expected')
                if any(r.get(c) is not None for r in self.rows)]
        width = {c: max(len(c), *(len(f'{r.get(c, "")}') for r in self.rows)) for c in cols}
        lines = ['  '.join(c.rjust(width[c]) for c in cols)]
        lines += ['  '.join(f'{r.get(c, "")}'.rjust(width[c]) for c in cols) for r in self.rows]
        if self.board is not None:
            lines.append(f'flash for the model {self.flash_budget:,} bytes: '
                         f'max_depth {self.max_depth}'
                         + (f' (expected {self.max_depth_expected})'
                            if self.max_depth_expected is not None else ''))
        return '\n'.join(lines)


def _as_list(value) -> List[Any]:
    """`value` as a list of candidates (a scalar or None is one candidate)."""
    if value is None or isinstance(value, (int, np.integer)):
        return [value]
    return list(value)


def suggest_complexity(X=None, y=None, *, algorithm: str = 'gbdt',
                       n_estimators: Any = 100, n_classes: Optional[int] = None,
                       n_features: Optional[int] = None,
                       depths: Iterable[int] = range(1, 13),
                       max_leaf_nodes: Any = None,
                       board: Optional[Any] = None,
                       flash_bytes: Optional[int] = None,
                       ram_bytes: Optional[int] = None,
                       firmware_flash: Optional[int] = None,
                       firmware_ram: Optional[int] = None,
                       probe: bool = True, probe_rows: Optional[int] = 50_000,
                       probe_estimators: int = 10, random_state: Any = 0,
                       missing_routing: bool = False,
                       **params) -> ComplexityReport:
    """Model size per configuration, and the largest a board can hold.

    Every combination of `n_estimators`, `max_leaf_nodes` and `depths` is a
    candidate (pass a list to search more than one value of the first two).
    The hardware is optional. Without it the report only lists the size of
    each depth. With it, a depth fits when the whole sketch (firmware plus
    model) fits both the flash and the RAM of the board.

    Args:
        X, y: Training data. Optional: they give `n_classes`, `n_features`,
            the row count that caps the leaves, and the probe.
        algorithm: 'gbdt', 'rf' or 'dt'.
        n_estimators: Boosting stages or forest trees, or a list of them.
        n_classes, n_features: Taken from `y` and `X` when omitted.
        depths: Candidate values of ``max_depth``.
        max_leaf_nodes: Leaf limit applied together with the depth (None: no
            limit), or a list of them.
        board: A key of :data:`BOARDS` or a :class:`Board`.
        flash_bytes, ram_bytes, firmware_flash, firmware_ram: Custom memory,
            alone or overriding the values of `board` (see :class:`Board`).
        probe: With `X` and `y`, fit `probe_estimators` stages on
            `probe_rows` rows per depth to estimate the expected size.
        random_state: Seed of the probe.
        missing_routing: See :func:`node_bytes`.
        **params: Further estimator parameters for the probe
            (``min_samples_leaf``, ``learning_rate``, ...).

    Returns:
        ComplexityReport
    """
    algo = _algorithm(algorithm)
    if n_classes is None:
        if y is None:
            raise ValueError('pass n_classes or y')
        n_classes = int(len(np.unique(np.asarray(y))))
    if n_features is None and X is not None:
        n_features = int(np.asarray(X).shape[1])
    target = make_board(board, flash_bytes=flash_bytes, ram_bytes=ram_bytes,
                        firmware_flash=firmware_flash, firmware_ram=firmware_ram)
    dbl = target.double_bytes if target is not None else 8
    n_samples = None if X is None else int(len(X))
    if params.get('min_samples_leaf') and n_samples:
        n_samples //= int(params['min_samples_leaf'])

    report = ComplexityReport(board=target)
    if target is not None:
        report.flash_budget = target.flash_bytes - target.firmware_flash
        ram_fixed = target.firmware_ram + runtime_ram_bytes(n_features or 0, n_classes, dbl)

    def fits(model: int):
        """(sketch flash, sketch RAM, both fit) for a model of `model` bytes."""
        flash = target.firmware_flash + model
        ram = ram_fixed + (model if target.constants_in_ram else 0)
        return flash, ram, flash <= target.flash_bytes and ram <= target.ram_bytes

    est_grid = _as_list(n_estimators)
    leaf_grid = _as_list(max_leaf_nodes)
    probed: Dict[Any, float] = {}
    for n_est, leaf_cap, d in itertools.product(est_grid, leaf_grid, depths):
        ub = estimate_footprint(algo, n_est, n_classes, d, leaf_cap,
                                n_samples, missing_routing=missing_routing,
                                double_bytes=dbl)
        row: Dict[str, Any] = {'n_estimators': n_est, 'max_leaf_nodes': leaf_cap,
                               'max_depth': d, 'trees': ub['trees'],
                               'max_leaves': ub['leaves'], 'max_bytes': ub['model_bytes']}
        if probe and X is not None and y is not None:
            # leaves per tree barely depend on the number of trees: probe once
            if (d, leaf_cap) not in probed:
                extra = {'max_leaf_nodes': leaf_cap} if leaf_cap is not None else {}
                probed[d, leaf_cap] = probe_leaves(
                    X, y, algo, d, min(probe_estimators, max(est_grid)),
                    probe_rows, random_state, **extra, **params)
            ex = estimate_footprint(algo, n_est, n_classes, d, leaf_cap,
                                    n_samples, leaves_per_tree=probed[d, leaf_cap],
                                    missing_routing=missing_routing, double_bytes=dbl)
            row['expected_leaves'] = ex['leaves']
            row['expected_bytes'] = ex['model_bytes']
        if target is not None:
            row['flash'], row['ram'], row['fits'] = fits(row['max_bytes'])
            if 'expected_bytes' in row:
                row['fits_expected'] = fits(row['expected_bytes'])[2]
        report.rows.append(row)

    if target is not None:
        ok = [r['max_depth'] for r in report.rows if r['fits']]
        report.max_depth = max(ok) if ok else None
        ok = [r['max_depth'] for r in report.rows if r.get('fits_expected')]
        report.max_depth_expected = max(ok) if ok else None
    return report


def max_depth_for_budget(flash_budget: int, algorithm: str = 'gbdt',
                         n_estimators: int = 100, n_classes: int = 2,
                         missing_routing: bool = False,
                         double_bytes: int = 8) -> Optional[int]:
    """Deepest ``max_depth`` whose full trees fit `flash_budget` bytes.

    Closed form of the bound: ``trees * (PER_TREE_BYTES + node * (2 ** (d + 1)
    - 1)) <= budget``. None when not even depth 0 (one leaf) fits.
    """
    trees = n_trees(algorithm, n_estimators, n_classes)
    node = node_bytes(algorithm, n_classes, missing_routing, double_bytes)
    per_tree = flash_budget / trees - PER_TREE_BYTES
    if per_tree < node:
        return None
    return int(math.floor(math.log2((per_tree / node + 1) / 2)))


def max_leaves_for_budget(flash_budget: int, algorithm: str = 'gbdt',
                          n_estimators: int = 100, n_classes: int = 2,
                          missing_routing: bool = False,
                          double_bytes: int = 8) -> Optional[int]:
    """Largest ``max_leaf_nodes`` whose trees fit `flash_budget` bytes.

    Finer than :func:`max_depth_for_budget`: a leaf limit can take any value,
    so the model fills the budget instead of stopping at a power of two.
    None when not even two leaves per tree fit.
    """
    trees = n_trees(algorithm, n_estimators, n_classes)
    node = node_bytes(algorithm, n_classes, missing_routing, double_bytes)
    nodes = int((flash_budget / trees - PER_TREE_BYTES) // node)
    leaves = (nodes + 1) // 2
    return leaves if leaves >= 2 else None


@dataclass
class BoardFit:
    """Result of :func:`fit_to_board`.

    Attributes:
        params: Estimator parameters of the best candidate (pass them to the
            scikit-learn estimator together with your other settings).
        candidates: One dict per candidate: ``n_estimators``,
            ``max_leaf_nodes``, the guaranteed sketch ``flash`` and ``ram``,
            the validation ``score`` and ``fit_s``.
        board: The target.
        model_budget: Bytes left for the model.
        scoring: The scikit-learn scorer used.
    """
    params: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    board: Board
    model_budget: int
    scoring: str

    def to_frame(self):
        """The candidates as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.candidates)

    def __str__(self) -> str:
        cols = ['n_estimators', 'max_leaf_nodes', 'flash', 'ram', 'score', 'fit_s']
        width = {c: max(len(c), *(len(f'{r.get(c, "")}') for r in self.candidates))
                 for c in cols}
        lines = ['  '.join(c.rjust(width[c]) for c in cols)]
        lines += ['  '.join(f'{r.get(c, "")}'.rjust(width[c]) for c in cols)
                  for r in self.candidates]
        lines.append(f'best ({self.scoring}): {self.params}')
        return '\n'.join(lines)


def fit_to_board(X, y, *, board: Optional[Any] = None,
                 flash_bytes: Optional[int] = None, ram_bytes: Optional[int] = None,
                 firmware_flash: Optional[int] = None, firmware_ram: Optional[int] = None,
                 algorithm: str = 'gbdt',
                 n_estimators: Sequence[int] = (25, 50, 100, 200),
                 X_val=None, y_val=None, sample_rows: Optional[int] = 50_000,
                 val_rows: Optional[int] = 20_000, scoring: str = 'f1_macro',
                 random_state: Any = 0, missing_routing: bool = False,
                 **params) -> BoardFit:
    """The size parameters that make the best model a board can hold.

    For each number of trees in `n_estimators` the leaves per tree are set
    to the largest ``max_leaf_nodes`` whose worst case fits the board
    (:func:`max_leaves_for_budget`), so every candidate fills the memory and
    any model trained with its parameters is guaranteed to fit. Candidates
    are then compared by `scoring` on validation data: more, smaller trees
    or fewer, larger ones is a question of accuracy, not of size. On boards
    that copy the model to RAM (AVR) the RAM bounds the model too.

    Args:
        X, y: Training data. Candidates are fitted on `sample_rows` rows
            drawn from it (None: all rows).
        board, flash_bytes, ram_bytes, firmware_flash, firmware_ram: The
            target, a preset of :data:`BOARDS` and/or custom memory sizes.
        algorithm: 'gbdt' or 'rf' ('dt' has a single candidate).
        n_estimators: Candidate numbers of trees (boosting stages).
        X_val, y_val: Validation data; when omitted, `val_rows` rows of `X`
            not used for fitting.
        scoring: A scikit-learn scorer name ('f1_macro', 'accuracy', ...).
        random_state: Seed of the sampling and of the estimators.
        missing_routing: See :func:`node_bytes`.
        **params: Further estimator parameters (``learning_rate``, ...).

    Returns:
        BoardFit: ``params`` holds ``n_estimators`` (not for 'dt'),
        ``max_leaf_nodes`` and ``max_depth=None``.
    """
    import time
    from sklearn.metrics import get_scorer

    algo = _algorithm(algorithm)
    target = make_board(board, flash_bytes=flash_bytes, ram_bytes=ram_bytes,
                        firmware_flash=firmware_flash, firmware_ram=firmware_ram)
    if target is None:
        raise ValueError('fit_to_board needs a board or flash_bytes and ram_bytes')
    X = np.asarray(X)
    y = np.asarray(y)
    n_classes = int(len(np.unique(y)))
    ram_fixed = target.firmware_ram + runtime_ram_bytes(X.shape[1], n_classes,
                                                        target.double_bytes)
    budget = target.flash_bytes - target.firmware_flash
    if target.constants_in_ram:
        budget = min(budget, target.ram_bytes - ram_fixed)

    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(X))
    n_fit = len(X) if sample_rows is None else min(int(sample_rows), len(X))
    fit_idx = order[:n_fit]
    if X_val is None:
        rest = order[n_fit:]
        if len(rest) == 0:  # small data: hold out a fifth of it
            cut = max(1, len(order) // 5)
            fit_idx, rest = order[cut:], order[:cut]
        val_idx = rest if val_rows is None else rest[:int(val_rows)]
        X_val, y_val = X[val_idx], y[val_idx]
    Xf, yf = X[fit_idx], y[fit_idx]
    scorer = get_scorer(scoring)

    grid = [1] if algo == 'dt' else sorted(set(int(n) for n in n_estimators))
    cands: List[Dict[str, Any]] = []
    for n in grid:
        leaves = max_leaves_for_budget(budget, algo, n, n_classes, missing_routing,
                                       target.double_bytes)
        if leaves is None:
            continue
        ub = estimate_footprint(algo, n, n_classes, max_leaf_nodes=leaves,
                                missing_routing=missing_routing,
                                double_bytes=target.double_bytes)
        model_params = dict(params, max_leaf_nodes=leaves, max_depth=None)
        if algo != 'dt':
            model_params['n_estimators'] = n
        est = _make_estimator(algo, model_params.pop('n_estimators', 1),
                              model_params.pop('max_depth'), random_state, model_params)
        t0 = time.perf_counter()
        est.fit(Xf, yf)
        row = {'n_estimators': n if algo != 'dt' else None, 'max_leaf_nodes': leaves,
               'model_bytes': ub['model_bytes'],
               'flash': target.firmware_flash + ub['model_bytes'],
               'ram': ram_fixed + (ub['model_bytes'] if target.constants_in_ram else 0),
               'score': round(float(scorer(est, X_val, y_val)), 6),
               'fit_s': round(time.perf_counter() - t0, 1)}
        cands.append(row)
    if not cands:
        raise ValueError('no candidate fits the board: fewer trees or a larger board')
    best = max(cands, key=lambda r: (r['score'], -r['model_bytes']))
    best_params = {'max_leaf_nodes': best['max_leaf_nodes'], 'max_depth': None}
    if algo != 'dt':
        best_params = {'n_estimators': best['n_estimators'], **best_params}
    return BoardFit(params=best_params, candidates=cands, board=target,
                    model_budget=int(budget), scoring=scoring)
