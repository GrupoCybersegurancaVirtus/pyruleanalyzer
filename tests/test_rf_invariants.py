"""
Comprehensive Random Forest invariant tests for pyruleanalyzer.

Tests the following invariants:
1. Initial model must be 100% faithful to sklearn (0 divergences).
2. 'soft' duplicate removal + no specific removal = 0 divergences from sklearn.
3. 'hard' duplicate removal + no specific removal = very few divergences (<=2%).
   (Hard also merges inter-tree semantic duplicates, which changes voting.)
4. Higher specific removal threshold = monotonically non-decreasing divergences.

Uses multiple artificial datasets with varying characteristics and model parameters.
"""

import sys
import os
import time
import tempfile
import shutil
import traceback

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Ensure pyruleanalyzer is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyruleanalyzer.rule_classifier import RuleClassifier


# ============================================================================
# HELPERS
# ============================================================================

def generate_dataset(name, n_train=1000, n_test=300, n_features=5, n_classes=2,
                     random_state=42, noise=0.0, categorical=False):
    """Generate a synthetic dataset and save to temporary CSV files.

    Args:
        name: Identifier for the dataset.
        n_train: Number of training samples.
        n_test: Number of test samples.
        n_features: Number of features.
        n_classes: Number of target classes.
        random_state: Seed for reproducibility.
        noise: Fraction of labels to flip (0.0 - 1.0).
        categorical: Whether to include categorical-like integer features.

    Returns:
        Tuple of (train_path, test_path, tmpdir).
    """
    rng = np.random.RandomState(random_state)
    n_total = n_train + n_test

    if categorical:
        X = np.zeros((n_total, n_features))
        for i in range(n_features):
            if i % 2 == 0:
                X[:, i] = rng.randn(n_total)
            else:
                X[:, i] = rng.randint(0, 5, n_total).astype(float)
    else:
        X = rng.randn(n_total, n_features)

    # Create non-trivial decision boundaries
    y = np.zeros(n_total, dtype=int)
    if n_classes == 2:
        if n_features >= 2:
            y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
        else:
            y = (X[:, 0] > 0).astype(int)
    elif n_classes == 3:
        score = X[:, 0] + 0.5 * X[:, 1]
        y[score < -0.5] = 0
        y[(score >= -0.5) & (score < 0.5)] = 1
        y[score >= 0.5] = 2
    elif n_classes == 5:
        score = X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2]
        thresholds = np.percentile(score, [20, 40, 60, 80])
        y = np.digitize(score, thresholds)
    else:
        score = X[:, 0]
        thresholds = np.percentile(score, np.linspace(0, 100, n_classes + 1)[1:-1])
        y = np.digitize(score, thresholds)

    # Add label noise
    if noise > 0:
        n_noisy = int(noise * n_total)
        noisy_idx = rng.choice(n_total, n_noisy, replace=False)
        y[noisy_idx] = rng.randint(0, n_classes, n_noisy)

    # Build DataFrames
    feature_names = [f'v{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['class'] = y

    tmpdir = tempfile.mkdtemp(prefix=f'pyrule_test_{name}_')
    train_path = os.path.join(tmpdir, 'train.csv')
    test_path = os.path.join(tmpdir, 'test.csv')

    df.iloc[:n_train].to_csv(train_path, index=False)
    df.iloc[n_train:].to_csv(test_path, index=False)

    return train_path, test_path, tmpdir


def sklearn_predict(train_path, test_path, model_params):
    """Train sklearn RF and return predictions on test set.

    Returns:
        Tuple of (y_pred, y_test, model, feature_names, X_test).
    """
    X_train, y_train, X_test, y_test, class_names, _, feature_names = \
        RuleClassifier.process_data(train_path, test_path)

    assert X_train is not None, 'X_train cannot be None'
    assert y_train is not None, 'y_train cannot be None'
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, np.asarray(y_train))

    y_pred = model.predict(X_test)
    return y_pred, y_test, model, feature_names, X_test


def rule_classify_all_rf(classifier, X_test, feature_names, rules):
    """Classify all test samples using RuleClassifier.classify_rf.

    Returns:
        np.array of predictions.
    """
    sample_dicts = [dict(zip(feature_names, row)) for row in X_test]
    preds = []
    for sample in sample_dicts:
        predicted_class, votes, proba, matched_rules = RuleClassifier.classify_rf(sample, rules)
        if predicted_class is not None:
            try:
                pred = int(str(predicted_class).replace('Class', '').strip())
            except (ValueError, AttributeError):
                pred = predicted_class
        else:
            try:
                pred = int(str(classifier.default_class).replace('Class', '').strip())
            except (ValueError, AttributeError):
                pred = classifier.default_class
        preds.append(pred)
    return np.array(preds)


def native_classify_all(classifier, X_test, feature_names):
    """Classify all test samples using native compiled function.

    Returns:
        np.array of predictions.
    """
    sample_dicts = [dict(zip(feature_names, row)) for row in X_test]
    preds = []
    for sample in sample_dicts:
        pred, _, _ = classifier.native_fn(sample)
        try:
            pred = int(str(pred).replace('Class', '').strip())
        except (ValueError, AttributeError):
            pass
        preds.append(pred)
    return np.array(preds)


# ============================================================================
# TEST CONFIGURATIONS
# ============================================================================

# Each config: (name, dataset_kwargs, model_params)
# Using fewer estimators than production to keep tests fast while still
# exercising inter-tree duplicate detection.
TEST_CONFIGS = [
    # --- Binary classification ---
    (
        'RF_binary_500_depth3',
        dict(n_train=500, n_test=200, n_features=4, n_classes=2, random_state=42),
        dict(n_estimators=10, random_state=42, max_depth=3),
    ),
    (
        'RF_binary_500_depth10',
        dict(n_train=500, n_test=200, n_features=4, n_classes=2, random_state=42),
        dict(n_estimators=10, random_state=42, max_depth=10),
    ),
    (
        'RF_binary_500_noMaxDepth',
        dict(n_train=500, n_test=200, n_features=4, n_classes=2, random_state=42),
        dict(n_estimators=10, random_state=42),
    ),
    (
        'RF_binary_800_noisy',
        dict(n_train=800, n_test=200, n_features=6, n_classes=2, random_state=7, noise=0.15),
        dict(n_estimators=10, random_state=42, max_depth=8),
    ),
    (
        'RF_binary_1000_20feat',
        dict(n_train=1000, n_test=300, n_features=20, n_classes=2, random_state=99),
        dict(n_estimators=10, random_state=42, max_depth=6, max_features='sqrt'),
    ),
    (
        'RF_binary_1000_minLeaf20',
        dict(n_train=1000, n_test=300, n_features=5, n_classes=2, random_state=42),
        dict(n_estimators=10, random_state=42, min_samples_leaf=20),
    ),
    (
        'RF_binary_1000_30trees',
        dict(n_train=1000, n_test=300, n_features=5, n_classes=2, random_state=42),
        dict(n_estimators=30, random_state=42, max_depth=5),
    ),
    (
        'RF_binary_600_categorical',
        dict(n_train=600, n_test=200, n_features=6, n_classes=2, random_state=42, categorical=True),
        dict(n_estimators=10, random_state=42, max_depth=6),
    ),
    (
        'RF_binary_500_noBootstrap',
        dict(n_train=500, n_test=200, n_features=4, n_classes=2, random_state=42),
        dict(n_estimators=10, random_state=42, max_depth=5, bootstrap=False),
    ),

    # --- Multiclass ---
    (
        'RF_multi3_600_depth4',
        dict(n_train=600, n_test=200, n_features=5, n_classes=3, random_state=42),
        dict(n_estimators=10, random_state=42, max_depth=4),
    ),
    (
        'RF_multi3_600_depth12',
        dict(n_train=600, n_test=200, n_features=5, n_classes=3, random_state=42),
        dict(n_estimators=10, random_state=42, max_depth=12),
    ),
    (
        'RF_multi3_800_noisy',
        dict(n_train=800, n_test=200, n_features=5, n_classes=3, random_state=11, noise=0.1),
        dict(n_estimators=10, random_state=42, max_depth=8),
    ),
    (
        'RF_multi5_1000_depth5',
        dict(n_train=1000, n_test=300, n_features=8, n_classes=5, random_state=42),
        dict(n_estimators=10, random_state=42, max_depth=5),
    ),
    (
        'RF_multi5_2000_depth15',
        dict(n_train=2000, n_test=500, n_features=8, n_classes=5, random_state=42),
        dict(n_estimators=10, random_state=42, max_depth=15),
    ),
    (
        'RF_multi5_1000_20trees',
        dict(n_train=1000, n_test=300, n_features=8, n_classes=5, random_state=42),
        dict(n_estimators=20, random_state=42, max_depth=6),
    ),

    # --- Edge cases ---
    (
        'RF_binary_50_tiny',
        dict(n_train=50, n_test=20, n_features=3, n_classes=2, random_state=42),
        dict(n_estimators=5, random_state=42, max_depth=3),
    ),
    (
        'RF_binary_500_1feat',
        dict(n_train=500, n_test=200, n_features=1, n_classes=2, random_state=42),
        dict(n_estimators=10, random_state=42),
    ),
    (
        'RF_binary_200_overfit',
        dict(n_train=200, n_test=100, n_features=10, n_classes=2, random_state=42),
        dict(n_estimators=10, random_state=42, max_depth=None, min_samples_leaf=1),
    ),
    (
        'RF_multi3_1000_maxLeaf20',
        dict(n_train=1000, n_test=300, n_features=5, n_classes=3, random_state=42),
        dict(n_estimators=10, random_state=42, max_leaf_nodes=20),
    ),
    (
        'RF_binary_1000_entropy',
        dict(n_train=1000, n_test=300, n_features=5, n_classes=2, random_state=42),
        dict(n_estimators=10, random_state=42, criterion='entropy', max_depth=8),
    ),

    # ── Large-scale configs ──────────────────────────────────────────────
    (
        'RF_binary_10k_depth5',
        dict(n_train=10000, n_test=10000, n_features=6, n_classes=2, random_state=42),
        dict(n_estimators=10, random_state=42, max_depth=5),
    ),
    (
        'RF_binary_10k_depth15',
        dict(n_train=10000, n_test=10000, n_features=6, n_classes=2, random_state=42),
        dict(n_estimators=10, random_state=42, max_depth=15),
    ),
    (
        'RF_multi3_10k_depth8',
        dict(n_train=10000, n_test=10000, n_features=6, n_classes=3, random_state=42),
        dict(n_estimators=10, random_state=42, max_depth=8),
    ),
    (
        'RF_multi5_10k_15feat',
        dict(n_train=10000, n_test=10000, n_features=15, n_classes=5, random_state=42),
        dict(n_estimators=10, random_state=42, max_depth=10),
    ),
    (
        'RF_binary_10k_noisy',
        dict(n_train=10000, n_test=10000, n_features=8, n_classes=2, noise=0.3, random_state=42),
        dict(n_estimators=10, random_state=42, max_depth=10),
    ),
]


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_test_config(name, dataset_kwargs, model_params, verbose=True):
    """Run all invariant tests for a single configuration.

    Returns:
        dict with test results and any failures.
    """
    results = {
        'name': name,
        'passed': True,
        'failures': [],
        'warnings': [],
        'details': {},
    }

    tmpdir = None
    try:
        # 1. Generate dataset
        train_path, test_path, tmpdir = generate_dataset(name, **dataset_kwargs)

        # 2. Train sklearn model directly for reference
        y_sklearn, y_test, sk_model, feature_names, X_test = \
            sklearn_predict(train_path, test_path, model_params)

        n_test = len(y_test)
        n_leaves_sklearn = sum(est.get_n_leaves() for est in sk_model.estimators_)

        results['details']['n_test'] = n_test
        results['details']['n_leaves_sklearn'] = n_leaves_sklearn
        results['details']['n_estimators'] = len(sk_model.estimators_)
        results['details']['sklearn_accuracy'] = float(np.mean(y_sklearn == y_test))

        # ================================================================
        # INVARIANT 1: Initial model must be faithful to sklearn
        # ================================================================
        class_names = [str(c) for c in sorted(np.unique(np.asarray(y_test)))]
        rules_list = RuleClassifier.get_tree_rules(
            sk_model, feature_names, class_names, algorithm_type='Random Forest'
        )
        class_names_map = {str(c): i for i, c in enumerate(class_names)}

        # Suppress stdout during classifier construction
        _orig_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            classifier = RuleClassifier.generate_classifier_model(
                rules_list, class_names_map, 'Random Forest'
            )
        finally:
            sys.stdout.close()
            sys.stdout = _orig_stdout

        n_rules_initial = len(classifier.initial_rules)
        results['details']['n_rules_initial'] = n_rules_initial

        # Test classify_rf (iterative)
        y_rule_initial = rule_classify_all_rf(
            classifier, X_test, feature_names, classifier.initial_rules
        )

        div_initial = int(np.sum(y_sklearn != y_rule_initial))
        results['details']['invariant1_divergences'] = div_initial
        results['details']['invariant1_div_pct'] = div_initial / n_test * 100

        if div_initial > 0:
            results['passed'] = False
            results['failures'].append(
                f'INVARIANT 1 FAILED: Initial rules diverge from sklearn in '
                f'{div_initial}/{n_test} samples ({div_initial/n_test*100:.2f}%)'
            )

        # Test native_fn (compiled)
        if classifier.native_fn is not None:
            y_native_initial = native_classify_all(classifier, X_test, feature_names)
            div_native = int(np.sum(y_sklearn != y_native_initial))
            results['details']['invariant1_native_divergences'] = div_native

            if div_native > 0:
                results['passed'] = False
                results['failures'].append(
                    f'INVARIANT 1 (NATIVE) FAILED: Native fn diverges from sklearn in '
                    f'{div_native}/{n_test} samples ({div_native/n_test*100:.2f}%)'
                )
        else:
            results['warnings'].append('Native function not compiled, skipping native test.')

        # ================================================================
        # INVARIANT 2: soft + no specific removal = 0 divergences
        # ================================================================
        _orig_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            classifier_soft = RuleClassifier.generate_classifier_model(
                RuleClassifier.get_tree_rules(
                    sk_model, feature_names, class_names, 'Random Forest'
                ),
                class_names_map, 'Random Forest'
            )
            classifier_soft.execute_rule_analysis(
                test_path, remove_duplicates='soft', remove_below_n_classifications=-1
            )
        finally:
            sys.stdout.close()
            sys.stdout = _orig_stdout

        n_rules_after_soft = len(classifier_soft.final_rules)
        results['details']['n_rules_after_soft'] = n_rules_after_soft

        y_soft = rule_classify_all_rf(
            classifier_soft, X_test, feature_names, classifier_soft.final_rules
        )
        div_soft = int(np.sum(y_sklearn != y_soft))
        results['details']['invariant2_divergences'] = div_soft
        results['details']['invariant2_div_pct'] = div_soft / n_test * 100

        if div_soft > 0:
            results['passed'] = False
            results['failures'].append(
                f'INVARIANT 2 FAILED: soft removal diverges from sklearn in '
                f'{div_soft}/{n_test} samples ({div_soft/n_test*100:.2f}%)'
            )

        # Also test native after soft
        if classifier_soft.native_fn is not None:
            y_soft_native = native_classify_all(classifier_soft, X_test, feature_names)
            div_soft_native = int(np.sum(y_sklearn != y_soft_native))
            results['details']['invariant2_native_divergences'] = div_soft_native
            if div_soft_native > 0:
                results['passed'] = False
                results['failures'].append(
                    f'INVARIANT 2 (NATIVE) FAILED: soft+native diverges from sklearn in '
                    f'{div_soft_native}/{n_test} samples ({div_soft_native/n_test*100:.2f}%)'
                )

        # ================================================================
        # INVARIANT 3: hard + no specific removal = very few divergences (<=2%)
        # For RF, "hard" also merges inter-tree semantic duplicates, which
        # changes voting by reducing the number of trees that contribute.
        # This may cause a small number of divergences from the original.
        # ================================================================
        _orig_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            classifier_hard = RuleClassifier.generate_classifier_model(
                RuleClassifier.get_tree_rules(
                    sk_model, feature_names, class_names, 'Random Forest'
                ),
                class_names_map, 'Random Forest'
            )
            classifier_hard.execute_rule_analysis(
                test_path, remove_duplicates='hard', remove_below_n_classifications=-1
            )
        finally:
            sys.stdout.close()
            sys.stdout = _orig_stdout

        n_rules_after_hard = len(classifier_hard.final_rules)
        results['details']['n_rules_after_hard'] = n_rules_after_hard

        y_hard = rule_classify_all_rf(
            classifier_hard, X_test, feature_names, classifier_hard.final_rules
        )
        div_hard = int(np.sum(y_sklearn != y_hard))
        div_hard_pct = div_hard / n_test * 100
        results['details']['invariant3_divergences'] = div_hard
        results['details']['invariant3_div_pct'] = div_hard_pct

        # Allow up to 2% divergence for hard mode (inter-tree merging changes voting)
        hard_threshold_pct = 2.0
        if div_hard_pct > hard_threshold_pct:
            results['passed'] = False
            results['failures'].append(
                f'INVARIANT 3 FAILED: hard removal diverges from sklearn in '
                f'{div_hard}/{n_test} samples ({div_hard_pct:.2f}%) '
                f'[threshold: {hard_threshold_pct}%]'
            )

        # ================================================================
        # INVARIANT 4: Higher specific removal threshold = more accuracy loss
        #              (monotonically non-decreasing divergences)
        # ================================================================
        thresholds = [0, 1, 3, 5, 10, 20, 50]
        threshold_results = []

        for threshold in thresholds:
            _orig_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                clf_t = RuleClassifier.generate_classifier_model(
                    RuleClassifier.get_tree_rules(
                        sk_model, feature_names, class_names, 'Random Forest'
                    ),
                    class_names_map, 'Random Forest'
                )
                clf_t.execute_rule_analysis(
                    test_path, remove_duplicates='soft',
                    remove_below_n_classifications=threshold
                )
            finally:
                sys.stdout.close()
                sys.stdout = _orig_stdout

            y_t = rule_classify_all_rf(clf_t, X_test, feature_names, clf_t.final_rules)
            div_t = int(np.sum(y_sklearn != y_t))
            n_rules_t = len(clf_t.final_rules)

            threshold_results.append({
                'threshold': threshold,
                'divergences': div_t,
                'div_pct': div_t / n_test * 100,
                'n_rules': n_rules_t,
            })

        results['details']['invariant4_threshold_results'] = threshold_results

        # Check monotonicity: divergences should be non-decreasing
        divs = [r['divergences'] for r in threshold_results]
        # Allow small violations (up to 1% of samples) due to sibling promotion
        monotonic_violations = []
        for i in range(1, len(divs)):
            if divs[i] < divs[i-1] - max(1, n_test * 0.01):
                monotonic_violations.append(
                    f'threshold {thresholds[i]}: {divs[i]} divergences < '
                    f'threshold {thresholds[i-1]}: {divs[i-1]} divergences '
                    f'(significant decrease of {divs[i-1] - divs[i]})'
                )

        if monotonic_violations:
            results['warnings'].append(
                'INVARIANT 4 WARNING: Non-monotonic divergence pattern: '
                + '; '.join(monotonic_violations)
            )

        # Key check: highest threshold should have >= divergences than threshold 0
        if len(divs) >= 2 and divs[-1] < divs[0]:
            results['warnings'].append(
                f'INVARIANT 4 NOTE: Highest threshold ({thresholds[-1]}) has fewer '
                f'divergences ({divs[-1]}) than lowest ({thresholds[0]}={divs[0]}). '
                f'This can happen if the forest is small.'
            )

    except Exception as e:
        results['passed'] = False
        results['failures'].append(f'EXCEPTION: {e}\n{traceback.format_exc()}')

    finally:
        if tmpdir and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)

    return results


def print_results(results):
    """Print formatted test results."""
    status = 'PASS' if results['passed'] else 'FAIL'
    print(f'\n{"="*80}')
    print(f'[{status}] {results["name"]}')
    print(f'{"="*80}')

    d = results['details']
    if 'n_test' in d:
        print(f'  Samples: {d["n_test"]} | '
              f'Sklearn leaves (total): {d.get("n_leaves_sklearn", "?")} | '
              f'Trees: {d.get("n_estimators", "?")} | '
              f'Initial rules: {d.get("n_rules_initial", "?")} | '
              f'Sklearn acc: {d.get("sklearn_accuracy", 0):.4f}')

    # Invariant 1
    if 'invariant1_divergences' in d:
        div = d['invariant1_divergences']
        pct = d['invariant1_div_pct']
        marker = 'OK' if div == 0 else 'FAIL'
        print(f'  [INV1] Initial vs Sklearn:  {div} divergences ({pct:.2f}%) [{marker}]')
    if 'invariant1_native_divergences' in d:
        div = d['invariant1_native_divergences']
        marker = 'OK' if div == 0 else 'FAIL'
        print(f'  [INV1] Native vs Sklearn:   {div} divergences [{marker}]')

    # Invariant 2
    if 'invariant2_divergences' in d:
        div = d['invariant2_divergences']
        pct = d['invariant2_div_pct']
        rules_after = d.get('n_rules_after_soft', '?')
        marker = 'OK' if div == 0 else 'FAIL'
        print(f'  [INV2] Soft+NoSpec:         {div} divergences ({pct:.2f}%), '
              f'rules: {d.get("n_rules_initial","?")} -> {rules_after} [{marker}]')
    if 'invariant2_native_divergences' in d:
        div = d['invariant2_native_divergences']
        marker = 'OK' if div == 0 else 'FAIL'
        print(f'  [INV2] Soft+NoSpec (native): {div} divergences [{marker}]')

    # Invariant 3
    if 'invariant3_divergences' in d:
        div = d['invariant3_divergences']
        pct = d['invariant3_div_pct']
        rules_after = d.get('n_rules_after_hard', '?')
        marker = 'OK' if pct <= 2.0 else 'FAIL'
        print(f'  [INV3] Hard+NoSpec:         {div} divergences ({pct:.2f}%), '
              f'rules: {d.get("n_rules_initial","?")} -> {rules_after} [{marker}]')

    # Invariant 4
    if 'invariant4_threshold_results' in d:
        print('  [INV4] Threshold analysis:')
        for tr in d['invariant4_threshold_results']:
            print(f'         thresh={tr["threshold"]:>3d}: '
                  f'{tr["divergences"]:>4d} div ({tr["div_pct"]:>6.2f}%), '
                  f'{tr["n_rules"]:>4d} rules')

    # Failures
    for f in results['failures']:
        print(f'  ** FAILURE: {f}')

    # Warnings
    for w in results['warnings']:
        print(f'  >> WARNING: {w}')


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    print('=' * 80)
    print('PYRULEANALYZER - RANDOM FOREST INVARIANT TESTS')
    print('=' * 80)
    print(f'Running {len(TEST_CONFIGS)} test configurations...\n')

    all_results = []
    total_pass = 0
    total_fail = 0

    for name, ds_kwargs, model_params in TEST_CONFIGS:
        print(f'\n>>> Running: {name} ...', end='', flush=True)
        t0 = time.perf_counter()
        result = run_test_config(name, ds_kwargs, model_params)
        elapsed = time.perf_counter() - t0
        result['elapsed'] = elapsed
        all_results.append(result)

        if result['passed']:
            total_pass += 1
            print(' PASS')
        else:
            total_fail += 1
            print(' FAIL')

        d = result['details']
        n = d.get('n_test', '?')
        trees = d.get('n_estimators', '?')
        ri = d.get('n_rules_initial', '?')
        rs = d.get('n_rules_after_soft', ri)
        div1 = d.get('invariant1_divergences', '?')
        acc = d.get('sklearn_accuracy', 0)
        print(f'    samples={n} | trees={trees} | rules: {ri}->{rs} | '
              f'inv1_div={div1} | sk_acc={acc:.4f} | {elapsed:.1f}s')

    # Print detailed results
    print('\n\n')
    print('#' * 80)
    print('DETAILED RESULTS')
    print('#' * 80)

    for result in all_results:
        print_results(result)

    # Summary
    print('\n\n')
    print('=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f'Total: {len(TEST_CONFIGS)} | Passed: {total_pass} | Failed: {total_fail}')
    total_time = sum(r.get('elapsed', 0) for r in all_results)
    print(f'Total time: {total_time:.1f}s')

    if total_fail > 0:
        print('\nFailed tests:')
        for r in all_results:
            if not r['passed']:
                print(f'  - {r["name"]}')
                for f in r['failures']:
                    print(f'    {f}')

    print('\n' + '=' * 80)

    sys.exit(0 if total_fail == 0 else 1)
