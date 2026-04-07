"""
Benchmark: Sklearn vs pyRuleAnalyzer using ARTIFICIAL (synthetic) datasets.

This script generates synthetic classification datasets with configurable
parameters and runs the full benchmark pipeline for any algorithm type
(Decision Tree, Random Forest, or Gradient Boosting Decision Trees).

Usage:
    python examples/sklearn_vs_ruleclassifier_synthetic.py

Configure the parameters in the CONFIGURATION section below.
"""

import sys
import os
import time
import pickle
import tempfile
import importlib

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Add parent directory to path to find pyruleanalyzer package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyruleanalyzer import PyRuleAnalyzer, RuleClassifier
from pyruleanalyzer._accel import HAS_C_EXTENSION

# ==============================================================================
# CONFIGURATION - SYNTHETIC DATASET PARAMETERS
# ==============================================================================

# --- Dataset Parameters ---
DATASET_PARAMS = {
    'n_samples': 100000,          # Total number of samples (train + test)
    'n_features': 20,            # Total number of features
    'n_informative': 10,         # Number of informative features
    'n_redundant': 5,            # Number of redundant features (linear combinations)
    'n_classes': 4,              # Number of target classes
    'n_clusters_per_class': 1,   # Number of clusters per class
    'class_sep': 1.0,            # Class separation (higher = easier problem)
    'flip_y': 0.01,             # Fraction of labels randomly flipped (noise)
    'random_state': 42,          # Reproducibility seed
}

# Train/test split ratio (fraction used for training)
TRAIN_RATIO = 0.7

# --- Algorithm ---
# Options: 'Decision Tree', 'Random Forest', 'Gradient Boosting Decision Trees'
# ALGORITHM_TYPE = 'Decision Tree'
# ALGORITHM_TYPE = 'Random Forest'
ALGORITHM_TYPE = 'Random Forest'

# --- Model Hyperparameters (passed to sklearn) ---
MODEL_PARAMS = {
    'random_state': 42,
    # 'max_depth': 10,
    # 'n_estimators': 100,       # For RF and GBDT
    # 'learning_rate': 0.1,      # For GBDT
}

# --- Rule Analysis Parameters ---
REMOVE_LOW_USAGE = 1  # -1 = no usage pruning, 0+ = threshold

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def generate_synthetic_dataset(params: dict, train_ratio: float) -> tuple:
    """
    Generates a synthetic classification dataset and saves it as temporary CSVs.

    Args:
        params (dict): Parameters for sklearn.datasets.make_classification.
        train_ratio (float): Fraction of data used for training.

    Returns:
        Tuple of (train_csv_path, test_csv_path, temp_dir).
        The caller is responsible for cleanup of temp_dir.
    """
    print("\n" + "=" * 100)
    print("SYNTHETIC DATASET GENERATION")
    print("=" * 100)

    X, y = make_classification(
        n_samples=params['n_samples'],
        n_features=params['n_features'],
        n_informative=params['n_informative'],
        n_redundant=params['n_redundant'],
        n_classes=params['n_classes'],
        n_clusters_per_class=params['n_clusters_per_class'],
        class_sep=params['class_sep'],
        flip_y=params['flip_y'],
        random_state=params['random_state'],
    )

    # Create feature names
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    # Build DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['class'] = y

    # Split into train/test
    n_train = int(len(df) * train_ratio)
    # Shuffle before splitting (already shuffled by make_classification, but explicit)
    df = df.sample(frac=1, random_state=params['random_state']).reset_index(drop=True)
    df_train = df.iloc[:n_train]
    df_test = df.iloc[n_train:]

    # Print dataset summary
    COL1 = 35
    print(f"\n{'PARAMETER':<{COL1}} | VALUE")
    print("-" * 60)
    print(f"{'Total samples':<{COL1}} | {params['n_samples']}")
    print(f"{'Train samples':<{COL1}} | {len(df_train)}")
    print(f"{'Test samples':<{COL1}} | {len(df_test)}")
    print(f"{'Features (total)':<{COL1}} | {params['n_features']}")
    print(f"{'Features (informative)':<{COL1}} | {params['n_informative']}")
    print(f"{'Features (redundant)':<{COL1}} | {params['n_redundant']}")
    print(f"{'Classes':<{COL1}} | {params['n_classes']}")
    print(f"{'Clusters per class':<{COL1}} | {params['n_clusters_per_class']}")
    print(f"{'Class separation':<{COL1}} | {params['class_sep']}")
    print(f"{'Label noise (flip_y)':<{COL1}} | {params['flip_y']}")
    print(f"{'Random state':<{COL1}} | {params['random_state']}")

    # Class distribution
    print(f"\n{'CLASS DISTRIBUTION (train)':<{COL1}}")
    print("-" * 40)
    for cls in sorted(df_train['class'].unique()):
        count = (df_train['class'] == cls).sum()
        pct = count / len(df_train) * 100
        print(f"  Class {cls}: {count} ({pct:.1f}%)")

    # Save to temporary CSV files
    temp_dir = tempfile.mkdtemp(prefix='pyruleanalyzer_synthetic_')
    train_path = os.path.join(temp_dir, 'train.csv')
    test_path = os.path.join(temp_dir, 'test.csv')

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"\nTemp files: {temp_dir}")

    return train_path, test_path, temp_dir


def count_leaves_in_file(filename: str) -> int:
    """Count how many 'return' exist in the generated file (proxy for leaf count)."""
    if not os.path.exists(filename):
        return 0
    with open(filename, 'r') as f:
        return f.read().count('return ')


def safe_speed(n: int, t: float) -> str:
    """Avoid division by zero in speed calculation."""
    if t <= 0:
        return "Inf"
    return f"{n / t:.0f}"


def count_sklearn_leaves(sk_model, algorithm_type: str):
    """Count total leaves in a sklearn model."""
    if algorithm_type == 'Decision Tree':
        if hasattr(sk_model, 'get_n_leaves'):
            return sk_model.get_n_leaves()
    elif algorithm_type == 'Random Forest':
        if hasattr(sk_model, 'estimators_'):
            return sum(tree.get_n_leaves() for tree in sk_model.estimators_)
    elif algorithm_type == 'Gradient Boosting Decision Trees':
        if hasattr(sk_model, 'estimators_'):
            n_estimators, n_cols = sk_model.estimators_.shape
            total = sum(
                sk_model.estimators_[i, j].get_n_leaves()
                for i in range(n_estimators)
                for j in range(n_cols)
            )
            n_classes = len(sk_model.classes_)
            total += n_classes if not (n_classes == 2 and n_cols == 1) else 1
            return total
    return "N/A"


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    # Ensure output directory exists
    os.makedirs('examples/files', exist_ok=True)

    # ======================================================================
    # 1. GENERATE SYNTHETIC DATASET
    # ======================================================================
    train_path, test_path, temp_dir = generate_synthetic_dataset(
        DATASET_PARAMS, TRAIN_RATIO
    )

    try:
        # ==================================================================
        # 2. TRAINING AND REFINEMENT
        # ==================================================================
        # Create analyzer using the new factory method
        analyzer = PyRuleAnalyzer.create(
            train_path=train_path,
            test_path=test_path,
            model=ALGORITHM_TYPE,
            params=MODEL_PARAMS,
            refine=True,  # Automatically refine rules after creation
            refine_params={'remove_low_usage': REMOVE_LOW_USAGE}
        )

        # Get classifier for further operations
        classifier = analyzer.classifier

        classifier.compare_initial_final_results(test_path)

        # ==================================================================
        # 3. EXPORT
        # ==================================================================
        X_train, _, X_test, y_test, _, _, feature_names = RuleClassifier.process_data(
            train_path, test_path
        )
        sample_dicts = pd.DataFrame(X_test, columns=feature_names).to_dict('records')

        # Determine export filename based on algorithm
        algo_prefix = {
            'Decision Tree': 'dt',
            'Random Forest': 'rf',
            'Gradient Boosting Decision Trees': 'gbdt',
        }[ALGORITHM_TYPE]

        export_file = f"files/{algo_prefix}_classifier.py"
        classifier.export_all(
            base_name=f"files/{algo_prefix}_model",
            feature_names=feature_names,
            export_python=True,
            export_binary=True,
            export_c=True
        )

        # Compile tree arrays for predict_batch
        classifier.compile_tree_arrays(feature_names=feature_names)

        # ==================================================================
        # 4. BENCHMARK AND PERFORMANCE VALIDATION
        # ==================================================================
        print("\n" + "=" * 100)
        print(f"PERFORMANCE COMPARISON REPORT: SKLEARN vs PYRULEANALYZER ({ALGORITHM_TYPE.upper()})")
        print(f"Dataset: SYNTHETIC ({DATASET_PARAMS['n_samples']} samples, "
              f"{DATASET_PARAMS['n_features']} features, "
              f"{DATASET_PARAMS['n_classes']} classes)")
        print("=" * 100)

        backend = "C" if HAS_C_EXTENSION else "NumPy"
        print(f"C extension available: {HAS_C_EXTENSION}")

        # A. Load original Sklearn model
        with open('files/sklearn_model.pkl', 'rb') as f:
            sk_orig = pickle.load(f)

        # B. Dynamic import of exported classifier
        exported_classifier = None
        option_exported = False
        try:
            if os.path.exists(export_file):
                export_dir = os.path.dirname(os.path.abspath(export_file))
                if export_dir not in sys.path:
                    sys.path.insert(0, export_dir)
                module_name = f'{algo_prefix}_classifier'
                exported_classifier = importlib.import_module(module_name)
                importlib.reload(exported_classifier)
                option_exported = True
            else:
                print(f"[ERROR] File {export_file} not found.")
        except Exception as e:
            print(f"[ERROR] Failed to import {export_file}: {e}")

        # --- STRUCTURE REPORT ---
        leaves_sklearn = count_sklearn_leaves(sk_orig, ALGORITHM_TYPE)
        leaves_native = count_leaves_in_file(export_file)

        COL1 = 45
        print(f"\n{'STRUCTURE':<{COL1}} | {'LEAVES/RULES':<15}")
        print("-" * 65)
        print(f"{'Sklearn (original leaves)':<{COL1}} | {leaves_sklearn:<15}")
        print(f"{'pyRuleAnalyzer Exported (.py)':<{COL1}} | {leaves_native:<15}")
        print(f"{'pyRuleAnalyzer Rules (after optimization)':<{COL1}} | {len(classifier.final_rules):<15}")

        # --- INFERENCE ENGINE BENCHMARK ---
        print("\n" + "-" * 100)
        print(f"{'INFERENCE ENGINE':<{COL1}} | {'ACCURACY':<15} | {'TIME (s)':<12} | {'SAMPLES/s':<12}")
        print("-" * 100)

        y_test_int = np.array([int(y) for y in y_test])

        # For GBDT, sklearn.predict needs DataFrame
        if ALGORITHM_TYPE == 'Gradient Boosting Decision Trees':
            X_test_predict = pd.DataFrame(X_test, columns=feature_names)
        else:
            X_test_predict = X_test

        # 1. Sklearn (Cython)
        start = time.time()
        y_orig = sk_orig.predict(X_test_predict)
        t_orig = time.time() - start
        y_orig = np.array([int(p) for p in y_orig])
        acc_orig = np.mean(y_orig == y_test_int)
        print(f"{'1. Sklearn (Cython)':<{COL1}} | {acc_orig:<15.5f} | {t_orig:<12.4f} | {safe_speed(len(y_test), t_orig)}")

        # 2. pyRuleAnalyzer Vectorized
        label_batch = f"2. pyRuleAnalyzer Vectorized ({backend})"
        start = time.time()
        y_batch = classifier.predict_batch(X_test, feature_names=feature_names)
        t_batch = time.time() - start
        acc_batch = np.mean(y_batch == y_test_int)
        print(f"{label_batch:<{COL1}} | {acc_batch:<15.5f} | {t_batch:<12.4f} | {safe_speed(len(y_test), t_batch)}")

        # 3. pyRuleAnalyzer Exported (.py)
        t_opt3 = None
        if option_exported and exported_classifier is not None:
            start = time.time()
            y_opt3 = [exported_classifier.predict(s) for s in sample_dicts]
            t_opt3 = time.time() - start
            y_opt3 = np.array(y_opt3)
            acc_opt3 = np.mean(y_opt3 == y_test_int)
            print(f"{'3. pyRuleAnalyzer Exported (.py)':<{COL1}} | {acc_opt3:<15.5f} | {t_opt3:<12.4f} | {safe_speed(len(y_test), t_opt3)}")

        # 4. pyRuleAnalyzer Rules (per sample)
        start = time.time()
        y_py = [classifier.classify(s, final=True)[0] for s in sample_dicts]
        t_py = time.time() - start
        # Normalize GBDT class labels if needed
        y_py_clean = []
        for pred in y_py:
            try:
                pred = int(str(pred).replace('Class', '').strip())
            except (ValueError, AttributeError):
                pass
            y_py_clean.append(pred)
        y_py = np.array(y_py_clean)
        acc_py = np.mean(y_py == y_test_int)
        print(f"{'4. pyRuleAnalyzer Rules (per sample)':<{COL1}} | {acc_py:<15.5f} | {t_py:<12.4f} | {safe_speed(len(y_test), t_py)}")

        print("=" * 100)

        # --- RELATIVE SPEEDUP ---
        print("\nSPEEDUP RELATIVE TO SKLEARN")
        print("-" * 70)
        print(f"  {'ENGINE':<42} | {'SPEED vs SKLEARN':>18}")
        print(f"  {'-'*42}-+-{'-'*18}")
        if t_orig > 0:
            pct_batch = ((t_orig - t_batch) / t_orig) * 100
            sign_batch = "+" if pct_batch >= 0 else ""
            print(f"  {'pyRuleAnalyzer Vectorized (' + backend + ')':<42} | {sign_batch}{pct_batch:>10.1f}% {'faster' if pct_batch >= 0 else 'slower'}")
            if t_opt3 is not None:
                pct_opt3 = ((t_orig - t_opt3) / t_orig) * 100
                sign_opt3 = "+" if pct_opt3 >= 0 else ""
                print(f"  {'pyRuleAnalyzer Exported (.py)':<42} | {sign_opt3}{pct_opt3:>10.1f}% {'faster' if pct_opt3 >= 0 else 'slower'}")
            pct_py = ((t_orig - t_py) / t_orig) * 100
            sign_py = "+" if pct_py >= 0 else ""
            print(f"  {'pyRuleAnalyzer Rules (per sample)':<42} | {sign_py}{pct_py:>10.1f}% {'faster' if pct_py >= 0 else 'slower'}")

        # --- DIVERGENCE ANALYSIS ---
        print("\nDIVERGENCE ANALYSIS")
        print("-" * 60)
        divergent_batch = int(np.sum(y_orig != y_batch))
        print(f"Sklearn vs Vectorized:          {divergent_batch}/{len(y_test)} ({divergent_batch / len(y_test) * 100:.2f}%)")
        divergent_py = int(np.sum(y_orig != y_py))
        print(f"Sklearn vs Rules (per sample):  {divergent_py}/{len(y_test)} ({divergent_py / len(y_test) * 100:.2f}%)")
        if option_exported and exported_classifier is not None:
            divergent_exp = int(np.sum(y_orig != y_opt3))
            print(f"Sklearn vs Exported (.py):      {divergent_exp}/{len(y_test)} ({divergent_exp / len(y_test) * 100:.2f}%)")

        # --- FILE SIZE COMPARISON ---
        print("\nFILE SIZE COMPARISON (DISK)")

        export_bin = f"files/{algo_prefix}_model.bin"
        export_h = f"files/{algo_prefix}_model.h"

        files = {
            "Sklearn Original (.pkl)":            "files/sklearn_model.pkl",
            "Binary (.bin) [Vectorized]":         export_bin,
            "Exported (.py) [Exported]":          export_file,
            "C Header (.h) [Embedded]":           export_h,
            "pyRuleAnalyzer Full (.pkl)":         "files/initial_model.pkl",
        }

        orig_size = (
            os.path.getsize(files["Sklearn Original (.pkl)"])
            if os.path.exists(files["Sklearn Original (.pkl)"])
            else 0
        )

        print(f"{'FILE':<{COL1}} | {'SIZE (KB)':>12} | {'% of Original':>14}")
        print("-" * 78)

        for label, path in files.items():
            if os.path.exists(path):
                size_bytes = os.path.getsize(path)
                size_kb = size_bytes / 1024
                if orig_size > 0:
                    pct = f"{(size_bytes / orig_size) * 100:6.2f}%"
                else:
                    pct = "N/A"
                print(f"{label:<{COL1}} | {size_kb:>12.2f} KB | {pct:>14}")
            else:
                print(f"{label:<{COL1}} | {'NOT FOUND':>12} | {'N/A':>14}")

    finally:
        # Cleanup temporary files
        import shutil
        try:
            shutil.rmtree(temp_dir)
            print(f"\nTemp files cleaned up: {temp_dir}")
        except Exception as e:
            print(f"\nWarning: Could not clean up temp files at {temp_dir}: {e}")


if __name__ == '__main__':
    main()
