import sys
import os
import time
import pickle
import importlib
import numpy as np
import pandas as pd

# Add parent directory to path to find pyruleanalyzer package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyruleanalyzer.rule_classifier import RuleClassifier
from pyruleanalyzer._accel import HAS_C_EXTENSION

# --- CONFIGURATION ---
# Choose dataset by uncommenting below
# train_path = "examples/data/covid_train.csv"
# test_path = "examples/data/covid_test.csv"

# train_path = "examples/data/train-set1.csv"
# test_path = "examples/data/test-set1.csv"

# train_path = "examples/data/ddos-train.csv"
# test_path = "examples/data/ddos-test.csv"

train_path = "examples/data/A Machine Learning-Based Classification and Prediction Technique for DDoS Attacks/train.csv"
test_path = "examples/data/A Machine Learning-Based Classification and Prediction Technique for DDoS Attacks/test.csv"

# train_path = "examples/data/DDoS Attack Classification Leveraging Data Balancing and Hyperparameter Tuning Approach Using Ensemble Machine Learning with XAI/train.csv"
# test_path = "examples/data/DDoS Attack Classification Leveraging Data Balancing and Hyperparameter Tuning Approach Using Ensemble Machine Learning with XAI/test.csv"

# Gradient Boosting Decision Trees specific configuration
model_params = {
    'random_state': 42,
    'n_estimators': 100,
    # 'max_depth': 3,           # Max depth (default = 3 for GBDT)
    # 'learning_rate': 0.1,     # Learning rate
}

# ==============================================================================
# 1. TRAINING AND REFINEMENT PROCESS (GBDT)
# ==============================================================================
# Create classifier with algorithm_type='Gradient Boosting Decision Trees'
classifier = RuleClassifier.new_classifier(
    train_path, test_path, model_params,
    algorithm_type='Gradient Boosting Decision Trees'
)

# Execute redundancy analysis
classifier.execute_rule_analysis(
    test_path, remove_duplicates="hard", remove_below_n_classifications=1
)

# Generate detailed report comparing Before x After
classifier.compare_initial_final_results(test_path)

# ==============================================================================
# 2. NATIVE CLASSIFIER EXPORT
# ==============================================================================
# Retrieve data for benchmark test
X_train, _, X_test, y_test, _, _, feature_names = RuleClassifier.process_data(
    train_path, test_path
)

# Convert to DataFrame and list of dicts
X_test_df = pd.DataFrame(X_test, columns=feature_names)
sample_dicts = X_test_df.to_dict('records')

export_file = "examples/files/gbdt_classifier.py"
classifier.export_to_native_python(feature_names, filename=export_file)

# Compile tree arrays for predict_batch
classifier.compile_tree_arrays(feature_names=feature_names)

# Dynamic import of exported classifier
gbdt_classifier = None
option_standalone = False
try:
    if os.path.exists(export_file):
        export_dir = os.path.dirname(os.path.abspath(export_file))
        if export_dir not in sys.path:
            sys.path.insert(0, export_dir)
        gbdt_classifier = importlib.import_module('gbdt_classifier')
        importlib.reload(gbdt_classifier)
        option_standalone = True
    else:
        print(f"[ERROR] File {export_file} not found.")
except Exception as e:
    print(f"[ERROR] Failed to import {export_file}: {e}")

# ==============================================================================
# 3. BENCHMARK AND PERFORMANCE VALIDATION
# ==============================================================================

print("\n" + "=" * 100)
print("PERFORMANCE COMPARISON REPORT: SKLEARN (GBDT) vs PYRULEANALYZER")
print("=" * 100)
backend = "C" if HAS_C_EXTENSION else "NumPy"
print(f"C extension available: {HAS_C_EXTENSION}")

# A. Load original Sklearn model
with open('examples/files/sklearn_model.pkl', 'rb') as f:
    sk_orig = pickle.load(f)

# --- HELPER FUNCTIONS ---

def safe_speed(n, t):
    """Avoid division by zero in speed calculation."""
    if t <= 0:
        return "Inf"
    return f"{n / t:.0f}"


# --- Leaf/rule count ---
if hasattr(sk_orig, 'estimators_'):
    n_estimators, n_cols = sk_orig.estimators_.shape
    leaves_sklearn = sum(
        sk_orig.estimators_[i, j].get_n_leaves()
        for i in range(n_estimators)
        for j in range(n_cols)
    )
    n_classes = len(sk_orig.classes_)
    leaves_sklearn += n_classes if not (n_classes == 2 and n_cols == 1) else 1
else:
    leaves_sklearn = "N/A"

rules_initial = len(classifier.initial_rules)
rules_refined = len(classifier.final_rules) if classifier.final_rules else rules_initial

# --- STRUCTURE REPORT ---
COL1 = 45
print(f"\n{'STRUCTURE':<{COL1}} | {'LEAVES/RULES':<15}")
print("-" * 65)
print(f"{'Sklearn (original leaves, total)':<{COL1}} | {leaves_sklearn:<15}")
print(f"{'pyRuleAnalyzer Rules (before optimization)':<{COL1}} | {rules_initial:<15}")
print(f"{'pyRuleAnalyzer Rules (after optimization)':<{COL1}} | {rules_refined:<15}")

print("\n" + "-" * 100)
print(
    f"{'INFERENCE ENGINE':<{COL1}} | {'ACCURACY':<15} | {'TIME (s)':<12} | {'SAMPLES/s':<12}"
)
print("-" * 100)

# 1. Sklearn (Cython)
start = time.time()
y_orig_raw = sk_orig.predict(X_test_df)
t_orig = time.time() - start
y_orig = np.array([int(p) for p in y_orig_raw])
y_test_int = np.array([int(y) for y in y_test])
acc_orig = np.mean(y_orig == y_test_int)
print(
    f"{'1. Sklearn (Cython)':<{COL1}} | {acc_orig:<15.5f} | {t_orig:<12.4f} | {safe_speed(len(y_test), t_orig)}"
)

# 2. pyRuleAnalyzer Vectorized
label_batch = f"2. pyRuleAnalyzer Vectorized ({backend})"
start = time.time()
y_batch = classifier.predict_batch(X_test, feature_names=feature_names)
t_batch = time.time() - start
acc_batch = np.mean(y_batch == y_test_int)
print(
    f"{label_batch:<{COL1}} | {acc_batch:<15.5f} | {t_batch:<12.4f} | {safe_speed(len(y_test), t_batch)}"
)

# 3. pyRuleAnalyzer Rules (Initial)
print("   Classifying with initial model...")
start = time.time()
y_initial = []
for sample in sample_dicts:
    pred, _, _ = RuleClassifier.classify_gbdt(
        sample, classifier.initial_rules,
        classifier._gbdt_init_scores,
        classifier._gbdt_is_binary,
        classifier._gbdt_classes,
    )
    try:
        pred = int(str(pred).replace('Class', '').strip())
    except (ValueError, AttributeError):
        pass
    y_initial.append(pred)
t_initial = time.time() - start

y_initial = np.array(y_initial)
acc_initial = np.mean(y_initial == y_test_int)
print(
    f"{'3. pyRuleAnalyzer Rules (Initial)':<{COL1}} | {acc_initial:<15.5f} | {t_initial:<12.4f} | {safe_speed(len(y_test), t_initial)}"
)

# 4. pyRuleAnalyzer Rules (Refined)
if classifier.final_rules:
    print("   Classifying with refined model...")
    start = time.time()
    y_refined = []
    use_native = classifier.native_fn is not None
    for sample in sample_dicts:
        if use_native:
            try:
                pred, _, _ = classifier.native_fn.classify(sample)  # type: ignore[union-attr]
            except Exception:
                pred, _, _ = RuleClassifier.classify_gbdt(
                    sample, classifier.final_rules,
                    classifier._gbdt_init_scores,
                    classifier._gbdt_is_binary,
                    classifier._gbdt_classes,
                )
        else:
            pred, _, _ = RuleClassifier.classify_gbdt(
                sample, classifier.final_rules,
                classifier._gbdt_init_scores,
                classifier._gbdt_is_binary,
                classifier._gbdt_classes,
            )
        try:
            pred = int(str(pred).replace('Class', '').strip())
        except (ValueError, AttributeError):
            pass
        y_refined.append(pred)
    t_refined = time.time() - start

    y_refined = np.array(y_refined)
    acc_refined = np.mean(y_refined == y_test_int)
    print(
        f"{'4. pyRuleAnalyzer Rules (Refined)':<{COL1}} | {acc_refined:<15.5f} | {t_refined:<12.4f} | {safe_speed(len(y_test), t_refined)}"
    )

# 5. pyRuleAnalyzer Exported (.py)
if option_standalone and gbdt_classifier is not None:
    start = time.time()
    y_standalone = []
    for sample in sample_dicts:
        pred = gbdt_classifier.predict(sample)
        try:
            pred = int(str(pred).replace('Class', '').strip())
        except (ValueError, AttributeError):
            pass
        y_standalone.append(pred)
    t_standalone = time.time() - start

    y_standalone = np.array(y_standalone)
    acc_standalone = np.mean(y_standalone == y_test_int)
    print(
        f"{'5. pyRuleAnalyzer Exported (.py)':<{COL1}} | {acc_standalone:<15.5f} | {t_standalone:<12.4f} | {safe_speed(len(y_test), t_standalone)}"
    )

print("=" * 100)

# --- RELATIVE SPEEDUP ---
print("\nSPEEDUP RELATIVE TO SKLEARN")
print("-" * 60)
if t_orig > 0:
    print(f"  pyRuleAnalyzer Vectorized ({backend}):  {t_orig/max(t_batch, 1e-9):.2f}x {'faster' if t_batch < t_orig else 'slower'}")
    print(f"  pyRuleAnalyzer Rules (Initial):    {t_orig/max(t_initial, 1e-9):.2f}x {'faster' if t_initial < t_orig else 'slower'}")
    if classifier.final_rules:
        print(f"  pyRuleAnalyzer Rules (Refined):   {t_orig/max(t_refined, 1e-9):.2f}x {'faster' if t_refined < t_orig else 'slower'}")
    if option_standalone:
        print(f"  pyRuleAnalyzer Exported (.py):     {t_orig/max(t_standalone, 1e-9):.2f}x {'faster' if t_standalone < t_orig else 'slower'}")

# --- DIVERGENCE ANALYSIS ---
print("\nDIVERGENCE ANALYSIS")
print("-" * 60)

divergent_initial = int(np.sum(y_orig != y_initial))
print(f"Sklearn vs Rules (Initial):     {divergent_initial}/{len(y_test)} ({divergent_initial / len(y_test) * 100:.2f}%)")

divergent_batch = int(np.sum(y_orig != y_batch))
print(f"Sklearn vs Vectorized:          {divergent_batch}/{len(y_test)} ({divergent_batch / len(y_test) * 100:.2f}%)")

if classifier.final_rules:
    divergent_refined = int(np.sum(y_orig != y_refined))
    divergent_init_ref = int(np.sum(y_initial != y_refined))
    print(f"Sklearn vs Rules (Refined):     {divergent_refined}/{len(y_test)} ({divergent_refined / len(y_test) * 100:.2f}%)")
    print(f"Rules Initial vs Refined:       {divergent_init_ref}/{len(y_test)} ({divergent_init_ref / len(y_test) * 100:.2f}%)")

if option_standalone and gbdt_classifier is not None:
    divergent_standalone = int(np.sum(y_orig != y_standalone))
    print(f"Sklearn vs Exported (.py):      {divergent_standalone}/{len(y_test)} ({divergent_standalone / len(y_test) * 100:.2f}%)")

# --- FILE SIZE COMPARISON (DISK) ---
print("\nFILE SIZE COMPARISON (DISK)")

export_bin = "examples/files/gbdt_model.bin"
classifier.export_to_binary(export_bin)

export_h = "examples/files/gbdt_model.h"
classifier.export_to_c_header(export_h)

files = {
    "Sklearn Original (.pkl)":              "examples/files/sklearn_model.pkl",
    "Binary (.bin) [Vectorized]":           export_bin,
    "Exported (.py) [Exported]":            export_file,
    "C Header (.h) [Embedded]":             export_h,
    "pyRuleAnalyzer Full (.pkl)":           "examples/files/initial_model.pkl",
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

# --- STRUCTURAL COMPLEXITY METRICS (SCS) ---
print(f"\n{'=' * 100}")
print("STRUCTURAL COMPLEXITY METRICS (SCS)")
print(f"{'=' * 100}")

n_features = len(feature_names)
metrics_init = RuleClassifier.calculate_structural_complexity(classifier.initial_rules, n_features)
metrics_final = RuleClassifier.calculate_structural_complexity(classifier.final_rules, n_features)

COL_M = 38
COL_V = 14
print(f"{'METRIC':<{COL_M}} | {'INITIAL':>{COL_V}} | {'FINAL':>{COL_V}} | {'CHANGE':>{COL_V}}")
print(f"{'-' * 88}")

for k in metrics_init:
    v_init = metrics_init[k]
    v_final = metrics_final.get(k, 0)
    if isinstance(v_init, float):
        s_init = f"{v_init:.4f}"
        s_final = f"{v_final:.4f}"
    else:
        s_init = str(v_init)
        s_final = str(v_final)
    if isinstance(v_init, (int, float)) and v_init != 0:
        pct = ((v_final - v_init) / v_init) * 100
        s_pct = f"{pct:+.1f}%"
    else:
        s_pct = "N/A"
    print(f"  {k:<{COL_M}} | {s_init:>{COL_V}} | {s_final:>{COL_V}} | {s_pct:>{COL_V}}")

print(f"{'=' * 100}")
