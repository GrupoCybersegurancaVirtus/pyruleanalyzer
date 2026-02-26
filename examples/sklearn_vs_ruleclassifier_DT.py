import sys
import os
import time
import pickle
import numpy as np
import pandas as pd
import importlib

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

model_params = {'random_state': 42}

# ==============================================================================
# 1. TRAINING AND REFINEMENT PROCESS
# ==============================================================================
# Create classifier, train model (or load), extract rules
classifier = RuleClassifier.new_classifier(train_path, test_path, model_params, algorithm_type='Decision Tree')

# Execute analysis (prune duplicates and low-usage rules)
classifier.execute_rule_analysis(test_path, remove_duplicates="soft", remove_below_n_classifications=-1)

# Generate detailed report comparing Before x After
classifier.compare_initial_final_results(test_path)

# ==============================================================================
# 2. EXPORT
# ==============================================================================
# Retrieve data for benchmark test
X_train, _, X_test, y_test, _, _, feature_names = RuleClassifier.process_data(train_path, test_path)

# Convert to list of dicts (standard input for native engine)
sample_dicts = pd.DataFrame(X_test, columns=feature_names).to_dict('records')

export_file = "examples/files/dt_classifier.py"
classifier.export_to_native_python(feature_names, filename=export_file)

# Compile tree arrays for predict_batch
classifier.compile_tree_arrays(feature_names=feature_names)

# ==============================================================================
# 3. BENCHMARK AND PERFORMANCE VALIDATION
# ==============================================================================
print("\n" + "="*100)
print("PERFORMANCE COMPARISON REPORT: SKLEARN vs PYRULEANALYZER (DT)")
print("="*100)
backend = "C" if HAS_C_EXTENSION else "NumPy"
print(f"C extension available: {HAS_C_EXTENSION}")

# A. Load original Sklearn model
with open('examples/files/sklearn_model.pkl', 'rb') as f:
    sk_orig = pickle.load(f)

# B. Dynamic import of exported classifier (Python Standalone)
dt_classifier = None
option3_available = False
try:
    if os.path.exists(export_file):
        export_dir = os.path.dirname(os.path.abspath(export_file))
        if export_dir not in sys.path:
            sys.path.insert(0, export_dir)
        dt_classifier = importlib.import_module('dt_classifier')
        importlib.reload(dt_classifier)  # Reload in case it changed
        option3_available = True
    else:
        print(f"[ERROR] File {export_file} not found.")
except Exception as e:
    print(f"[ERROR] Failed to import {export_file}: {e}")

# --- HELPER FUNCTIONS ---

def count_leaves_in_file(filename):
    """Count how many 'return' exist in the generated file (proxy for leaf count)."""
    if not os.path.exists(filename):
        return 0
    with open(filename, 'r') as f:
        return f.read().count('return ')

leaves_native = count_leaves_in_file(export_file)
leaves_sklearn = sk_orig.get_n_leaves() if hasattr(sk_orig, 'get_n_leaves') else "N/A"

# --- STRUCTURE REPORT ---
COL1 = 42
print(f"\n{'STRUCTURE':<{COL1}} | {'LEAVES/RULES':<15}")
print("-" * 62)
print(f"{'Sklearn (original leaves)':<{COL1}} | {leaves_sklearn:<15}")
print(f"{'pyRuleAnalyzer Exported (.py)':<{COL1}} | {leaves_native:<15}")
print(f"{'pyRuleAnalyzer Rules (after optimization)':<{COL1}} | {len(classifier.final_rules):<15}")


print("\n" + "-" * 100)
print(f"{'INFERENCE ENGINE':<{COL1}} | {'ACCURACY':<15} | {'TIME (s)':<12} | {'SAMPLES/s':<12}")
print("-" * 100)

# Helper function to avoid division by zero
def safe_speed(n, t):
    if t <= 0:
        return "Inf"
    return f"{n/t:.0f}"

# 1. Sklearn (Cython)
start = time.time()
y_orig = sk_orig.predict(X_test)
t_orig = time.time() - start
acc_orig = np.mean(y_orig == y_test)
print(f"{'1. Sklearn (Cython)':<{COL1}} | {acc_orig:<15.5f} | {t_orig:<12.4f} | {safe_speed(len(y_test), t_orig)}")

# 2. pyRuleAnalyzer Vectorized
label_batch = f"2. pyRuleAnalyzer Vectorized ({backend})"
start = time.time()
y_batch = classifier.predict_batch(X_test, feature_names=feature_names)
t_batch = time.time() - start
acc_batch = np.mean(y_batch == y_test)
print(f"{label_batch:<{COL1}} | {acc_batch:<15.5f} | {t_batch:<12.4f} | {safe_speed(len(y_test), t_batch)}")

# 3. pyRuleAnalyzer Exported (.py)
if option3_available and dt_classifier is not None:
    start = time.time()
    y_opt3 = [dt_classifier.predict(s) for s in sample_dicts]
    t_opt3 = time.time() - start

    y_opt3 = np.array(y_opt3)
    acc_opt3 = np.mean(y_opt3 == y_test)
    print(f"{'3. pyRuleAnalyzer Exported (.py)':<{COL1}} | {acc_opt3:<15.5f} | {t_opt3:<12.4f} | {safe_speed(len(y_test), t_opt3)}")

# 4. pyRuleAnalyzer Rules (per sample)
start = time.time()
y_py = [classifier.classify(s, final=True)[0] for s in sample_dicts]
t_py = time.time() - start

y_py = np.array(y_py)
acc_py = np.mean(y_py == y_test)
print(f"{'4. pyRuleAnalyzer Rules (per sample)':<{COL1}} | {acc_py:<15.5f} | {t_py:<12.4f} | {safe_speed(len(y_test), t_py)}")

print("="*100)

# --- RELATIVE SPEEDUP ---
print("\nSPEEDUP RELATIVE TO SKLEARN")
print("-" * 55)
if t_orig > 0:
    print(f"  pyRuleAnalyzer Vectorized ({backend}):  {t_orig/max(t_batch, 1e-9):.2f}x {'faster' if t_batch < t_orig else 'slower'}")
    if option3_available:
        print(f"  pyRuleAnalyzer Exported (.py):     {t_orig/max(t_opt3, 1e-9):.2f}x {'faster' if t_opt3 < t_orig else 'slower'}")
    print(f"  pyRuleAnalyzer Rules (per sample): {t_orig/max(t_py, 1e-9):.2f}x {'faster' if t_py < t_orig else 'slower'}")

# --- FILE SIZE COMPARISON (DISK) ---
print("\nFILE SIZE COMPARISON (DISK)")

export_bin = "examples/files/dt_model.bin"
classifier.export_to_binary(export_bin)

export_h = "examples/files/dt_model.h"
classifier.export_to_c_header(export_h)

files = {
    "Sklearn Original (.pkl)":              "examples/files/sklearn_model.pkl",
    "Binary (.bin) [Vectorized]":           export_bin,
    "Exported (.py) [Exported]":            export_file,
    "C Header (.h) [Embedded]":             export_h,
    "pyRuleAnalyzer Full (.pkl)":           "examples/files/final_model.pkl",
}

orig_size = os.path.getsize(files["Sklearn Original (.pkl)"]) if os.path.exists(files["Sklearn Original (.pkl)"]) else 0
print(f"{'FILE':<{COL1}} | {'SIZE (KB)':>12} | {'% of Original':>14}")
print("-" * 75)

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

COL_M = 35
COL_V = 14
print(f"{'METRIC':<{COL_M}} | {'INITIAL':>{COL_V}} | {'FINAL':>{COL_V}} | {'CHANGE':>{COL_V}}")
print(f"{'-' * 85}")

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
