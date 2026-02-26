import sys
import os
import time
import pickle
import numpy as np
import pandas as pd
import importlib

# Adiciona o diretorio pai ao path para encontrar o pacote pyruleanalyzer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pyruleanalyzer.rule_classifier import RuleClassifier
from pyruleanalyzer._accel import HAS_C_EXTENSION

# --- CONFIGURACAO ---
# Escolha o dataset descomentando abaixo
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

# Configuracao Especifica para Random Forest
model_params = {
    # "random_state": 42,
    # "n_estimators": 10,
    # "max_depth": 10,
}

# ==============================================================================
# 1. PROCESSO DE TREINAMENTO E REFINAMENTO (RANDOM FOREST)
# ==============================================================================
# Cria o classificador definindo algorithm_type='Random Forest'
classifier = RuleClassifier.new_classifier(
    train_path, test_path, model_params, algorithm_type="Random Forest"
)

# Executa a analise
classifier.execute_rule_analysis(
    test_path, remove_duplicates="soft", remove_below_n_classifications=1
)

# Gera relatorio detalhado
classifier.compare_initial_final_results(test_path)

# ==============================================================================
# 2. EXPORTACAO
# ==============================================================================
X_train, _, X_test, y_test, _, _, feature_names = RuleClassifier.process_data(
    train_path, test_path
)
sample_dicts = pd.DataFrame(X_test, columns=feature_names).to_dict("records")

export_file = "examples/files/rf_classifier.py"
classifier.export_to_native_python(feature_names, filename=export_file)

# Compilar arrays de arvore para predict_batch
classifier.compile_tree_arrays(feature_names=feature_names)

# ==============================================================================
# 3. BENCHMARK E VALIDACAO DE PERFORMANCE
# ==============================================================================
print("\n" + "=" * 90)
print("RELATORIO DE COMPARACAO DE DESEMPENHO: SKLEARN (RF) vs PYRULEANALYZER")
print("=" * 90)
print(f"Extensao C disponivel: {HAS_C_EXTENSION}")

# A. Carregamento do Sklearn Original
with open("examples/files/sklearn_model.pkl", "rb") as f:
    sk_orig = pickle.load(f)

# B. Importacao dinamica do classificador exportado
rf_classifier = None
option3_available = False
try:
    if os.path.exists(export_file):
        export_dir = os.path.dirname(os.path.abspath(export_file))
        if export_dir not in sys.path:
            sys.path.insert(0, export_dir)
        rf_classifier = importlib.import_module('rf_classifier')
        importlib.reload(rf_classifier)
        option3_available = True
    else:
        print(f"[ERRO] Arquivo {export_file} nao encontrado.")
except Exception as e:
    print(f"[ERRO] Falha ao importar {export_file}: {e}")

# --- FUNCOES AUXILIARES ---

def count_leaves_in_file(filename):
    if not os.path.exists(filename):
        return 0
    with open(filename, "r") as f:
        return f.read().count("return ")


def safe_speed(n, t):
    if t <= 0:
        return "Inf"
    return f"{n / t:.0f}"


leaves_native = count_leaves_in_file(export_file)

# Logica ajustada para contar folhas em Random Forest
if hasattr(sk_orig, "estimators_"):
    leaves_sklearn = sum([tree.get_n_leaves() for tree in sk_orig.estimators_])
elif hasattr(sk_orig, "get_n_leaves"):
    leaves_sklearn = sk_orig.get_n_leaves()
else:
    leaves_sklearn = "N/A"

# --- RELATORIO DE ESTRUTURA ---
print(f"\n{'ESTRUTURA':<30} | {'FOLHAS/REGRAS':<15}")
print("-" * 50)
print(f"{'Sklearn Original (Total)':<30} | {leaves_sklearn:<15}")
print(f"{'Python Standalone':<30} | {leaves_native:<15}")
print(f"{'pyRuleAnalyzer (Regras)':<30} | {len(classifier.final_rules):<15}")


print("\n" + "-" * 90)
print(
    f"{'MOTOR DE INFERENCIA':<30} | {'ACURACIA':<15} | {'TEMPO (s)':<12} | {'SAMPLES/s':<12}"
)
print("-" * 90)

# 1. Sklearn Original
start = time.time()
y_orig = sk_orig.predict(X_test)
t_orig = time.time() - start
acc_orig = np.mean(y_orig == y_test)
print(
    f"{'1. Sklearn Original':<30} | {acc_orig:<15.5f} | {t_orig:<12.4f} | {safe_speed(len(y_test), t_orig)}"
)

# 2. predict_batch (Vetorizado com extensao C)
start = time.time()
y_batch = classifier.predict_batch(X_test, feature_names=feature_names)
t_batch = time.time() - start
acc_batch = np.mean(y_batch == y_test)
backend = "C ext" if HAS_C_EXTENSION else "numpy"
print(
    f"{'2. predict_batch (' + backend + ')':<30} | {acc_batch:<15.5f} | {t_batch:<12.4f} | {safe_speed(len(y_test), t_batch)}"
)

# 3. Python Standalone
if option3_available and rf_classifier is not None:
    start = time.time()
    y_opt3 = [rf_classifier.predict(s) for s in sample_dicts]
    t_opt3 = time.time() - start

    y_opt3 = np.array(y_opt3)
    acc_opt3 = np.mean(y_opt3 == y_test)
    print(
        f"{'3. Python Standalone (.py)':<30} | {acc_opt3:<15.5f} | {t_opt3:<12.4f} | {safe_speed(len(y_test), t_opt3)}"
    )

# 4. pyRuleAnalyzer classify() per-sample
start = time.time()
y_py = [classifier.classify(s, final=True)[0] for s in sample_dicts]
t_py = time.time() - start

y_py = np.array(y_py)
acc_py = np.mean(y_py == y_test)
print(
    f"{'4. classify() per-sample':<30} | {acc_py:<15.5f} | {t_py:<12.4f} | {safe_speed(len(y_test), t_py)}"
)

print("=" * 90)

# --- SPEEDUP RELATIVO ---
print(f"\n{'SPEEDUP RELATIVO AO SKLEARN'}")
print("-" * 50)
if t_orig > 0:
    print(f"  predict_batch ({backend}): {t_orig/max(t_batch, 1e-9):.2f}x {'mais rapido' if t_batch < t_orig else 'mais lento'}")
    if option3_available:
        print(f"  Python Standalone:        {t_orig/max(t_opt3, 1e-9):.2f}x {'mais rapido' if t_opt3 < t_orig else 'mais lento'}")
    print(f"  classify() per-sample:    {t_orig/max(t_py, 1e-9):.2f}x {'mais rapido' if t_py < t_orig else 'mais lento'}")

# --- COMPARATIVO DE TAMANHO (DISCO) ---
print(f"\n{'COMPARATIVO DE TAMANHO (DISCO)'}")

export_bin = "examples/files/rf_model.bin"
classifier.export_to_binary(export_bin)

export_h = "examples/files/rf_model.h"
classifier.export_to_c_header(export_h)

files = {
    "Sklearn Original (.pkl)": "examples/files/sklearn_model.pkl",
    "predict_batch (.bin)": export_bin,
    "Python Standalone (.py)": export_file,
    "C Header (.h)": export_h,
    "pyRuleAnalyzer (.pkl)": "examples/files/final_model.pkl",
}

orig_size = (
    os.path.getsize(files["Sklearn Original (.pkl)"])
    if os.path.exists(files["Sklearn Original (.pkl)"])
    else 0
)

print(f"{'ARQUIVO':<30} | {'TAMANHO (KB)':>12} | {'% do Original':>14}")
print("-" * 65)

for label, path in files.items():
    if os.path.exists(path):
        size_bytes = os.path.getsize(path)
        size_kb = size_bytes / 1024

        if orig_size > 0:
            pct = f"{(size_bytes / orig_size) * 100:6.2f}%"
        else:
            pct = "N/A"

        print(f"{label:<30} | {size_kb:>12.2f} KB | {pct:>14}")
    else:
        print(f"{label:<30} | {'NOT FOUND':>12} | {'N/A':>14}")
