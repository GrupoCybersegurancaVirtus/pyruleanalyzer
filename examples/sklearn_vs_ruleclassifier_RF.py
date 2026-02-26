import sys
import os
import time
import pickle
import numpy as np
import pandas as pd
import importlib

# Adiciona o diretório pai ao path para encontrar o pacote pyruleanalyzer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pyruleanalyzer.rule_classifier import RuleClassifier

# --- CONFIGURAÇÃO ---
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

# Configuração Específica para Random Forest
# Recomendado limitar profundidade e estimadores para testes iniciais para não gerar milhões de regras
model_params = {
    # "random_state": 42,
    # "n_estimators": 10,
    # # ,'max_depth': 10       # Profundidade máxima para manter as regras legíveis
}

# ==============================================================================
# 1. PROCESSO DE TREINAMENTO E REFINAMENTO (RANDOM FOREST)
# ==============================================================================
# Cria o classificador definindo algorithm_type='Random Forest'
classifier = RuleClassifier.new_classifier(
    train_path, test_path, model_params, algorithm_type="Random Forest"
)

# Executa a análise
# Em RF, 'remove_duplicates' geralmente tem menos impacto visual que em DT,
# mas ajuda a limpar regras redundantes entre árvores se usar "hard"
classifier.execute_rule_analysis(
    test_path, remove_duplicates="soft", remove_below_n_classifications=1
)

# Gera relatório detalhado
classifier.compare_initial_final_results(test_path)

# ==============================================================================
# 2. EXPORTAÇÃO
# ==============================================================================
X_train, _, X_test, y_test, _, _, feature_names = RuleClassifier.process_data(
    train_path, test_path
)
sample_dicts = pd.DataFrame(X_test, columns=feature_names).to_dict("records")

# Nome do arquivo de exportação para RF
export_file = "examples/files/rf_classifier.py"

# Nota: A exportação nativa converte a Floresta em uma lista sequencial de regras (Decision List)
classifier.export_to_native_python(feature_names, filename=export_file)

# ==============================================================================
# 3. BENCHMARK E VALIDAÇÃO DE PERFORMANCE
# ==============================================================================
print("\n" + "=" * 80)
print("RELATÓRIO DE COMPARAÇÃO DE DESEMPENHO: SKLEARN (RF) vs NATIVE PYTHON")
print("=" * 80)

# A. Carregamento do Sklearn Original
with open("examples/files/sklearn_model.pkl", "rb") as f:
    sk_orig = pickle.load(f)

# B. Importação dinâmica do classificador exportado
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
        print(f"[ERRO] Arquivo {export_file} não encontrado.")
except Exception as e:
    print(f"[ERRO] Falha ao importar {export_file}: {e}")

# --- FUNÇÕES AUXILIARES ---


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

# Lógica ajustada para contar folhas em Random Forest (Soma das folhas de todas as árvores)
if hasattr(sk_orig, "estimators_"):
    leaves_sklearn = sum([tree.get_n_leaves() for tree in sk_orig.estimators_])
elif hasattr(sk_orig, "get_n_leaves"):
    leaves_sklearn = sk_orig.get_n_leaves()
else:
    leaves_sklearn = "N/A"

# --- RELATÓRIO DE ESTRUTURA ---
print(f"{'ESTRUTURA':<30} | {'FOLHAS/REGRAS':<15}")
print("-" * 50)
print(f"{'Sklearn Original (Total)':<30} | {leaves_sklearn:<15}")
print(f"{'pyRuleAnalyzer (Novo)':<30} | {leaves_native:<15}")
print(f"{'pyRuleAnalyzer (Antigo)':<30} | {len(classifier.final_rules):<15}")


print("\n" + "-" * 80)
print(
    f"{'MOTOR DE INFERÊNCIA':<30} | {'ACURÁCIA':<15} | {'TEMPO (s)':<12} | {'SAMPLES/s':<12}"
)
print("-" * 80)

# 1. Sklearn Original
start = time.time()
y_orig = sk_orig.predict(X_test)
t_orig = time.time() - start
acc_orig = np.mean(y_orig == y_test)
print(
    f"{'1. Sklearn Original':<30} | {acc_orig:<15.5f} | {t_orig:<12.4f} | {safe_speed(len(y_test), t_orig)}"
)

# 2. Python Standalone
if option3_available and rf_classifier is not None:
    start = time.time()
    # Simula inferência
    y_opt3 = [rf_classifier.predict(s) for s in sample_dicts]
    t_opt3 = time.time() - start

    y_opt3 = np.array(y_opt3)
    acc_opt3 = np.mean(y_opt3 == y_test)
    print(
        f"{'2. pyRuleAnalyzer (Novo)':<30} | {acc_opt3:<15.5f} | {t_opt3:<12.4f} | {safe_speed(len(y_test), t_opt3)}"
    )

# 3. pyRuleAnalyzer Internal
start = time.time()
# classify retorna tupla (pred, votes, proba), pegamos [0]
y_py = [classifier.classify(s, final=True)[0] for s in sample_dicts]
t_py = time.time() - start

y_py = np.array(y_py)
acc_py = np.mean(y_py == y_test)
print(
    f"{'3. pyRuleAnalyzer (Antigo)':<30} | {acc_py:<15.5f} | {t_py:<12.4f} | {safe_speed(len(y_test), t_py)}"
)

print("=" * 80)

# --- COMPARATIVO DE TAMANHO (DISCO) ---
files = {
    "Sklearn Original": "examples/files/sklearn_model.pkl",
    "pyRuleAnalyzer (Novo)": export_file,
    "pyRuleAnalyzer (Antigo)": "examples/files/final_model.pkl",
}

orig_size = (
    os.path.getsize(files["Sklearn Original"])
    if os.path.exists(files["Sklearn Original"])
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
