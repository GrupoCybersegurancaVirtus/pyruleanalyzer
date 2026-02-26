import sys
import os
import time
import pickle
import importlib
import numpy as np
import pandas as pd

# Adiciona o diretorio pai ao path para encontrar o pacote pyruleanalyzer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyruleanalyzer.rule_classifier import RuleClassifier

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

# Configuracao Especifica para Gradient Boosting Decision Trees
model_params = {
    'random_state': 42,
    'n_estimators': 100,
    # 'max_depth': 3,           # Profundidade maxima (padrao = 3 para GBDT)
    # 'learning_rate': 0.1,     # Taxa de aprendizado
}

# ==============================================================================
# 1. PROCESSO DE TREINAMENTO E REFINAMENTO (GBDT)
# ==============================================================================
# Cria o classificador definindo algorithm_type='Gradient Boosting Decision Trees'
classifier = RuleClassifier.new_classifier(
    train_path, test_path, model_params,
    algorithm_type='Gradient Boosting Decision Trees'
)

# Executa a analise de redundancia
# Em GBDT, remove_duplicates="hard" ativa deteccao intra-tree
# remove_below_n_classifications controla poda de regras pouco usadas
classifier.execute_rule_analysis(
    test_path, remove_duplicates="hard", remove_below_n_classifications=1
)

# Gera relatorio detalhado comparando Antes x Depois
classifier.compare_initial_final_results(test_path)

# ==============================================================================
# 2. EXPORTACAO DO CLASSIFICADOR NATIVO
# ==============================================================================
# Recuperacao de dados para o teste de benchmark
X_train, _, X_test, y_test, _, _, feature_names = RuleClassifier.process_data(
    train_path, test_path
)

# Converter para DataFrame e lista de dicionarios
X_test_df = pd.DataFrame(X_test, columns=feature_names)
sample_dicts = X_test_df.to_dict('records')

export_file = "examples/files/gbdt_classifier.py"
classifier.export_to_native_python(feature_names, filename=export_file)

# Importacao dinamica do classificador exportado
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
        print(f"[ERRO] Arquivo {export_file} nao encontrado.")
except Exception as e:
    print(f"[ERRO] Falha ao importar {export_file}: {e}")

# ==============================================================================
# 3. BENCHMARK E VALIDACAO DE PERFORMANCE
# ==============================================================================

print("\n" + "=" * 80)
print("RELATORIO DE COMPARACAO DE DESEMPENHO: SKLEARN (GBDT) vs PYRULEANALYZER")
print("=" * 80)

# A. Carregamento do Sklearn Original
with open('examples/files/sklearn_model.pkl', 'rb') as f:
    sk_orig = pickle.load(f)

# --- FUNCOES AUXILIARES ---

def safe_speed(n, t):
    """Evita divisao por zero no calculo de velocidade."""
    if t <= 0:
        return "Inf"
    return f"{n / t:.0f}"


# --- Contagem de regras/folhas ---
# Sklearn: GBDT tem estimators_ como array 2D (n_estimators, n_classes) para multiclass
# ou (n_estimators, 1) para binario.
# Devemos contar as folhas de TODAS as colunas (classes), nao apenas da coluna 0.
if hasattr(sk_orig, 'estimators_'):
    n_estimators, n_cols = sk_orig.estimators_.shape
    leaves_sklearn = sum(
        sk_orig.estimators_[i, j].get_n_leaves()
        for i in range(n_estimators)
        for j in range(n_cols)
    )
    # Adicionar as init rules (1 por classe) para comparacao justa com o pyRuleAnalyzer
    n_classes = len(sk_orig.classes_)
    leaves_sklearn += n_classes if not (n_classes == 2 and n_cols == 1) else 1
else:
    leaves_sklearn = "N/A"

# pyRuleAnalyzer: contar regras do modelo inicial e refinado
rules_initial = len(classifier.initial_rules)
rules_refined = len(classifier.final_rules) if classifier.final_rules else rules_initial

# --- RELATORIO DE ESTRUTURA ---
print(f"\n{'ESTRUTURA':<35} | {'FOLHAS/REGRAS':<15}")
print("-" * 55)
print(f"{'Sklearn Original (Total)':<35} | {leaves_sklearn:<15}")
print(f"{'pyRuleAnalyzer Inicial':<35} | {rules_initial:<15}")
print(f"{'pyRuleAnalyzer Refinado':<35} | {rules_refined:<15}")

print("\n" + "-" * 80)
print(
    f"{'MOTOR DE INFERENCIA':<35} | {'ACURACIA':<15} | {'TEMPO (s)':<12} | {'SAMPLES/s':<12}"
)
print("-" * 80)

# 1. Sklearn Original (Vetorizado em C)
start = time.time()
y_orig_raw = sk_orig.predict(X_test_df)  # DataFrame para evitar warning
t_orig = time.time() - start
y_orig = np.array([int(p) for p in y_orig_raw])
y_test_int = np.array([int(y) for y in y_test])
acc_orig = np.mean(y_orig == y_test_int)
print(
    f"{'1. Sklearn Original':<35} | {acc_orig:<15.5f} | {t_orig:<12.4f} | {safe_speed(len(y_test), t_orig)}"
)

# 2. pyRuleAnalyzer (Modelo Inicial - usando classify_gbdt)
print("   Classificando com modelo inicial...")
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
    f"{'2. pyRuleAnalyzer (Inicial)':<35} | {acc_initial:<15.5f} | {t_initial:<12.4f} | {safe_speed(len(y_test), t_initial)}"
)

# 3. pyRuleAnalyzer (Modelo Refinado - usando native_fn ou classify_gbdt)
if classifier.final_rules:
    print("   Classificando com modelo refinado...")
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
        f"{'3. pyRuleAnalyzer (Refinado)':<35} | {acc_refined:<15.5f} | {t_refined:<12.4f} | {safe_speed(len(y_test), t_refined)}"
    )

# 4. Python Standalone (Arquivo exportado)
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
        f"{'4. Python Standalone (Arquivo)':<35} | {acc_standalone:<15.5f} | {t_standalone:<12.4f} | {safe_speed(len(y_test), t_standalone)}"
    )

print("=" * 80)

# --- ANALISE DE DIVERGENCIAS ---
print(f"\n{'ANALISE DE DIVERGENCIAS'}")
print("-" * 55)

divergent_initial = int(np.sum(y_orig != y_initial))
print(f"Sklearn vs Inicial:  {divergent_initial}/{len(y_test)} ({divergent_initial / len(y_test) * 100:.2f}%)")

if classifier.final_rules:
    divergent_refined = int(np.sum(y_orig != y_refined))
    divergent_init_ref = int(np.sum(y_initial != y_refined))
    print(f"Sklearn vs Refinado: {divergent_refined}/{len(y_test)} ({divergent_refined / len(y_test) * 100:.2f}%)")
    print(f"Inicial vs Refinado: {divergent_init_ref}/{len(y_test)} ({divergent_init_ref / len(y_test) * 100:.2f}%)")

if option_standalone and gbdt_classifier is not None:
    divergent_standalone = int(np.sum(y_orig != y_standalone))
    print(f"Sklearn vs Standalone: {divergent_standalone}/{len(y_test)} ({divergent_standalone / len(y_test) * 100:.2f}%)")

# --- COMPARATIVO DE TAMANHO (DISCO) ---
print(f"\n{'COMPARATIVO DE TAMANHO (DISCO)'}")

files = {
    "Sklearn Original": "examples/files/sklearn_model.pkl",
    "pyRuleAnalyzer (Modelo)": "examples/files/initial_model.pkl",
    "Python Standalone (Arquivo)": export_file,
}

orig_size = (
    os.path.getsize(files["Sklearn Original"])
    if os.path.exists(files["Sklearn Original"])
    else 0
)

print(f"{'ARQUIVO':<35} | {'TAMANHO (KB)':>12} | {'% do Original':>14}")
print("-" * 70)

for label, path in files.items():
    if os.path.exists(path):
        size_bytes = os.path.getsize(path)
        size_kb = size_bytes / 1024

        if orig_size > 0:
            pct = f"{(size_bytes / orig_size) * 100:6.2f}%"
        else:
            pct = "N/A"

        print(f"{label:<35} | {size_kb:>12.2f} KB | {pct:>14}")
    else:
        print(f"{label:<35} | {'NOT FOUND':>12} | {'N/A':>14}")
