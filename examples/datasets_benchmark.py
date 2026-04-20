import sys
import os
import time
import pickle
import numpy as np
import pandas as pd

# Adiciona o diretório pai ao sys.path for importar pyruleanalyzer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyruleanalyzer import PyRuleAnalyzer, RuleClassifier

# ==============================================================================
# CONFIGURAÇÃO of DATASETS
# Preencha os caminhos for os arquivos CSV of treino and test of cada dataset,
# bem as os parâmetros for cada algoritmo.
# ==============================================================================
DATASETS = [
    {
        "nome": "DDoS Attack Classification Leveraging Data Balancing (Artigo)",
        "train": "examples/data/DDoS Attack Classification Leveraging Data Balancing and Hyperparameter Tuning Approach Using Ensemble Machine Learning with XAI/train.csv",
        "test": "examples/data/DDoS Attack Classification Leveraging Data Balancing and Hyperparameter Tuning Approach Using Ensemble Machine Learning with XAI/test.csv",
        "param_dt": {"random_state": 42,"max_depth": 15},
        "param_rf": {"random_state": 42, "n_estimators": 10},
        "param_gbdt": {"random_state": 42, "n_estimators": 10}
    },
    {
        "nome": "A Machine Learning-Based Classification for DDoS (Artigo)",
        "train": "examples/data/A Machine Learning-Based Classification and Prediction Technique for DDoS Attacks/train.csv",
        "test": "examples/data/A Machine Learning-Based Classification and Prediction Technique for DDoS Attacks/test.csv",
        "param_dt": {"random_state": 42,"max_depth": 15},
        "param_rf": {"random_state": 42, "n_estimators": 10},
        "param_gbdt": {"random_state": 42, "n_estimators": 10}
    },
    {
        "nome": "Detecting DDoS Attacks using Decision Tree (Artigo)",
        "train": "examples/data/Detecting_DDoS_Attacks_using_Decision_Tree_Algorithm/train-Detecting_DDoS_Attacks_using_Decision_Tree_Algorithm.csv",
        "test": "examples/data/Detecting_DDoS_Attacks_using_Decision_Tree_Algorithm/test-Detecting_DDoS_Attacks_using_Decision_Tree_Algorithm.csv",
        "param_dt": {"random_state": 42,"max_depth": 15},
        "param_rf": {"random_state": 42, "n_estimators": 10},
        "param_gbdt": {"random_state": 42, "n_estimators": 10, "max_depth": 5}
    },
    {
        "nome": "KDDCup99 - Network Intrusion & DoS (Public Benchmark)",
        "train": "examples/data/KDDCup99/train.csv",
        "test": "examples/data/KDDCup99/test.csv",
        "param_dt": {"random_state": 42,"max_depth": 15},
        "param_rf": {"random_state": 42, "n_estimators": 10, "max_depth": 10},
        "param_gbdt": {"random_state": 42, "n_estimators": 10, "max_depth": 5}
    }
]

MODELS_TO_TEST = [
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting Decision Trees"
]

OUTPUT_FILE = "files/benchmark_results.txt"

# Method to ensure dir.
def ensure_dir(file_path):
    """Ensure dir.

    Args:
        file_path: The file_path parameter.

    Returns:
        The result.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

# Method to format cell 3.
def format_cell_3(sk, unref, ref, format_str="{:.2f}"):
    """Format cell 3.

    Args:
        sk: The sk parameter.
        unref: The unref parameter.
        ref: The ref parameter.
        format_str: The format_str parameter.

    Returns:
        The result.
    """
    if isinstance(sk, (int, float)) and isinstance(unref, (int, float)) and isinstance(ref, (int, float)):
        return f"{format_str.format(sk)} / {format_str.format(unref)} / {format_str.format(ref)}"
    return f"{sk} / {unref} / {ref}"

# Method to format cell.
def format_cell(orig, opt, format_str="{:.2f}"):
    """Format cell.

    Args:
        orig: The orig parameter.
        opt: The opt parameter.
        format_str: The format_str parameter.

    Returns:
        The result.
    """
    if isinstance(orig, (int, float)) and isinstance(opt, (int, float)):
        return f"{format_str.format(orig)} ({format_str.format(opt)})"
    return f"{orig} ({opt})"

# Method to run benchmark.
def run_benchmark():
    """Run benchmark.


    Returns:
        The result.
    """
    ensure_dir(OUTPUT_FILE)
    
    # Cabeçalho da tabela em Markdown
    header = "| Estudo | Modelo | Acurácia (%) (Sk / Antes / Depois) | Regras (Sk / Antes / Depois) | Complexidade (Sk / Antes / Depois) | Tempo (s) (Sk / Antes / Depois) | Tamanho bytes (Sk / Antes / Depois) | Regras Duplicadas | Regras Específicas |\n"
    separator = "|---|---|---|---|---|---|---|---|---|\n"
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(separator)
    
    print(header.strip())
    print(separator.strip())

    for ds in DATASETS:
        if not ds["train"] or not ds["test"]:
            continue
            
        try:
            # Processa data for obter os arrays and nomes of features
            X_train, _, X_test, y_test, _, _, feature_names = RuleClassifier.process_data(ds["train"], ds["test"])
            n_features = len(feature_names)
            
            # Normalizar y_test for garantir compatibilidade (evitar string vs int failures)
            y_test_arr = np.array([int(float(p)) if isinstance(p, (str, float)) and str(p).replace('.','',1).isdigit() else p for p in y_test])
        except Exception as e:
            print(f"Erro ao carregar dados do dataset {ds['nome']}: {e}")
            continue

        for model_name in MODELS_TO_TEST:
            # Identifica a sigla and coleta os parâmetros adequados do dicionário do dataset
            if model_name == "Decision Tree":
                model_abbr = "DT"
                params = ds.get("param_dt", {})
                if not params: params = {'random_state': 42, 'max_depth': 10}
            elif model_name == "Random Forest":
                model_abbr = "RF"
                params = ds.get("param_rf", {})
                if not params: params = {'random_state': 42, 'n_estimators': 10, 'max_depth': 10}
            elif model_name == "Gradient Boosting Decision Trees":
                model_abbr = "GBDT"
                params = ds.get("param_gbdt", {})
                if not params: params = {'random_state': 42, 'n_estimators': 10, 'max_depth': 5}
                
            try:
                # 1. Cria and treina o model base (Sklearn) através do PyRuleAnalyzer, without refinar ainda
                # Passamos save_models=True for que ele exporte arquivos temporários as o sklearn_model.pkl
                analyzer = PyRuleAnalyzer.create(
                    train_path=ds["train"],
                    test_path=ds["test"],
                    model=model_name,
                    params=params,
                    refine=False,
                    save_models=True
                )
                
                classifier = analyzer.classifier
                
                # --- MÉTRICAS ORIGINAIS ---
                sk_path = 'files/sklearn_model.pkl'
                with open(sk_path, 'rb') as f:
                    sk_orig = pickle.load(f)
                
                # Acurácia and Tempo Original (Sklearn)
                start_time = time.time()
                # Agora que PyRuleAnalyzer treina todos os modelos (DT, RF, GBDT) usando pandas DataFrame,
                # temos que usar um DataFrame no predict for evitar UserWarnings do Scikit-Learn
                X_test_predict = pd.DataFrame(X_test, columns=feature_names)
                y_pred_orig = sk_orig.predict(X_test_predict)
                t_orig = time.time() - start_time
                
                y_pred_orig = np.array([int(float(p)) if isinstance(p, (str, float)) and str(p).replace('.','',1).isdigit() else p for p in y_pred_orig])
                acc_orig = np.mean(y_pred_orig == y_test_arr) * 100
                
                # rules and Complexidade Originais
                rules_orig = len(classifier.initial_rules)
                comp_dict_orig = RuleClassifier.calculate_structural_complexity(classifier.initial_rules, n_features)
                comp_orig = comp_dict_orig.get("complexity_score", 0.0)
                
                # Tamanho Original (Sklearn Pickle)
                size_orig = os.path.getsize(sk_path) if os.path.exists(sk_path) else 0

                # --- MÉTRICAS PRÉ-REFINAMENTO (PYRULEANALYZER) ---
                classifier.compile_tree_arrays(feature_names=feature_names)
                
                # Tempo Pré-Refinamento
                start_time = time.time()
                y_pred_unref = classifier.predict_batch(X_test, feature_names=feature_names)
                t_unref = time.time() - start_time
                
                # Acurácia Pré-Refinamento
                y_pred_unref_arr = np.array([int(float(p)) if isinstance(p, (str, float)) and str(p).replace('.','',1).isdigit() else p for p in y_pred_unref])
                acc_unref = np.mean(y_pred_unref_arr == y_test_arr) * 100
                
                # rules and Complexidade Pré-Refinamento (iguais às originais do Sklearn)
                rules_unref = rules_orig
                comp_unref = comp_orig
                
                # Tamanho Pré-Refinamento
                bin_path_unref = "files/temp_model_unref.bin"
                classifier.export_all(base_name="files/temp_model_unref", feature_names=feature_names, export_binary=True, export_python=False, export_c=False)
                size_unref = os.path.getsize(bin_path_unref) if os.path.exists(bin_path_unref) else 0

                # --- REFINAMENTO (REMOÇÃO of REDUNDÂNCIAS) ---
                if model_name == "Decision Tree":
                    from pyruleanalyzer.dt_analyzer import DTAnalyzer
                    model_analyzer = DTAnalyzer(classifier)
                    threshold = 1
                elif model_name == "Random Forest":
                    from pyruleanalyzer.rf_analyzer import RFAnalyzer
                    model_analyzer = RFAnalyzer(classifier)
                    threshold = 1
                elif model_name == "Gradient Boosting Decision Trees":
                    from pyruleanalyzer.gbdt_analyzer import GBDTAnalyzer
                    model_analyzer = GBDTAnalyzer(classifier)
                    threshold = 1
                
                # Executa o refinamento internamente for contabilizar as rules
                model_analyzer.execute_rule_refinement(
                    file_path=ds["test"],
                    remove_below_n_classifications=threshold,
                    save_final_model=True,
                    save_report=False
                )
                
                # Contabilização of Redundâncias
                dup_rules = model_analyzer.redundancy_counts.get("intra_tree", 0) + model_analyzer.redundancy_counts.get("inter_tree", 0)
                spec_rules = model_analyzer.redundancy_counts.get("low_usage", 0)
                
                # --- MÉTRICAS OTIMIZADAS ---
                # Compila primeiro for permitir a predição vetorizada and a exportação binária
                classifier.compile_tree_arrays(feature_names=feature_names)
                
                # Exporta for binário for obter o tamanho em bytes do model PyRuleAnalyzer final
                bin_path = "files/temp_model.bin"
                classifier.export_all(base_name="files/temp_model", feature_names=feature_names, export_binary=True, export_python=False, export_c=False)
                size_opt = os.path.getsize(bin_path) if os.path.exists(bin_path) else 0
                
                # Tempo and Acurácia Otimizada (Vectorized Prediction)
                start_time = time.time()
                y_pred_opt = classifier.predict_batch(X_test, feature_names=feature_names)
                t_opt = time.time() - start_time
                
                y_pred_opt = np.array([int(float(p)) if isinstance(p, (str, float)) and str(p).replace('.','',1).isdigit() else p for p in y_pred_opt])
                acc_opt = np.mean(y_pred_opt == y_test_arr) * 100
                
                # rules and Complexidade Otimizadas
                rules_opt = len(classifier.final_rules)
                comp_dict_opt = RuleClassifier.calculate_structural_complexity(classifier.final_rules, n_features)
                comp_opt = comp_dict_opt.get("complexity_score", 0.0)
                
                # --- FORMATAÇÃO and SALVAMENTO ---
                acc_str = format_cell_3(acc_orig, acc_unref, acc_opt, "{:.2f}")
                rules_str = format_cell_3(rules_orig, rules_unref, rules_opt, "{}")
                comp_str = format_cell_3(comp_orig, comp_unref, comp_opt, "{:.2f}")
                time_str = format_cell_3(t_orig, t_unref, t_opt, "{:.4f}")
                size_str = format_cell_3(size_orig, size_unref, size_opt, "{}")
                
                row = f"| {ds['nome']} | {model_abbr} | {acc_str} | {rules_str} | {comp_str} | {time_str} | {size_str} | {dup_rules} | {spec_rules} |\n"
                
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(row)
                
                print(row.strip())
                
            except Exception as e:
                print(f"| {ds['nome']} | {model_abbr} | ERRO | - | - | - | - | - | - |")
                print(f"    Detalhes do erro ({model_abbr}): {e}")

if __name__ == "__main__":
    print("="*80)
    print("BENCHMARK PYRULEANALYZER - TESTE DE BASES DE DADOS")
    print("="*80)
    print("Por favor, preencha os caminhos de 'train' e 'test', além dos parâmetros de treino, na lista DATASETS no arquivo do script.")
    print("Os resultados serão iterados e salvos formatados no diretório files/\n")
    
    run_benchmark()
