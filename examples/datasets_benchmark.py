import sys
import os
import time
import pickle
import numpy as np
import pandas as pd

# Adiciona o diretório pai ao sys.path para importar pyruleanalyzer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyruleanalyzer import PyRuleAnalyzer, RuleClassifier

# ==============================================================================
# CONFIGURAÇÃO DE DATASETS
# Preencha os caminhos para os arquivos CSV de treino e teste de cada dataset,
# bem como os parâmetros para cada algoritmo.
# ==============================================================================
DATASETS = [
    {"nome": "Agha e Rehman", "train": "", "test": "", "param_dt": {}, "param_rf": {}, "param_gbdt": {}},
    {"nome": "Almaraz-Rivera et al.", "train": "", "test": "", "param_dt": {}, "param_rf": {}, "param_gbdt": {}},
    {"nome": "Gao et al.", "train": "", "test": "", "param_dt": {}, "param_rf": {}, "param_gbdt": {}},
    {"nome": "Al-Fayoumi e Al-Haija", "train": "", "test": "", "param_dt": {}, "param_rf": {}, "param_gbdt": {}},
    {"nome": "Amalraj e Madhusankha", "train": "", "test": "", "param_dt": {}, "param_rf": {}, "param_gbdt": {}},
    {"nome": "Chen et al.", "train": "", "test": "", "param_dt": {}, "param_rf": {}, "param_gbdt": {}},
    {"nome": "Maniriho et al.", "train": "", "test": "", "param_dt": {}, "param_rf": {}, "param_gbdt": {}},
    {"nome": "Alqahtani et al.", "train": "", "test": "", "param_dt": {}, "param_rf": {}, "param_gbdt": {}},
    {"nome": "Becerra-Suarez et al.", "train": "", "test": "", "param_dt": {}, "param_rf": {}, "param_gbdt": {}},
    {"nome": "Slyamkhanov et al.", "train": "", "test": "", "param_dt": {}, "param_rf": {}, "param_gbdt": {}}
]

MODELS_TO_TEST = [
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting Decision Trees"
]

OUTPUT_FILE = "files/benchmark_results.txt"

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def format_cell(orig, opt, format_str="{:.2f}"):
    if isinstance(orig, (int, float)) and isinstance(opt, (int, float)):
        return f"{format_str.format(orig)} ({format_str.format(opt)})"
    return f"{orig} ({opt})"

def run_benchmark():
    ensure_dir(OUTPUT_FILE)
    
    # Cabeçalho da tabela em Markdown
    header = "| Estudo | Modelo | Acurácia (%) | Regras | Complexidade | Tempo de Inferência (s) | Tamanho em bytes | Regras Duplicadas | Regras Específicas |\n"
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
            # Processa dados para obter os arrays e nomes de features
            X_train, _, X_test, y_test, _, _, feature_names = RuleClassifier.process_data(ds["train"], ds["test"])
            n_features = len(feature_names)
            
            # Normalizar y_test para garantir compatibilidade (evitar string vs int failures)
            y_test_arr = np.array([int(float(p)) if isinstance(p, (str, float)) and str(p).replace('.','',1).isdigit() else p for p in y_test])
        except Exception as e:
            print(f"Erro ao carregar dados do dataset {ds['nome']}: {e}")
            continue

        for model_name in MODELS_TO_TEST:
            # Identifica a sigla e coleta os parâmetros adequados do dicionário do dataset
            if model_name == "Decision Tree":
                model_abbr = "DT"
                params = ds.get("param_dt", {})
                if not params: params = {'random_state': 42}
            elif model_name == "Random Forest":
                model_abbr = "RF"
                params = ds.get("param_rf", {})
                if not params: params = {'random_state': 42, 'n_estimators': 100}
            elif model_name == "Gradient Boosting Decision Trees":
                model_abbr = "GBDT"
                params = ds.get("param_gbdt", {})
                if not params: params = {'random_state': 42, 'n_estimators': 100}
                
            try:
                # 1. Cria e treina o modelo base (Sklearn) através do PyRuleAnalyzer, sem refinar ainda
                analyzer = PyRuleAnalyzer.create(
                    train_path=ds["train"],
                    test_path=ds["test"],
                    model=model_name,
                    params=params,
                    refine=False
                )
                
                classifier = analyzer.classifier
                
                # --- MÉTRICAS ORIGINAIS ---
                sk_path = 'files/sklearn_model.pkl'
                with open(sk_path, 'rb') as f:
                    sk_orig = pickle.load(f)
                
                # Acurácia e Tempo Original (Sklearn)
                start_time = time.time()
                X_test_df = pd.DataFrame(X_test, columns=feature_names)
                y_pred_orig = sk_orig.predict(X_test_df)
                t_orig = time.time() - start_time
                
                y_pred_orig = np.array([int(float(p)) if isinstance(p, (str, float)) and str(p).replace('.','',1).isdigit() else p for p in y_pred_orig])
                acc_orig = np.mean(y_pred_orig == y_test_arr) * 100
                
                # Regras e Complexidade Originais
                rules_orig = len(classifier.initial_rules)
                comp_dict_orig = RuleClassifier.calculate_structural_complexity(classifier.initial_rules, n_features)
                comp_orig = comp_dict_orig.get("complexity_score", 0.0)
                
                # Tamanho Original (Sklearn Pickle)
                size_orig = os.path.getsize(sk_path) if os.path.exists(sk_path) else 0

                # --- REFINAMENTO (REMOÇÃO DE REDUNDÂNCIAS) ---
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
                
                # Executa o refinamento internamente para contabilizar as regras
                model_analyzer.execute_rule_refinement(
                    file_path=ds["test"],
                    remove_below_n_classifications=threshold,
                    save_final_model=True,
                    save_report=False
                )
                
                # Contabilização de Redundâncias
                dup_rules = model_analyzer.redundancy_counts.get("intra_tree", 0) + model_analyzer.redundancy_counts.get("inter_tree", 0)
                spec_rules = model_analyzer.redundancy_counts.get("low_usage", 0)
                
                # --- MÉTRICAS OTIMIZADAS ---
                # Compila primeiro para permitir a predição vetorizada e a exportação binária
                classifier.compile_tree_arrays(feature_names=feature_names)
                
                # Exporta para binário para obter o tamanho em bytes do modelo PyRuleAnalyzer final
                bin_path = "files/temp_model.bin"
                classifier.export_all(base_name="files/temp_model", feature_names=feature_names, export_binary=True, export_python=False, export_c=False)
                size_opt = os.path.getsize(bin_path) if os.path.exists(bin_path) else 0
                
                # Tempo e Acurácia Otimizada (Vectorized Prediction)
                start_time = time.time()
                y_pred_opt = classifier.predict_batch(X_test, feature_names=feature_names)
                t_opt = time.time() - start_time
                
                y_pred_opt = np.array([int(float(p)) if isinstance(p, (str, float)) and str(p).replace('.','',1).isdigit() else p for p in y_pred_opt])
                acc_opt = np.mean(y_pred_opt == y_test_arr) * 100
                
                # Regras e Complexidade Otimizadas
                rules_opt = len(classifier.final_rules)
                comp_dict_opt = RuleClassifier.calculate_structural_complexity(classifier.final_rules, n_features)
                comp_opt = comp_dict_opt.get("complexity_score", 0.0)
                
                # --- FORMATAÇÃO E SALVAMENTO ---
                acc_str = format_cell(acc_orig, acc_opt, "{:.2f}")
                rules_str = format_cell(rules_orig, rules_opt, "{}")
                comp_str = format_cell(comp_orig, comp_opt, "{:.2f}")
                time_str = format_cell(t_orig, t_opt, "{:.3f}")
                size_str = format_cell(size_orig, size_opt, "{}")
                
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
