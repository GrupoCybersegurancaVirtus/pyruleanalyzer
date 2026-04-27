import sys
import os
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent directory to sys.path to import pyruleanalyzer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyruleanalyzer import PyRuleAnalyzer, RuleClassifier

# ==============================================================================
# CONFIGURATION of DATASETS
# Fill in the paths for train and test CSV files of each dataset,
# as well as the parameters for each algorithm.
# ==============================================================================
DATASETS = [
    {
        "nome": "Usage of Machine Learning in DDOS Attack Detection (Article)",
        "train": "examples/data/Usage_of_Machine_Learning_in_DDOS_Attack_Detection/train.csv",
        "test": "examples/data/Usage_of_Machine_Learning_in_DDOS_Attack_Detection/test.csv",
        "param_dt": {"random_state": 42},
        "param_rf": {"n_estimators": 300, "random_state": 42},
        "param_gbdt": {"random_state":42}
    },
    {
        "nome": "Improvement of Distributed Denial of Service Attack Detection through Machine Learning and Data Processing (Article)",
        "train": "examples/data/Improvement_of_Distributed_Denial_of_Service_Attack_Detection_through_Machine_Learning_and_Data_Processing/train.csv",
        "test": "examples/data/Improvement_of_Distributed_Denial_of_Service_Attack_Detection_through_Machine_Learning_and_Data_Processing/test.csv",
        "param_dt": {"criterion":'gini', "splitter":'best', "max_depth":7, "min_samples_split":24, "min_samples_leaf":10, "random_state":42},
        "param_rf": {"criterion":'entropy', "n_estimators":43, "max_depth":12, "max_features":0.9145, "random_state":42},
        "param_gbdt": {"random_state":42}
    },
    {
        "nome": "Machine Learning Techniques for Detecting DDOS Attacks (Article)",
        "train": "examples/data/Machine_Learning_Techniques_for_Detecting_DDOS_Attacks/train.csv",
        "test": "examples/data/Machine_Learning_Techniques_for_Detecting_DDOS_Attacks/test.csv",
        "param_dt": {"random_state":42},
        "param_rf": {"random_state":42},
        "param_gbdt": {"random_state":42}
    },
    {
        "nome": "A Paradigm for DoS Attack Disclosure using Machine Learning Techniques (Article)",
        "train": "examples/data/A_Paradigm_for_DoS_Attack_Disclosure_using_Machine_Learning_Techniques/train.csv",
        "test": "examples/data/A_Paradigm_for_DoS_Attack_Disclosure_using_Machine_Learning_Techniques/test.csv",
        "param_dt": {"random_state":42}, # only dt
        "param_rf": {"random_state":42},
        "param_gbdt": {"random_state":42}
    },
    {
        "nome": "Anomaly detection in NetFlow network traffic using supervised machine learning algorithms (Article)",
        "train": "examples/data/Anomaly_detection_in_NetFlow_network_traffic_using_supervised_machine_learning_algorithms/train.csv",
        "test": "examples/data/Anomaly_detection_in_NetFlow_network_traffic_using_supervised_machine_learning_algorithms/test.csv",
        "param_dt": {"random_state": 42},
        "param_rf": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        "param_gbdt": {"random_state":42}
    },
    # PW bases
    {
        "nome": "A Comprehensive Analysis of Network Security Attack Classification using Machine Learning Algorithms (Article)",
        "train": "examples/data/A_Comprehensive_Analysis_of_Network_Security_Attack_Classification_using_Machine_Learning_Algorithms/train.csv",
        "test": "examples/data/A_Comprehensive_Analysis_of_Network_Security_Attack_Classification_using_Machine_Learning_Algorithms/test.csv",
        "param_dt": {"random_state":42},
        "param_rf": {"random_state":42}, # only rf
        "param_gbdt": {"random_state":42},
    },
    {
        "nome": "An Evaluation of Machine Learning Methods for Classifying Bot Traffic in Software Defined Networks (Article)",
        "train": "examples/data/An_Evaluation_of_Machine_Learning_Methods_for_Classifying_Bot_Traffic_in_Software_Defined_Networks/train.csv",
        "test": "examples/data/An_Evaluation_of_Machine_Learning_Methods_for_Classifying_Bot_Traffic_in_Software_Defined_Networks/test.csv",
        "param_dt": {"random_state":42},
        "param_rf": {"random_state":42}, # only rf
        "param_gbdt": {"random_state":42}
    },
    {
        "nome": "Capturing low-rate DDoS attack based on MQTT protocol in software Defined-IoT environment (Article)",
        "train": "examples/data/Capturing_low-rate_DDoS_attack_based_on_MQTT_protocol_in_software_Defined-IoT_environment/train.csv",
        "test": "examples/data/Capturing_low-rate_DDoS_attack_based_on_MQTT_protocol_in_software_Defined-IoT_environment/test.csv",
        "param_dt": {"random_state":42}, # only dt
        "param_rf": {"random_state":42},
        "param_gbdt": {"random_state":42}
    },
    {
        "nome": "Detecting DDoS Attacks using Decision Tree Algorithm (Article)",
        "train": "examples/data/Detecting_DDoS_Attacks_using_Decision_Tree_Algorithm/train.csv",
        "test": "examples/data/Detecting_DDoS_Attacks_using_Decision_Tree_Algorithm/test.csv",
        "param_dt": {"random_state":42},
        "param_rf": {"random_state":42},
        "param_gbdt": {"random_state":42}
    },
    {
        "nome": "Detection of DDoS Attacks using Machine Learning Algorithms (Article)",
        "train": "examples/data/Detection_of_DDoS_Attacks_using_Machine_Learning_Algorithms/train.csv",
        "test": "examples/data/Detection_of_DDoS_Attacks_using_Machine_Learning_Algorithms/test.csv",
        "param_dt": {"random_state":42},
        "param_rf": {"random_state":42},
        "param_gbdt": {"random_state":42}
    },
    {
        "nome": "Enhancing DDoS Attack Detection via Blending Ensemble Learning (Article)",
        "train": "examples/data/Enhancing_DDoS_Attack_Detection_via_Blending_Ensemble_Learning/train.csv",
        "test": "examples/data/Enhancing_DDoS_Attack_Detection_via_Blending_Ensemble_Learning/test.csv",
        "param_dt": {"random_state":42},
        "param_rf": {"random_state":42},
        "param_gbdt": {"random_state":42}
    },
]

MODELS_TO_TEST = [
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting Decision Trees"
]

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = f"files/benchmark_results_{current_time}.txt"

# Method to ensure directory exists.
def ensure_dir(file_path):
    """Ensure directory exists.

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
    
    # Markdown table header
    header = "| Study | Model | Accuracy (%) (Sk / Before / After) | Rules (Sk / Before / After) | Complexity (Sk / Before / After) | Time (s) (Sk / Before / After) | Size bytes (Sk / Before / After) | Duplicated Rules | Specific Rules |\n"
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
            # Process data to obtain arrays and feature names
            X_train, _, X_test, y_test, _, _, feature_names = RuleClassifier.process_data(ds["train"], ds["test"])
            n_features = len(feature_names)
            
            # Normalize y_test to ensure compatibility (avoid string vs int failures)
            y_test_arr = np.array([int(float(p)) if isinstance(p, (str, float)) and str(p).replace('.','',1).isdigit() else p for p in y_test])
        except Exception as e:
            print(f"Error loading data from dataset {ds['nome']}: {e}")
            continue

        for model_name in MODELS_TO_TEST:
            # Identify the abbreviation and collect appropriate parameters from the dataset dictionary
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
                # 1. Create and train the base model (Sklearn) via PyRuleAnalyzer, without refining yet
                # We pass save_models=True so it exports temporary files to sklearn_model.pkl
                analyzer = PyRuleAnalyzer.create(
                    train_path=ds["train"],
                    test_path=ds["test"],
                    model=model_name,
                    params=params,
                    refine=False,
                    save_models=True
                )
                
                classifier = analyzer.classifier
                
                # --- ORIGINAL METRICS ---
                sk_path = 'files/sklearn_model.pkl'
                with open(sk_path, 'rb') as f:
                    sk_orig = pickle.load(f)
                
                # Accuracy and Original Time (Sklearn)
                start_time = time.time()
                # Now that PyRuleAnalyzer trains all models (DT, RF, GBDT) using pandas DataFrame,
                # we need to use a DataFrame in predict to avoid UserWarnings from Scikit-Learn
                X_test_predict = pd.DataFrame(X_test, columns=feature_names)
                y_pred_orig = sk_orig.predict(X_test_predict)
                t_orig = time.time() - start_time
                
                y_pred_orig = np.array([int(float(p)) if isinstance(p, (str, float)) and str(p).replace('.','',1).isdigit() else p for p in y_pred_orig])
                acc_orig = np.mean(y_pred_orig == y_test_arr) * 100
                
                # Original rules and Complexity
                rules_orig = len(classifier.initial_rules)
                comp_dict_orig = RuleClassifier.calculate_structural_complexity(classifier.initial_rules, n_features)
                comp_orig = comp_dict_orig.get("complexity_score", 0.0)
                
                # Original Size (Sklearn Pickle)
                size_orig = os.path.getsize(sk_path) if os.path.exists(sk_path) else 0

                # --- PRE-REFINEMENT METRICS (PYRULEANALYZER) ---
                classifier.compile_tree_arrays(feature_names=feature_names)
                
                # Pre-Refinement Time
                start_time = time.time()
                y_pred_unref = classifier.predict_batch(X_test, feature_names=feature_names)
                t_unref = time.time() - start_time
                
                # Pre-Refinement Accuracy
                y_pred_unref_arr = np.array([int(float(p)) if isinstance(p, (str, float)) and str(p).replace('.','',1).isdigit() else p for p in y_pred_unref])
                acc_unref = np.mean(y_pred_unref_arr == y_test_arr) * 100
                
                # Pre-Refinement rules and Complexity (same as original Sklearn metrics)
                rules_unref = rules_orig
                comp_unref = comp_orig
                
                # Pre-Refinement Size
                bin_path_unref = "files/temp_model_unref.bin"
                classifier.export_all(base_name="files/temp_model_unref", feature_names=feature_names, export_binary=True, export_python=False, export_c=False)
                size_unref = os.path.getsize(bin_path_unref) if os.path.exists(bin_path_unref) else 0

                # --- REFINEMENT (REDUNDANCY REMOVAL) ---
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
                
                # Execute refinement internally to count rules
                model_analyzer.execute_rule_refinement(
                    file_path=ds["test"],
                    remove_below_n_classifications=threshold,
                    save_final_model=True,
                    save_report=False
                )
                
                # Redundancy counting
                dup_rules = model_analyzer.redundancy_counts.get("intra_tree", 0) + model_analyzer.redundancy_counts.get("inter_tree", 0)
                spec_rules = model_analyzer.redundancy_counts.get("low_usage", 0)
                
                # --- OPTIMIZED METRICS ---
                # Compile first to allow vectorized prediction and binary export
                classifier.compile_tree_arrays(feature_names=feature_names)
                
                # Export to binary to get the final PyRuleAnalyzer model size in bytes
                bin_path = "files/temp_model.bin"
                classifier.export_all(base_name="files/temp_model", feature_names=feature_names, export_binary=True, export_python=False, export_c=False)
                size_opt = os.path.getsize(bin_path) if os.path.exists(bin_path) else 0
                
                # Optimized Time and Accuracy (Vectorized Prediction)
                start_time = time.time()
                y_pred_opt = classifier.predict_batch(X_test, feature_names=feature_names)
                t_opt = time.time() - start_time
                
                y_pred_opt = np.array([int(float(p)) if isinstance(p, (str, float)) and str(p).replace('.','',1).isdigit() else p for p in y_pred_opt])
                acc_opt = np.mean(y_pred_opt == y_test_arr) * 100
                
                # Optimized rules and Complexity
                rules_opt = len(classifier.final_rules)
                comp_dict_opt = RuleClassifier.calculate_structural_complexity(classifier.final_rules, n_features)
                comp_opt = comp_dict_opt.get("complexity_score", 0.0)
                
                # --- FORMATTING AND SAVING ---
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
                print(f"| {ds['nome']} | {model_abbr} | ERROR | - | - | - | - | - | - |")
                print(f"Error details ({model_abbr}): {e}")

if __name__ == "__main__":
    print("="*80)
    print("PYRULEANALYZER BENCHMARK - DATABASE TESTS")
    print("="*80)
    print("Please fill in the 'train' and 'test' paths, as well as training parameters, in the DATASETS list in the script file.")
    print("Results will be iterated and saved formatted in the files/ directory\n")
    
    run_benchmark()