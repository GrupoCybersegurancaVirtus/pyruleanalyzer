"""
Full-metric benchmark across the article datasets.

For every (dataset x model) pair it trains the model, extracts rules, refines
them, exports every artifact format and records everything measurable:
timings of each phase, model/rule structure, structural complexity, predictive
quality of the three engines (sklearn, initial rules, final rules), inference
throughput, peak memory, the size of every generated file and the structure of
the generated Coloured Petri Net.

Outputs (in --out-dir, default files/, one set per run):
    benchmark_<timestamp>.csv    one flat row per (dataset x model) - every metric
    benchmark_<timestamp>.json   the same plus nested detail (confusion matrix,
                                 per-class metrics, hyperparameters, environment)
    benchmark_<timestamp>.md     the readable markdown table (as before) plus a
                                 summary section

Usage:
    python examples/datasets_benchmark_example.py
    python examples/datasets_benchmark_example.py --dataset Detecting --model DT
    python examples/datasets_benchmark_example.py --no-cpn --no-arduino
    python examples/datasets_benchmark_example.py --keep-artifacts --timing-repeats 3

Configure the datasets and their hyperparameters in the DATASETS list below.
"""

import argparse
import contextlib
import json
import os
import pickle
import platform
import re
import shutil
import statistics
import sys
import time
import tracemalloc
from datetime import datetime

import numpy as np
import pandas as pd

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

MODEL_ABBR = {
    "Decision Tree": "DT",
    "Random Forest": "RF",
    "Gradient Boosting Decision Trees": "GBDT",
}

DEFAULT_PARAMS = {
    "Decision Tree": {'random_state': 42, 'max_depth': 10},
    "Random Forest": {'random_state': 42, 'n_estimators': 10, 'max_depth': 10},
    "Gradient Boosting Decision Trees": {'random_state': 42, 'n_estimators': 10, 'max_depth': 5},
}

# The .cpn grows with the rule count; a 300-tree forest would produce a net that
# takes minutes to write and hundreds of MB on disk. Above this many rules the
# Petri net stage is skipped and the reason is recorded.
MAX_RULES_FOR_CPN = 5000


# ==============================================================================
# SMALL HELPERS
# ==============================================================================

def ensure_dir(path):
    """Create the directory of `path` (or `path` itself when it has no extension)."""
    directory = path if not os.path.splitext(path)[1] else os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def slug(text, maxlen=60):
    """Filesystem-safe short name."""
    return re.sub(r'[^0-9A-Za-z]+', '_', text).strip('_')[:maxlen]


def file_size(path):
    """Size in bytes, or 0 when the file is missing."""
    return os.path.getsize(path) if path and os.path.exists(path) else 0


def count_lines(path):
    """Line count of a text artifact, or 0."""
    if not path or not os.path.exists(path):
        return 0
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def to_int_array(values):
    """Normalize labels to ints where possible (avoids str-vs-int mismatches)."""
    out = []
    for v in values:
        if v is None:
            out.append(-1)
            continue
        try:
            out.append(int(float(v)))
        except (TypeError, ValueError):
            out.append(v)
    return np.array(out)


class Phase:
    """Context manager measuring wall time and (optionally) peak Python memory.

    Usage:
        with Phase('refine', metrics, track_memory=True):
            ...
    Writes `time_<name>_s` and, when tracking, `peak_mem_<name>_mb`.
    """

    def __init__(self, name, sink, track_memory=False):
        self.name = name
        self.sink = sink
        self.track_memory = track_memory
        self.elapsed = 0.0

    def __enter__(self):
        if self.track_memory:
            tracemalloc.start()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
        self.sink[f'time_{self.name}_s'] = round(self.elapsed, 6)
        if self.track_memory:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.sink[f'peak_mem_{self.name}_mb'] = round(peak / (1024 * 1024), 3)
        return False


def rss_mb():
    """Resident set size in MB (0.0 when psutil is unavailable)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


# ==============================================================================
# METRIC COLLECTORS
# ==============================================================================

def environment_info():
    """Versions and machine identification, recorded once per run."""
    import sklearn
    return {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'python': platform.python_version(),
        'sklearn': sklearn.__version__,
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
    }


def dataset_info(ds, X_train, y_train, X_test, y_test, feature_names):
    """Shape and class balance of the data, plus the size of the CSVs."""
    y_train_arr = to_int_array(y_train) if y_train is not None else np.array([])
    y_test_arr = to_int_array(y_test)
    classes, counts = np.unique(y_test_arr, return_counts=True)
    dist = {str(c): int(n) for c, n in zip(classes, counts)}
    majority = max(counts) / counts.sum() * 100 if counts.size else 0.0
    return {
        'n_train_samples': int(len(X_train)) if X_train is not None else 0,
        'n_test_samples': int(len(X_test)),
        'n_features': len(feature_names),
        'n_classes': int(len(classes)),
        'class_distribution_test': dist,
        'majority_class_percent': round(float(majority), 4),
        'train_csv_bytes': file_size(ds['train']),
        'test_csv_bytes': file_size(ds['test']),
        'n_train_labels': int(len(y_train_arr)),
    }


def sklearn_structure(sk_model):
    """Node/leaf/depth statistics of the underlying scikit-learn estimator."""
    trees = []
    if hasattr(sk_model, 'tree_'):
        trees = [sk_model.tree_]
    elif hasattr(sk_model, 'estimators_'):
        estimators = np.ravel(sk_model.estimators_)
        trees = [e.tree_ for e in estimators if hasattr(e, 'tree_')]

    if not trees:
        return {'sk_n_trees': 0, 'sk_total_nodes': 0, 'sk_total_leaves': 0,
                'sk_max_depth': 0, 'sk_mean_depth': 0.0}

    nodes = [int(t.node_count) for t in trees]
    leaves = [int(np.sum(t.children_left == -1)) for t in trees]
    depths = [int(t.max_depth) for t in trees]
    return {
        'sk_n_trees': len(trees),
        'sk_total_nodes': int(sum(nodes)),
        'sk_total_leaves': int(sum(leaves)),
        'sk_mean_nodes_per_tree': round(float(statistics.mean(nodes)), 3),
        'sk_max_depth': max(depths),
        'sk_mean_depth': round(float(statistics.mean(depths)), 3),
    }


def rule_stats(rules, prefix):
    """Counts and condition statistics of a rule set."""
    if not rules:
        return {f'{prefix}_rules': 0}
    depths = [len(r.conditions) for r in rules]
    feats = set()
    per_tree = {}
    for r in rules:
        feats.update(item[0] for item in r.parsed_conditions)
        tid = r.name.split('_')[0] if '_' in r.name else 'tree0'
        per_tree[tid] = per_tree.get(tid, 0) + 1
    counts = list(per_tree.values())
    return {
        f'{prefix}_rules': len(rules),
        f'{prefix}_conditions_total': int(sum(depths)),
        f'{prefix}_conditions_mean': round(float(statistics.mean(depths)), 4),
        f'{prefix}_conditions_median': float(statistics.median(depths)),
        f'{prefix}_conditions_max': int(max(depths)),
        f'{prefix}_features_used': len(feats),
        f'{prefix}_n_tree_groups': len(per_tree),
        f'{prefix}_rules_per_tree_mean': round(float(statistics.mean(counts)), 3),
        f'{prefix}_rules_per_tree_max': int(max(counts)),
    }


def complexity_stats(rules, n_features, prefix):
    """Structural Complexity Score and its companion metrics."""
    d = RuleClassifier.calculate_structural_complexity(rules, n_features)
    return {
        f'{prefix}_scs': round(float(d.get('complexity_score', 0.0)), 6),
        f'{prefix}_scs_max_depth': int(d.get('max_depth', 0)),
        f'{prefix}_scs_mean_depth': round(float(d.get('mean_rule_depth', 0.0)), 4),
        f'{prefix}_scs_features_used': int(d.get('features_used', 0)),
    }


def quality_stats(y_true, y_pred, prefix):
    """Accuracy, precision/recall/F1 (macro and weighted), specificity, MCC."""
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 matthews_corrcoef, precision_score, recall_score)

    flat = {
        f'{prefix}_accuracy': round(float(accuracy_score(y_true, y_pred)) * 100, 4),
        f'{prefix}_precision_macro': round(float(precision_score(
            y_true, y_pred, average='macro', zero_division=0.0)) * 100, 4),
        f'{prefix}_recall_macro': round(float(recall_score(
            y_true, y_pred, average='macro', zero_division=0.0)) * 100, 4),
        f'{prefix}_f1_macro': round(float(f1_score(
            y_true, y_pred, average='macro', zero_division=0.0)) * 100, 4),
        f'{prefix}_f1_weighted': round(float(f1_score(
            y_true, y_pred, average='weighted', zero_division=0.0)) * 100, 4),
        f'{prefix}_mcc': round(float(matthews_corrcoef(y_true, y_pred)), 6),
    }

    labels = sorted(set(np.concatenate([np.unique(y_true), np.unique(y_pred)]).tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificities = []
    per_class = {}
    total = cm.sum()
    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(total - tp - fp - fn)
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        specificities.append(spec)
        support = int(cm[i, :].sum())
        per_class[str(label)] = {
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'support': support,
            'precision': round(tp / (tp + fp), 6) if (tp + fp) else 0.0,
            'recall': round(tp / (tp + fn), 6) if (tp + fn) else 0.0,
            'specificity': round(spec, 6),
        }
    flat[f'{prefix}_specificity_macro'] = round(
        float(statistics.mean(specificities)) * 100 if specificities else 0.0, 4)

    nested = {'labels': [str(l) for l in labels],
              'confusion_matrix': cm.tolist(),
              'per_class': per_class}
    return flat, nested


def timed_predict(fn, repeats):
    """Run `fn` `repeats` times; return (predictions, best_seconds, mean_seconds)."""
    times = []
    preds = None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        preds = fn()
        times.append(time.perf_counter() - t0)
    return preds, min(times), float(statistics.mean(times))


def throughput_stats(prefix, best_s, mean_s, n_samples):
    """Latency per sample and throughput derived from a prediction timing."""
    per_sample_us = (best_s / n_samples) * 1e6 if n_samples else 0.0
    return {
        f'time_predict_{prefix}_s': round(best_s, 6),
        f'time_predict_{prefix}_mean_s': round(mean_s, 6),
        f'latency_{prefix}_us_per_sample': round(per_sample_us, 4),
        f'throughput_{prefix}_samples_per_s': round(n_samples / best_s, 2) if best_s else 0.0,
    }


CPN_TAGS = ('page', 'place', 'trans', 'arc', 'subst', 'color', 'var', 'globbox',
            'instance', 'port', 'portsock')


def cpn_metrics(path, prefix):
    """Count the structural elements of a generated CPN Tools .cpn file.

    The exporter returns only the file path, so the net is measured by counting
    the XML elements it wrote. Ratios (places per rule, arcs per transition)
    are added by the caller, which knows the rule count.
    """
    out = {f'{prefix}_bytes': file_size(path), f'{prefix}_generated': bool(path)}
    if not path or not os.path.exists(path):
        return out
    try:
        import xml.etree.ElementTree as ET
        counts = {t: 0 for t in CPN_TAGS}
        for _, elem in ET.iterparse(path, events=('end',)):
            tag = elem.tag.split('}')[-1]
            if tag in counts:
                counts[tag] += 1
            elem.clear()
    except Exception:
        # Malformed or partially written file: fall back to a textual count
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        counts = {t: len(re.findall(rf'<{t}[\s>]', text)) for t in CPN_TAGS}

    out.update({
        f'{prefix}_pages': counts['page'],
        f'{prefix}_places': counts['place'],
        f'{prefix}_transitions': counts['trans'],
        f'{prefix}_arcs': counts['arc'],
        f'{prefix}_subst_transitions': counts['subst'],
        f'{prefix}_colorsets': counts['color'],
        f'{prefix}_variables': counts['var'],
        f'{prefix}_port_sockets': counts['portsock'],
        f'{prefix}_nodes': counts['place'] + counts['trans'],
        f'{prefix}_arcs_per_transition': round(counts['arc'] / counts['trans'], 4)
        if counts['trans'] else 0.0,
    })
    return out


# ==============================================================================
# ONE (DATASET x MODEL) RUN
# ==============================================================================

def run_case(ds, model_name, data, args, artifact_dir):
    """Collect every metric for one dataset/model pair.

    Returns (flat, nested). `flat` is one CSV row; `nested` carries the
    structures that do not fit a column (confusion matrix, per-class metrics,
    hyperparameters).
    """
    X_train, y_train, X_test, y_test, feature_names = data
    n_features = len(feature_names)
    n_test = len(X_test)
    y_test_arr = to_int_array(y_test)

    abbr = MODEL_ABBR[model_name]
    params = ds.get(f'param_{abbr.lower()}') or DEFAULT_PARAMS[model_name]

    flat = {'dataset': ds['nome'], 'model': abbr, 'model_full': model_name,
            'status': 'ok', 'error': ''}
    nested = {'hyperparameters': dict(params)}
    flat.update(dataset_info(ds, X_train, y_train, X_test, y_test, feature_names))

    ensure_dir(artifact_dir)
    base = os.path.join(artifact_dir, f'{abbr.lower()}')
    rss_before = rss_mb()

    # --- 1. training + rule extraction -------------------------------------
    with Phase('train_extract', flat, args.memory):
        analyzer = PyRuleAnalyzer.create(
            train_path=ds['train'], test_path=ds['test'], model=model_name,
            params=params, refine=False, save_models=True,
        )
    classifier = analyzer.classifier

    sk_path = 'files/sklearn_model.pkl'
    with open(sk_path, 'rb') as f:
        sk_model = pickle.load(f)
    flat.update(sklearn_structure(sk_model))
    flat['size_sklearn_pkl_bytes'] = file_size(sk_path)

    # --- 2. initial rules ---------------------------------------------------
    initial_rules = list(classifier.initial_rules)
    flat.update(rule_stats(initial_rules, 'initial'))
    flat.update(complexity_stats(initial_rules, n_features, 'initial'))

    with Phase('compile_arrays_initial', flat):
        classifier.compile_tree_arrays(feature_names=feature_names)

    # A DataFrame avoids sklearn's "fitted with feature names" warning, but only
    # when the estimator was actually fitted with them.
    sk_input = (pd.DataFrame(X_test, columns=feature_names)
                if hasattr(sk_model, 'feature_names_in_') else X_test)

    # --- 3. predictions: sklearn, initial rules ----------------------------
    y_sk, sk_best, sk_mean = timed_predict(
        lambda: sk_model.predict(sk_input), args.timing_repeats)
    y_sk = to_int_array(y_sk)
    flat.update(throughput_stats('sklearn', sk_best, sk_mean, n_test))
    q, n = quality_stats(y_test_arr, y_sk, 'sklearn')
    flat.update(q)
    nested['sklearn'] = n

    mem_sink = {}
    with Phase('predict_initial', mem_sink, args.memory):
        y_ini, ini_best, ini_mean = timed_predict(
            lambda: classifier.predict_batch(X_test, feature_names=feature_names),
            args.timing_repeats)
    if 'peak_mem_predict_initial_mb' in mem_sink:
        flat['peak_mem_predict_initial_mb'] = mem_sink['peak_mem_predict_initial_mb']
    y_ini = to_int_array(y_ini)
    flat.update(throughput_stats('initial', ini_best, ini_mean, n_test))
    q, n = quality_stats(y_test_arr, y_ini, 'initial')
    flat.update(q)
    nested['initial'] = n
    flat['fidelity_initial_vs_sklearn'] = round(float(np.mean(y_ini == y_sk)) * 100, 4)
    flat['disagreements_initial_vs_sklearn'] = int(np.sum(y_ini != y_sk))

    # --- 4. artifacts of the initial model ---------------------------------
    with Phase('export_initial', flat):
        classifier.export_all(base_name=f'{base}_initial', feature_names=feature_names,
                              export_binary=True, export_python=True, export_c=True)
    flat['size_initial_bin_bytes'] = file_size(f'{base}_initial.bin')
    flat['size_initial_py_bytes'] = file_size(f'{base}_initial.py')
    flat['size_initial_h_bytes'] = file_size(f'{base}_initial.h')
    flat['lines_initial_py'] = count_lines(f'{base}_initial.py')
    flat['lines_initial_h'] = count_lines(f'{base}_initial.h')

    if args.cpn:
        flat.update(cpn_for(classifier, initial_rules, feature_names,
                            f'{base}_initial.cpn', 'cpn_initial', flat))

    # --- 5. refinement ------------------------------------------------------
    analyzer_cls = {
        'DT': ('pyruleanalyzer.dt_analyzer', 'DTAnalyzer'),
        'RF': ('pyruleanalyzer.rf_analyzer', 'RFAnalyzer'),
        'GBDT': ('pyruleanalyzer.gbdt_analyzer', 'GBDTAnalyzer'),
    }[abbr]
    module = __import__(analyzer_cls[0], fromlist=[analyzer_cls[1]])
    model_analyzer = getattr(module, analyzer_cls[1])(classifier)

    with Phase('refine', flat, args.memory):
        model_analyzer.execute_rule_refinement(
            file_path=ds['test'], remove_below_n_classifications=args.threshold,
            save_final_model=True, save_report=False,
        )

    counts = model_analyzer.redundancy_counts
    flat['redundancies_intra_tree'] = int(counts.get('intra_tree', 0))
    flat['redundancies_inter_tree'] = int(counts.get('inter_tree', 0))
    flat['redundancies_low_usage'] = int(counts.get('low_usage', 0))
    flat['redundancies_total'] = int(sum(counts.values()))
    flat['duplicated_rules'] = flat['redundancies_intra_tree'] + flat['redundancies_inter_tree']
    flat['specific_rules'] = flat['redundancies_low_usage']

    # --- 6. final rules -----------------------------------------------------
    final_rules = list(classifier.final_rules) if classifier.final_rules else initial_rules
    flat.update(rule_stats(final_rules, 'final'))
    flat.update(complexity_stats(final_rules, n_features, 'final'))

    with Phase('compile_arrays_final', flat):
        classifier.compile_tree_arrays(feature_names=feature_names)

    mem_sink = {}
    with Phase('predict_final', mem_sink, args.memory):
        y_fin, fin_best, fin_mean = timed_predict(
            lambda: classifier.predict_batch(X_test, feature_names=feature_names),
            args.timing_repeats)
    if 'peak_mem_predict_final_mb' in mem_sink:
        flat['peak_mem_predict_final_mb'] = mem_sink['peak_mem_predict_final_mb']
    y_fin = to_int_array(y_fin)
    flat.update(throughput_stats('final', fin_best, fin_mean, n_test))
    q, n = quality_stats(y_test_arr, y_fin, 'final')
    flat.update(q)
    nested['final'] = n
    flat['fidelity_final_vs_sklearn'] = round(float(np.mean(y_fin == y_sk)) * 100, 4)
    flat['disagreements_final_vs_sklearn'] = int(np.sum(y_fin != y_sk))
    flat['agreement_final_vs_initial'] = round(float(np.mean(y_fin == y_ini)) * 100, 4)

    # --- 7. artifacts of the final model -----------------------------------
    with Phase('export_final', flat):
        classifier.export_all(base_name=f'{base}_final', feature_names=feature_names,
                              export_binary=True, export_python=True, export_c=True)
    flat['size_final_bin_bytes'] = file_size(f'{base}_final.bin')
    flat['size_final_py_bytes'] = file_size(f'{base}_final.py')
    flat['size_final_h_bytes'] = file_size(f'{base}_final.h')
    flat['lines_final_py'] = count_lines(f'{base}_final.py')
    flat['lines_final_h'] = count_lines(f'{base}_final.h')
    flat['size_initial_model_pkl_bytes'] = file_size('files/initial_model.pkl')
    flat['size_final_model_pkl_bytes'] = file_size('files/final_model.pkl')

    if args.arduino:
        try:
            with Phase('export_arduino', flat):
                res = classifier.export_to_arduino_ino(
                    filepath=f'{base}_final.ino', board_model='auto',
                    include_memory_check=True)
            mc = res.get('memory_check', {})
            flat['size_final_ino_bytes'] = file_size(f'{base}_final.ino')
            flat['arduino_board'] = mc.get('board_model', '')
            flat['arduino_flash_bytes'] = int(mc.get('flash_bytes', 0))
            flat['arduino_ram_bytes'] = int(mc.get('ram_bytes', 0))
            flat['arduino_flash_percent'] = round(float(mc.get('flash_percent', 0.0)), 3)
            flat['arduino_ram_percent'] = round(float(mc.get('ram_percent', 0.0)), 3)
            flat['arduino_fits_flash'] = bool(mc.get('fits_flash', False))
            flat['arduino_fits_ram'] = bool(mc.get('fits_ram', False))
        except Exception as exc:
            flat['arduino_error'] = str(exc)[:200]

    if args.cpn:
        flat.update(cpn_for(classifier, final_rules, feature_names,
                            f'{base}_final.cpn', 'cpn_final', flat))

    # --- 8. derived ratios --------------------------------------------------
    ini_r, fin_r = flat.get('initial_rules', 0), flat.get('final_rules', 0)
    flat['rules_removed'] = ini_r - fin_r
    flat['rules_reduction_percent'] = round(
        (ini_r - fin_r) / ini_r * 100, 4) if ini_r else 0.0
    ini_scs, fin_scs = flat.get('initial_scs', 0.0), flat.get('final_scs', 0.0)
    flat['scs_reduction_percent'] = round(
        (ini_scs - fin_scs) / ini_scs * 100, 4) if ini_scs else 0.0
    flat['accuracy_delta_final_minus_sklearn'] = round(
        flat['final_accuracy'] - flat['sklearn_accuracy'], 4)
    flat['size_reduction_bin_percent'] = round(
        (flat['size_initial_bin_bytes'] - flat['size_final_bin_bytes'])
        / flat['size_initial_bin_bytes'] * 100, 4) if flat['size_initial_bin_bytes'] else 0.0
    flat['bytes_per_rule_final'] = round(
        flat['size_final_bin_bytes'] / fin_r, 3) if fin_r else 0.0
    flat['speedup_sklearn_over_final'] = round(
        flat['time_predict_sklearn_s'] / flat['time_predict_final_s'], 4) \
        if flat.get('time_predict_final_s') else 0.0
    if flat.get('cpn_final_places') and fin_r:
        flat['cpn_final_places_per_rule'] = round(flat['cpn_final_places'] / fin_r, 4)
        flat['cpn_final_bytes_per_rule'] = round(flat['cpn_final_bytes'] / fin_r, 2)

    flat['total_artifact_bytes'] = sum(
        v for k, v in flat.items()
        if k.endswith('_bytes') and isinstance(v, (int, float))
        and (k.startswith('size_') or k.startswith('cpn_')))
    flat['time_total_s'] = round(sum(
        v for k, v in flat.items() if k.startswith('time_') and isinstance(v, (int, float))), 6)
    flat['rss_delta_mb'] = round(rss_mb() - rss_before, 3)

    if not args.keep_artifacts:
        shutil.rmtree(artifact_dir, ignore_errors=True)

    return flat, nested


def cpn_for(classifier, rules, feature_names, path, prefix, flat):
    """Export one Coloured Petri Net and measure it (guarded by rule count)."""
    out = {}
    if len(rules) > MAX_RULES_FOR_CPN:
        out[f'{prefix}_generated'] = False
        out[f'{prefix}_skipped_reason'] = f'{len(rules)} rules > MAX_RULES_FOR_CPN'
        return out
    try:
        t0 = time.perf_counter()
        classifier.to_cpn_tools(filepath=path, rules=rules,
                                feature_names=feature_names,
                                use_final=(prefix == 'cpn_final'))
        out[f'time_{prefix}_export_s'] = round(time.perf_counter() - t0, 6)
        out.update(cpn_metrics(path, prefix))
    except Exception as exc:
        out[f'{prefix}_generated'] = False
        out[f'{prefix}_error'] = str(exc)[:200]
    return out


# ==============================================================================
# OUTPUT
# ==============================================================================

MD_COLUMNS = [
    ('Study', lambda r: r['dataset']),
    ('Model', lambda r: r['model']),
    ('Accuracy (%) (Sk / Before / After)', lambda r: '{:.2f} / {:.2f} / {:.2f}'.format(
        r['sklearn_accuracy'], r['initial_accuracy'], r['final_accuracy'])),
    ('F1 macro (Sk / Before / After)', lambda r: '{:.2f} / {:.2f} / {:.2f}'.format(
        r['sklearn_f1_macro'], r['initial_f1_macro'], r['final_f1_macro'])),
    ('Rules (Before / After)', lambda r: '{} / {}'.format(
        r['initial_rules'], r['final_rules'])),
    ('SCS (Before / After)', lambda r: '{:.2f} / {:.2f}'.format(
        r['initial_scs'], r['final_scs'])),
    ('Predict time s (Sk / Before / After)', lambda r: '{:.4f} / {:.4f} / {:.4f}'.format(
        r['time_predict_sklearn_s'], r['time_predict_initial_s'], r['time_predict_final_s'])),
    ('Gen time s (train / refine)', lambda r: '{:.2f} / {:.2f}'.format(
        r.get('time_train_extract_s', 0.0), r.get('time_refine_s', 0.0))),
    ('Size bytes (Sk / Before / After)', lambda r: '{} / {} / {}'.format(
        r['size_sklearn_pkl_bytes'], r['size_initial_bin_bytes'], r['size_final_bin_bytes'])),
    ('CPN final (places/trans/arcs)', lambda r: '{} / {} / {}'.format(
        r.get('cpn_final_places', '-'), r.get('cpn_final_transitions', '-'),
        r.get('cpn_final_arcs', '-'))),
    ('Duplicated Rules', lambda r: r['duplicated_rules']),
    ('Specific Rules', lambda r: r['specific_rules']),
]


def write_outputs(rows, nested_rows, env, paths):
    """Write the CSV, the JSON and the markdown table."""
    # CSV: union of every key, stable order (first-seen)
    columns = []
    for row in rows:
        for k in row:
            if k not in columns:
                columns.append(k)
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(paths['csv'], index=False, encoding='utf-8')

    with open(paths['json'], 'w', encoding='utf-8') as f:
        json.dump({'environment': env, 'results': nested_rows}, f, indent=2,
                  ensure_ascii=False, default=str)

    ok = [r for r in rows if r.get('status') == 'ok']
    lines = ['| ' + ' | '.join(name for name, _ in MD_COLUMNS) + ' |',
             '|' + '---|' * len(MD_COLUMNS)]
    for r in rows:
        if r.get('status') != 'ok':
            lines.append(f"| {r['dataset']} | {r['model']} | ERROR: {r.get('error', '')[:80]} |"
                         + ' - |' * (len(MD_COLUMNS) - 3))
            continue
        lines.append('| ' + ' | '.join(str(fn(r)) for _, fn in MD_COLUMNS) + ' |')

    if ok:
        lines += ['', '## Summary', '']
        lines.append(f'- Runs: {len(ok)} ok, {len(rows) - len(ok)} failed')
        lines.append('- Mean rule reduction: {:.2f}%'.format(
            statistics.mean(r['rules_reduction_percent'] for r in ok)))
        lines.append('- Mean SCS reduction: {:.2f}%'.format(
            statistics.mean(r['scs_reduction_percent'] for r in ok)))
        lines.append('- Mean accuracy delta (final - sklearn): {:.3f} pp'.format(
            statistics.mean(r['accuracy_delta_final_minus_sklearn'] for r in ok)))
        lines.append('- Total wall time: {:.1f} s'.format(
            sum(r['time_total_s'] for r in ok)))
        lines.append('')
        lines.append(f"- Environment: python {env['python']}, sklearn {env['sklearn']}, "
                     f"{env['platform']}")
        lines.append(f"- Full metrics: `{os.path.basename(paths['csv'])}` and "
                     f"`{os.path.basename(paths['json'])}`")

    with open(paths['md'], 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    p = argparse.ArgumentParser(
        description='Full-metric benchmark of pyruleanalyzer over the article datasets.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Configure the datasets in the DATASETS list inside this file.')
    p.add_argument('--dataset', default=None,
                   help='only datasets whose name contains this substring')
    p.add_argument('--model', default=None, choices=['DT', 'RF', 'GBDT'],
                   help='only this model')
    p.add_argument('--out-dir', default='files', help='where the reports go')
    p.add_argument('--threshold', type=int, default=1,
                   help='remove_below_n_classifications used in the refinement')
    p.add_argument('--timing-repeats', type=int, default=1,
                   help='how many times each prediction is timed (best is reported)')
    p.add_argument('--no-cpn', dest='cpn', action='store_false',
                   help='skip the Petri net export and its metrics')
    p.add_argument('--no-arduino', dest='arduino', action='store_false',
                   help='skip the .ino export and the Flash/SRAM estimate')
    p.add_argument('--no-memory', dest='memory', action='store_false',
                   help='skip the tracemalloc peak-memory measurements')
    p.add_argument('--keep-artifacts', action='store_true',
                   help='keep the generated .bin/.py/.h/.ino/.cpn files')
    p.add_argument('--verbose', action='store_true',
                   help="let the library print to the console instead of the run log")
    args = p.parse_args()

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(args.out_dir, exist_ok=True)
    paths = {
        'csv': os.path.join(args.out_dir, f'benchmark_{stamp}.csv'),
        'json': os.path.join(args.out_dir, f'benchmark_{stamp}.json'),
        'md': os.path.join(args.out_dir, f'benchmark_results_{stamp}.md'),
    }
    artifacts_root = os.path.join(args.out_dir, f'benchmark_{stamp}_artifacts')
    log_path = os.path.join(args.out_dir, f'benchmark_{stamp}.log')

    env = environment_info()
    print('=' * 80)
    print('PYRULEANALYZER BENCHMARK - FULL METRICS')
    print('=' * 80)
    print(f"python {env['python']} | sklearn {env['sklearn']} | {env['platform']}")
    print(f"reports: {paths['csv']}\n         {paths['json']}\n         {paths['md']}")
    if not args.verbose:
        print(f"library output: {log_path}")
    print()

    datasets = [d for d in DATASETS if d['train'] and d['test']]
    if args.dataset:
        datasets = [d for d in datasets if args.dataset.lower() in d['nome'].lower()]
    models = [m for m in MODELS_TO_TEST
              if not args.model or MODEL_ABBR[m] == args.model]

    rows, nested_rows = [], []
    # The library prints a lot; send it to a per-run log so the progress
    # lines stay readable (--verbose keeps it on the console).
    log = None if args.verbose else open(log_path, 'w', encoding='utf-8')

    for ds in datasets:
        print(f"\n### {ds['nome']}")
        try:
            X_train, y_train, X_test, y_test, _, _, feature_names = \
                RuleClassifier.process_data(ds['train'], ds['test'])
            data = (X_train, y_train, X_test, y_test, feature_names)
        except Exception as exc:
            print(f'  [!] could not load the data: {exc}')
            for model_name in models:
                rows.append({'dataset': ds['nome'], 'model': MODEL_ABBR[model_name],
                             'status': 'load_error', 'error': str(exc)[:300]})
            continue

        for model_name in models:
            abbr = MODEL_ABBR[model_name]
            print(f'  - {abbr} ...', end='', flush=True)
            art_dir = os.path.join(artifacts_root, f"{slug(ds['nome'])}_{abbr}")
            t0 = time.perf_counter()
            try:
                if log is None:
                    flat, nested = run_case(ds, model_name, data, args, art_dir)
                else:
                    log.write(f"{chr(10)}===== {ds['nome']} | {abbr} ====={chr(10)}")
                    log.flush()
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        flat, nested = run_case(ds, model_name, data, args, art_dir)
                rows.append(flat)
                nested_rows.append({'dataset': ds['nome'], 'model': abbr,
                                    'metrics': flat, 'detail': nested})
                print(f" ok ({time.perf_counter() - t0:.1f}s) "
                      f"acc {flat['final_accuracy']:.2f}% | "
                      f"rules {flat['initial_rules']}->{flat['final_rules']}")
            except Exception as exc:
                rows.append({'dataset': ds['nome'], 'model': abbr,
                             'status': 'error', 'error': str(exc)[:300]})
                print(f' ERROR: {exc}')
            finally:
                write_outputs(rows, nested_rows, env, paths)  # partial results survive

    if log is not None:
        log.close()
    if not args.keep_artifacts:
        shutil.rmtree(artifacts_root, ignore_errors=True)

    print('\nDone.')
    print(f"  {paths['csv']}")
    print(f"  {paths['json']}")
    print(f"  {paths['md']}")
    if not args.verbose:
        print(f"  {log_path}")


if __name__ == '__main__':
    main()
