"""
pyruleanalyzer test over the federated-learning IDS papers in
C:\\Users\\caiod\\Documents\\papers.

Every paper folder holds ``processed/<dataset>/{train,test[,val]}.csv.gz``
(already preprocessed as described in the folder's preprocessing.txt). The
compressed CSVs are read directly by pandas, so nothing has to be unzipped.

For each dataset of the selected paper and each model (DT / RF / GBDT) the
script:
    1. trains the scikit-learn model,
    2. extracts the rules (PyRuleAnalyzer.from_sklearn),
    3. refines them (on val.csv when the paper has one, else on test.csv,
       as in datasets_benchmark_example.py),
    4. compares sklearn x initial rules x final rules on the test set
       (accuracy, F1, fidelity, rule count, SCS, prediction time).

Results are printed and saved to files/papers/<paper>_<timestamp>.csv.

HOW TO USE: pick ONE paper in the "SELECT THE PAPER" section below -- leave its
PAPER = {...} block uncommented and keep all the others commented out.

    python examples/papers_benchmark.py
"""

import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyruleanalyzer import PyRuleAnalyzer, RuleClassifier

# ==============================================================================
# GENERAL CONFIGURATION
# ==============================================================================
PAPERS_ROOT = r"C:\Users\caiod\Documents\papers"
OUT_DIR = "files/papers"

MODELS = ["DT", "RF", "GBDT"]

# The papers use DL/FL models, so they report no tree hyperparameters; these are
# the defaults. A dataset entry may override them with "param_dt"/"param_rf"/
# "param_gbdt".
DEFAULT_PARAMS = {
    "DT": {"random_state": 42},
    "RF": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
    "GBDT": {"n_estimators": 50, "max_depth": 3, "random_state": 42},
}

# Several datasets have millions of rows (CICIoT2023, Bot-IoT, SWaT, Edge-IIoTset
# raw, ...). Stratified subsamples keep the run tractable; None = use everything.
# A dataset entry may override them with "max_train_rows"/"max_test_rows".
MAX_TRAIN_ROWS = 200_000
MAX_TEST_ROWS = 100_000

# remove_below_n_classifications of the refinement (-1 disables it).
REFINE_THRESHOLD = 1

# Export the final rules as .py/.bin/.h next to the report.
EXPORT_ARTIFACTS = False

# ==============================================================================
# SELECT THE PAPER (keep exactly one PAPER block uncommented)
#
# Dataset keys:
#   name     sub-folder of processed/
#   target   label column
#   drop     columns removed before training (label leaks, IDs, raw text)
#   resplit  True when the paper's train split has a single class (one-class /
#            anomaly-detection setups): all splits are merged and re-split 80/20
#            stratified, because a supervised tree needs every class in train.
# ==============================================================================

PAPER = {
    "title": "2DF-IDS - Decentralized and differentially private federated learning-based intrusion detection system for industrial IoT",
    "datasets": [
        {"name": "edge_iiotset", "target": "Attack_type"},
    ],
}

# PAPER = {
#     "title": "A cutting-edge framework for industrial intrusion detection - Privacy-preserving, cost-friendly, and powered by federated learning",
#     "datasets": [
#         {"name": "gas_pipeline", "target": "result"},
#         {"name": "water_storage", "target": "result"},
#     ],
# }

# PAPER = {
#     "title": "A federated learning-based approach for improving intrusion detection in industrial internet of things networks",
#     "datasets": [
#         {"name": "edge_iiotset", "target": "Attack_type"},
#     ],
# }

# PAPER = {
#     "title": "An ensemble deep federated learning cyber-threat hunting model for industrial internet of things",
#     "datasets": [
#         {"name": "gas_pipeline", "target": "result"},
#         {"name": "swat", "target": "Normal/Attack"},
#     ],
# }

# PAPER = {
#     "title": "An optimal federated learning-based intrusion detection for iot environment",
#     "datasets": [
#         {"name": "mqttset", "target": "target"},
#     ],
# }

# PAPER = {
#     "title": "Asynchronous Peer-to-Peer Federated Capability-Based Targeted Ransomware Detection Model for Industrial IoT",
#     "datasets": [
#         {"name": "nsl_kdd", "target": "label"},
#         {"name": "x_iiotid", "target": "class1"},
#     ],
# }

# PAPER = {
#     "title": "CFL-IDS - An Effective Clustered Federated Learning Framework for Industrial Internet of Things Intrusion Detection",
#     "datasets": [
#         {"name": "gas_pipeline", "target": "result"},
#         {"name": "unsw_nb15", "target": "attack_cat", "drop": ["label"]},
#     ],
# }

# PAPER = {
#     "title": "Data-Centric Federated Learning for Anomaly Detection in Smart Grids and Other Industrial Control Systems",
#     "datasets": [
#         {"name": "bot_iot", "target": "category"},
#         {"name": "nsl_kdd", "target": "label"},
#         {"name": "unsw_nb15", "target": "attack_cat"},
#     ],
# }

# PAPER = {
#     # The paper trains on normal data only, so train.csv has no attack samples.
#     "title": "Deep Federated Learning-Based Cyber-Attack Detection in Industrial Control Systems",
#     "datasets": [
#         {"name": "swat", "target": "Normal/Attack", "resplit": True},
#     ],
# }

# PAPER = {
#     "title": "DeepFed_ Federated Deep Learning for Intrusion Detection in Industrial Cyber–Physical Systems",
#     "datasets": [
#         {"name": "gas_pipeline", "target": "result"},
#     ],
# }

# PAPER = {
#     "title": "Delay and Energy-Efficient Asynchronous Federated Learning for Intrusion Detection in Heterogeneous Industrial Internet of Things",
#     "datasets": [
#         {"name": "nsl_kdd", "target": "label"},
#     ],
# }

# PAPER = {
#     "title": "Design of a federated ensemble model for intrusion detection in distributed iiot networks for enhancing cybersecurity",
#     "datasets": [
#         {"name": "edge_iiotset", "target": "Attack_type"},
#         {"name": "ton_iot", "target": "type"},
#     ],
# }

# PAPER = {
#     "title": "Efficient privacy-preserving federated deep learning for network intrusion of industrial iot",
#     "datasets": [
#         {"name": "cicids2017", "target": "label"},
#         {"name": "kddcup99", "target": "label"},
#         {"name": "nsl_kdd", "target": "label"},
#     ],
# }

# PAPER = {
#     "title": "Enhancing industrial iot security - Utilizing blockchain-assisted deep federated learning for collaborative intrusion detection",
#     "datasets": [
#         {"name": "ton_iot", "target": "type"},
#         {"name": "unsw_nb15", "target": "attack_cat"},
#     ],
# }

# PAPER = {
#     "title": "Fed-IIoT_ A Robust Federated Malware Detection Architecture in Industrial IoT",
#     "datasets": [
#         {"name": "drebin", "target": "class"},
#     ],
# }

# PAPER = {
#     "title": "Federated Learning Models for Intrusion Detection in Industrial IoT Networks",
#     "datasets": [
#         {"name": "ciciot2023", "target": "label"},
#     ],
# }

# PAPER = {
#     "title": "Federated Learning for Network Anomaly Detection in a Distributed Industrial Environment",
#     "datasets": [
#         {"name": "westermo", "target": "NST_M_Label"},
#     ],
# }

# PAPER = {
#     "title": "Federated Semisupervised Learning for Attack Detection in Industrial Internet of Things",
#     "datasets": [
#         {"name": "gas_pipeline", "target": "result"},
#         {"name": "water_storage", "target": "result"},
#     ],
# }

# PAPER = {
#     "title": "Federated Threat-Hunting Approach for Microservice-Based Industrial Cyber-Physical System",
#     "datasets": [
#         {"name": "litnet_2020", "target": "attack_type"},
#         {"name": "ton_iot", "target": "type"},
#     ],
# }

# PAPER = {
#     "title": "Federated learning for network attack detection using attention-based graph neural networks",
#     "datasets": [
#         {"name": "nsl_kdd", "target": "label"},
#     ],
# }

# PAPER = {
#     "title": "Federated transfer learning for intrusion detection system in industrial iot 4.0",
#     "datasets": [
#         {"name": "bot_iot", "target": "category"},
#         {"name": "water_pipeline", "target": "result"},
#     ],
# }

# PAPER = {
#     # No processed/ folder: the Modbus PCAPs (ICS_PCAPS) were never converted.
#     "title": "Federated-Learning-Based Anomaly Detection for IoT Security Attacks",
#     "datasets": [],
# }

# PAPER = {
#     "title": "Federated-SRUs - A Federated-Simple-Recurrent-Units-Based IDS for Accurate Detection of Cyber Attacks Against IoT-Augmented Industrial Control Systems",
#     "datasets": [
#         {"name": "gas_pipeline", "target": "result"},
#     ],
# }

# PAPER = {
#     # Private DNP3 dataset generated by the authors: nothing to test.
#     "title": "Ids for industrial applications_ A federated learning approach with active personalization",
#     "datasets": [],
# }

# PAPER = {
#     "title": "Internet of Things Intrusion Detection_ Centralized, On-Device, or Federated Learning.pdf",
#     "datasets": [
#         {"name": "nsl_kdd", "target": "label"},
#     ],
# }

# PAPER = {
#     # Autoencoder setup: train/val hold benign traffic only, hence resplit.
#     "title": "Interpretable Anomaly Detection in Industrial Control Systems Using Federated Learning",
#     "datasets": [
#         {"name": f"n_baiot_{device}", "target": "is_attack", "resplit": True}
#         for device in [
#             "Danmini_Doorbell",
#             "Ecobee_Thermostat",
#             "Ennio_Doorbell",
#             "Philips_B120N10_Baby_Monitor",
#             "Provision_PT_737E_Security_Camera",
#             "Provision_PT_838_Security_Camera",
#             "Samsung_SNH_1011_N_Webcam",
#             "SimpleHome_XCS7_1002_WHT_Security_Camera",
#             "SimpleHome_XCS7_1003_WHT_Security_Camera",
#         ]
#     ],
# }

# PAPER = {
#     "title": "Intrusion Detection Approach for Industrial Internet of Things Traffic Using Deep Recurrent Reinforcement Learning Assisted Federated Learning",
#     "datasets": [
#         {"name": "edge_iiotset", "target": "Attack_type"},
#         {"name": "ton_iot", "target": "type"},
#         {"name": "x_iiotid", "target": "class1"},
#     ],
# }

# PAPER = {
#     # Raw Edge-IIoTset (no preprocessing in the paper): drop the binary label
#     # (leaks the target) and the identifier / free-text columns.
#     "title": "Ppss_ A privacy-preserving secure framework using blockchain-enabled federated deep learning for industrial iots",
#     "datasets": [
#         {"name": "edge_iiotset", "target": "Attack_type",
#          "drop": ["Attack_label", "frame.time", "ip.src_host", "ip.dst_host",
#                   "arp.src.proto_ipv4", "arp.dst.proto_ipv4", "http.file_data",
#                   "http.request.full_uri", "http.request.uri.query", "http.referer",
#                   "tcp.options", "tcp.payload", "tcp.srcport", "mqtt.msg"]},
#     ],
# }

# PAPER = {
#     "title": "Privacy-Preserved Cyberattack Detection in Industrial Edge of Things (IEoT) - A Blockchain-Orchestrated Federated Learning Approach",
#     "datasets": [
#         {"name": "litnet_2020", "target": "attack_type"},
#         {"name": "ton_iot", "target": "type"},
#     ],
# }


# ==============================================================================
# DATA LOADING
# ==============================================================================

def safe_names(columns):
    """Feature names usable in rules and in the .py/.h exports (unique identifiers)."""
    out, seen = [], set()
    for col in columns:
        name = re.sub(r'[^0-9A-Za-z_]', '_', str(col)).strip('_') or 'f'
        if name[0].isdigit():
            name = f'f_{name}'
        base, k = name, 1
        while name in seen:
            k += 1
            name = f'{base}_{k}'
        seen.add(name)
        out.append(name)
    return out


def stratified_cap(X, y, max_rows, seed=42):
    """Stratified subsample of at most max_rows rows (plain random if a class is too rare)."""
    if max_rows is None or len(y) <= max_rows:
        return X, y
    try:
        X, _, y, _ = train_test_split(X, y, train_size=max_rows, stratify=y, random_state=seed)
    except ValueError:
        X, _, y, _ = train_test_split(X, y, train_size=max_rows, random_state=seed)
    return X.reset_index(drop=True), y


def load_dataset(folder, ds):
    """Read the splits, encode labels/categoricals and return numeric DataFrames.

    Returns (X_train, y_train, X_refine, y_refine, X_test, y_test, class_names).
    X_refine is val.csv when present, otherwise the test set.
    """
    base = os.path.join(folder, 'processed', ds['name'])
    splits = {}
    for split in ('train', 'val', 'test'):
        path = os.path.join(base, f'{split}.csv.gz')
        if os.path.exists(path):
            splits[split] = pd.read_csv(path, low_memory=False)
    if 'train' not in splits or 'test' not in splits:
        raise FileNotFoundError(f'train/test not found in {base}')

    target = ds['target']
    drop = [c for c in ds.get('drop', []) if c in splits['train'].columns]
    for name, df in splits.items():
        splits[name] = df.drop(columns=drop)

    if ds.get('resplit'):
        full = pd.concat(list(splits.values()), ignore_index=True)
        try:
            tr, te = train_test_split(full, test_size=0.2, stratify=full[target], random_state=42)
        except ValueError:
            tr, te = train_test_split(full, test_size=0.2, random_state=42)
        splits = {'train': tr.reset_index(drop=True), 'test': te.reset_index(drop=True)}

    # Label encoding over the union of splits so no split sees an unknown class.
    labels = sorted(pd.concat([df[target] for df in splits.values()]).astype(str).unique())
    label_map = {lab: i for i, lab in enumerate(labels)}

    features = [c for c in splits['train'].columns if c != target]
    categories = {}
    for col in features:
        if splits['train'][col].dtype != object:
            continue
        numeric = pd.to_numeric(splits['train'][col], errors='coerce')
        if numeric.notna().mean() > 0.99:
            continue  # numbers stored as text: coerced below
        categories[col] = {v: i for i, v in enumerate(sorted(splits['train'][col].astype(str).unique()))}

    names = safe_names(features)
    out = {}
    for split, df in splits.items():
        cols = {}
        for col, name in zip(features, names):
            if col in categories:
                cols[name] = df[col].astype(str).map(categories[col]).fillna(-1)
            else:
                cols[name] = pd.to_numeric(df[col], errors='coerce')
        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float64)
        y = df[target].astype(str).map(label_map).to_numpy(dtype=int)
        out[split] = (X, y)

    X_train, y_train = stratified_cap(*out['train'], ds.get('max_train_rows', MAX_TRAIN_ROWS))
    X_test, y_test = stratified_cap(*out['test'], ds.get('max_test_rows', MAX_TEST_ROWS))
    if 'val' in out:
        X_ref, y_ref = stratified_cap(*out['val'], ds.get('max_test_rows', MAX_TEST_ROWS))
    else:
        X_ref, y_ref = X_test, y_test
    return X_train, y_train, X_ref, y_ref, X_test, y_test, labels


# ==============================================================================
# ONE (DATASET x MODEL) RUN
# ==============================================================================

ESTIMATORS = {
    "DT": DecisionTreeClassifier,
    "RF": RandomForestClassifier,
    "GBDT": GradientBoostingClassifier,
}


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def scores(y_true, y_pred):
    return (accuracy_score(y_true, y_pred) * 100,
            f1_score(y_true, y_pred, average='macro', zero_division=0) * 100)


def run_model(abbr, ds, data, artifact_base):
    X_train, y_train, X_ref, y_ref, X_test, y_test, _ = data
    features = list(X_train.columns)
    params = ds.get(f'param_{abbr.lower()}', DEFAULT_PARAMS[abbr])

    sk_model, t_train = timed(lambda: ESTIMATORS[abbr](**params).fit(X_train, y_train))
    class_names = [str(c) for c in sk_model.classes_]

    analyzer, t_extract = timed(lambda: PyRuleAnalyzer.from_sklearn(sk_model, features, class_names))
    clf = analyzer.classifier
    initial_rules = list(clf.initial_rules)

    y_sk, t_sk = timed(lambda: sk_model.predict(X_test))
    y_ini, t_ini = timed(lambda: analyzer.predict(X_test.values, use_refined=False))

    stats, t_refine = timed(lambda: analyzer.execute_rule_refinement(
        X=X_ref, y=y_ref, remove_below_n_classifications=REFINE_THRESHOLD))
    final_rules = list(clf.final_rules) if clf.final_rules else initial_rules

    y_fin, t_fin = timed(lambda: analyzer.predict(X_test.values, use_refined=True))

    y_sk, y_ini, y_fin = (np.asarray(v).astype(int) for v in (y_sk, y_ini, y_fin))
    acc_sk, f1_sk = scores(y_test, y_sk)
    acc_ini, f1_ini = scores(y_test, y_ini)
    acc_fin, f1_fin = scores(y_test, y_fin)
    scs_ini = RuleClassifier.calculate_structural_complexity(initial_rules, len(features))
    scs_fin = RuleClassifier.calculate_structural_complexity(final_rules, len(features))

    if EXPORT_ARTIFACTS:
        analyzer.export(artifact_base, formats=["python", "binary", "c"], use_refined=True)

    return {
        'model': abbr,
        'acc_sklearn': round(acc_sk, 4), 'acc_initial': round(acc_ini, 4), 'acc_final': round(acc_fin, 4),
        'f1_sklearn': round(f1_sk, 4), 'f1_initial': round(f1_ini, 4), 'f1_final': round(f1_fin, 4),
        'fidelity_initial': round(float(np.mean(y_ini == y_sk)) * 100, 4),
        'fidelity_final': round(float(np.mean(y_fin == y_sk)) * 100, 4),
        'rules_initial': len(initial_rules), 'rules_final': len(final_rules),
        'rules_reduction_pct': round(stats['reduction_percent'], 2),
        'scs_initial': round(float(scs_ini.get('complexity_score', 0.0)), 4),
        'scs_final': round(float(scs_fin.get('complexity_score', 0.0)), 4),
        'time_train_s': round(t_train, 3), 'time_extract_s': round(t_extract, 3),
        'time_refine_s': round(t_refine, 3),
        'time_pred_sklearn_s': round(t_sk, 4), 'time_pred_initial_s': round(t_ini, 4),
        'time_pred_final_s': round(t_fin, 4),
    }


# ==============================================================================
# MAIN
# ==============================================================================

def slug(text, maxlen=60):
    return re.sub(r'[^0-9A-Za-z]+', '_', text).strip('_')[:maxlen]


def main():
    title = PAPER['title']
    folder = os.path.join(PAPERS_ROOT, title)
    print('=' * 80)
    print(title)
    print('=' * 80)
    if not PAPER['datasets']:
        print('This paper has no processed dataset to test.')
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report = os.path.join(OUT_DIR, f'{slug(title)}_{stamp}.csv')
    rows = []

    for ds in PAPER['datasets']:
        print(f"\n### {ds['name']}  (target: {ds['target']})")
        try:
            data, t_load = timed(lambda: load_dataset(folder, ds))
        except Exception as exc:
            print(f'  [!] could not load: {exc}')
            rows.append({'dataset': ds['name'], 'status': 'load_error', 'error': str(exc)[:300]})
            continue
        X_train, y_train, X_ref, _, X_test, y_test, labels = data
        print(f'  loaded in {t_load:.1f}s | train {X_train.shape} | test {X_test.shape} | '
              f'refine on {"val" if X_ref is not X_test else "test"} | {len(labels)} classes')
        print(f'  classes: {dict(enumerate(labels))}')

        for abbr in MODELS:
            print(f'  - {abbr} ...', end='', flush=True)
            base = os.path.join(OUT_DIR, f"{slug(title, 40)}_{ds['name']}_{abbr.lower()}")
            try:
                row = run_model(abbr, ds, data, base)
                print(f" acc sk/ini/fin {row['acc_sklearn']:.2f}/{row['acc_initial']:.2f}/"
                      f"{row['acc_final']:.2f} | fidelity {row['fidelity_final']:.2f}% | "
                      f"rules {row['rules_initial']}->{row['rules_final']}")
                rows.append({'dataset': ds['name'], 'status': 'ok',
                             'n_train': len(X_train), 'n_test': len(X_test),
                             'n_features': X_train.shape[1], 'n_classes': len(labels), **row})
            except Exception as exc:
                print(f' ERROR: {exc}')
                rows.append({'dataset': ds['name'], 'model': abbr, 'status': 'error',
                             'error': str(exc)[:300]})
            pd.DataFrame(rows).to_csv(report, index=False)  # partial results survive

    print(f'\nReport: {report}')


if __name__ == '__main__':
    main()
