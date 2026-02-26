"""
Verification script: predict_batch() vs sklearn.predict() for DT, RF, GBDT.
Tests correctness (100% match on initial rules) and speed.
Uses lower-level APIs to build classifiers directly from sklearn models.
"""
import numpy as np
import pandas as pd
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from pyruleanalyzer.rule_classifier import RuleClassifier

np.random.seed(42)
all_pass = True
feat_names = [f'f{i}' for i in range(10)]
class_names_2 = ['class_0', 'class_1']
class_names_5 = [f'class_{i}' for i in range(5)]


def build_clf_dt_rf(model, feat_names_list, class_names_list, algorithm_type):
    """Build a RuleClassifier from a fitted DT or RF model."""
    rules = RuleClassifier.get_tree_rules(model, feat_names_list, class_names_list, algorithm_type=algorithm_type)
    class_names_map = {str(name): i for i, name in enumerate(class_names_list)}
    clf = RuleClassifier.generate_classifier_model(rules, class_names_map, algorithm_type)
    return clf


def build_clf_gbdt(model, feat_names_list, class_names_list):
    """Build a RuleClassifier from a fitted GBDT model."""
    rules, init_scores, is_binary, gbdt_classes = RuleClassifier.get_gbdt_rules(
        model, feat_names_list, class_names_list
    )
    clf = RuleClassifier(rules, algorithm_type='Gradient Boosting Decision Trees')
    clf._gbdt_init_scores = init_scores
    clf._gbdt_is_binary = is_binary
    clf._gbdt_classes = gbdt_classes
    clf.update_native_model(clf.initial_rules)
    return clf


def run_test(test_name, sk_pred, clf, X_test, feat_names_list):
    global all_pass
    print('=' * 70)
    print(f'TEST: {test_name}')
    print('=' * 70)
    clf.compile_tree_arrays(rules=clf.initial_rules, feature_names=feat_names_list)
    batch_pred = clf.predict_batch(X_test, feature_names=feat_names_list)
    match = np.sum(batch_pred == sk_pred)
    total = len(sk_pred)
    print(f'  sklearn vs predict_batch: {match}/{total} match ({match/total*100:.2f}%)')
    if match == total:
        print('  PASS')
    else:
        print('  FAIL -- mismatches at indices:')
        mismatches = np.where(batch_pred != sk_pred)[0]
        for idx in mismatches[:10]:
            print(f'    idx={idx}: sklearn={sk_pred[idx]}, batch={batch_pred[idx]}')
        all_pass = False
    print()


# ==== DT Binary ====
X, y = make_classification(n_samples=2000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
clf = build_clf_dt_rf(dt, feat_names, class_names_2, 'Decision Tree')
run_test('Decision Tree - Binary', dt.predict(X_test), clf, X_test, feat_names)

# ==== DT Multiclass ====
X, y = make_classification(n_samples=2000, n_features=10, n_classes=5, n_informative=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dt = DecisionTreeClassifier(max_depth=8, random_state=42)
dt.fit(X_train, y_train)
clf = build_clf_dt_rf(dt, feat_names, class_names_5, 'Decision Tree')
run_test('Decision Tree - Multiclass (5)', dt.predict(X_test), clf, X_test, feat_names)

# ==== RF Binary ====
X, y = make_classification(n_samples=2000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
clf = build_clf_dt_rf(rf, feat_names, class_names_2, 'Random Forest')
run_test('Random Forest - Binary', rf.predict(X_test), clf, X_test, feat_names)

# ==== RF Multiclass ====
X, y = make_classification(n_samples=2000, n_features=10, n_classes=5, n_informative=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=10, max_depth=8, random_state=42)
rf.fit(X_train, y_train)
clf = build_clf_dt_rf(rf, feat_names, class_names_5, 'Random Forest')
run_test('Random Forest - Multiclass (5)', rf.predict(X_test), clf, X_test, feat_names)

# ==== RF with 30 trees ====
X, y = make_classification(n_samples=3000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
clf = build_clf_dt_rf(rf, feat_names, class_names_2, 'Random Forest')
run_test('Random Forest - Binary (30 trees, depth 10)', rf.predict(X_test), clf, X_test, feat_names)

# ==== GBDT Binary ====
X, y = make_classification(n_samples=2000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
gb = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=42)
gb.fit(X_train, y_train)
clf = build_clf_gbdt(gb, feat_names, class_names_2)
run_test('GBDT - Binary', gb.predict(X_test), clf, X_test, feat_names)

# ==== GBDT Binary (more estimators) ====
X, y = make_classification(n_samples=3000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
gb = GradientBoostingClassifier(n_estimators=30, max_depth=5, random_state=42)
gb.fit(X_train, y_train)
clf = build_clf_gbdt(gb, feat_names, class_names_2)
run_test('GBDT - Binary (30 est, depth 5)', gb.predict(X_test), clf, X_test, feat_names)

# ==== GBDT Multiclass ====
X, y = make_classification(n_samples=2000, n_features=10, n_classes=5, n_informative=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
gb = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=42)
gb.fit(X_train, y_train)
clf = build_clf_gbdt(gb, feat_names, class_names_5)
run_test('GBDT - Multiclass (5)', gb.predict(X_test), clf, X_test, feat_names)

# ==== GBDT Multiclass (more estimators) ====
X, y = make_classification(n_samples=3000, n_features=10, n_classes=5, n_informative=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
gb = GradientBoostingClassifier(n_estimators=20, max_depth=5, random_state=42)
gb.fit(X_train, y_train)
clf = build_clf_gbdt(gb, feat_names, class_names_5)
run_test('GBDT - Multiclass (5, 20 est, depth 5)', gb.predict(X_test), clf, X_test, feat_names)


# ==== SPEED BENCHMARK ====
print('=' * 70)
print('SPEED BENCHMARK: predict_batch vs sklearn (RF, 5k test samples)')
print('=' * 70)
X, y = make_classification(n_samples=10000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
rf = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
clf = build_clf_dt_rf(rf, feat_names, class_names_2, 'Random Forest')
clf.compile_tree_arrays(rules=clf.initial_rules, feature_names=feat_names)

# Warm up
_ = rf.predict(X_test)
_ = clf.predict_batch(X_test, feature_names=feat_names)

N_ITER = 20
t0 = time.perf_counter()
for _ in range(N_ITER):
    sk_pred = rf.predict(X_test)
t_sk = (time.perf_counter() - t0) / N_ITER

t0 = time.perf_counter()
for _ in range(N_ITER):
    batch_pred = clf.predict_batch(X_test, feature_names=feat_names)
t_batch = (time.perf_counter() - t0) / N_ITER

print(f'  sklearn.predict:  {t_sk*1000:.2f} ms')
print(f'  predict_batch:    {t_batch*1000:.2f} ms')
print(f'  Ratio (batch/sk): {t_batch/t_sk:.2f}x')
match_count = np.sum(batch_pred == sk_pred)
print(f'  Match: {match_count}/{len(sk_pred)} ({match_count/len(sk_pred)*100:.2f}%)')
print()

print('=' * 70)
if all_pass:
    print('ALL CORRECTNESS TESTS PASSED')
else:
    print('SOME TESTS FAILED')
print('=' * 70)
