import sys
import os
import time
import numpy as np
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier

# Ajuste de path para encontrar a biblioteca
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from pyruleanalyzer.rule_classifier import RuleClassifier

# --- CONFIGURAÇÃO ---
train_path = "examples/data/DDoS Attack Classification Leveraging Data Balancing and Hyperparameter Tuning Approach Using Ensemble Machine Learning with XAI/train.csv"
test_path = "examples/data/DDoS Attack Classification Leveraging Data Balancing and Hyperparameter Tuning Approach Using Ensemble Machine Learning with XAI/test.csv"
model_params = {'random_state': 42, 'max_depth': 5}

# 1. Geração do Modelo e Extração de Regras
# O RuleClassifier extrai os caminhos lógicos do modelo original[cite: 115, 123].
classifier = RuleClassifier.new_classifier(train_path, test_path, model_params, algorithm_type='Decision Tree')

# 2. Refinamento (Análise de Regras)
# Aqui eliminamos duplicatas e regras com baixíssima cobertura (overfitting)[cite: 182, 184].
classifier.execute_rule_analysis(test_path, remove_duplicates="soft", remove_below_n_classifications=1)

# 3. Preparação dos Dados para o Benchmark Final
_, _, X_test, y_test, _, _, feature_names = RuleClassifier.process_data(train_path, test_path, is_test_only=True)
sample_dicts = pd.DataFrame(X_test, columns=feature_names).to_dict('records')

# --- INFERÊNCIA PYRULEANALYZER (Baseado em Regras/Python) ---
start_py = time.time()
# O classify percorre a lista de regras linearmente[cite: 143].
py_preds = [classifier.classify(sample, final=True)[0] for sample in sample_dicts]
end_py = time.time()
time_py = end_py - start_py

# --- INFERÊNCIA SKLEARN (Nativo/C) ---
# Carregamos o modelo salvo durante o new_classifier[cite: 178].
with open('examples/files/sklearn_model.pkl', 'rb') as f:
    sk_model = pickle.load(f)

start_sk = time.time()
sk_preds = sk_model.predict(X_test)
end_sk = time.time()
time_sk = end_sk - start_sk

# --- PRINT FINAL DA COMPARAÇÃO ---
print("\n" + "="*50)
print("COMPARAÇÃO FINAL DE PERFORMANCE E ACURÁCIA")
print("="*50)

# Métricas pyRuleAnalyzer
py_acc = np.mean([p == t for p, t in zip(py_preds, y_test)])
print(f"MODELO REFINADO (pyRuleAnalyzer):")
print(f"  - Tempo de Inferência: {time_py:.4f} segundos")
print(f"  - Acurácia:           {py_acc:.5f}")
print(f"  - Número de Regras:    {len(classifier.final_rules)} [cite: 199]")

# Métricas Sklearn
sk_acc = np.mean(sk_preds == y_test)
print(f"\nMODELO NATIVO (Scikit-Learn):")
print(f"  - Tempo de Inferência: {time_sk:.4f} segundos")
print(f"  - Acurácia:           {sk_acc:.5f}")
print(f"  - Estrutura:          Árvore Binária Completa [cite: 665]")

print("-" * 50)
speedup = time_py / time_sk if time_sk > 0 else 0
print(f"O Scikit-Learn foi {speedup:.1f}x mais rápido que o loop Python.")
print(f"Diferença de Acurácia: {abs(py_acc - sk_acc):.5f}")
print("="*50)

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# --- 1. Visualizar Árvore Original (Sklearn) ---
plt.figure(figsize=(20,10))
plot_tree(sk_model, 
          feature_names=feature_names, 
          class_names=[str(c) for c in np.unique(y_test)], 
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title("Estrutura Original do Scikit-Learn")
plt.savefig("tree_original_sklearn.png")
print("Árvore original salva como 'tree_original_sklearn.png'")

# --- 2. Visualizar Árvore Refinada (pyRuleAnalyzer) ---
print("\n" + "="*50)
print("ESTRUTURA LÓGICA REFINADA (pyRuleAnalyzer)")
print("="*50)

# Agrupando por profundidade para facilitar a visualização da hierarquia
for rule in sorted(classifier.final_rules, key=lambda x: len(x.conditions)):
    prefix = "  " * (len(rule.conditions) - 1)
    status = "[REFINADA]" if "_&_" in rule.name else "[MANTIDA]"
    print(f"{prefix}└─ {status} {rule.name}:")
    for cond in rule.conditions:
        print(f"{prefix}   IF {cond}")
    print(f"{prefix}   THEN Class {rule.class_}")
    print("-" * 30)