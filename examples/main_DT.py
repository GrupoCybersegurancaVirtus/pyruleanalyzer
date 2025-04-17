from rule_classifier import RuleClassifier

# ************************ EXECUÇÃO ************************

train_path = "data/rapid_balanceado_treinamento.csv"
test_path = "data/rapid_balanceado_teste.csv"

# Parâmetros do modelo
model_parameters = {
    'criterion': 'gini',
    'splitter': 'best',
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0,
    'class_weight': None,
    'ccp_alpha': 0.0,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 2,
    'max_features': None,
    'random_state': 42
}

# Treinamento do modelo inicial
classifier = RuleClassifier.new_classifier(train_path, test_path, model_parameters, algorithm_type='Decision Tree')

# Análise e remoção de regras
classifier.execute_rule_analysis(test_path, remove_duplicates=True, remove_below_n_classifications=-1)

# Comparação dos resultados iniciais e finais
classifier.compare_initial_final_results(test_path)