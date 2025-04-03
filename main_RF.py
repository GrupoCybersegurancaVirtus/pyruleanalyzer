from rule_classifier import RuleClassifier

# ************************ EXECUÇÃO ************************

train_path = "G:/Meu Drive/Mestrado/VIRTUS/Código/Novo RuleXtract/data/rapid_balanceado_treinamento.csv"
test_path = "G:/Meu Drive/Mestrado/VIRTUS/Código/Novo RuleXtract/data/rapid_balanceado_teste.csv"

# Parâmetros do modelo
model_parameters = {
    'n_estimators': 101,
    'min_samples_leaf': 2,
    'min_samples_split': 2,
    'max_depth': None, 
    'max_features': 'sqrt',
    'random_state': 42
}

# Treinamento do modelo inicial
classifier = RuleClassifier.new_classifier(train_path, test_path, model_parameters, algorithm_type='Random Forest')

# Análise e remoção de regras
classifier.execute_rule_analysis(test_path, remove_below_n_classifications=-1)

# Comparação dos resultados iniciais e finais
classifier.compare_initial_final_results(test_path)