from pyruleanalyzer import RuleClassifier

# ************************ EXECUÇÃO ************************

train_path = "data/rapid_balanceado_treinamento.csv"
test_path = "data/rapid_balanceado_teste.csv"

# Parâmetros do modelo
model_parameters = {
    'criterion': 'gini',
    'bootstrap': True,
    'oob_score': False,
    'n_jobs': None,
    'verbose': 0,
    'warm_start': False,
    'ccp_alpha': 0.0,
    'max_samples': None,
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
#(exact, strict, relaxed,...)
classifier.execute_rule_analysis(test_path, remove_duplicates=True, remove_below_n_classifications=-1)

# Comparação dos resultados iniciais e finais
classifier.compare_initial_final_results(test_path)