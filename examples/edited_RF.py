from pyruleanalyzer.rule_classifier import RuleClassifier

# Caminho para os dados de teste
test_path = "examples/data/test-set1.csv"

# # Carregando o modelo já existente
# classifier = RuleClassifier.load("examples/files/final_model.pkl")

# Carregando o modelo já existente
classifier = RuleClassifier.load("examples/files/edited_model.pkl")

# Comparando resultados iniciais e finais
classifier.compare_initial_final_results(test_path)

# # Editando as regras
# classifier.edit_rules()