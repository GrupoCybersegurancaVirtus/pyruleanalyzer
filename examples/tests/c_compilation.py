def reconstruct_sklearn_model_dt(self, X_train, feature_names):
        import sys
        import os
        from sklearn.tree import DecisionTreeClassifier
        import numpy as np
        import pickle

        def trace(msg):
            print(f"[TRACE] {msg}")
            sys.stdout.flush()

        num_classes = self.num_classes
        num_features = len(feature_names)
        trace(f"Reconstrução Blindada: {len(self.final_rules)} regras, {num_classes} classes.")

        # 1. Construção do Grafo (Sem recursão para evitar stack overflow)
        tree_dict = {0: {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': np.zeros(num_classes)}}
        next_id = 1
        max_d_found = 0

        for rule in self.final_rules:
            curr = 0
            depth = 0
            for var, op, val in rule.parsed_conditions:
                f_idx = feature_names.index(var)
                depth += 1
                if tree_dict[curr]['f'] == -2:
                    tree_dict[curr].update({'f': f_idx, 't': val, 'l': next_id, 'r': next_id + 1})
                    tree_dict[next_id] = {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': np.zeros(num_classes)}
                    tree_dict[next_id + 1] = {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': np.zeros(num_classes)}
                    next_id += 2
                curr = tree_dict[curr]['l'] if op in ['<=', '<'] else tree_dict[curr]['r']
            
            max_d_found = max(max_d_found, depth)
            c_idx = int(rule.class_.replace('Class', ''))
            tree_dict[curr]['v'][c_idx] = 1.0

        node_count = len(tree_dict)
        trace(f"Grafo: {node_count} nós. Profundidade: {max_d_found}")

        # 2. Definição do DType com Offsets Fixos (Crucial para evitar crash)
        # Este formato é o padrão para Sklearn 1.2+
        node_dtype = np.dtype({
            'names': ['left_child', 'right_child', 'feature', 'threshold', 'impurity', 'n_node_samples', 'weighted_n_node_samples', 'missing_go_to_left'],
            'formats': ['<i8', '<i8', '<i8', '<f8', '<f8', '<i8', '<f8', 'u1'],
            'offsets': [0, 8, 16, 24, 32, 40, 48, 56],
            'itemsize': 64
        })

        # Criamos os arrays e forçamos a ordem 'C' (contígua em memória)
        nodes_array = np.empty(node_count, dtype=node_dtype)
        values_array = np.empty((node_count, 1, num_classes), dtype='<f8')

        for i in range(node_count):
            n = tree_dict[i]
            nodes_array[i] = (n['l'], n['r'], n['f'], n['t'], 0.0, 100, 100.0, 0)
            values_array[i, 0, :] = n['v']

        # Forçamos os arrays a serem contíguos para o C não se perder
        nodes_array = np.ascontiguousarray(nodes_array)
        values_array = np.ascontiguousarray(values_array)

        # 3. Injeção Protegida
        new_model = DecisionTreeClassifier(max_depth=max_d_found)
        # O fit deve ter o X com o número correto de colunas
        new_model.fit(np.zeros((num_classes, num_features)), np.arange(num_classes))

        trace("Injetando estado...")
        state = {
            'max_depth': max_d_found,
            'node_count': node_count,
            'nodes': nodes_array,
            'values': values_array
        }

        try:
            # Mantemos referências vivas para evitar Garbage Collection precoce
            new_model.tree_.__setstate__(state)
            trace("Setstate OK!")
            
            # Salva imediatamente antes que qualquer outra operação ocorra
            save_path = 'examples/files/sklearn_adapted_model.pkl'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(new_model, f)
            trace("Modelo salvo com sucesso.")
            
        except Exception as e:
            trace(f"Erro fatal: {e}")
            return None

        return new_model
    
    #DA ERROO fato de o script chegar no "Setstate OK!" e fechar logo em seguida é a prova final de que a árvore foi montada corretamente na memória, mas o motor do Scikit-Learn (em C++) entra em colapso assim que tenta acessar esses dados para qualquer operação posterior (seja um predict, um print do objeto ou o encerramento do script que tenta limpar a memória). Como o seu grafo tem 6.363 nós e apenas 43 de profundidade, o problema não é estouro de pilha (stack overflow), mas sim um corrompimento de memória (Memory Corruption).
    
    
def reconstruct_sklearn_model_dt(self, X_train, feature_names):
        import sys
        import numpy as np
        import pickle
        from sklearn.tree import DecisionTreeClassifier

        def trace(msg):
            print(f"[TRACE] {msg}")
            sys.stdout.flush()

        num_classes = self.num_classes
        trace(f"Iniciando Mapeador Reverso: {len(self.final_rules)} regras.")

        # --- PASSO 1: CONSTRUIR O GRAFO LÓGICO (Dicionário) ---
        tree_dict = {0: {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': np.zeros(num_classes)}}
        next_id = 1
        max_d = 0

        for rule in self.final_rules:
            curr = 0
            depth = 0
            for var, op, val in rule.parsed_conditions:
                f_idx = feature_names.index(var)
                depth += 1
                if tree_dict[curr]['f'] == -2:
                    tree_dict[curr].update({'f': f_idx, 't': val, 'l': next_id, 'r': next_id + 1})
                    tree_dict[next_id] = {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': np.zeros(num_classes)}
                    tree_dict[next_id + 1] = {'l': -1, 'r': -1, 'f': -2, 't': -2.0, 'v': np.zeros(num_classes)}
                    next_id += 2
                curr = tree_dict[curr]['l'] if op in ['<=', '<'] else tree_dict[curr]['r']
            max_d = max(max_d, depth)
            c_idx = int(rule.class_.replace('Class', ''))
            tree_dict[curr]['v'][c_idx] = 1.0

        node_count = len(tree_dict)
        trace(f"Grafo construído: {node_count} nós. Obtendo DNA do Sklearn...")

        # --- PASSO 2: OBTER O DTYPE REAL DO SISTEMA ---
        # Criamos um modelo temporário para "roubar" o formato exato dos nós
        dummy = DecisionTreeClassifier(max_depth=1)
        dummy.fit(X_train[:2], np.arange(2))
        dummy_state = dummy.tree_.__getstate__()
        
        # O segredo: usamos o DType que o Sklearn do seu PC gerou
        native_dtype = dummy_state['nodes'].dtype
        trace(f"DType Nativo Identificado: {native_dtype}")

        # --- PASSO 3: POPULAR ARRAYS NATIVOS ---
        nodes_array = np.zeros(node_count, dtype=native_dtype)
        values_array = np.zeros((node_count, 1, num_classes))

        for i in range(node_count):
            n = tree_dict[i]
            # Mapeamos para os nomes exatos do DType nativo
            nodes_array[i]['left_child'] = n['l']
            nodes_array[i]['right_child'] = n['r']
            nodes_array[i]['feature'] = n['f']
            nodes_array[i]['threshold'] = n['t']
            nodes_array[i]['impurity'] = 1e-7 # Valor pequeno para estabilidade
            nodes_array[i]['n_node_samples'] = 100
            nodes_array[i]['weighted_n_node_samples'] = 100.0
            
            # Se a sua versão tiver o campo de valores ausentes, preenchemos
            if 'missing_go_to_left' in native_dtype.names:
                nodes_array[i]['missing_go_to_left'] = 0
            
            values_array[i, 0, :] = n['v']

        # --- PASSO 4: TRANSPLANTE ---
        new_model = DecisionTreeClassifier(max_depth=max_d)
        new_model.fit(X_train[:num_classes], np.arange(num_classes))
        
        # Montamos o estado final usando a estrutura 'saudável'
        final_state = {
            'max_depth': max_d,
            'node_count': node_count,
            'nodes': nodes_array,
            'values': values_array
        }

        try:
            trace("Realizando transplante de estado...")
            new_model.tree_.__setstate__(final_state)
            trace("Transplante concluído com sucesso!")
            
            # Salva o arquivo pkl
            with open('examples/files/sklearn_adapted_model.pkl', 'wb') as f:
                pickle.dump(new_model, f)
            trace("Modelo salvo e pronto para uso.")
            
        except Exception as e:
            trace(f"Falha no transplante: {e}")
            return None

        return new_model