"""
Pipeline completo: de dados brutos a modelo Arduino/ESP32.

Esta módulo fornece a função `full_pipeline` que automatiza todo o fluxo:
  1. Carregar dados (CSV ou DataFrame)
  2. Treinar modelo (Decision Tree / Random Forest / GBDT)
  3. Refinar regras
  4. Validar com dataset de teste
  5. Gerar sketch Arduino/ESP32 pronto
  6. Estimar uso de memória

Uso programático:
    from pyruleanalyzer.full_pipeline import full_pipeline
    
    results = full_pipeline(
        train_csv='train.csv',
        target_feature='Target',
        model_type='Random Forest',
        test_csv='test.csv',
        generate_sketch=True,
        memory_report=True,
        output_dir='files',
    )
"""

import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def full_pipeline(
    train_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    target_feature: str = "Target",
    model_type: str = "Decision Tree",
    n_estimators: Optional[int] = None,
    max_depth: Optional[int] = None,
    min_samples_split: Optional[int] = None,
    remove_below_n_classifications: int = 0,
    output_dir: str = "files",
    output_name: str = "model",
    save_model: bool = False,
    generate_arduino_sketch: bool = False,
    board_model: str = "auto",
    serial_baud: int = 115200,
    include_sensor_placeholders: bool = True,
    memory_report: bool = True,
    target_platform: str = "auto",
    max_flash_percent: float = 90.0,
    max_ram_percent: float = 85.0,
    random_seed: int = 42,
    model_pkl: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the complete pipeline: train, validate, and optionally export to Arduino/ESP32.

    Parameters:
        train_csv: Path to training CSV file. Required if model_pkl is not provided.
        test_csv: Path to test CSV file for validation.
        target_feature: Name of the target column in the data.
        model_type: Model type ("Decision Tree", "Random Forest", "Gradient Boosting Decision Trees").
        n_estimators: Number of estimators (for Random Forest / GBDT).
        max_depth: Maximum tree depth.
        min_samples_split: Minimum samples for internal node split.
        remove_below_n_classifications: Remove rules with fewer than N classifications.
        output_dir: Output directory for generated files.
        output_name: Base filename for generated files.
        save_model: Save the model as .pkl after training.
        generate_arduino_sketch: Generate Arduino/ESP32 .ino sketch from the trained model.
        board_model: Target board ("auto", "uno", "nano", "mega", "leonardo", "esp32").
            Used only when generate_arduino_sketch=True. Default is "auto" (detects automatically).
        serial_baud: Serial baud rate for Arduino output (default 115200).
            Used only when generate_arduino_sketch=True.
        include_sensor_placeholders: Include TODO placeholders in read_features() function
            (default True). Used only when generate_arduino_sketch=True.
        memory_report: Show memory estimation report and check compatibility.
        target_platform: Target platform for compatibility check ("auto", "avr", "esp32").
        max_flash_percent: Maximum percentage of Flash memory to allow (for board check, default 90%).
        max_ram_percent: Maximum percentage of SRAM to allow (for board check, default 85%).
        random_seed: Random seed for train/test split.
        model_pkl: Path to a saved .pkl model file to load instead of training.

    Returns:
        Dictionary with pipeline results, including:
            - 'model': Trained or loaded PyRuleAnalyzer model
            - 'validation': Validation metrics (if test_csv was provided)
            - 'memory_check': Memory estimation and board compatibility results
            - 'generated_files': Dictionary with paths of generated files (.ino, .pkl, etc.)
            - 'refine_stats': Rule refinement statistics
            - 'elapsed': Total execution time in seconds
    """
    # Importações internas
    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    from . import PyRuleAnalyzer
    
    results: Dict[str, Any] = {
        'model': None,
        'validation': None,
        'memory_check': None,
        'generated_files': {},
        'elapsed': 0.0,
    }
    
    start_time = time.time()
    
    # Garantir que diretório de saída existe
    os.makedirs(output_dir, exist_ok=True)
    
    X_test = None
    y_test = None
    
    # --------------------------------------------------------
    # Etapa 1: Carregar dados ou carregar modelo existente
    # --------------------------------------------------------
    if model_pkl:
        # Carregar modelo existente
        with open(model_pkl, 'rb') as f:
            model = pickle.load(f)
    else:
        # Validar entrada de dados
        if train_csv is None or not os.path.exists(train_csv):
            raise ValueError(
                "--train-csv é obrigatório para treinar um novo modelo. "
                "Ou forneça --model-pkl para carregar um modelo existente."
            )
        
        import pandas as pd
        df_train = pd.read_csv(train_csv)
        
        # Verificar se o dataset tem coluna target ou é sem labels
        if target_feature and target_feature in df_train.columns:
            has_target = True
            X_train = df_train.drop(columns=[target_feature])
            y_train = df_train[target_feature]
            
            # Converter tipos
            X_train = X_train.select_dtypes([np.number]).fillna(0)
            if y_train.dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_train = le.fit_transform(y_train)
        else:
            has_target = False
            numeric_cols = df_train.select_dtypes([np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                raise ValueError(
                    "Dataset precisa ter pelo menos 2 colunas numéricas."
                )
            y_col = numeric_cols[-1]
            X_train = df_train[numeric_cols[:-1]].fillna(0)
            y_train = df_train[y_col].astype(int)
            print(f'  ⚠️  Usando coluna "{y_col}" como target proxy (dataset sem labels)')
        
        X_train.columns = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Se houver dados de teste com target, separar para validação
        if test_csv and os.path.exists(test_csv):
            df_test = pd.read_csv(test_csv)
            
            if has_target and target_feature in df_test.columns:
                X_test = df_test.drop(columns=[target_feature])
                y_test = df_test[target_feature]
                
                if y_test.dtype == 'object':
                    from sklearn.preprocessing import LabelEncoder
                    le2 = LabelEncoder()
                    y_test = le2.fit_transform(y_test)
            else:
                from sklearn.model_selection import train_test_split
                numeric_cols_test = df_test.select_dtypes([np.number]).columns.tolist()
                X_test_raw = df_test[numeric_cols_test[:-1]].fillna(0)
                _, X_test_temp, _, y_test = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=random_seed
                )
                X_test = X_test_temp
            X_test = X_test.select_dtypes([np.number]).fillna(0)
            X_test.columns = [f"feature_{i}" for i in range(X_test.shape[1])]
        elif has_target:
            # Validação cruzada se não houver dados de teste
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=0.2, random_state=random_seed,
                stratify=y_train if len(np.unique(y_train)) > 1 else None
            )
        
        # Treinar modelo
        model = PyRuleAnalyzer.new_model(model=model_type)
        
        # Ajustar parâmetros específicos por tipo via classifier interno
        if hasattr(model, 'classifier'):
            estimator_params = {}
            if n_estimators is not None:
                estimator_params['n_estimators'] = n_estimators
            if max_depth is not None:
                estimator_params['max_depth'] = max_depth
            if min_samples_split is not None:
                estimator_params['min_samples_split'] = min_samples_split
            
            for key, val in estimator_params.items():
                clf = model.classifier
                if hasattr(clf, 'set_params'):
                    try:
                        clf.set_params(**{key: val})
                    except Exception:
                        pass
        
        model.fit(X_train, y_train)
        
        # Garantir que arrays estão compilados
        if hasattr(model.classifier, 'compile_tree_arrays'):
            model.classifier.compile_tree_arrays()
        
        # Refinar regras
        refine_stats = model.execute_rule_refinement(
            X=X_train,
            y=y_train,
            remove_below_n_classifications=remove_below_n_classifications,
            save_final_model=False,
            save_report=False,
        )
        results['refine_stats'] = refine_stats
    
    # --------------------------------------------------------
    # Etapa 2: Validar modelo
    # --------------------------------------------------------
    validation = None
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        validation = {
            'accuracy': float(accuracy),
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred.tolist(),
        }
        results['validation'] = validation
    
    # --------------------------------------------------------
    # Etapa 3: Verificar memória
    # --------------------------------------------------------
    memory_check = None
    if memory_report:
        memory_check = _estimate_and_check_memory(
            model, 
            target_platform=target_platform
        )
        results['memory_check'] = memory_check
    
    # --------------------------------------------------------
    # Etapa 4: Gerar sketch Arduino (se solicitado)
    # --------------------------------------------------------
    if generate_arduino_sketch:
        generated_files = _generate_arduino_sketch(
            model,
            base_name=output_name,
            output_dir=output_dir,
            board_model=board_model,
            serial_baud=serial_baud,
            include_sensor_placeholders=include_sensor_placeholders,
            max_flash_percent=max_flash_percent,
            max_ram_percent=max_ram_percent,
        )
        results['generated_files'] = generated_files
    
    # --------------------------------------------------------
    # Salvar modelo se solicitado
    # --------------------------------------------------------
    if save_model:
        save_path = os.path.join(output_dir, f'{output_name}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        results['saved_model_path'] = save_path
    
    elapsed = time.time() - start_time
    results['elapsed'] = elapsed
    results['model'] = model
    
    return results


def _estimate_and_check_memory(
    model,
    target_platform: str = "auto",
) -> Dict[str, Any]:
    """Estimate memory and check compatibility with Arduino/ESP32 platforms."""
    try:
        # Importar de simulate_arduino relativo ao pacote
        import sys
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        examples_dir = os.path.join(pkg_dir, '..', '..')  # ajusta conforme necessário
        
        # Tentar importar do módulo interno primeiro
        from ..pyruleanalyzer import PyRuleAnalyzer as _PyRA
    except Exception:
        pass
    
    # Função inline de estimativa de memória baseada na estrutura do modelo
    estimates = _compute_memory_estimates(model)
    
    results = {}
    
    if target_platform == "auto":
        avr_ok = (estimates['estimated_flash_bytes'] <= estimates['avr_max_flash'] and
                  estimates['total_ram_bytes'] <= estimates['avr_max_sram'])
        esp32_ok = (estimates['estimated_flash_bytes'] <= estimates['esp32_max_flash'] and
                    estimates['total_ram_bytes'] <= estimates['esp32_max_sram'])
        
        results['avr_compatible'] = avr_ok
        results['esp32_compatible'] = esp32_ok
    else:
        if target_platform == 'avr':
            ok = (estimates['estimated_flash_bytes'] <= estimates['avr_max_flash'] and
                  estimates['total_ram_bytes'] <= estimates['avr_max_sram'])
            results['avr_compatible'] = ok
        elif target_platform == 'esp32':
            ok = (estimates['estimated_flash_bytes'] <= estimates['esp32_max_flash'] and
                  estimates['total_ram_bytes'] <= estimates['esp32_max_sram'])
            results['esp32_compatible'] = ok
    
    results['estimates'] = estimates
    
    return results


def _compute_memory_estimates(model) -> Dict[str, Any]:
    """Compute memory usage estimates for the model."""
    # Valores máximos das plataformas
    avr_max_flash = 143360  # Arduino Uno: ~140KB disponível
    avr_max_sram = 2048     # Arduino Uno: 2KB SRAM
    
    esp32_max_flash = 4 * 1024 * 1024  # ESP32: 4MB Flash
    esp32_max_sram = 524288             # ESP32: 512KB SRAM
    
    estimated_flash_bytes = 0
    total_ram_bytes = 0
    
    if hasattr(model, 'classifier'):
        clf = model.classifier
        
        # Estimar tamanho dos arrays do classificador
        if hasattr(clf, 'n_classes_'):
            n_classes = int(clf.n_classes_)
        else:
            n_classes = 2
            
        if hasattr(clf, 'n_features_in_'):
            n_features = int(clf.n_features_in_)
        else:
            n_features = 0
        
        # Estimativa baseada na estrutura do Decision Tree
        if hasattr(clf, 'tree_'):
            tree = clf.tree_
            n_nodes = tree.node_count
            
            # Cada nodo: feature_id (4B) + threshold (8B) + class (4B) + impurity (8B)
            node_size = 24  
            estimated_flash_bytes += n_nodes * node_size
            
            # Arrays de children/parents
            estimated_flash_bytes += n_nodes * 4 * 2  # left/right children
            estimated_flash_bytes += n_nodes * 4  # parent pointer
            
            # Header e metadados
            estimated_flash_bytes += 512
            
        elif hasattr(clf, 'estimators_'):
            # Random Forest / GBDT - somar estimadores
            for est in clf.estimators_:
                if hasattr(est, 'tree_'):
                    tree = est.tree_
                    n_nodes = tree.node_count
                    estimated_flash_bytes += n_nodes * 24 + n_nodes * 12 + 512
        
        # Estimativa de overhead do runtime C
        total_ram_bytes = estimated_flash_bytes * 0.1  # ~10% do flash como RAM temporária
    
    return {
        'estimated_flash_bytes': estimated_flash_bytes,
        'total_ram_bytes': total_ram_bytes,
        'avr_max_flash': avr_max_flash,
        'avr_max_sram': avr_max_sram,
        'esp32_max_flash': esp32_max_flash,
        'esp32_max_sram': esp32_max_sram,
    }


def _generate_arduino_sketch(
    model,
    base_name: str = "model",
    output_dir: str = "files",
    board_model: str = "auto",
    serial_baud: int = 115200,
    include_sensor_placeholders: bool = True,
    max_flash_percent: float = 90.0,
    max_ram_percent: float = 85.0,
) -> Dict[str, str]:
    """Generate Arduino/ESP32 .ino sketch from the model (autocontida)."""
    files = {}
    
    # Try to export Arduino .ino sketch (autocontida — no external dependencies)
    if hasattr(model, 'classifier') and hasattr(model.classifier, 'export_to_arduino_ino'):
        try:
            ino_path = os.path.join(output_dir, f"{base_name}.ino")
            result = model.classifier.export_to_arduino_ino(
                filepath=ino_path,
                board_model=board_model,
                serial_baud=serial_baud,
                include_sensor_placeholders=include_sensor_placeholders,
                include_memory_check=True,
                max_flash_percent=max_flash_percent,
                max_ram_percent=max_ram_percent,
            )
            files['arduino'] = ino_path
            files['memory_check'] = result.get('memory_check', {})
            return files
        except Exception as e:
            print(f'  [!] Error generating .ino sketch: {e}')
    
    # Fallback: generate C header only via PyRuleAnalyzer
    h_path = os.path.join(output_dir, f"{base_name}.h")
    
    if hasattr(model, 'export_to_c_header'):
        try:
            model.export_to_c_header(filepath=h_path)
            files['header'] = h_path
            return files
        except Exception as e:
            raise RuntimeError(f'Error exporting C header: {e}')
    
    # If no export_to_c_header, try manually
    if hasattr(model, 'classifier') and hasattr(model.classifier, 'tree_'):
        from .pyruleanalyzer import PyRuleAnalyzer
        model.export_to_c_header(filepath=h_path)
        files['header'] = h_path
    
    return files
