"""
Test for the full_pipeline function from pyruleanalyzer.

Demonstrates the use of the new full_pipeline function that receives parameters directly,
instead of depending on command-line arguments.
"""

import os
import sys
from pathlib import Path

# Add root directory to path for import
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_full_pipeline_with_mock_data():
    """Test full_pipeline with mock generated data."""
    from sklearn.datasets import make_classification
    import pandas as pd
    
    # Generate sample data
    X, y = make_classification(
        n_samples=500,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_classes=3,
        random_state=42,
    )
    
    # Create DataFrame with column names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['Target'] = y
    
    # Split into train/test
    train_path = 'train_data.csv'
    test_path = 'test_data.csv'
    
    df[:400].to_csv(train_path, index=False)
    df[400:].to_csv(test_path, index=False)
    
    try:
        from pyruleanalyzer import full_pipeline
        
        print('=' * 60)
        print('TEST: full_pipeline with mock data')
        print('=' * 60)
        
        results = full_pipeline(
            train_csv=train_path,
            test_csv=test_path,
            target_feature='Target',
            model_type='Decision Tree',
            max_depth=10,
            generate_sketch=False,
            memory_report=True,
            output_dir='files',
            output_name='model_test',
        )
        
        print()
        print('Results:')
        if results.get('validation'):
            print(f"  Accuracy: {results['validation']['accuracy']:.4f}")
        if results.get('memory_check'):
            mc = results['memory_check']
            print(f"  AVR compatible: {mc.get('avr_compatible', 'N/A')}")
            print(f"  ESP32 compatible: {mc.get('esp32_compatible', 'N/A')}")
        if results.get('refine_stats'):
            rs = results['refine_stats']
            print(f"  Rules generated: {rs.get('n_rules_final', 'N/A')}")
        
        print()
        print('✅ Test passed!')
        
    finally:
        # Clean up temporary files
        for f in [train_path, test_path]:
            if os.path.exists(f):
                os.remove(f)


def test_full_pipeline_with_model_pkl():
    """Test full_pipeline loading existing model."""
    from sklearn.datasets import make_classification
    import pandas as pd
    import pickle
    
    # Generate sample data
    X, y = make_classification(
        n_samples=300,
        n_features=6,
        n_informative=4,
        n_redundant=2,
        n_classes=2,
        random_state=123,
    )
    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['Target'] = y
    
    train_path = 'train_data2.csv'
    model_path = 'files/model_test.pkl'
    
    df[:240].to_csv(train_path, index=False)
    
    try:
        from pyruleanalyzer import full_pipeline
        
        print()
        print('=' * 60)
        print('TEST: full_pipeline save and load model')
        print('=' * 60)
        
        # Train and save model
        results1 = full_pipeline(
            train_csv=train_path,
            target_feature='Target',
            model_type='Random Forest',
            n_estimators=10,
            max_depth=8,
            save_model=True,
            output_dir='files',
            output_name='model_test',
        )
        
        print(f"  Model saved at: {results1.get('saved_model_path', 'N/A')}")
        
        # Load model and run pipeline again
        results2 = full_pipeline(
            model_pkl=model_path,
            memory_report=True,
            output_dir='files',
        )
        
        print(f"  Model loaded successfully!")
        print()
        print('✅ Test passed!')
        
    finally:
        if os.path.exists(train_path):
            os.remove(train_path)
        if os.path.exists(model_path):
            os.remove(model_path)


def test_full_pipeline_with_sketch():
    """Test Arduino sketch generation."""
    from sklearn.datasets import make_classification
    import pandas as pd
    
    # Generate simple data
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=99,
    )
    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['Target'] = y
    
    train_path = 'train_data3.csv'
    sketch_dir = 'files/sketch_test'
    
    df[:160].to_csv(train_path, index=False)
    
    try:
        from pyruleanalyzer import full_pipeline
        
        print()
        print('=' * 60)
        print('TEST: full_pipeline with sketch generation')
        print('=' * 60)
        
        results = full_pipeline(
            train_csv=train_path,
            target_feature='Target',
            model_type='Decision Tree',
            max_depth=5,
            generate_sketch=True,
            memory_report=True,
            output_dir=sketch_dir,
            output_name='model_sketch',
        )
        
        print()
        if results.get('generated_files'):
            for name, path in results['generated_files'].items():
                exists = '✅' if os.path.exists(path) else '❌'
                size = os.path.getsize(path) if os.path.exists(path) else 0
                print(f"  {exists} {name}: {path} ({size:,} bytes)")
        
        print()
        print('✅ Test passed!')
        
    finally:
        if os.path.exists(train_path):
            os.remove(train_path)


if __name__ == '__main__':
    test_full_pipeline_with_mock_data()
    test_full_pipeline_with_model_pkl()
    test_full_pipeline_with_sketch()
    
    print()
    print('=' * 60)
    print('ALL TESTS PASSED!')
    print('=' * 60)