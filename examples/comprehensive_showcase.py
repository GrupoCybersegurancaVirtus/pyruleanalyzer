import os
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Ensure pyruleanalyzer can be imported from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyruleanalyzer import PyRuleAnalyzer
from pyruleanalyzer import DTAnalyzer

def main():
    print("="*80)
    print("PYRULEANALYZER COMPREHENSIVE SHOWCASE")
    print("="*80)

    # =========================================================================
    # 1. DATA PREPARATION
    # =========================================================================
    print("\n[1] Preparing synthetic data...")
    # Create a synthetic dataset for demonstration purposes
    X_array, y_array = make_classification(
        n_samples=1000, n_features=5, n_informative=3, n_classes=2, random_state=42
    )
    
    # Convert to pandas DataFrame for realistic usage
    feature_names = [f"Feature_{i}" for i in range(1, 6)]
    X = pd.DataFrame(X_array, columns=feature_names)
    y = pd.Series(y_array, name="Target")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # For demonstrating the legacy CSV workflow, let's save the splits to files
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    os.makedirs("files", exist_ok=True)
    train_csv_path = "files/temp_train.csv"
    test_csv_path = "files/temp_test.csv"
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    # =========================================================================
    # 2. MODEL TRAINING (Scikit-Learn API)
    # =========================================================================
    print("\n[2] Training the Model...")
    
    # Instantiate the classifier (Supports: 'Decision Tree', 'Random Forest', 'Gradient Boosting Decision Trees')
    model = PyRuleAnalyzer.new_model(model="Decision Tree")
    
    # METHOD A: Train using in-memory DataFrames (Modern approach)
    print(" -> Fitting model using Pandas DataFrames (X_train, y_train)...")
    model.fit(X_train, y_train)

    # METHOD B (Commented out): Train using CSV paths (Legacy/Overloaded approach)
    # print(" -> Fitting model using CSV paths...")
    # model.fit(train_csv_path, test_csv_path)

    # =========================================================================
    # 3. RULE ANALYSIS
    # =========================================================================
    print("\n[3] Analyzing and Refining Rules...")
    
    # Create an analyzer specific to the model type
    analyzer = DTAnalyzer(model.classifier)
    
    # Execute analysis to remove duplicate and low-usage rules
    # It uses the test dataset to evaluate which rules are actually useful
    analyzer.execute_rule_analysis(
        file_path=test_csv_path,          # Test data to evaluate rule usage
        remove_below_n_classifications=0, # Refine rules that are never used (0 classifications)
        save_final_model=False,           # We won't save the pkl here
        save_report=False                 # We won't generate the .txt report
    )

    # =========================================================================
    # 4. INTERACTIVE REPORTING (Jupyter Friendly)
    # =========================================================================
    print("\n[4] Generating Summary Report...")
    
    report = model.summary_report()
    print(f" -> Model Type: {report['model_type']}")
    print(f" -> Total Active Rules: {report['total_rules']}")
    print(f" -> Detected Classes: {report['classes']}")
    
    # Show the first rule as an example
    if report['rules']:
        first_rule = report['rules'][0]
        print(f" -> Example Rule [{first_rule['name']}]:")
        print(f"    IF {first_rule['conditions']} THEN Class = {first_rule['class']}")

    # =========================================================================
    # 5. PREDICTIONS (Single & Batch)
    # =========================================================================
    print("\n[5] Making Predictions...")
    
    # A. Single Sample Prediction (Dictionary)
    sample_dict = X_test.iloc[0].to_dict()
    single_pred = model.predict(sample_dict)
    print(f" -> Single Sample Prediction (Dict): Class {single_pred}")

    # B. Batch Prediction (DataFrame/NumPy)
    # This automatically compiles the rules to C/NumPy under the hood!
    batch_preds = model.predict(X_test)
    accuracy = (batch_preds == y_test).mean()
    print(f" -> Batch Prediction Accuracy: {accuracy:.4f} (Tested on {len(y_test)} samples)")

    # C. Probabilities (Works for all, but most useful for Random Forest/GBDT)
    probs = model.predict_proba(X_test)
    print(f" -> First sample probabilities: {probs[0]}")

    # =========================================================================
    # 6. COMPARING METRICS
    # =========================================================================
    print("\n[6] Comparing Initial vs Final (Refined) Metrics...")
    # This prints out a full comparison including confusion matrices and complexity scores
    analyzer.compare_initial_final_results(test_csv_path)

    # =========================================================================
    # 7. EXPORTING THE MODEL
    # =========================================================================
    print("\n[7] Exporting the Classifier...")
    
    # Export as a standalone Python script
    py_path = "files/my_exported_model.py"
    model.to_python(py_path)
    print(f" -> Exported to Python: {py_path}")

    # Export as a C header file (Great for embedded systems like Arduino)
    c_path = "files/my_exported_model.h"
    model.to_c_header(c_path)
    print(f" -> Exported to C Header: {c_path}")

    # Export as a compact Binary file (PYRA format)
    bin_path = "files/my_exported_model.bin"
    model.to_binary(bin_path)
    print(f" -> Exported to Binary: {bin_path}")

    # =========================================================================
    # 8. LOADING THE BINARY MODEL
    # =========================================================================
    print("\n[8] Loading Binary Model...")
    
    # Load the binary model without needing Scikit-Learn
    loaded_model = PyRuleAnalyzer.load_binary(bin_path)
    loaded_preds = loaded_model.predict(X_test)
    
    match = (loaded_preds == batch_preds).all()
    print(f" -> Binary loaded successfully. Predictions match original: {match}")

    # =========================================================================
    # 9. INTERACTIVE EDITING (Uncomment to try)
    # =========================================================================
    # print("\n[9] Interactive Editing (Uncomment to test)...")
    # model.edit_rules()

    print("\n" + "="*80)
    print("SHOWCASE COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()