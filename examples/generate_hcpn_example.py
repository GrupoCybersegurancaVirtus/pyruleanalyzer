"""Generate CPN Tools HCPN models from a pyRuleAnalyzer model.

This example shows how to convert a trained model into a **Hierarchical
Coloured Petri Net** (HCPN) in the native CPN Tools 4.x ``.cpn`` format, for
both the *initial* (unrefined) model and the *final* (post-refinement) model,
so they can be opened, visualised, simulated and verified in
`CPN Tools <https://cpntools.org>`_.

The mapping implements the proofs of correctness from the accompanying article
(Theorem 1: Decision Tree -> CPN; Theorem 2: GBDT -> HCPN with sequential
boosting-stage accumulation and a NumPy-compatible argmax / sigmoid decision).

Run:
    python examples/generate_hcpn_example.py
    python examples/generate_hcpn_example.py --model "Decision Tree"
    python examples/generate_hcpn_example.py --model "Random Forest"
    python examples/generate_hcpn_example.py --classes 3        # multiclass GBDT
"""

import argparse
import os
import sys

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Make pyruleanalyzer importable when run from the repository.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyruleanalyzer import PyRuleAnalyzer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Gradient Boosting Decision Trees",
                        choices=["Gradient Boosting Decision Trees",
                                 "Decision Tree", "Random Forest"])
    parser.add_argument("--classes", type=int, default=2,
                        help="Number of target classes (>=2 enables multiclass).")
    parser.add_argument("--out", default="files/hcpn",
                        help="Output base name (under files/ by default).")
    args = parser.parse_args()

    os.makedirs("files", exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Synthetic dataset (replace with your own IDS / DDoS CSVs).
    # ------------------------------------------------------------------
    print("[1] Preparing data...")
    X_arr, y_arr = make_classification(
        n_samples=1200, n_features=6, n_informative=4,
        n_classes=args.classes, n_clusters_per_class=1, random_state=42)
    feature_names = [f"feat_{i}" for i in range(X_arr.shape[1])]
    X = pd.DataFrame(X_arr, columns=feature_names)
    y = pd.Series(y_arr, name="target")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    os.makedirs("files", exist_ok=True)
    train_csv, test_csv = "files/hcpn_train.csv", "files/hcpn_test.csv"
    pd.concat([X_train, y_train], axis=1).to_csv(train_csv, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_csv, index=False)

    # ------------------------------------------------------------------
    # 2. Train + extract rules (initial model).
    # ------------------------------------------------------------------
    print(f"[2] Training {args.model} and extracting rules...")
    params = {
        "Gradient Boosting Decision Trees": {
            "n_estimators": 8, "max_depth": 3, "learning_rate": 0.1,
            "random_state": 42},
        "Decision Tree": {"max_depth": 5, "random_state": 42},
        "Random Forest": {"n_estimators": 6, "max_depth": 4, "random_state": 42},
    }[args.model]

    analyzer = PyRuleAnalyzer.create(
        train_path=train_csv, test_path=test_csv,
        model=args.model, params=params, refine=False)

    # ------------------------------------------------------------------
    # 3. Refine the rules (final model).
    # ------------------------------------------------------------------
    print("[3] Refining rules (duplicate + low-usage removal)...")
    stats = analyzer.execute_rule_refinement(
        test_path=test_csv, remove_below_n_classifications=1,
        save_final_model=False, save_report=False)
    print(f"    rules: {stats['rules_before']} -> {stats['rules_after']} "
          f"({stats['reduction_percent']:.1f}% reduction)")

    # ------------------------------------------------------------------
    # 4. Export both HCPN models for CPN Tools.
    #    A real test sample is used as the initial marking so the net can
    #    be simulated immediately after opening it in CPN Tools.
    # ------------------------------------------------------------------
    print("[4] Generating CPN Tools .cpn HCPN models...")
    sample = X_test.iloc[0]
    paths = analyzer.export_hcpn(
        base_name=args.out, which="both", sample=sample,
        feature_names=feature_names, model_name="GBDT")

    print("\nDone. Open these files in CPN Tools (File -> Load Net):")
    for which, path in paths.items():
        print(f"  - {which:7s}: {os.path.abspath(path)}")
    print("\nThe top page distributes the input sample to each class channel;")
    print("each boosting stage is a substitution transition bound to its tree")
    print("subpage, and the decision transition implements the argmax / sigmoid")
    print("policy. Use the simulation tools in CPN Tools to step through it.")


if __name__ == "__main__":
    main()
