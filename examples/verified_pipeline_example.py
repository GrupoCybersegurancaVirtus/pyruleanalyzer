"""End-to-end pipeline with automatic model checking.

Runs the whole flow the tool is built around, in one call:

    train (or reuse) a scikit-learn model
      -> extract rules
      -> export the initial HCPN and model check it
      -> refine the rules
      -> export the refined HCPN and model check it again
      -> compare the two verdicts
      -> export Python / binary / C / Arduino
      -> optionally compile and upload to the board

The point of running the checker twice is the comparison at the end. Refinement
removes rules, merges siblings and promotes survivors, and when it goes wrong
the accuracy usually does not move -- overlapping rules tend to agree on the
class. The verification does move: ``B1`` fails the moment the refined rule set
stops being a partition of the feature space, and the pipeline then refuses to
export the model instead of shipping a net that is nondeterministic.

Run:
    python examples/verified_pipeline_example.py
    python examples/verified_pipeline_example.py --model "Random Forest"
    python examples/verified_pipeline_example.py --classes 3 --arduino
    python examples/verified_pipeline_example.py --askctl        # + CPN Tools scripts

To also flash a board (needs arduino-cli and a connected device):
    python examples/verified_pipeline_example.py --arduino \\
        --fqbn arduino:avr:nano --port COM3 --upload
"""

import argparse
import os
import sys

import pandas as pd
from sklearn.datasets import make_classification

# Make pyruleanalyzer importable when run from the repository.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyruleanalyzer import verified_pipeline
from pyruleanalyzer.verified_pipeline import VerificationError


# Function to parse the command-line arguments of the example.
def parse_args():
    """Read the example's options.

    Returns:
        argparse.Namespace: The parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Gradient Boosting Decision Trees",
                        choices=["Gradient Boosting Decision Trees",
                                 "Decision Tree", "Random Forest"])
    parser.add_argument("--classes", type=int, default=2,
                        help="Number of target classes.")
    parser.add_argument("--samples", type=int, default=3,
                        help="Test samples fixed, one at a time, in the net's "
                             "initial marking during verification.")
    parser.add_argument("--max-nodes", type=int, default=200_000,
                        help="Occurrence-graph budget.")
    parser.add_argument("--arduino", action="store_true",
                        help="Also export a ready-to-flash .ino sketch.")
    parser.add_argument("--askctl", action="store_true",
                        help="Also write the ASK-CTL scripts for CPN Tools.")
    parser.add_argument("--fqbn", default=None,
                        help="Board FQBN for arduino-cli, e.g. arduino:avr:nano.")
    parser.add_argument("--port", default=None, help="Serial port, e.g. COM3.")
    parser.add_argument("--upload", action="store_true",
                        help="Upload after compiling (needs --fqbn and --port).")
    parser.add_argument("--out", default="files/verified",
                        help="Output base path.")
    return parser.parse_args()


# Function to run the example.
def main():
    """Build a synthetic dataset and run the verified pipeline on it."""
    args = parse_args()

    X_arr, y_arr = make_classification(
        n_samples=1200, n_features=6, n_informative=4,
        n_classes=args.classes, n_clusters_per_class=1, random_state=42)
    X = pd.DataFrame(X_arr, columns=[f"feat_{i}" for i in range(6)])
    y = pd.Series(y_arr, name="Target")

    params = {
        "Gradient Boosting Decision Trees": {
            "n_estimators": 8, "max_depth": 3, "learning_rate": 0.1,
            "random_state": 42},
        "Decision Tree": {"max_depth": 5, "random_state": 42},
        "Random Forest": {"n_estimators": 6, "max_depth": 4, "random_state": 42},
    }[args.model]

    formats = ["python", "binary", "c"]
    if args.arduino or args.fqbn:
        formats.append("arduino")

    out_dir = os.path.dirname(args.out) or "files"
    out_name = os.path.basename(args.out)

    try:
        result = verified_pipeline(
            X=X, y=y,
            model_type=args.model, params=params,
            remove_below_n_classifications=1,
            verify_samples=args.samples,
            max_nodes=args.max_nodes,
            export_askctl=args.askctl,
            export_formats=formats,
            deploy_fqbn=args.fqbn,
            deploy_port=args.port,
            upload=args.upload,
            output_dir=out_dir,
            output_name=out_name,
            verbose=True,
        )
    except VerificationError as exc:
        # The pipeline refused to export: the net no longer satisfies what the
        # initial one did. The failing report is attached to the exception.
        print(f"\n[!] {exc}")
        print(exc.result.report())
        return 1

    print("\nGenerated files:")
    for label, path in result["hcpn"].items():
        print(f"  hcpn ({label:7s}) {os.path.abspath(path)}")
    for fmt, path in result["exports"].items():
        if isinstance(path, str):
            print(f"  {fmt:<14} {os.path.abspath(path)}")

    print("\nThe .cpn files open directly in CPN Tools (File -> Load Net).")
    if args.askctl:
        print("The .sml files next to them ask CPN Tools the same questions:")
        print("  open the net, apply Enter State Space, then evaluate")
        print('  use "<file>.sml";  in an auxiliary text (it builds the graphs itself).')
    return 0


if __name__ == "__main__":
    sys.exit(main())
