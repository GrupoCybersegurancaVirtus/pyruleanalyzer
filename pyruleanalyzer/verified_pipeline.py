"""End-to-end pipeline with formal verification as a build step.

The tool already knew how to go from a scikit-learn model to rules, from rules
to a Coloured Petri Net, and from rules to code that runs on a microcontroller.
What it did not know how to do was *check*, without leaving Python, that the net
still behaves like a classifier after the rules were refined. This module chains
the whole thing together and makes that check a gate rather than an afterthought:

.. code-block:: text

    1. train a scikit-learn model, or take one that is already trained
    2. extract the rules and export the initial HCPN
    3. model check the initial model                      (optional baseline)
    4. refine the rules
    5. model check the refined model                      (the acceptance test)
    6. export: Python, binary, C header, Arduino sketch
    7. deploy to the edge device                          (optional)

Step 5 is the point of the pipeline. Refinement removes rules, merges siblings
and promotes survivors; accuracy barely moves when it goes wrong, because
overlapping rules usually still agree on the class. The verification does move:
``B1`` (leaf mutual exclusion) fails the moment the refined rule set stops being
a partition of the feature space, and with ``fail_on_violation`` set the pipeline
refuses to export a model whose net no longer satisfies what the initial one did.

Usage:
    from pyruleanalyzer import verified_pipeline

    result = verified_pipeline(
        train_csv="data/train.csv", test_csv="data/test.csv",
        target_feature="Target", model_type="Random Forest",
        export_formats=("python", "c", "arduino"),
    )
    print(result["verification"]["final"].report())
"""

import os
import pickle
import shutil
import subprocess
import time
from typing import Any, Dict, Optional, Sequence

__all__ = ["verified_pipeline", "VerificationError"]


# Exception raised when the verified pipeline finds a violated property.
class VerificationError(RuntimeError):
    """A model-checking property was violated and the pipeline was gated on it.

    Attributes:
        result: The :class:`~pyruleanalyzer.model_checker.ModelCheckResult` that
            failed, so the caller can inspect the counterexample.
        stage (str): ``"initial"`` or ``"final"``.
    """

    # Method to build the exception with the failing result attached.
    def __init__(self, message, result=None, stage=""):
        """Create the exception.

        Args:
            message (str): Human-readable summary.
            result: The failing model-checking result.
            stage (str): Which model failed.
        """
        super().__init__(message)
        self.result = result
        self.stage = stage


# Function to print a pipeline step header.
def _step(verbose: bool, number: int, title: str) -> None:
    """Print a numbered pipeline step header.

    Args:
        verbose (bool): Whether to print anything at all.
        number (int): Step number.
        title (str): Step title.
    """
    if verbose:
        print(f"\n[{number}] {title}")
        print("-" * (len(title) + 6))


# Function to load a dataset from CSV files or in-memory arrays.
def _load_data(X, y, train_csv, test_csv, target_feature, test_size, random_seed):
    """Resolve the pipeline's data arguments into train/test splits.

    Args:
        X: Feature matrix, or None when CSV files are used.
        y: Target vector, or None when CSV files are used.
        train_csv (str): Training CSV path.
        test_csv (str): Test CSV path, optional.
        target_feature (str): Target column name.
        test_size (float): Hold-out fraction when no test set is given.
        random_seed (int): Split seed.

    Returns:
        tuple: ``(X_train, X_test, y_train, y_test)`` as pandas objects.

    Raises:
        ValueError: If neither arrays nor a training CSV were provided.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    if X is not None:
        X = pd.DataFrame(X) if not hasattr(X, "columns") else X
        y = pd.Series(y) if not hasattr(y, "iloc") else y
        return train_test_split(X, y, test_size=test_size,
                                random_state=random_seed)

    if not train_csv:
        raise ValueError(
            "verified_pipeline needs data: pass X and y, or train_csv, or an "
            "already-trained model together with X/y for validation."
        )

    df = pd.read_csv(train_csv)
    if target_feature not in df.columns:
        raise ValueError(f"column '{target_feature}' not found in {train_csv}")
    X_train = df.drop(columns=[target_feature])
    y_train = df[target_feature]

    if test_csv and os.path.exists(test_csv):
        dft = pd.read_csv(test_csv)
        return X_train, dft.drop(columns=[target_feature]), y_train, \
            dft[target_feature]
    return train_test_split(X_train, y_train, test_size=test_size,
                            random_state=random_seed)


# Function to pick the samples used as initial markings during verification.
def _pick_samples(X_test, n: int):
    """Select the samples fixed in the net's initial marking.

    Args:
        X_test: Test features, or None.
        n (int): How many samples to take.

    Returns:
        The selected rows, or None to use the net's own initial marking.
    """
    if X_test is None or n <= 0:
        return None
    return X_test.iloc[:n] if hasattr(X_test, "iloc") else X_test[:n]


# Function to compile and upload a generated sketch with arduino-cli.
def _deploy_arduino(ino_path: str, fqbn: str, port: str, cli: str = "arduino-cli",
                    upload: bool = True, verbose: bool = True) -> Dict[str, Any]:
    """Compile and optionally upload an exported sketch to a board.

    Args:
        ino_path (str): The generated ``.ino`` file.
        fqbn (str): Fully qualified board name, e.g. ``arduino:avr:nano``.
        port (str): Serial port of the board, e.g. ``COM3``.
        cli (str): Path to the ``arduino-cli`` executable.
        upload (bool): Upload after a successful compile.
        verbose (bool): Print the commands and their outcome.

    Returns:
        dict: ``{"sketch_dir", "compiled", "uploaded", "commands", "error"}``.
    """
    result: Dict[str, Any] = {"sketch_dir": None, "compiled": False,
                              "uploaded": False, "commands": [], "error": None}

    # arduino-cli expects <dir>/<dir>.ino, so the sketch is placed in its own
    # folder named after the file.
    base = os.path.splitext(os.path.basename(ino_path))[0]
    sketch_dir = os.path.join(os.path.dirname(os.path.abspath(ino_path)), base)
    os.makedirs(sketch_dir, exist_ok=True)
    target = os.path.join(sketch_dir, base + ".ino")
    if os.path.abspath(target) != os.path.abspath(ino_path):
        shutil.copyfile(ino_path, target)
    result["sketch_dir"] = sketch_dir

    compile_cmd = [cli, "compile", "--fqbn", fqbn, sketch_dir]
    upload_cmd = [cli, "upload", "-p", port, "--fqbn", fqbn, sketch_dir]
    result["commands"] = [" ".join(compile_cmd), " ".join(upload_cmd)]

    if shutil.which(cli) is None:
        result["error"] = (f"{cli} not found on PATH; run the commands above "
                           "manually once it is installed")
        if verbose:
            print(f"  [!] {result['error']}")
        return result

    if verbose:
        print(f"  $ {result['commands'][0]}")
    proc = subprocess.run(compile_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        result["error"] = proc.stderr.strip() or proc.stdout.strip()
        if verbose:
            print(f"  [!] compilation failed:\n{result['error']}")
        return result
    result["compiled"] = True
    if verbose:
        print("  compiled")

    if not upload or not port:
        return result
    if verbose:
        print(f"  $ {result['commands'][1]}")
    proc = subprocess.run(upload_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        result["error"] = proc.stderr.strip() or proc.stdout.strip()
        if verbose:
            print(f"  [!] upload failed:\n{result['error']}")
        return result
    result["uploaded"] = True
    if verbose:
        print("  uploaded")
    return result


# Function to run the whole verified pipeline from data to edge device.
def verified_pipeline(
    X=None,
    y=None,
    train_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    target_feature: str = "Target",
    sklearn_model=None,
    model_pkl: Optional[str] = None,
    model_type: str = "Decision Tree",
    params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.25,
    random_seed: int = 42,
    refine: bool = True,
    remove_below_n_classifications: int = -1,
    verify_initial: bool = True,
    verify_final: bool = True,
    verify_samples: int = 3,
    max_nodes: int = 200_000,
    reduction: Optional[str] = None,
    check_consistency: bool = True,
    export_askctl: bool = False,
    fail_on_violation: bool = True,
    export_formats: Sequence[str] = ("python",),
    board_model: str = "auto",
    serial_baud: int = 115200,
    deploy_fqbn: Optional[str] = None,
    deploy_port: Optional[str] = None,
    arduino_cli: str = "arduino-cli",
    upload: bool = False,
    output_dir: str = "files",
    output_name: str = "model",
    save_model: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full pipeline: train, verify, refine, verify again, export, deploy.

    Args:
        X: Feature matrix (DataFrame or array). Alternative to ``train_csv``.
        y: Target vector. Required together with ``X``.
        train_csv: Training CSV path. Alternative to ``X``/``y``.
        test_csv: Test CSV path. When omitted a hold-out split is taken.
        target_feature: Target column name in the CSV files.
        sklearn_model: An already-fitted scikit-learn estimator to use instead
            of training a new one. ``X``/``y`` (or the CSVs) are then only used
            for validation and for the rule-usage statistics of refinement.
        model_pkl: A saved :class:`PyRuleAnalyzer` to load instead of training.
        model_type: ``"Decision Tree"``, ``"Random Forest"`` or
            ``"Gradient Boosting Decision Trees"``.
        params: Hyperparameters for the scikit-learn estimator.
        test_size: Hold-out fraction when no test set is given.
        random_seed: Seed for the train/test split.
        refine: Run rule refinement. When False the pipeline stops at the
            initial model and verifies only that one.
        remove_below_n_classifications: Usage threshold below which a rule is
            removed during refinement. ``-1`` disables it.
        verify_initial: Model check the unrefined model, to have a baseline to
            compare the refined one against.
        verify_final: Model check the refined model.
        verify_samples: How many test samples are model checked, one
            occurrence graph each (the sample sits in the initial marking).
        max_nodes: Occurrence-graph budget; beyond it the state-space properties
            get no verdict.
        reduction: ``"stubborn"`` to model check on stubborn-set reduced
            occurrence graphs (same verdicts, far fewer markings for forests
            and multiclass boosting); None for the full graph.
        check_consistency: Compare the class the net computes against the
            classifier's own prediction (property PC) on the whole test set.
        export_askctl: Also write the ASK-CTL (SML) script of each net, ready to
            run in CPN Tools.
        fail_on_violation: Raise :class:`VerificationError` instead of exporting
            a model whose net violates a property.
        export_formats: Any of ``"python"``, ``"binary"``, ``"c"``,
            ``"arduino"``, ``"cpn"``.
        board_model: Target board for the Arduino sketch.
        serial_baud: Serial baud rate written into the sketch.
        deploy_fqbn: Fully qualified board name for ``arduino-cli`` (for example
            ``arduino:avr:nano``). Deployment is skipped when omitted.
        deploy_port: Serial port of the board. Required to upload.
        arduino_cli: Path to the ``arduino-cli`` executable.
        upload: Upload the compiled sketch to the board. Compilation alone runs
            whenever ``deploy_fqbn`` is given.
        output_dir: Directory for every generated file.
        output_name: Base name for every generated file.
        save_model: Pickle the analyzer next to the other outputs.
        verbose: Print the progress and the verification reports.

    Returns:
        dict: With keys ``model``, ``metrics``, ``hcpn``, ``verification``,
        ``exports``, ``deployment``, ``passed`` and ``elapsed``. The
        ``verification`` entry holds the
        :class:`~pyruleanalyzer.model_checker.ModelCheckResult` of each stage
        plus a ``comparison`` string when both were verified.

    Raises:
        VerificationError: If ``fail_on_violation`` is set and a checked model
            violates one of its properties.

    Example:
        >>> result = verified_pipeline(
        ...     train_csv="data/train.csv", test_csv="data/test.csv",
        ...     target_feature="Target",
        ...     model_type="Gradient Boosting Decision Trees",
        ...     export_formats=("python", "c", "arduino"),
        ... )
        >>> result["passed"], result["metrics"]["accuracy_final"]
        (True, 0.94)
    """
    from sklearn.metrics import accuracy_score

    from .model_checker import compare_results
    from .pyruleanalyzer import PyRuleAnalyzer

    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, output_name)

    results: Dict[str, Any] = {
        "model": None,
        "metrics": {},
        "hcpn": {},
        "verification": {},
        "exports": {},
        "deployment": None,
        "passed": True,
        "elapsed": 0.0,
    }

    # ------------------------------------------------------------------
    # 1. Data and model
    # ------------------------------------------------------------------
    _step(verbose, 1, "Data and model")
    X_train = X_test = y_train = y_test = None
    if X is not None or train_csv:
        X_train, X_test, y_train, y_test = _load_data(
            X, y, train_csv, test_csv, target_feature, test_size, random_seed)
        if verbose:
            print(f"  train: {len(X_train)} rows   test: {len(X_test)} rows   "
                  f"features: {X_train.shape[1]}")

    if model_pkl:
        with open(model_pkl, "rb") as fh:
            analyzer = pickle.load(fh)
        if verbose:
            print(f"  loaded analyzer from {model_pkl}")
    elif sklearn_model is not None:
        names = list(X_train.columns) if X_train is not None else None
        analyzer = PyRuleAnalyzer.from_sklearn(sklearn_model, feature_names=names)
        if verbose:
            print(f"  wrapped a fitted {type(sklearn_model).__name__}")
    else:
        if X_train is None:
            raise ValueError("training data is required when no model is given")
        analyzer = PyRuleAnalyzer.new_model(model=model_type, params=params)
        analyzer.fit(X_train, y_train)
        if verbose:
            print(f"  trained {model_type} with "
                  f"{len(analyzer.classifier.initial_rules)} rules")

    results["model"] = analyzer
    results["metrics"]["rules_initial"] = len(analyzer.classifier.initial_rules)
    if X_test is not None:
        acc = accuracy_score(y_test, analyzer.predict(X_test, use_refined=False))
        results["metrics"]["accuracy_initial"] = float(acc)
        if verbose:
            print(f"  accuracy (initial): {acc:.4f}")

    samples = _pick_samples(X_test, verify_samples)

    # ------------------------------------------------------------------
    # 2-3. Initial HCPN and its verification
    # ------------------------------------------------------------------
    _step(verbose, 2, "Initial model -> HCPN")
    first = samples.iloc[0] if samples is not None and hasattr(samples, "iloc") \
        else None
    paths = analyzer.export_hcpn(base_name=base, which="initial", sample=first)
    results["hcpn"]["initial"] = paths["initial"]

    if verify_initial:
        _step(verbose, 3, "Model checking the initial model")
        initial_result = analyzer.model_check(
            which="initial", cpn_path=paths["initial"], samples=samples,
            max_nodes=max_nodes, reduction=reduction,
            check_consistency=check_consistency,
            test_samples=X_test, askctl=export_askctl, verbose=verbose)
        results["verification"]["initial"] = initial_result
        if not initial_result.passed:
            results["passed"] = False
            if fail_on_violation:
                raise VerificationError(
                    "the initial model violates "
                    f"{[f.id for f in initial_result.failures]}",
                    initial_result, "initial")

    # ------------------------------------------------------------------
    # 4-5. Refinement and its verification
    # ------------------------------------------------------------------
    if refine:
        _step(verbose, 4, "Rule refinement")
        stats = analyzer.execute_rule_refinement(
            X=X_test, y=y_test,
            remove_below_n_classifications=remove_below_n_classifications,
            save_final_model=False, save_report=False)
        results["metrics"].update({
            "rules_final": stats["rules_after"],
            "rules_removed": stats["rules_removed"],
            "reduction_percent": stats["reduction_percent"],
        })
        if verbose:
            print(f"  rules: {stats['rules_before']} -> {stats['rules_after']} "
                  f"({stats['reduction_percent']:.1f}% reduction)")
        if X_test is not None:
            acc = accuracy_score(y_test, analyzer.predict(X_test))
            results["metrics"]["accuracy_final"] = float(acc)
            if verbose:
                print(f"  accuracy (refined): {acc:.4f}")

        _step(verbose, 5, "Refined model -> HCPN and model checking")
        paths = analyzer.export_hcpn(base_name=base, which="final", sample=first)
        results["hcpn"]["final"] = paths["final"]

        if verify_final:
            final_result = analyzer.model_check(
                which="final", cpn_path=paths["final"], samples=samples,
                max_nodes=max_nodes, reduction=reduction,
                check_consistency=check_consistency,
                test_samples=X_test, askctl=export_askctl, verbose=verbose)
            results["verification"]["final"] = final_result
            if "initial" in results["verification"]:
                comparison = compare_results(
                    results["verification"]["initial"], final_result)
                results["verification"]["comparison"] = comparison
                if verbose:
                    print()
                    print(comparison)
            if not final_result.passed:
                results["passed"] = False
                if fail_on_violation:
                    raise VerificationError(
                        "refinement broke the model: the refined net violates "
                        f"{[f.id for f in final_result.failures]}",
                        final_result, "final")

    # ------------------------------------------------------------------
    # 6. Exports
    # ------------------------------------------------------------------
    formats = [f.lower() for f in (export_formats or ())]
    if formats:
        _step(verbose, 6, "Exports")
        plain = [f for f in formats if f in ("python", "binary", "bin", "c",
                                             "header")]
        if plain:
            exported = analyzer.export(base_name=base, formats=plain)
            results["exports"].update(exported)
        if "arduino" in formats:
            ino = f"{base}.ino"
            info = analyzer.classifier.export_to_arduino_ino(
                filepath=ino, board_model=board_model, serial_baud=serial_baud)
            results["exports"]["arduino"] = ino
            results["exports"]["memory_check"] = info.get("memory_check", {})
        if "cpn" in formats:
            results["exports"].update(
                {f"cpn_{k}": v for k, v in results["hcpn"].items()})
        if verbose:
            for fmt, path in results["exports"].items():
                if isinstance(path, str):
                    print(f"  {fmt:<10} {path}")

    # ------------------------------------------------------------------
    # 7. Edge deployment
    # ------------------------------------------------------------------
    if deploy_fqbn:
        _step(verbose, 7, "Edge deployment")
        ino = results["exports"].get("arduino")
        if not ino:
            ino = f"{base}.ino"
            analyzer.classifier.export_to_arduino_ino(
                filepath=ino, board_model=board_model, serial_baud=serial_baud)
            results["exports"]["arduino"] = ino
        results["deployment"] = _deploy_arduino(
            ino, deploy_fqbn, deploy_port or "", cli=arduino_cli,
            upload=upload, verbose=verbose)

    if save_model:
        path = f"{base}.pkl"
        analyzer.save(path)
        results["exports"]["pkl"] = path

    results["elapsed"] = time.time() - t0
    if verbose:
        verdict = "VERIFIED" if results["passed"] else "VIOLATION FOUND"
        print(f"\nPipeline finished in {results['elapsed']:.1f}s -- {verdict}")
    return results
