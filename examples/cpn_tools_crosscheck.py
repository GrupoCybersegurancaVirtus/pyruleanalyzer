"""Cross-validation of the model checker against CPN Tools.

Two claims have to hold before the model checker's answers can be reported in
a paper instead of CPN Tools' own:

1. **Agreement.** On every net, the occurrence graph pyRuleAnalyzer builds has
   the size CPN Tools reports (nodes, arcs, SCC nodes, SCC arcs, dead markings,
   home markings), and every CTL property gets the same verdict from both --
   CPN Tools answering through the ASK-CTL formulas generated for that net.
2. **Sensitivity.** Every property is falsifiable: for each one there is a
   mutant net that violates it, and the violation is reported. Agreement on
   correct nets alone would also be reached by two checkers that always answer
   "true".

The benchmark is every model family (Decision Tree, Random Forest, binary and
3-class GBDT), initial and refined, plus the 3-class, 8-stage GBDT of the
article (19,685 markings), plus every mutant of every refined net (see
:mod:`pyruleanalyzer.cpn_mutants`). CPN Tools is run headlessly through
:mod:`pyruleanalyzer.cpntools_oracle`: the CPN Tools 4.0.1 simulator compiles
each net, the state-space tool is entered as the GUI does it, and ASK-CTL
evaluates the queries.

Run:
    python examples/cpn_tools_crosscheck.py                 # everything
    python examples/cpn_tools_crosscheck.py --no-mutants    # base nets only
    python examples/cpn_tools_crosscheck.py --python-only   # skip CPN Tools

Output (``files/crosscheck/``): ``results.json`` with every value from both
engines, ``report.md`` with the agreement and sensitivity tables, and the nets
and mutants themselves, so any row can be reopened in CPN Tools.
"""

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyruleanalyzer import PyRuleAnalyzer
from pyruleanalyzer.cpn_mutants import make_mutants
from pyruleanalyzer.cpntools_oracle import CPNToolsOracle
from pyruleanalyzer.model_checker import PROPERTY_CATALOG, CPNModelChecker

MODELS = [
    ("dt", "Decision Tree", {"max_depth": 4, "random_state": 0}, 2),
    ("rf", "Random Forest", {"n_estimators": 4, "max_depth": 3, "random_state": 0}, 2),
    ("gbdt2", "Gradient Boosting Decision Trees",
     {"n_estimators": 4, "max_depth": 2, "learning_rate": 0.2, "random_state": 0}, 2),
    ("gbdt3", "Gradient Boosting Decision Trees",
     {"n_estimators": 3, "max_depth": 2, "learning_rate": 0.2, "random_state": 0}, 3),
]


# Function to parse the command-line options.
def parse_args():
    """Read the options.

    Returns:
        argparse.Namespace: The parsed options.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="files/crosscheck", help="Output directory.")
    p.add_argument("--no-mutants", action="store_true", help="Base nets only.")
    p.add_argument("--python-only", action="store_true",
                   help="Do not run CPN Tools (sensitivity table only).")
    p.add_argument("--article-net", default="files/hcpn_mc3_final.cpn",
                   help="Also cross-check this net (the article's case study).")
    return p.parse_args()


# Function to build a small dataset for one model family.
def dataset(n_classes: int, seed: int):
    """A reproducible dataset with a known decision boundary.

    Args:
        n_classes (int): Number of classes.
        seed (int): Random seed.

    Returns:
        tuple: ``(X, y)`` as pandas objects.
    """
    rng = np.random.RandomState(seed)
    X = rng.rand(300, 4)
    if n_classes == 2:
        y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    else:
        y = (X[:, 0] * n_classes).astype(int).clip(0, n_classes - 1)
    return pd.DataFrame(X, columns=[f"feat_{i}" for i in range(4)]), pd.Series(y)


# Function to train, refine and export the benchmark nets.
def build_benchmark(out_dir: str, with_mutants: bool):
    """Create every base net and, optionally, every mutant.

    Args:
        out_dir (str): Output directory.
        with_mutants (bool): Also derive the mutants of the refined nets.

    Returns:
        list: One job per net: path, family, stage, mutant info, labels,
        analyzer.
    """
    jobs = []
    for key, model, params, k in MODELS:
        X, y = dataset(k, zlib.crc32(key.encode()) % 50)
        analyzer = PyRuleAnalyzer.new_model(model=model, params=params)
        analyzer.fit(X[:220], y[:220])
        analyzer.execute_rule_refinement(X=X[220:], y=y[220:],
                                         remove_below_n_classifications=1,
                                         save_final_model=False, save_report=False)
        paths = analyzer.export_hcpn(base_name=os.path.join(out_dir, key),
                                     which="both", sample=X.iloc[230])
        labels = [int(c) for c in analyzer.class_names]
        for stage in ("initial", "final"):
            jobs.append({"net": paths[stage], "model": key, "family": model,
                         "stage": stage, "mutant": None, "targets": [],
                         "labels": labels, "analyzer": analyzer})
        if with_mutants:
            for m in make_mutants(paths["final"], os.path.join(out_dir, "mutants")):
                jobs.append({"net": m["path"], "model": key, "family": model,
                             "stage": "final", "mutant": m["name"],
                             "targets": list(m["targets"]),
                             "labels": labels, "analyzer": analyzer})
    return jobs


# Function to run both engines on one net.
def run_job(job, oracle):
    """Verify a net with pyRuleAnalyzer and, if available, with CPN Tools.

    Args:
        job (dict): The net and its context.
        oracle (CPNToolsOracle|None): The CPN Tools oracle.

    Returns:
        dict: Values from both engines and the per-property verdicts.
    """
    analyzer = job.get("analyzer")
    checker = CPNModelChecker(job["net"], class_labels=job["labels"],
                              feature_names=analyzer.feature_names if analyzer
                              else None)
    t0 = time.time()
    result = checker.check(classifier=analyzer.classifier if analyzer else None,
                           use_final=(job["stage"] == "final"))
    pyra_s = time.time() - t0
    decl, queries = checker.ml_program()
    record = {
        "net": os.path.relpath(job["net"]), "model": job["model"],
        "family": job["family"], "stage": job["stage"], "mutant": job["mutant"],
        "targets": job["targets"], "pyra": result.oracle,
        "verdicts": {pid: r.status for pid, r in result.properties.items()},
        "pyra_seconds": pyra_s, "cpntools": None, "cpntools_errors": None,
    }
    if oracle is not None:
        answer = oracle.run(job["net"], decl, queries)
        record["cpntools"] = answer["values"]
        record["cpntools_errors"] = answer.get("errors") or None
        record["cpntools_seconds"] = answer.get("elapsed")
        record["banner"] = (answer.get("banner") or "").splitlines()[:2]
    return record


# Function to write the Markdown report of a run.
def write_report(records, path: str):
    """Summarise agreement and sensitivity.

    Args:
        records (list): One record per net.
        path (str): Destination ``.md`` file.

    Returns:
        dict: Headline numbers.
    """
    lines = ["# Cross-validation against CPN Tools", ""]
    compared = agreed = 0
    mismatches = []
    for r in records:
        if not r["cpntools"]:
            continue
        for key, mine in r["pyra"].items():
            theirs = r["cpntools"].get(key)
            compared += 1
            if theirs == mine:
                agreed += 1
            else:
                mismatches.append((r["net"], key, mine, theirs))
    banner = next((r.get("banner") for r in records if r.get("banner")), None)
    lines.append(f"Engine: {' / '.join(banner) if banner else 'CPN Tools not run'}")
    lines.append("")
    lines.append(f"- nets: {len(records)} "
                 f"({sum(1 for r in records if not r['mutant'])} correct, "
                 f"{sum(1 for r in records if r['mutant'])} mutants)")
    lines.append(f"- values compared with CPN Tools: {compared}, identical: {agreed}")
    lines.append("")
    lines.append("## Agreement per net")
    lines.append("")
    lines.append("| net | mutant | nodes | arcs | dead | home | values | agree |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for r in records:
        c = r["cpntools"] or {}
        keys = list(r["pyra"])
        same = sum(1 for k in keys if c.get(k) == r["pyra"][k])
        lines.append(f"| {os.path.basename(r['net'])} | {r['mutant'] or '-'} | "
                     f"{r['pyra'].get('nodes')} | {r['pyra'].get('arcs')} | "
                     f"{r['pyra'].get('dead')} | {r['pyra'].get('home')} | "
                     f"{len(keys)} | "
                     f"{'n/a' if not c else ('yes' if same == len(keys) else f'{same}/{len(keys)}')} |")
    lines.append("")
    lines.append("## Sensitivity: every property fails on its mutant")
    lines.append("")
    lines.append("| model | mutant | must fail | pyRuleAnalyzer | CPN Tools |")
    lines.append("|---|---|---|---|---|")
    detected = total = 0
    for r in records:
        if not r["mutant"]:
            continue
        for target in r["targets"]:
            total += 1
            mine = r["verdicts"].get(target)
            ok = mine == "FAIL"
            detected += ok
            if r["cpntools"] and target in r["cpntools"]:
                theirs = "FAIL" if r["cpntools"][target] == "false" else "PASS"
            else:
                theirs = f"n/a ({PROPERTY_CATALOG[target].method})"
            lines.append(f"| {r['model']} | {r['mutant']} | {target} | {mine} | "
                         f"{theirs} |")
    lines.append("")
    lines.append(f"Targets detected by pyRuleAnalyzer: {detected}/{total}")
    lines.append("")
    ev_rows = [r for r in records if r["cpntools"] and r["mutant"] == "deadlock"]
    if ev_rows:
        lines.append("## ASK-CTL's EV at dead markings")
        lines.append("")
        lines.append("On the `deadlock` mutants the net stops without a prediction. "
                     "Plain `EV(PRED)` still holds in CPN Tools, because ASK-CTL's "
                     "`FORALL_UNTIL` is vacuously true at a dead marking; the "
                     "textbook `AF pred` (encoded as `AFmax PRED`) does not.")
        lines.append("")
        lines.append("| net | EV(PRED) CPN Tools | EV(PRED) predicted | AF pred CPN Tools | "
                     "AF pred pyRuleAnalyzer |")
        lines.append("|---|---|---|---|---|")
        for r in ev_rows:
            lines.append(f"| {os.path.basename(r['net'])} | {r['cpntools'].get('A3_ev')} "
                         f"| {r['pyra'].get('A3_ev')} | {r['cpntools'].get('A3')} | "
                         f"{r['pyra'].get('A3')} |")
        lines.append("")
    if mismatches:
        lines.append("## Mismatches")
        lines.append("")
        for net, key, mine, theirs in mismatches:
            lines.append(f"- {net} `{key}`: pyRuleAnalyzer={mine} CPN Tools={theirs}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return {"nets": len(records), "compared": compared, "agreed": agreed,
            "mismatches": len(mismatches), "detected": detected, "targets": total}


# Function to run the cross-validation.
def main():
    """Build the benchmark, run both engines and write the report."""
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    oracle = None
    if not args.python_only:
        oracle = CPNToolsOracle()
        if not oracle.available():
            print("[!] CPN Tools engine unavailable:", "; ".join(oracle.missing()))
            print("    running the Python side only")
            oracle = None
        else:
            print(f"CPN Tools simulator image sha256={oracle.simulator_digest()[:16]}...")

    jobs = build_benchmark(args.out, with_mutants=not args.no_mutants)
    if args.article_net and os.path.exists(args.article_net):
        jobs.append({"net": args.article_net, "model": "article",
                     "family": "Gradient Boosting Decision Trees", "stage": "final",
                     "mutant": None, "targets": [], "labels": [0, 1, 2],
                     "analyzer": None})

    records = []
    for i, job in enumerate(jobs, start=1):
        label = os.path.basename(job["net"])
        print(f"[{i}/{len(jobs)}] {label}", flush=True)
        try:
            rec = run_job(job, oracle)
        except Exception as exc:
            print(f"    error: {exc}")
            rec = {"net": os.path.relpath(job["net"]), "model": job["model"],
                   "family": job["family"], "stage": job["stage"],
                   "mutant": job["mutant"], "targets": job["targets"],
                   "pyra": {}, "verdicts": {}, "cpntools": None,
                   "cpntools_errors": {"exception": str(exc)}}
        if rec.get("cpntools"):
            bad = [k for k, v in rec["pyra"].items() if rec["cpntools"].get(k) != v]
            print(f"    {rec['pyra'].get('nodes')} nodes; "
                  f"{'all ' + str(len(rec['pyra'])) + ' values agree' if not bad else 'MISMATCH ' + str(bad)}"
                  f"  ({rec.get('cpntools_seconds', 0):.0f}s)", flush=True)
        if rec.get("cpntools_errors"):
            print(f"    CPN Tools errors: {list(rec['cpntools_errors'])}")
        records.append(rec)
        with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=1)

    summary = write_report(records, os.path.join(args.out, "report.md"))
    print()
    print(f"nets: {summary['nets']}   values compared: {summary['compared']}   "
          f"identical: {summary['agreed']}   mismatches: {summary['mismatches']}")
    print(f"mutant targets detected: {summary['detected']}/{summary['targets']}")
    print(f"report: {os.path.join(args.out, 'report.md')}")


if __name__ == "__main__":
    main()
