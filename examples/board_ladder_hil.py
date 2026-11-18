"""
Board ladder, hardware in the loop: one model sized for each board, run on all.

For every board of the ladder the script sizes a model with `fit_to_board`
(the board preset, firmware included), refines it, exports the Arduino sketch
for that board, wraps it in the serial test harness of
`arduino_hardware_test.py` and compiles it. With the boards connected (one
USB hub is enough), it uploads each sketch, sends the same test samples to
every board and compares each answer with the prediction of the refined rules
on the PC, measuring the inference time on the board.

Two phases, so everything that needs no hardware is done before the boards
arrive:

    # 1. no hardware: train, size, refine, export, build and compile
    python examples/board_ladder_hil.py prepare --split-dir data/my_ids --target label
    python examples/board_ladder_hil.py prepare --dataset wine      # sklearn data

    # 2. list the serial ports of the connected boards
    python examples/board_ladder_hil.py ports

    # 3. upload and test the boards that are connected (any subset)
    python examples/board_ladder_hil.py run --port mega2560=COM5 --port esp32=COM7 \
        --port nrf52840=COM9 --port rp2040_16mb=COM11

Output, in --out (default files/board_ladder): one folder per board with the
production sketch and the HIL sketch, manifest.json (models, memory, PC
predictions), and after `run` report.json and report.md (parity, accuracy and
latency per board).

FIRST UPLOAD, BOARD BY BOARD
----------------------------
* Mega 2560 (CH340 clone): Windows may need the WCH CH340 driver.
* ESP32 DevKit: if the upload stalls at "Connecting...", hold BOOT until it
  starts writing.
* nRF52840 (nice!nano clone): double-tap RESET; the board shows up as a USB
  drive and a new COM port. Needs the Adafruit nRF52 core:
      arduino-cli core install adafruit:nrf52 --additional-urls \
          https://adafruit.github.io/arduino-board-index/package_adafruit_index.json
* RP2040 16 MB: it ships with MicroPython, which does not answer the reset
  the uploader sends. The first time, hold BOOT while plugging it in (it
  shows up as RPI-RP2) and pass that drive's port, or run the upload once
  from the Arduino IDE; later uploads are automatic.

The boards answer in float32 on AVR (double is 4 bytes there) and in float64
elsewhere. Splits are exact on every board (the export writes, on AVR, the
largest float32 not above each threshold); a boosting model sums its scores
in float32 on AVR, so on AVR a sample whose two best scores are within float32
rounding may come out different. Such samples are reported, not hidden.
"""
import argparse
import contextlib
import importlib.util
import io
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'examples'))

from pyruleanalyzer import PyRuleAnalyzer  # noqa: E402
from pyruleanalyzer.footprint import BOARDS, fit_to_board  # noqa: E402

# The HIL harness, compile and upload helpers of the single-board script.
_spec = importlib.util.spec_from_file_location('hil', ROOT / 'examples' / 'arduino_hardware_test.py')
hil = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hil)

# name -> (FQBN, footprint preset). The default ladder: 256 KB, 1, 4, 16 MB.
LADDER = {
    'mega2560': ('arduino:avr:mega:cpu=atmega2560', 'mega2560'),
    'nrf52840': ('adafruit:nrf52:feather52840', 'nrf52840'),
    'esp32': ('esp32:esp32:esp32:PartitionScheme=huge_app', 'esp32_huge_app'),
    'rp2040_16mb': ('rp2040:rp2040:vccgnd_yd_rp2040:flash=16777216_0', 'rp2040_16mb'),
    # also available
    'uno': ('arduino:avr:uno', 'uno'),
    'esp32s3_16mb': ('esp32:esp32:esp32s3:FlashSize=16M,PartitionScheme=custom,PSRAM=opi',
                     'esp32s3_16mb_custom'),
}
DEFAULT_BOARDS = ['mega2560', 'nrf52840', 'esp32', 'rp2040_16mb']
GBDT_TRAIN_ROWS = 50_000  # scikit-learn's GBDT is single-threaded


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def load_splits(folder, target, drop=()):
    """A dataset already split into train/val/test files of `folder`.

    Reads train.csv and test.csv (or .csv.gz) and, when present, val.csv,
    which becomes the refinement set; without it a stratified fifth of the
    training data is held out for refinement. Labels are coded over the union
    of the splits (sorted), text columns as the sorted codes of their training
    values (-1 for a value not seen in training), and missing or infinite
    values become 0.

    Returns:
        (X_tr, y_tr, X_ref, y_ref, X_te, y_te, feature names).
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    splits = {}
    for split in ('train', 'val', 'test'):
        for ext in ('.csv', '.csv.gz'):
            path = Path(folder) / f'{split}{ext}'
            if path.exists():
                splits[split] = pd.read_csv(path, low_memory=False).drop(
                    columns=[c for c in drop], errors='ignore')
                break
    if 'train' not in splits or 'test' not in splits:
        raise FileNotFoundError(f'train.csv and test.csv not found in {folder}')
    labels = sorted(pd.concat([df[target] for df in splits.values()]).astype(str).unique())
    code = {lab: i for i, lab in enumerate(labels)}
    feats = [c for c in splits['train'].columns if c != target]
    cats = {}
    for col in feats:
        if splits['train'][col].dtype == object and                 pd.to_numeric(splits['train'][col], errors='coerce').notna().mean() <= 0.99:
            cats[col] = {v: i for i, v in enumerate(sorted(splits['train'][col].astype(str).unique()))}
    out = {}
    for split, df in splits.items():
        X = pd.DataFrame({c: (df[c].astype(str).map(cats[c]).fillna(-1) if c in cats
                              else pd.to_numeric(df[c], errors='coerce')) for c in feats})
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)
        out[split] = (X, df[target].astype(str).map(code).to_numpy(dtype=int))
    X_tr, y_tr = out['train']
    if 'val' in out:
        X_ref, y_ref = out['val']
    else:
        X_tr, X_ref, y_tr, y_ref = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42,
                                                    stratify=y_tr)
    return (X_tr, y_tr, X_ref, y_ref) + out['test'] + (feats,)


def load_data(args):
    """(X_tr, y_tr, X_ref, y_ref, X_te, y_te, feature names, dataset name)."""
    if args.split_dir is not None:
        return load_splits(args.split_dir, args.target, args.drop) + (Path(args.split_dir).name,)
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        y = df.pop(args.target)
        X, feats, name = df.values.astype(float), list(df.columns), Path(args.csv).stem
        y = np.unique(y, return_inverse=True)[1]
    else:
        d = getattr(datasets, f'load_{args.dataset}')()
        X, y, name = d.data, d.target, args.dataset
        feats = [f'f{i}' for i in range(X.shape[1])]
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, test_size=0.4, random_state=args.seed,
                                                  stratify=y)
    X_ref, X_te, y_ref, y_te = train_test_split(X_rest, y_rest, test_size=0.5,
                                                random_state=args.seed, stratify=y_rest)
    return X_tr, y_tr, X_ref, y_ref, X_te, y_te, feats, name


# ----------------------------------------------------------------------------
# prepare: size, refine, export, compile (no hardware)
# ----------------------------------------------------------------------------

def compile_usage(cli, fqbn, sketch_dir):
    """Compile and return the memory the toolchain reports (None on failure)."""
    cp = subprocess.run([cli, 'compile', '--fqbn', fqbn, sketch_dir], capture_output=True, text=True)
    out = (cp.stdout or '') + (cp.stderr or '')
    if cp.returncode != 0:
        return {'ok': False, 'log': out[-2000:]}
    import re
    usage = {'ok': True}
    for key, pat in (('flash', r'Sketch uses (\d+) bytes.*?Maximum is (\d+)'),
                     ('ram', r'Global variables use (\d+) bytes.*?Maximum is (\d+)')):
        m = re.search(pat, out, re.S)
        if m:
            usage[f'{key}_bytes'], usage[f'{key}_max'] = int(m.group(1)), int(m.group(2))
    return usage


def firmware(cli, fqbn, work):
    """Flash and RAM of a sketch with Serial and no model, for this core."""
    sk = work / 'firmware'
    sk.mkdir(parents=True, exist_ok=True)
    (sk / 'firmware.ino').write_text(
        '#if defined(USE_TINYUSB)\n#include <Adafruit_TinyUSB.h>\n#endif\n'
        'void setup() { Serial.begin(115200); }\n'
        'void loop() { if (Serial.available()) { Serial.println(Serial.read()); } }\n')
    return compile_usage(cli, fqbn, str(sk))


def prepare(args):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, f1_score
    cli = shutil.which(args.arduino_cli) or args.arduino_cli
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    X_tr, y_tr, X_ref, y_ref, X_te, y_te, feats, name = load_data(args)
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(X_te), min(args.n_samples, len(X_te)), replace=False)
    # The trees compare float32 inputs: send each value as the shortest decimal
    # of its float32 (a few characters, so 100+ features fit the serial buffer
    # of an AVR), and predict on the PC from exactly the same values.
    X_hil = np.array([[float(str(v)) for v in row] for row in X_te[pick].astype(np.float32)])
    y_hil = y_te[pick]
    np.savez(out / 'samples.npz', X=X_hil, y=y_hil)
    print(f'{name}: train {len(X_tr)}, refinement {len(X_ref)}, test {len(X_te)}, '
          f'{len(feats)} features; {len(X_hil)} samples for the boards')
    manifest = {'dataset': name, 'features': feats, 'n_samples': int(len(X_hil)),
                'refine_threshold': args.refine_threshold, 'boards': {}}
    for board in args.boards:
        fqbn, preset = LADDER[board]
        work = out / board
        t0 = time.time()
        fw = firmware(cli, fqbn, work)
        if not fw.get('ok'):
            print(f'[{board}] the core does not compile a plain sketch:\n{fw["log"][-600:]}')
            continue
        target = replace(BOARDS[preset], firmware_flash=fw['flash_bytes'], firmware_ram=fw['ram_bytes'])
        model = row = None
        for algo in ([args.algorithm] if args.algorithm != 'auto' else ['gbdt', 'dt']):
            try:
                fit = quiet(fit_to_board, X_tr, y_tr, board=target, algorithm=algo,
                            n_estimators=tuple(args.n_estimators), sample_rows=20_000,
                            val_rows=10_000, scoring='f1_macro', random_state=args.seed)
            except ValueError:
                continue
            params = dict(fit.params)
            Xf, yf = X_tr, y_tr
            if algo == 'gbdt':
                cls = GradientBoostingClassifier
                if len(X_tr) > GBDT_TRAIN_ROWS:
                    idx = rng.choice(len(X_tr), GBDT_TRAIN_ROWS, replace=False)
                    Xf, yf = X_tr[idx], y_tr[idx]
            elif algo == 'rf':
                cls = RandomForestClassifier
                params['n_jobs'] = 1
            else:
                cls = DecisionTreeClassifier
                params.pop('n_estimators', None)
            model = cls(random_state=args.seed, **params).fit(Xf, yf)
            row = {'algorithm': algo, 'params': {k: v for k, v in params.items() if k != 'n_jobs'},
                   'train_rows': int(len(Xf))}
            break
        if model is None:
            print(f'[{board}] no model fits')
            continue
        an = quiet(PyRuleAnalyzer.from_sklearn, model, feats, [str(c) for c in model.classes_])
        clf = an.classifier
        rules_before = len(clf.initial_rules)
        y_sk = model.predict(X_te)
        if args.refine_threshold >= 0:
            quiet(an.execute_rule_refinement, X=X_ref, y=y_ref.astype(str),
                  remove_below_n_classifications=args.refine_threshold,
                  refine_between_trees=(row['algorithm'] == 'gbdt'))
        y_ref_pred = np.asarray(quiet(an.predict, X_te, use_refined=True)).astype(int)
        row.update(rules_before=rules_before, rules_after=len(clf.final_rules),
                   acc_original=float(accuracy_score(y_te, y_sk)),
                   acc_refined=float(accuracy_score(y_te, y_ref_pred)),
                   f1_refined=float(f1_score(y_te, y_ref_pred, average='macro')),
                   fidelity=float((y_ref_pred == y_sk).mean()),
                   pc_predictions=[int(v) for v in np.asarray(
                       quiet(an.predict, X_hil, use_refined=True)).astype(int)],
                   firmware=fw, fqbn=fqbn, preset=preset)
        # production sketch, then the HIL sketch around its model section
        prod = work / 'production' / 'production.ino'
        prod.parent.mkdir(parents=True, exist_ok=True)
        quiet(clf.compile_tree_arrays, rules=clf.final_rules, feature_names=feats)
        quiet(clf.export_to_arduino_ino, filepath=str(prod), board_model=preset)
        hil_name = f'hil_{board}'
        quiet(hil.build_hil_sketch, str(prod), str(work), hil_name, args.baud, row['algorithm'])
        part = prod.parent / 'partitions.csv'
        if part.exists():  # custom partition table travels with the sketch
            shutil.copy(part, work / hil_name / 'partitions.csv')
        row['sketch'] = str(work / hil_name)
        row['compile'] = compile_usage(cli, fqbn, row['sketch'])
        row['seconds'] = round(time.time() - t0, 1)
        c = row['compile']
        mem = (f"flash {c['flash_bytes']:,}/{c['flash_max']:,} B, RAM {c['ram_bytes']:,} B"
               if c.get('ok') else 'DOES NOT COMPILE')
        print(f"[{board}] {row['algorithm']} {row['params']}: rules {rules_before} -> "
              f"{row['rules_after']}, accuracy {row['acc_refined']:.4f}, fidelity "
              f"{row['fidelity']:.5f}; {mem} ({row['seconds']} s)")
        manifest['boards'][board] = row
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(f'manifest: {out / "manifest.json"}')


# ----------------------------------------------------------------------------
# run: upload, send the samples, compare (hardware)
# ----------------------------------------------------------------------------

def run(args):
    cli = shutil.which(args.arduino_cli) or args.arduino_cli
    out = Path(args.out)
    manifest = json.loads((out / 'manifest.json').read_text())
    data = np.load(out / 'samples.npz')
    X, y = data['X'], data['y']
    ports = dict(p.split('=', 1) for p in args.port)
    report = {'dataset': manifest['dataset'], 'n_samples': int(len(X)), 'reps': args.reps,
              'boards': {}}
    for board, port in ports.items():
        row = manifest['boards'].get(board)
        if row is None:
            print(f'[{board}] not in the manifest; run prepare with --boards {board}')
            continue
        print(f'[{board}] {port}')
        if not args.skip_upload and not hil.upload_sketch(cli, row['fqbn'], port, row['sketch']):
            report['boards'][board] = {'error': 'upload failed'}
            continue
        res = hil.run_hil(port, args.baud, X, args.reps, args.timeout,
                          args.boot_delay if 'avr' in row['fqbn'] else 1.0)
        pc = np.asarray(row['pc_predictions'])
        got = res['predictions']
        ok = [i for i, v in enumerate(got) if v is not None]
        same = [i for i in ok if got[i] == pc[i]]
        lat = [t for t in res['latency_us'] if t is not None]
        entry = {
            'algorithm': row['algorithm'], 'params': row['params'],
            'rules_after': row['rules_after'], 'flash_bytes': row['compile'].get('flash_bytes'),
            'ram_bytes': row['compile'].get('ram_bytes'),
            'answered': len(ok), 'parity': len(same) / len(ok) if ok else None,
            'board_accuracy': float(np.mean([got[i] == y[i] for i in ok])) if ok else None,
            'pc_accuracy': float(np.mean(pc == y)),
            'latency_us': {'mean': statistics.mean(lat), 'p50': statistics.median(lat),
                           'p95': float(np.percentile(lat, 95)), 'max': max(lat)} if lat else None,
            'mismatches': [{'i': i, 'board': got[i], 'pc': int(pc[i]), 'true': int(y[i])}
                           for i in ok if got[i] != pc[i]][:50],
            'errors': res['errors'][:20], 'handshake': res['info'],
        }
        report['boards'][board] = entry
        lat_s = f"{entry['latency_us']['mean']:.1f} us" if lat else '-'
        print(f'  parity {entry["parity"]:.4%}, board accuracy {entry["board_accuracy"]:.4f}, '
              f'mean latency {lat_s}, errors {len(res["errors"])}')
    (out / 'report.json').write_text(json.dumps(report, indent=1, default=str))
    lines = ['| Board | Model | Rules | Flash (B) | RAM (B) | Parity with PC | Accuracy | '
             'Latency mean / p95 (us) |', '|---|---|---|---|---|---|---|---|']
    for b, e in report['boards'].items():
        if 'error' in e:
            lines.append(f'| {b} | {e["error"]} |||||||')
            continue
        lat = e['latency_us'] or {}
        lines.append(f"| {b} | {e['algorithm']} {e['params']} | {e['rules_after']} | "
                     f"{e['flash_bytes']:,} | {e['ram_bytes']:,} | {e['parity']:.2%} | "
                     f"{e['board_accuracy']:.4f} | {lat.get('mean', 0):.1f} / {lat.get('p95', 0):.1f} |")
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    failed = any('error' in e or (e['parity'] or 0) < args.min_parity / 100 or e['errors']
                 for e in report['boards'].values())
    return 1 if failed else 0


def ports(args):
    cli = shutil.which(args.arduino_cli) or args.arduino_cli
    cp = subprocess.run([cli, 'board', 'list'], capture_output=True, text=True)
    print(cp.stdout or cp.stderr)
    print('Clones with a CH340/CH9102 chip show as "Unknown": unplug one board at a time '
          'to see which port is which.')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--out', default=str(ROOT / 'files' / 'board_ladder'))
    common.add_argument('--arduino-cli', default='arduino-cli')
    common.add_argument('--baud', type=int, default=115200)

    pp = sub.add_parser('prepare', parents=[common], help='size, refine, export and compile')
    src = pp.add_mutually_exclusive_group()
    src.add_argument('--split-dir', default=None,
                     help='a folder with train.csv, test.csv and optionally val.csv (or .csv.gz)')
    src.add_argument('--dataset', default='wine', help='a scikit-learn dataset (load_<name>)')
    src.add_argument('--csv', default=None)
    pp.add_argument('--target', default='Target', help='label column of --csv or --split-dir')
    pp.add_argument('--drop', nargs='*', default=[], help='columns of --split-dir to ignore')
    pp.add_argument('--boards', nargs='+', default=DEFAULT_BOARDS, choices=sorted(LADDER))
    pp.add_argument('--algorithm', default='auto', choices=['auto', 'gbdt', 'rf', 'dt'],
                    help='auto: a GBDT, or a decision tree where no GBDT fits')
    pp.add_argument('--n-estimators', type=int, nargs='+', default=[5, 10, 25, 50])
    pp.add_argument('--refine-threshold', type=int, default=1,
                    help='remove rules used at most this many times (-1: merge only)')
    pp.add_argument('--n-samples', type=int, default=300, help='test samples sent to the boards')
    pp.add_argument('--seed', type=int, default=42)

    pr = sub.add_parser('run', parents=[common], help='upload and test the connected boards')
    pr.add_argument('--port', action='append', default=[], metavar='BOARD=PORT', required=True)
    pr.add_argument('--reps', type=int, default=20, help='inferences per sample, for the timing')
    pr.add_argument('--timeout', type=float, default=10.0)
    pr.add_argument('--boot-delay', type=float, default=2.5, help='AVR boards reset on open')
    pr.add_argument('--skip-upload', action='store_true')
    pr.add_argument('--min-parity', type=float, default=100.0)

    sub.add_parser('ports', parents=[common], help='list the serial ports')
    args = p.parse_args()
    if args.cmd == 'prepare':
        if args.split_dir is not None:
            args.dataset = None
        return prepare(args)
    if args.cmd == 'run':
        return run(args)
    return ports(args)


if __name__ == '__main__':
    sys.exit(main() or 0)
