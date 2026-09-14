"""
Hardware-in-the-Loop (HIL) test runner — pyruleanalyzer on real Arduino/ESP32.

This script takes a model trained with `full_pipeline`, generates a test sketch,
compiles it, flashes it to the board and checks — against real hardware — whether
the embedded inference agrees with the Python inference, while also measuring
latency and memory usage.

===============================================================================
 TUTORIAL — HOW THE TEST PIPELINE WORKS
===============================================================================

OVERVIEW
--------
The sketch produced by `full_pipeline(generate_arduino_sketch=True)` is a
*production* sketch: it reads sensors in `read_features()` and prints the class.
That is great for deployment but impossible to test automatically — the data
comes from the physical world.

To test on a real board we invert the flow: the PC becomes the data source. The
test (HIL) sketch reuses EXACTLY the same tree arrays and the same
`pyra_predict()` function as the production sketch, but replaces
`read_features()` with a serial protocol. The PC sends a feature vector, the
board answers with the class and the inference time. What gets validated is
therefore the very code that ships to the field.

    +-------------------+  P,<id>,<reps>,f0,f1,...   +--------------------+
    |  PC (this script) | -------------------------> |  Arduino / ESP32   |
    |  sklearn + rules  | <------------------------- |  pyra_predict()    |
    +-------------------+  R,<id>,<class>,<us>       +--------------------+
              |                                                |
              +---------- class-by-class comparison -----------+

THE 7 STAGES
------------
Stage 0 — PREREQUISITES (once)
    pip install pyserial
    arduino-cli:  https://arduino.github.io/arduino-cli/  (or: winget install ArduinoSA.CLI)
    arduino-cli core update-index
    arduino-cli core install arduino:avr          # Uno / Nano / Mega / Leonardo
    arduino-cli core install esp32:esp32          # ESP32 (add the board manager URL)
    Windows: CH340/CP2102 driver for Nano/ESP32 clones.
    Check the board:  arduino-cli board list

Stage 1 — TRAINING + EXPORT (host, no hardware)
    The script trains the model (or loads a .pkl with --model-pkl), holds out a
    test set that was NOT used for training, and calls `full_pipeline` with
    `generate_arduino_sketch=True`. That produces the production .ino plus the
    memory_check (Flash/SRAM estimate).

Stage 2 — HIL SKETCH GENERATION (host, no hardware)
    The script cuts the "MODEL DATA + PREDICTION ENGINE" section out of the
    production .ino (everything before "ARDUINO SKETCH SECTION"), patches two
    defects of older generators (the "Feature order:" lines emitted without
    "//"; a float feature buffer passed through a cast to const double*, which
    reads garbage where float != double -- both fixed in the generator since
    2026-09; the patch is a no-op on current output) and appends the serial
    harness on top. Result:
    files/hil/<name>_hil/<name>_hil.ino.
    Arduino rule: the folder must have the same name as the .ino.

Stage 3 — HOST-CHECK WITH g++ (host, no hardware)  [--host-check]
    Before spending time on flashing, the same C code is compiled on the PC with
    g++ and fed the test vectors. If C vs Python parity already fails here, the
    problem is in the model/exporter, not on the board. This stage is the one you
    run in CI — no hardware required at all.

Stage 4 — COMPILING FOR THE BOARD (arduino-cli)
    arduino-cli compile --fqbn <fqbn> <sketch_dir>
    The script reads "Sketch uses X bytes / Global variables use Y bytes" from the
    output and compares it against the memory_check estimate. The real number
    being larger than the estimate is expected (the Arduino runtime is included);
    what matters is the final percentage.

Stage 5 — UPLOAD
    arduino-cli upload -p <port> --fqbn <fqbn> <sketch_dir>
    Close the IDE Serial Monitor first: the port is exclusive.

Stage 6 — THE HIL RUN ITSELF
    Opens the serial port (opening it resets AVR boards — the script waits for the
    boot), handshakes with the "I" command, then sends N vectors from the test
    set. For each vector it collects: the board class, the Python class, the true
    label and the inference time (averaged over --reps repetitions, so that the
    resolution of micros() does not dominate the measurement).

Stage 7 — REPORT AND PASS CRITERIA
    Writes a .json and a .txt into --report-dir with parity, on-board accuracy,
    host accuracy, latency (mean/p50/p95/max) and memory. The process exits with
    code 0 (pass) or 1 (fail), ready to use as a CI gate:
      - parity >= --min-parity  (default 100%)
      - no invalid responses / timeouts
      - Flash and SRAM within the limits of the board

USAGE EXAMPLES
--------------
    # only generate the HIL sketch and validate it on the PC — no board needed
    python examples/arduino_hardware_test.py --dataset iris --host-check --no-hardware

    # full cycle on an Uno at COM5
    python examples/arduino_hardware_test.py --dataset iris --board uno --port COM5

    # ESP32, Random Forest, 200 samples, 50 repetitions per sample
    python examples/arduino_hardware_test.py --dataset wine --model "Random Forest" \
        --n-estimators 10 --max-depth 6 --board esp32 --port COM7 \
        --n-samples 200 --reps 50

    # your own CSV, skipping the upload (board already flashed)
    python examples/arduino_hardware_test.py --csv files/dataset_iris.csv \
        --target Target --board nano --port COM3 --skip-upload

SERIAL PROTOCOL (text, one line per message, terminated by \n)
-------------------------------------------------------------
    PC    -> board : I
    board -> PC    : I,<algorithm>,<n_features>,<n_classes>,<n_trees>
    PC    -> board : P,<id>,<reps>,<f0>,<f1>,...,<fN-1>
    board -> PC    : R,<id>,<class>,<total_microseconds>,<reps>
    board -> PC    : E,<reason>          (parse error / unknown command)

PITFALLS YOU WILL RUN INTO
--------------------------
* double == float on AVR (4 bytes). Thresholds are exported with 17 digits and
  lose precision when parsed by the atof() of the board. Samples sitting almost
  exactly on a threshold may diverge from Python. If parity comes out at 99.x%
  with every divergence on a boundary, that is the reason — not a logic bug. On
  the ESP32 (real 8-byte double) the effect disappears.
* FEATURE ORDER is a contract. The board knows no names, only indices. The vector
  you send must follow the same order used at training time; `read_features()` in
  the production sketch carries the same obligation.
* Uno/Nano have 2 KB of SRAM. Each feature costs 4 bytes in the buffer plus ~14
  bytes of text on the serial line. Past ~40 features the Uno gets tight; use a
  Mega or an ESP32.
* AVR boards reset when the serial port is opened. The script waits --boot-delay
  seconds and discards the banner before the handshake.
* An ESP32 with too many repetitions can trip the watchdog. Keep --reps modest
  (<= 200) or use smaller batches.
* Response timeout: large models (an RF with 40 trees) take milliseconds per
  inference on an Uno; multiply that by --reps before blaming --timeout.

CONTINUOUS INTEGRATION
----------------------
On CI without a board, run Stage 3 (--host-check --no-hardware): it guarantees the
exporter keeps producing valid C that is equivalent to Python. On a runner with a
physical board attached (self-hosted), run the full command with a fixed --port;
the exit code already works as the gate.

Requirements: sklearn, pandas, numpy (from the project) + pyserial (hardware) +
arduino-cli (compile/upload) + g++ (optional, for the host-check).
"""
import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Allows running straight from examples/ without installing the package
PROJECT_ROOT = str(Path(__file__).parent.parent.resolve())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==============================================================================
# 1. DATASETS
# ==============================================================================

BOARD_FQBN = {
    'uno': 'arduino:avr:uno',
    'nano': 'arduino:avr:nano:cpu=atmega328',
    'nano_old': 'arduino:avr:nano:cpu=atmega328old',
    'mega': 'arduino:avr:mega',
    'leonardo': 'arduino:avr:leonardo',
    'esp32': 'esp32:esp32:esp32',
}


def load_dataset(name, csv_path, target, out_dir):
    """Return (csv_path, target, feature_names). sklearn datasets are dumped to CSV."""
    import pandas as pd

    if csv_path:
        df = pd.read_csv(csv_path)
        if target not in df.columns:
            raise SystemExit(f'[!] Target column "{target}" not found in {csv_path}')
        return csv_path, target, [c for c in df.columns if c != target]

    from sklearn import datasets as skds
    loaders = {
        'iris': skds.load_iris,
        'wine': skds.load_wine,
        'breast_cancer': skds.load_breast_cancer,
        'digits': skds.load_digits,
    }
    data = loaders[name]()
    cols = [re.sub(r'[^0-9a-zA-Z_]', '_', c) for c in data.feature_names]
    df = pd.DataFrame(data.data, columns=cols)
    df['Target'] = data.target

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'dataset_{name}.csv')
    df.to_csv(path, index=False)
    print(f'  dataset "{name}": {len(df)} samples, {len(cols)} features, '
          f'{df["Target"].nunique()} classes -> {path}')
    return path, 'Target', cols


def split_holdout(csv_path, target, test_size, seed, out_dir):
    """Split train/test and write both CSVs (the test one feeds the HIL vectors)."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)
    strat = df[target] if df[target].value_counts().min() >= 2 else None
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=strat
    )
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, 'hil_train.csv')
    test_path = os.path.join(out_dir, 'hil_test.csv')
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, test_path


# ==============================================================================
# 2. HIL SKETCH GENERATION
# ==============================================================================

# Marker separating "model data + inference engine" from the sketch part in the
# file produced by export_to_arduino_ino().
SKETCH_SECTION_MARKER = 'ARDUINO SKETCH SECTION'

HIL_HARNESS = r'''
// ============================================================
// HIL TEST HARNESS (generated by examples/arduino_hardware_test.py)
// Replaces the production setup()/loop() with a serial protocol:
//   I                        -> I,<algo>,<n_feat>,<n_classes>,<n_trees>
//   P,<id>,<reps>,f0,...,fN  -> R,<id>,<class>,<total_us>,<reps>
//   anything else            -> E,<reason>
// The pyra_predict() above is exactly the one from the production sketch.
// ============================================================

#define SERIAL_BAUD %(baud)d
#define PYRA_RX_BUF (N_FEATURES * 14 + 32)

static char   pyra_rx[PYRA_RX_BUF];
static double pyra_feat[N_FEATURES];

// Read one serial line (up to '\n'), with timeout. Returns its length or -1.
static int pyra_read_line(void) {
    int n = 0;
    unsigned long t0 = millis();
    while (millis() - t0 < 3000UL) {
        if (!Serial.available()) continue;
        char c = (char)Serial.read();
        t0 = millis();
        if (c == '\r') continue;
        if (c == '\n') { pyra_rx[n] = '\0'; return n; }
        if (n < (int)sizeof(pyra_rx) - 1) pyra_rx[n++] = c;
    }
    pyra_rx[n] = '\0';
    return n > 0 ? n : -1;
}

static void pyra_send_info(void) {
    Serial.print(F("I,"));
    Serial.print(F(PYRA_ALGORITHM));
    Serial.print(',');  Serial.print(N_FEATURES);
    Serial.print(',');  Serial.print(N_CLASSES);
    Serial.print(',');  Serial.println(N_TREES);
}

static void pyra_handle(char *s) {
    if (s[0] == 'I') { pyra_send_info(); return; }
    if (s[0] != 'P') { Serial.println(F("E,unknown_command")); return; }

    char *tok = strtok(s, ",");            // "P"
    tok = strtok(NULL, ",");
    if (!tok) { Serial.println(F("E,missing_id")); return; }
    long id = atol(tok);

    tok = strtok(NULL, ",");
    if (!tok) { Serial.println(F("E,missing_reps")); return; }
    long reps = atol(tok);
    if (reps < 1) reps = 1;

    int k = 0;
    while ((tok = strtok(NULL, ",")) != NULL) {
        if (k < N_FEATURES) pyra_feat[k] = atof(tok);
        k++;
    }
    if (k != N_FEATURES) {
        Serial.print(F("E,bad_feature_count,")); Serial.println(k);
        return;
    }

    // volatile keeps the compiler from optimizing the repetition loop away
    volatile int32_t sink = 0;
    unsigned long t0 = micros();
    for (long r = 0; r < reps; r++) sink = pyra_predict(pyra_feat);
    unsigned long dt = micros() - t0;

    Serial.print(F("R,"));   Serial.print(id);
    Serial.print(',');       Serial.print((long)sink);
    Serial.print(',');       Serial.print(dt);
    Serial.print(',');       Serial.println(reps);
}

void setup(void) {
    Serial.begin(SERIAL_BAUD);
    while (!Serial) delay(10);     // boards with native USB (Leonardo/Micro)
    delay(50);
    Serial.println(F("PYRA-HIL-READY"));
}

void loop(void) {
    if (!Serial.available()) return;
    int n = pyra_read_line();
    if (n <= 0) return;
    pyra_handle(pyra_rx);
}
'''


def extract_model_section(ino_text):
    """Cut the reusable part of the .ino (data + engine) and patch its defects.

    Two defects of older generators are fixed here (current output is
    already correct, and the patch leaves it unchanged):
      1. the "Feature order:" block is emitted as "  [0] name", without "//",
         which does not compile;
      2. the production sketch declares `float features[]` and calls
         `pyra_predict((const double*)features)` — an invalid cast that reads
         garbage. The harness declares the buffer as double and uses no cast.
    """
    lines = ino_text.splitlines()

    cut = None
    for i, line in enumerate(lines):
        if SKETCH_SECTION_MARKER in line:
            cut = i
            break
    if cut is None:
        raise SystemExit(f'[!] Marker "{SKETCH_SECTION_MARKER}" not found in the generated .ino.')

    # step back over the "// ====" separator line that opens the block
    while cut > 0 and lines[cut - 1].lstrip().startswith('// ==='):
        cut -= 1

    body = lines[:cut]

    # comment out the "  [i] name" lines of the feature-order block
    fixed = []
    for line in body:
        if re.match(r'^\s*\[\d+\]\s', line):
            line = '// ' + line.strip()
        fixed.append(line)

    # sanity check: the whole engine must be inside the slice
    joined = '\n'.join(fixed)
    for required in ('pyra_traverse_tree', 'pyra_predict', '#define N_FEATURES'):
        if required not in joined:
            raise SystemExit(f'[!] The .ino slice does not contain "{required}" — '
                             'did the generator format change?')

    return joined


def build_hil_sketch(ino_path, sketch_root, sketch_name, baud, algorithm):
    """Assemble <sketch_root>/<sketch_name>/<sketch_name>.ino from the generated .ino."""
    with open(ino_path, 'r', encoding='utf-8') as f:
        model_section = extract_model_section(f.read())

    sketch_dir = os.path.join(sketch_root, sketch_name)
    os.makedirs(sketch_dir, exist_ok=True)
    sketch_path = os.path.join(sketch_dir, f'{sketch_name}.ino')

    header = (
        '// ============================================================\n'
        f'// HIL test sketch — generated at {datetime.now().isoformat(timespec="seconds")}\n'
        f'// Source: {os.path.basename(ino_path)}\n'
        '// DO NOT edit by hand: regenerate with examples/arduino_hardware_test.py\n'
        '// ============================================================\n'
        '#include <Arduino.h>\n'
        '#include <stdint.h>\n'
        '#include <stdlib.h>\n'
        '#include <string.h>\n'
        f'#define PYRA_ALGORITHM "{algorithm}"\n\n'
    )
    harness = HIL_HARNESS % {'baud': baud}

    with open(sketch_path, 'w', encoding='utf-8') as f:
        f.write(header + model_section + '\n' + harness)

    print(f'  HIL sketch: {sketch_path} ({os.path.getsize(sketch_path):,} bytes)')
    return sketch_path, model_section


# ==============================================================================
# 3. HOST-CHECK — compile the same C on the PC with g++ (Stage 3, no hardware)
# ==============================================================================

HOST_MAIN = r'''
// test main(): reads "f0,f1,...\n" from stdin and prints one class per line.
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(void) {
    static char line[1 << 16];
    double feat[N_FEATURES];
    while (fgets(line, sizeof(line), stdin)) {
        int k = 0;
        char *tok = strtok(line, ",\n");
        while (tok && k < N_FEATURES) { feat[k++] = atof(tok); tok = strtok(NULL, ",\n"); }
        if (k != N_FEATURES) { fprintf(stderr, "bad line\n"); return 2; }
        printf("%d\n", (int)pyra_predict(feat));
    }
    return 0;
}
'''


def host_check(model_section, X, work_dir):
    """Compile the exported engine with g++ and return its native predictions."""
    gpp = shutil.which('g++') or shutil.which('clang++')
    if not gpp:
        print('  [!] g++ not found in PATH — host-check skipped.')
        return None

    os.makedirs(work_dir, exist_ok=True)
    src = os.path.join(work_dir, 'host_check.cpp')
    exe = os.path.join(work_dir, 'host_check.exe' if os.name == 'nt' else 'host_check')

    with open(src, 'w', encoding='utf-8') as f:
        f.write('#include <cstdint>\n#include <cmath>\n')
        f.write(model_section)
        f.write('\n')
        f.write(HOST_MAIN)

    # -w: infinite thresholds are emitted as 1e500 and raise overflow warnings
    cp = subprocess.run([gpp, '-O2', '-w', '-o', exe, src],
                        capture_output=True, text=True)
    if cp.returncode != 0:
        print('  [!] g++ failed to compile the exported engine:')
        print(cp.stderr[-2000:])
        return False

    payload = '\n'.join(','.join(repr(float(v)) for v in row) for row in X) + '\n'
    run = subprocess.run([exe], input=payload, capture_output=True, text=True)
    if run.returncode != 0:
        print(f'  [!] host_check exited with {run.returncode}: {run.stderr[-500:]}')
        return False

    return [int(v) for v in run.stdout.split()]


# ==============================================================================
# 4. ARDUINO-CLI — compile and upload (Stages 4 and 5)
# ==============================================================================

def arduino_cli(args_list, cli):
    return subprocess.run([cli] + args_list, capture_output=True, text=True)


def detect_port(cli, fqbn):
    """Find the first port holding a recognized board."""
    cp = arduino_cli(['board', 'list', '--format', 'json'], cli)
    if cp.returncode != 0:
        return None
    try:
        data = json.loads(cp.stdout)
    except json.JSONDecodeError:
        return None
    entries = data.get('detected_ports', data) if isinstance(data, dict) else data
    for entry in entries or []:
        port = (entry.get('port') or {}).get('address') or entry.get('address')
        boards = entry.get('matching_boards') or entry.get('boards') or []
        if port and boards:
            print(f'  board detected: {boards[0].get("name", "?")} on {port}')
            return port
    return None


def compile_sketch(cli, fqbn, sketch_dir):
    """Compile and extract the REAL Flash/SRAM usage reported by the toolchain."""
    print(f'  arduino-cli compile --fqbn {fqbn} {sketch_dir}')
    cp = arduino_cli(['compile', '--fqbn', fqbn, sketch_dir], cli)
    out = (cp.stdout or '') + (cp.stderr or '')
    if cp.returncode != 0:
        print(out[-4000:])
        return None
    usage = {}
    m = re.search(r'Sketch uses (\d+) bytes.*?(\d+)% of.*?(\d+) bytes', out, re.S)
    if m:
        usage['flash_bytes'] = int(m.group(1))
        usage['flash_percent'] = float(m.group(2))
        usage['flash_max'] = int(m.group(3))
    m = re.search(r'Global variables use (\d+) bytes.*?(\d+)% of.*?(\d+) bytes', out, re.S)
    if m:
        usage['ram_bytes'] = int(m.group(1))
        usage['ram_percent'] = float(m.group(2))
        usage['ram_max'] = int(m.group(3))
    if usage:
        print(f'  measured memory: Flash {usage.get("flash_bytes", 0):,} B '
              f'({usage.get("flash_percent", 0)}%) | '
              f'SRAM {usage.get("ram_bytes", 0):,} B ({usage.get("ram_percent", 0)}%)')
    else:
        print('  compiled (no memory report in the expected format)')
    return usage


def upload_sketch(cli, fqbn, port, sketch_dir):
    print(f'  arduino-cli upload -p {port} --fqbn {fqbn} {sketch_dir}')
    cp = arduino_cli(['upload', '-p', port, '--fqbn', fqbn, sketch_dir], cli)
    if cp.returncode != 0:
        print(((cp.stdout or '') + (cp.stderr or ''))[-3000:])
        return False
    print('  upload finished')
    return True


# ==============================================================================
# 5. HIL — serial conversation with the board (Stage 6)
# ==============================================================================

def run_hil(port, baud, X, reps, timeout, boot_delay):
    """Send every row of X and collect (class, microseconds). Returns a dict."""
    try:
        import serial  # pyserial
    except ImportError:
        raise SystemExit('[!] pyserial is not installed. Run: pip install pyserial')

    print(f'  opening {port} @ {baud} (board reset: waiting {boot_delay}s)')
    with serial.Serial(port, baud, timeout=timeout) as ser:
        time.sleep(boot_delay)
        ser.reset_input_buffer()

        # handshake
        ser.write(b'I\n')
        info = None
        deadline = time.time() + timeout + 2
        while time.time() < deadline:
            line = ser.readline().decode('utf-8', 'replace').strip()
            if not line:
                continue
            if line.startswith('I,'):
                info = line
                break
        if not info:
            raise SystemExit('[!] The board did not answer the "I" handshake. '
                             'Check port, baud rate and whether the HIL sketch was flashed.')
        print(f'  handshake: {info}')

        preds, times, errors = [], [], []
        t_start = time.time()
        for i, row in enumerate(X):
            payload = 'P,%d,%d,%s\n' % (i, reps, ','.join(repr(float(v)) for v in row))
            ser.write(payload.encode('ascii'))
            ser.flush()

            got = None
            deadline = time.time() + timeout
            while time.time() < deadline:
                line = ser.readline().decode('utf-8', 'replace').strip()
                if not line:
                    continue
                if line.startswith('E,'):
                    errors.append((i, line))
                    break
                if line.startswith('R,'):
                    got = line
                    break
            if got is None:
                errors.append((i, 'timeout'))
                preds.append(None)
                times.append(None)
                continue

            parts = got.split(',')
            try:
                rid, cls, us, r = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            except (IndexError, ValueError):
                errors.append((i, f'invalid response: {got}'))
                preds.append(None)
                times.append(None)
                continue
            if rid != i:
                errors.append((i, f'out-of-order id: expected {i}, got {rid}'))
            preds.append(cls)
            times.append(us / max(r, 1))

            if (i + 1) % 25 == 0 or i + 1 == len(X):
                print(f'    {i + 1}/{len(X)} samples', end='\r')

        print()
        return {
            'predictions': preds,
            'latency_us': times,
            'errors': errors,
            'info': info,
            'wall_seconds': time.time() - t_start,
        }


# ==============================================================================
# 6. REPORT (Stage 7)
# ==============================================================================

def summarize(y_true, y_host, y_board, latency, memory_est, memory_real, extra):
    valid = [i for i, p in enumerate(y_board or []) if p is not None]
    parity = None
    mismatches = []
    if y_board:
        agree = sum(1 for i in valid if int(y_board[i]) == int(y_host[i]))
        parity = 100.0 * agree / len(valid) if valid else 0.0
        mismatches = [
            {'index': i, 'host': int(y_host[i]), 'board': int(y_board[i]),
             'true': int(y_true[i])}
            for i in valid if int(y_board[i]) != int(y_host[i])
        ][:20]

    lat = [t for t in (latency or []) if t is not None]
    lat_stats = {}
    if lat:
        ordered = sorted(lat)
        lat_stats = {
            'mean_us': statistics.mean(lat),
            'p50_us': ordered[len(ordered) // 2],
            'p95_us': ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
            'max_us': ordered[-1],
        }

    def accuracy(pred, idxs):
        if not pred or not idxs:
            return None
        return 100.0 * sum(1 for i in idxs if int(pred[i]) == int(y_true[i])) / len(idxs)

    host_idxs = valid if y_board else list(range(len(y_true)))

    return {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'n_samples': len(y_true),
        'n_answered': len(valid) if y_board else 0,
        'parity_percent': parity,
        'mismatches': mismatches,
        'accuracy_host_percent': accuracy(y_host, host_idxs),
        'accuracy_board_percent': accuracy(y_board, valid) if y_board else None,
        'latency': lat_stats,
        'memory_estimated': memory_est,
        'memory_measured': memory_real,
        **extra,
    }


def write_report(report, report_dir, name):
    os.makedirs(report_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = os.path.join(report_dir, f'{name}_hil_{stamp}.json')
    txt_path = os.path.join(report_dir, f'{name}_hil_{stamp}.txt')

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)

    lines = [
        '=' * 62,
        'HIL REPORT — pyruleanalyzer on hardware',
        '=' * 62,
        f'Date            : {report["timestamp"]}',
        f'Model           : {report.get("model_type")} ({report.get("algorithm")})',
        f'Board / FQBN    : {report.get("board")} / {report.get("fqbn")}',
        f'Port            : {report.get("port")}',
        f'Samples         : {report["n_samples"]} (answered: {report["n_answered"]})',
        '',
        f'C/Python parity   : {_fmt(report["parity_percent"])}%',
        f'Accuracy (host)   : {_fmt(report["accuracy_host_percent"])}%',
        f'Accuracy (board)  : {_fmt(report["accuracy_board_percent"])}%',
        f'Host-check (g++)  : {report.get("host_check", "not run")}',
    ]
    if report['latency']:
        l = report['latency']
        lines += ['',
                  f'Latency per inference: mean {l["mean_us"]:.1f} us | '
                  f'p50 {l["p50_us"]:.1f} | p95 {l["p95_us"]:.1f} | max {l["max_us"]:.1f}']
    if report.get('memory_measured'):
        m = report['memory_measured']
        lines += ['',
                  f'Measured memory: Flash {m.get("flash_bytes", 0):,} B '
                  f'({m.get("flash_percent", 0)}%) | SRAM {m.get("ram_bytes", 0):,} B '
                  f'({m.get("ram_percent", 0)}%)']
    if report.get('memory_estimated'):
        e = report['memory_estimated']
        lines.append(f'Memory estimated by pyruleanalyzer: '
                     f'Flash {e.get("flash_bytes", 0):,.0f} B '
                     f'| SRAM {e.get("ram_bytes", 0):,.0f} B')
    if report['mismatches']:
        lines += ['', 'Mismatches (up to 20):']
        for m in report['mismatches']:
            lines.append(f'  sample {m["index"]:>4}: host={m["host"]} board={m["board"]} '
                         f'true={m["true"]}')
    if report.get('errors'):
        lines += ['', f'Communication errors: {len(report["errors"])}']
        for idx, msg in report['errors'][:10]:
            lines.append(f'  sample {idx}: {msg}')
    lines += ['', f'RESULT: {"PASS" if report["passed"] else "FAIL"}', '=' * 62]

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print('\n'.join(lines))
    print(f'\n  reports: {json_path}\n           {txt_path}')


def _fmt(v):
    return 'n/a' if v is None else f'{v:.2f}'


# ==============================================================================
# 7. MAIN
# ==============================================================================

def main():
    p = argparse.ArgumentParser(
        description='Hardware-in-the-loop test for pyruleanalyzer models on Arduino/ESP32.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Read the docstring at the top of this file: it describes the whole pipeline.')

    g = p.add_argument_group('data and model')
    g.add_argument('--dataset', default='iris',
                   choices=['iris', 'wine', 'breast_cancer', 'digits'],
                   help='built-in dataset (ignored when --csv is given)')
    g.add_argument('--csv', default=None, help='your own CSV')
    g.add_argument('--target', default='Target', help='target column of --csv')
    g.add_argument('--model', default='Decision Tree',
                   choices=['Decision Tree', 'Random Forest',
                            'Gradient Boosting Decision Trees'])
    g.add_argument('--n-estimators', type=int, default=None)
    g.add_argument('--max-depth', type=int, default=6)
    g.add_argument('--test-size', type=float, default=0.3)
    g.add_argument('--seed', type=int, default=42)
    g.add_argument('--model-pkl', default=None, help='reuse an already trained model')

    g = p.add_argument_group('hardware')
    g.add_argument('--board', default='uno', choices=sorted(BOARD_FQBN),
                   help='target board (sets the FQBN and the memory limits)')
    g.add_argument('--fqbn', default=None, help='explicit FQBN, overrides --board')
    g.add_argument('--port', default=None,
                   help='serial port (COM5, /dev/ttyUSB0); auto-detected when omitted')
    g.add_argument('--baud', type=int, default=115200)
    g.add_argument('--arduino-cli', default='arduino-cli', help='path to arduino-cli')
    g.add_argument('--boot-delay', type=float, default=2.5,
                   help='wait after opening the serial port (AVR reset)')
    g.add_argument('--timeout', type=float, default=5.0, help='per-response timeout (s)')

    g = p.add_argument_group('test')
    g.add_argument('--n-samples', type=int, default=50,
                   help='how many test-set samples to send to the board')
    g.add_argument('--reps', type=int, default=20,
                   help='repetitions per sample when measuring latency')
    g.add_argument('--min-parity', type=float, default=100.0,
                   help='minimum C/Python parity for the test to pass (%%)')
    g.add_argument('--host-check', action='store_true',
                   help='Stage 3: compile the engine with g++ and validate it on the PC')
    g.add_argument('--no-hardware', action='store_true',
                   help='stop after Stage 3 (no compiling, no flashing)')
    g.add_argument('--skip-upload', action='store_true',
                   help='skip flashing (the board already runs the HIL sketch)')
    g.add_argument('--skip-compile', action='store_true',
                   help='skip compile+upload and go straight to the serial port')

    g = p.add_argument_group('output')
    g.add_argument('--output-dir', default='files/hil')
    g.add_argument('--report-dir', default='files/hil/reports')
    g.add_argument('--name', default='hil_model')

    args = p.parse_args()
    fqbn = args.fqbn or BOARD_FQBN[args.board]

    print('=' * 62)
    print('HARDWARE-IN-THE-LOOP TEST — pyruleanalyzer')
    print('=' * 62)

    # ---------- Stage 1: data + training + export ----------
    print('\n[1/7] Data and training')
    csv_path, target, _ = load_dataset(args.dataset, args.csv, args.target, args.output_dir)
    train_csv, test_csv = split_holdout(csv_path, target, args.test_size,
                                        args.seed, args.output_dir)

    from pyruleanalyzer import full_pipeline
    params = dict(
        train_csv=train_csv,
        test_csv=test_csv,
        target_feature=target,
        model_type=args.model,
        max_depth=args.max_depth,
        output_dir=args.output_dir,
        output_name=args.name,
        generate_arduino_sketch=True,
        board_model=args.board if args.board in BOARD_FQBN and args.board != 'nano_old'
        else 'auto',
        serial_baud=args.baud,
        memory_report=True,
        random_seed=args.seed,
    )
    if args.n_estimators is not None:
        params['n_estimators'] = args.n_estimators
    if args.model_pkl:
        params['model_pkl'] = args.model_pkl

    results = full_pipeline(**params)
    model = results['model']
    ino_path = results['generated_files'].get('arduino')
    if not ino_path or not os.path.exists(ino_path):
        raise SystemExit('[!] full_pipeline produced no .ino — nothing to test.')
    memory_est = results['generated_files'].get('memory_check') or results.get('memory_check')

    # ---------- Stage 2: HIL sketch ----------
    print('\n[2/7] HIL sketch generation')
    algorithm = getattr(model.classifier, 'algorithm_type', args.model)
    sketch_name = f'{args.name}_hil'
    sketch_path, model_section = build_hil_sketch(
        ino_path, args.output_dir, sketch_name, args.baud, algorithm)
    sketch_dir = os.path.dirname(sketch_path)

    # test vectors + Python reference
    import pandas as pd
    import numpy as np
    test_df = pd.read_csv(test_csv)
    if args.n_samples > 0:
        test_df = test_df.head(args.n_samples)
    y_true = test_df[target].astype(int).tolist()
    X = test_df.drop(columns=[target]).values.astype(float)
    y_host = [int(v) for v in np.asarray(model.predict(X)).ravel()]
    print(f'  {len(X)} test samples, {X.shape[1]} features')

    # ---------- Stage 3: host-check ----------
    print('\n[3/7] Host-check (g++)')
    host_status = 'skipped'
    host_parity = None
    if args.host_check:
        hc = host_check(model_section, X, os.path.join(args.output_dir, 'hostcheck'))
        if hc is None:
            host_status = 'g++ missing'
        elif hc is False:
            host_status = 'FAILED (compilation/execution)'
        else:
            agree = sum(1 for a, b in zip(hc, y_host) if a == b)
            host_parity = 100.0 * agree / len(y_host)
            host_status = f'{host_parity:.2f}% C/Python parity'
            print(f'  native C vs Python parity: {host_parity:.2f}% '
                  f'({agree}/{len(y_host)})')
    else:
        print('  (use --host-check to validate the C code on the PC before flashing)')

    common = {'model_type': args.model, 'algorithm': algorithm, 'board': args.board,
              'fqbn': fqbn, 'host_check': host_status,
              'host_check_parity_percent': host_parity, 'sketch': sketch_path}

    # the host-check only fails the run when it actually ran and came out too low
    host_failed = (host_status.startswith('FAILED')
                   or (host_parity is not None and host_parity < args.min_parity))

    if args.no_hardware:
        report = summarize(y_true, y_host, None, None, memory_est, None,
                           dict(common, port=None, errors=[], passed=not host_failed))
        write_report(report, args.report_dir, args.name)
        return 0 if report['passed'] else 1

    # ---------- Stages 4 and 5: compile + upload ----------
    cli = shutil.which(args.arduino_cli) or args.arduino_cli
    has_cli = shutil.which(args.arduino_cli) is not None
    memory_real = None
    if not args.skip_compile:
        if not has_cli:
            raise SystemExit('[!] arduino-cli not found. Install it or use '
                             '--skip-compile (board already flashed) / --no-hardware.')
        print('\n[4/7] Compilation')
        memory_real = compile_sketch(cli, fqbn, sketch_dir)
        if memory_real is None:
            raise SystemExit('[!] Compilation failed — see the arduino-cli output above.')

        port = args.port or detect_port(cli, fqbn)
        if not port:
            raise SystemExit('[!] No board detected. Pass --port explicitly.')

        print('\n[5/7] Upload')
        if args.skip_upload:
            print('  (--skip-upload)')
        elif not upload_sketch(cli, fqbn, port, sketch_dir):
            raise SystemExit('[!] Upload failed.')
    else:
        print('\n[4/7] Compilation — skipped (--skip-compile)')
        print('[5/7] Upload — skipped')
        port = args.port or (detect_port(cli, fqbn) if has_cli else None)
        if not port:
            raise SystemExit('[!] Please pass --port.')

    # ---------- Stage 6: HIL ----------
    print('\n[6/7] Test on the board')
    hil = run_hil(port, args.baud, X, args.reps, args.timeout, args.boot_delay)

    # ---------- Stage 7: report ----------
    print('\n[7/7] Report')
    report = summarize(
        y_true, y_host, hil['predictions'], hil['latency_us'], memory_est, memory_real,
        dict(common, port=port, errors=hil['errors'], board_info=hil['info'],
             wall_seconds=hil['wall_seconds'], reps_per_sample=args.reps, passed=False))

    over_memory = bool(memory_real and (memory_real.get('flash_percent', 0) > 100
                                        or memory_real.get('ram_percent', 0) > 100))
    report['passed'] = (
        not hil['errors']
        and report['parity_percent'] is not None
        and report['parity_percent'] >= args.min_parity
        and not host_failed
        and not over_memory
    )
    write_report(report, args.report_dir, args.name)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
