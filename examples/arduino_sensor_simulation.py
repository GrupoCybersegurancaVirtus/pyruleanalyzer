"""
Continuous-signal simulation of a pyruleanalyzer Arduino sketch, on the PC.

The HIL runner (`arduino_hardware_test.py`) checks the exported engine one
vector at a time. This script checks it the way it runs in the field: the
board receives inputs that evolve in time, classifies every
SAMPLE_INTERVAL_MS and prints the class over the serial port.

No board is needed. The production sketch's own setup()/loop() run unchanged on
the PC, compiled with g++ against a small Arduino shim (virtual clock,
analogRead/digitalRead and the serial RX line served from a recording, Serial
output to stdout). Only read_features() is replaced, by the function a real
deployment has to write -- so the generated .ino also builds for a real board
with arduino-cli.

===============================================================================
 WHAT FEEDS THE SKETCH
===============================================================================

Two independent choices: where the values come from (--signal) and how they
reach the board (--input).

--signal synthetic   Sensors drifting between operating regimes, one regime per
                     class (Markov chain, random dwell time). In each regime
                     every feature follows an Ornstein-Uhlenbeck process pulled
                     toward a training sample of that class. The process runs
                     in the same per-feature scale as the ADC (linear or log),
                     so heavy-tailed features move by orders of magnitude, not
                     by a fixed step.
--signal replay      Real rows of the test set, each held for --row-ms. With
                     --time-col the rows play in time order; otherwise they are
                     grouped into class episodes (random dwell time, like an
                     attack that lasts a while). The true label is the row's.

--input adc          Feature i is wired to pin A0+i. Analog features go through
                     an --adc-bits ADC, with a linear or logarithmic transfer
                     (--adc auto picks log for heavy-tailed features); 0/1
                     features become digital pins.
--input serial       A gateway sends the features as text, one line per update:
                     "f0,f1,...,fN-1*CS\\n" (CS = XOR checksum, as in NMEA),
                     values with 9 significant digits (float32 round-trip). The
                     shim models the UART (--baud) and the board's RX buffer
                     (--rx-buffer, 64 bytes on AVR): bytes that arrive while the
                     buffer is full are lost, exactly like on the board, and
                     the checksum makes the sketch drop the damaged line.

===============================================================================
 WHAT IS CHECKED
===============================================================================

For every loop() iteration:
    * the class the sketch printed vs model.predict() on the exact feature
      buffer the sketch classified (parity, must be 100%);
    * adc: the buffer vs the value Python computes from the same ADC counts
      (bit-exact for linear and digital channels, 1e-12 relative for log ones,
      whose exp() comes from the C library);
      serial: the buffer is always one of the lines the gateway sent (never a
      mix of two), and how often it is stale (a newer line had arrived);
    * the class vs the true label: accuracy, settled accuracy, detection
      latency after each change, class flips beyond the label changes.

Then it draws the timeline (signals of the most used features over the true
label, and the true / Python / sketch class over time).

Time is virtual: delay() advances the clock and nothing else does, so the
"(N us)" the sketch prints is 0 here. Inference latency on the target is
measured by the HIL runner on a real board.

USAGE
-----
    # built-in dataset, synthetic sensors on the ADC
    python examples/arduino_sensor_simulation.py

    # real dataset with its own split, rows replayed in class episodes,
    # features sent over serial
    python examples/arduino_sensor_simulation.py \\
        --train-csv examples/data/KDDCup99/train.csv \\
        --test-csv examples/data/KDDCup99/test.csv --target target \\
        --signal replay --input serial --rx-buffer 0

    # rows in capture order, text columns encoded, time column not a feature
    python examples/arduino_sensor_simulation.py \\
        --train-csv ".../Detection_of_DDoS_Attacks_using_Machine_Learning_Algorithms/train.csv" \\
        --test-csv  ".../Detection_of_DDoS_Attacks_using_Machine_Learning_Algorithms/test.csv" \\
        --target PKT_CLASS --signal replay --time-col PKT_SEND_TIME \\
        --drop-cols PKT_SEND_TIME --input serial

    # CSV without a header: auto-detected, columns become col_0..col_N,
    # target defaults to the last column
    python examples/arduino_sensor_simulation.py --csv examples/data/covid_train.csv

Outputs (in --output-dir, default files/sim):
    <name>_sim/<name>_sim.ino   wired sketch (builds with arduino-cli)
    <name>_sim/arduino_shim.h   Arduino API shim for the PC
    <name>_feed.txt             ADC counts or serial lines fed to the sketch
    <name>_timeline.csv         one row per loop(): time, true class, sketch, Python
    <name>_timeline.png         timeline plot
    <name>_report.json          metrics

Exit code 0 when parity and the input check pass, 1 otherwise.

Requirements: the project (sklearn, pandas, numpy, matplotlib) + g++ in PATH.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Allows running straight from examples/ without installing the package
PROJECT_ROOT = str(Path(__file__).parent.parent.resolve())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
EXAMPLES_DIR = str(Path(__file__).parent.resolve())
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

from arduino_hardware_test import load_dataset  # noqa: E402

KIND_LINEAR, KIND_LOG, KIND_DIGITAL = 0, 1, 2


# ==============================================================================
# 1. ARDUINO SHIM (compiled with the sketch on the PC)
# ==============================================================================

ARDUINO_SHIM = r'''
// ============================================================
// Minimal Arduino API for running a sketch on the PC
// (generated by examples/arduino_sensor_simulation.py).
// Time is virtual: only delay()/delayMicroseconds() advance it.
// analogRead()/digitalRead() return the recorded signal at the
// current virtual time (zero-order hold). Serial input replays
// recorded lines byte by byte at the UART rate into an RX buffer
// of fixed size; bytes arriving while it is full are dropped.
// ============================================================
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#define F(s) (s)
#define PROGMEM
#define HIGH 1
#define LOW 0
#define INPUT 0
#define OUTPUT 1
#define A0 14

static uint64_t shim_now_us = 0;

unsigned long millis(void) { return (unsigned long)(shim_now_us / 1000ULL); }
unsigned long micros(void) { return (unsigned long)shim_now_us; }
void delay(unsigned long ms) { shim_now_us += (uint64_t)ms * 1000ULL; }
void delayMicroseconds(unsigned int us) { shim_now_us += us; }
void pinMode(int, int) {}

// ---- analog / digital inputs ----
static std::vector<int> shim_counts;
static int shim_channels = 0;
static long shim_steps = 0;
static long shim_signal_ms = 1;

static int shim_read(int pin) {
    int ch = pin >= A0 ? pin - A0 : pin;
    if (ch < 0 || ch >= shim_channels || shim_steps == 0) return 0;
    long k = (long)(millis() / (unsigned long)shim_signal_ms);
    if (k >= shim_steps) k = shim_steps - 1;
    return shim_counts[(size_t)k * shim_channels + ch];
}
int analogRead(int pin) { return shim_read(pin); }
int digitalRead(int pin) { return shim_read(pin) ? HIGH : LOW; }

// ---- serial input: recorded lines through a UART and an RX buffer ----
struct ShimLine { double start_us; std::string bytes; };
static std::vector<ShimLine> shim_lines;
static double shim_byte_us = 0.0;
static size_t shim_rx_cap = 64;          // 0 = unlimited
static size_t shim_line = 0, shim_off = 0;
static std::string shim_rx;              // bytes in the RX buffer, oldest first
static unsigned long shim_rx_dropped = 0;

static void shim_ingest(void) {
    while (shim_line < shim_lines.size()) {
        const ShimLine &L = shim_lines[shim_line];
        double t = L.start_us + (shim_off + 1) * shim_byte_us;
        if (t > (double)shim_now_us) break;
        if (shim_rx_cap == 0 || shim_rx.size() < shim_rx_cap) shim_rx.push_back(L.bytes[shim_off]);
        else shim_rx_dropped++;
        if (++shim_off >= L.bytes.size()) { shim_line++; shim_off = 0; }
    }
}

struct ShimSerial {
    void begin(unsigned long) {}
    operator bool() const { return true; }
    int available(void) { shim_ingest(); return (int)shim_rx.size(); }
    int read(void) {
        shim_ingest();
        if (shim_rx.empty()) return -1;
        int c = (unsigned char)shim_rx[0];
        shim_rx.erase(0, 1);
        return c;
    }
    void print(const char *s) { fputs(s, stdout); }
    void print(char c) { fputc(c, stdout); }
    void print(int v) { printf("%d", v); }
    void print(unsigned int v) { printf("%u", v); }
    void print(long v) { printf("%ld", v); }
    void print(unsigned long v) { printf("%lu", v); }
    void print(long long v) { printf("%lld", v); }
    void print(unsigned long long v) { printf("%llu", v); }
    void print(double v, int digits = 2) { printf("%.*f", digits, v); }
    template <typename T> void println(T v) { print(v); fputc('\n', stdout); }
    void println(double v, int digits = 2) { print(v, digits); fputc('\n', stdout); }
    void println(void) { fputc('\n', stdout); }
};
static ShimSerial Serial;
'''

SIM_MAIN = r'''
// Simulation driver: loads the recording, runs setup() once and loop() until
// the recording ends. Before each loop() it prints the virtual time, after it
// the feature buffer the sketch classified.
#include "arduino_shim.h"
#include "%(sketch)s"

static int load_feed(const char *path, uint64_t *end_us) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %%s\n", path); return 2; }
    char kind[16];
    if (fscanf(f, "%%15s", kind) != 1) return 2;
    if (strcmp(kind, "ADC") == 0) {
        if (fscanf(f, "%%ld %%d %%ld", &shim_steps, &shim_channels, &shim_signal_ms) != 3) return 2;
        if (shim_channels != N_FEATURES) { fprintf(stderr, "channel count mismatch\n"); return 2; }
        shim_counts.resize((size_t)shim_steps * shim_channels);
        for (size_t i = 0; i < shim_counts.size(); i++)
            if (fscanf(f, "%%d", &shim_counts[i]) != 1) { fprintf(stderr, "short file\n"); return 2; }
        *end_us = (uint64_t)shim_steps * shim_signal_ms * 1000ULL;
    } else if (strcmp(kind, "SERIAL") == 0) {
        long n; unsigned long long dur_us; unsigned long cap;
        if (fscanf(f, "%%ld %%lf %%lu %%llu", &n, &shim_byte_us, &cap, &dur_us) != 4) return 2;
        shim_rx_cap = cap;
        *end_us = dur_us;
        static char buf[1 << 17];
        for (long i = 0; i < n; i++) {
            ShimLine L;
            if (fscanf(f, "%%lf %%131000s", &L.start_us, buf) != 2) { fprintf(stderr, "short file\n"); return 2; }
            L.bytes = std::string(buf) + "\n";
            shim_lines.push_back(L);
        }
    } else {
        fprintf(stderr, "unknown feed kind %%s\n", kind);
        return 2;
    }
    fclose(f);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: sim <feed.txt>\n"); return 2; }
    uint64_t end_us = 0;
    int rc = load_feed(argv[1], &end_us);
    if (rc) return rc;

    setup();
    while (shim_now_us < end_us) {
        printf("#T %%lu\n", millis());
        uint64_t before = shim_now_us;
        loop();
        printf("#X");
        for (int i = 0; i < N_FEATURES; i++) printf("%%c%%.17g", i ? ',' : ' ', features[i]);
        printf("\n");
        if (shim_now_us == before) delay(1);  // loop() without delay(): keep time moving
    }
%(epilogue)s
    return 0;
}
'''

SERIAL_EPILOGUE = (
    '    printf("#S %lu %lu %lu\\n", feed_accepted, feed_rejected, shim_rx_dropped);')


# ==============================================================================
# 2. DATA (any CSV: own split, no header, text columns, time column)
# ==============================================================================

# Function to tell whether a CSV starts with a header row.
def has_header(path: str) -> bool:
    """
    A file has a header unless every field of its first line is a number.

    Args:
        path (str): CSV file.

    Returns:
        bool: True when the first line is a header.
    """
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        first = f.readline().strip()
    for tok in first.split(','):
        try:
            float(tok)
        except ValueError:
            return True
    return False


# Function to read a CSV with or without a header.
def read_table(path: str, header: str):
    """
    Read a CSV; headerless files get columns col_0..col_N.

    Args:
        path (str): CSV file.
        header (str): 'auto', 'yes' or 'no'.

    Returns:
        pd.DataFrame: The table with stripped, unique column names.
    """
    import pandas as pd

    present = has_header(path) if header == 'auto' else header == 'yes'
    df = pd.read_csv(path, header=0 if present else None, low_memory=False)
    if not present:
        df.columns = [f'col_{i}' for i in range(df.shape[1])]
    names, seen = [], {}
    for c in df.columns:
        base = str(c).strip() or 'col'
        k = seen.get(base, 0)
        seen[base] = k + 1
        names.append(base if k == 0 else f'{base}_{k}')
    df.columns = names
    print(f'  {path}: {len(df):,} rows, {df.shape[1]} columns'
          f'{"" if present else " (no header: col_0..col_N)"}')
    return df


# Function to resolve the target column from a name, an index or the default.
def resolve_target(columns: List[str], target: Optional[str]) -> str:
    """
    Pick the target column: the given name, a column index, or the last column.

    Args:
        columns (list): Column names.
        target (str): User choice or None.

    Returns:
        str: Target column name.
    """
    if target is None:
        return columns[-1]
    if target in columns:
        return target
    if target.strip() in columns:
        return target.strip()
    if re.fullmatch(r'-?\d+', target):
        return columns[int(target)]
    raise SystemExit(f'[!] Target column "{target}" not found. Columns: {columns[:20]}...')


# Function to prepare train/test CSVs that full_pipeline can consume.
def prepare_data(args, out_dir: str) -> Dict:
    """
    Load the data, encode text columns and labels, and write clean CSVs.

    Text feature columns get one ordinal code per distinct value (shared by
    train and test); labels are mapped to 0..k-1 over the union of both sets;
    missing and infinite values become 0. The time column, if any, orders the
    replay and is dropped from the features only when listed in --drop-cols.

    Args:
        args (argparse.Namespace): Command-line options.
        out_dir (str): Where the clean CSVs go.

    Returns:
        dict: Paths, target name, feature names, class names, test times.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    if args.train_csv:
        if not args.test_csv:
            raise SystemExit('[!] --train-csv needs --test-csv (or use --csv to split one file).')
        train_df = read_table(args.train_csv, args.header)
        test_df = read_table(args.test_csv, args.header)
        if list(train_df.columns) != list(test_df.columns):
            raise SystemExit('[!] train and test CSVs have different columns.')
        target = resolve_target(list(train_df.columns), args.target)
    else:
        if args.csv:
            path = args.csv
        else:
            path, _, _ = load_dataset(args.dataset, None, 'Target', out_dir)
        df = read_table(path, args.header)
        target = resolve_target(list(df.columns), args.target if args.csv else 'Target')
        counts = df[target].value_counts()
        strat = df[target] if counts.min() >= 2 else None
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=args.seed,
                                             stratify=strat)

    if args.max_train_rows and len(train_df) > args.max_train_rows:
        train_df = train_df.sample(args.max_train_rows, random_state=args.seed)
        print(f'  train subsampled to {args.max_train_rows:,} rows (--max-train-rows)')

    test_time = None
    if args.time_col:
        if args.time_col not in test_df.columns:
            raise SystemExit(f'[!] --time-col "{args.time_col}" not found.')
        col = test_df[args.time_col]
        t = pd.to_numeric(col, errors='coerce')
        if t.isna().mean() > 0.5:
            t = pd.to_datetime(col, errors='coerce').astype('int64').astype(float)
        test_time = t.fillna(t.max()).to_numpy()

    drop = [c.strip() for c in (args.drop_cols.split(',') if args.drop_cols else [])]
    missing = [c for c in drop if c not in train_df.columns]
    if missing:
        raise SystemExit(f'[!] --drop-cols not found: {missing}')
    features = [c for c in train_df.columns if c != target and c not in drop]

    # labels: one code per distinct label over train + test
    y_all = pd.concat([train_df[target], test_df[target]])
    num = pd.to_numeric(y_all, errors='coerce')
    if num.notna().all():
        classes = sorted(num.unique())
        to_code = {v: i for i, v in enumerate(classes)}
        y_train = pd.to_numeric(train_df[target]).map(to_code)
        y_test = pd.to_numeric(test_df[target]).map(to_code)
    else:
        classes = sorted(y_all.astype(str).str.strip().unique())
        to_code = {v: i for i, v in enumerate(classes)}
        y_train = train_df[target].astype(str).str.strip().map(to_code)
        y_test = test_df[target].astype(str).str.strip().map(to_code)

    # features: numbers stay numbers, text columns get ordinal codes
    X_train, X_test, encoded = {}, {}, []
    for c in features:
        a = pd.to_numeric(train_df[c], errors='coerce')
        b = pd.to_numeric(test_df[c], errors='coerce')
        bad = (a.isna() & train_df[c].notna()).mean()
        if bad > 0.01:
            cats = sorted(set(train_df[c].astype(str)) | set(test_df[c].astype(str)))
            code = {v: i for i, v in enumerate(cats)}
            a = train_df[c].astype(str).map(code).astype(float)
            b = test_df[c].astype(str).map(code).astype(float)
            encoded.append(f'{c} ({len(cats)} values)')
        X_train[c], X_test[c] = a, b
    X_train = pd.DataFrame(X_train).replace([np.inf, -np.inf], np.nan)
    X_test = pd.DataFrame(X_test).replace([np.inf, -np.inf], np.nan)
    n_nan = int(X_train.isna().sum().sum() + X_test.isna().sum().sum())
    X_train, X_test = X_train.fillna(0.0), X_test.fillna(0.0)
    if encoded:
        print(f'  text columns encoded: {", ".join(encoded)}')
    if n_nan:
        print(f'  {n_nan:,} missing/infinite values set to 0')
    if drop:
        print(f'  dropped: {", ".join(drop)}')

    train_out = X_train.assign(**{target: y_train.values})
    test_out = X_test.assign(**{target: y_test.values})
    train_csv = os.path.join(out_dir, 'sim_train.csv')
    test_csv = os.path.join(out_dir, 'sim_test.csv')
    train_out.to_csv(train_csv, index=False)
    test_out.to_csv(test_csv, index=False)
    print(f'  {len(features)} features, {len(classes)} classes, '
          f'train {len(train_out):,} / test {len(test_out):,} rows')

    return {'train_csv': train_csv, 'test_csv': test_csv, 'target': target,
            'features': features, 'classes': [str(c) for c in classes],
            'X_train': X_train.to_numpy(float), 'X_test': X_test.to_numpy(float),
            'y_test': y_test.to_numpy(int), 'test_time': test_time}


# ==============================================================================
# 3. INPUT CHANNELS
# ==============================================================================

# Class holding the per-feature scale and ADC transfer function.
class SensorMap:
    """
    Per-feature channel description, shared by the generator and the ADC.

    Linear channels: value = lo + counts * scale over the training range plus a
    margin. Log channels: value = lo + (exp(counts * scale) - 1), lo = training
    minimum, so zero offsets stay exact and every decade gets the same number
    of counts. Digital channels: features that are only 0 or 1.

    Args:
        X_train (np.ndarray): Training features.
        adc_bits (int): ADC resolution.
        mode (str): 'linear', 'log' or 'auto' (log for heavy-tailed features).
        margin (float): Extra range above (and, linear, below), as a fraction.
    """

    # Method to size every channel from the training data.
    def __init__(self, X_train: np.ndarray, adc_bits: int = 10, mode: str = 'auto',
                 margin: float = 0.1):
        """
        Choose the kind, offset and step of every channel.

        Args:
            X_train (np.ndarray): Training features.
            adc_bits (int): ADC resolution in bits.
            mode (str): 'linear', 'log' or 'auto'.
            margin (float): Extra range, as a fraction of the span.
        """
        self.max_counts = (1 << adc_bits) - 1
        lo = X_train.min(axis=0)
        hi = X_train.max(axis=0)
        span = hi - lo
        q1, q3 = np.percentile(X_train, [25, 75], axis=0)
        iqr = q3 - q1
        digital = np.array([set(np.unique(col)) <= {0.0, 1.0} for col in X_train.T])
        if mode == 'log':
            heavy = span > 0
        elif mode == 'auto':
            heavy = span > 100 * np.maximum(iqr, span * 1e-12)
        else:
            heavy = np.zeros(len(lo), dtype=bool)
        self.kind = np.where(digital, KIND_DIGITAL,
                             np.where(heavy & (span > 0), KIND_LOG, KIND_LINEAR))

        safe_span = np.where(span > 0, span, 1.0)
        lin_lo = lo - margin * safe_span
        lin_scale = safe_span * (1 + 2 * margin) / self.max_counts
        log_scale = np.log1p(safe_span * (1 + margin)) / self.max_counts
        self.lo = np.select([self.kind == KIND_DIGITAL, self.kind == KIND_LOG],
                            [0.0, lo], lin_lo)
        self.scale = np.select([self.kind == KIND_DIGITAL, self.kind == KIND_LOG],
                               [1.0, log_scale], lin_scale)

    # Method to map values into the generator's working scale.
    def warp(self, values: np.ndarray) -> np.ndarray:
        """
        Identity for linear/digital channels, log1p(value - lo) for log ones.

        Args:
            values (np.ndarray): Physical values, (n, n_features).

        Returns:
            np.ndarray: Warped values.
        """
        logv = np.log1p(np.maximum(values - self.lo, 0.0))
        return np.where(self.kind == KIND_LOG, logv, values)

    # Method to map warped values back to physical units.
    def unwarp(self, u: np.ndarray) -> np.ndarray:
        """
        Inverse of warp().

        Args:
            u (np.ndarray): Warped values.

        Returns:
            np.ndarray: Physical values.
        """
        return np.where(self.kind == KIND_LOG, self.lo + np.expm1(u), u)

    # Method to quantize values the way the board's ADC would.
    def to_counts(self, values: np.ndarray) -> np.ndarray:
        """
        Physical values -> ADC counts (0/1 for digital pins).

        Args:
            values (np.ndarray): Signal, (n_steps, n_features).

        Returns:
            np.ndarray: Integer counts, same shape.
        """
        lin = (values - self.lo) / self.scale
        logc = np.log1p(np.maximum(values - self.lo, 0.0)) / self.scale
        counts = np.where(self.kind == KIND_LOG, logc, lin)
        counts = np.clip(np.rint(counts), 0, self.max_counts).astype(int)
        return np.where(self.kind == KIND_DIGITAL, (values > 0.5).astype(int), counts)

    # Method to convert ADC counts into what read_features() computes.
    def to_values(self, counts: np.ndarray) -> np.ndarray:
        """
        Same arithmetic as the generated read_features(), in double precision.

        Args:
            counts (np.ndarray): Integer counts.

        Returns:
            np.ndarray: Feature values seen by the sketch.
        """
        c = counts.astype(float)
        lin = self.lo + c * self.scale
        logv = self.lo + (np.exp(c * self.scale) - 1.0)
        return np.select([self.kind == KIND_DIGITAL, self.kind == KIND_LOG], [c, logv], lin)

    # Method to emit the ADC read_features() for this mapping.
    def c_code(self, feature_names: List[str]) -> str:
        """
        Generate the ADC read_features() that replaces the sketch placeholder.

        Args:
            feature_names (list): Feature names in model order.

        Returns:
            str: C source.
        """
        n = len(self.lo)
        kinds = {KIND_LINEAR: 'analog ', KIND_LOG: 'log    ', KIND_DIGITAL: 'digital'}
        lo = ', '.join(repr(float(v)) for v in self.lo)
        scale = ', '.join(repr(float(v)) for v in self.scale)
        kind = ', '.join(str(int(v)) for v in self.kind)
        wiring = '\n'.join(f'//   A0+{i:<3d} {kinds[int(self.kind[i])]}  {name}'
                           for i, name in enumerate(feature_names))
        return (
            '// ============================================================\n'
            '// Sensor wiring (generated by examples/arduino_sensor_simulation.py)\n'
            '// feature i <- pin A0+i\n'
            '//   analog:  value = SENSOR_LO[i] + counts * SENSOR_SCALE[i]\n'
            '//   log:     value = SENSOR_LO[i] + (exp(counts * SENSOR_SCALE[i]) - 1)\n'
            '//   digital: value = digitalRead() (0 or 1)\n'
            + wiring + '\n'
            + (f'// NOTE: {n} channels; an Uno has 6 analog inputs (use a Mega or a mux).\n'
               if n > 6 else '')
            + '// ============================================================\n'
            f'static const double SENSOR_LO[N_FEATURES] = {{{lo}}};\n'
            f'static const double SENSOR_SCALE[N_FEATURES] = {{{scale}}};\n'
            f'static const uint8_t SENSOR_KIND[N_FEATURES] = {{{kind}}};  '
            '// 0 analog, 1 log, 2 digital\n'
            '\n'
            'void read_features(void) {\n'
            '    for (int i = 0; i < N_FEATURES; i++) {\n'
            '        if (SENSOR_KIND[i] == 2)\n'
            '            features[i] = digitalRead(A0 + i) == HIGH ? 1.0 : 0.0;\n'
            '        else if (SENSOR_KIND[i] == 1)\n'
            '            features[i] = SENSOR_LO[i] + (exp(analogRead(A0 + i) * SENSOR_SCALE[i]) - 1.0);\n'
            '        else\n'
            '            features[i] = SENSOR_LO[i] + analogRead(A0 + i) * SENSOR_SCALE[i];\n'
            '    }\n'
            '}\n'
        )


SERIAL_READ_FEATURES = r'''// ============================================================
// Serial feature feed (generated by examples/arduino_sensor_simulation.py)
// One line per update: "f0,f1,...,fN-1*CS\n", CS = XOR of the bytes before
// '*', two hex digits. A line that overflows, fails the checksum or has the
// wrong field count is dropped and the previous features stay. Call
// read_features() often enough that the RX buffer (64 bytes on AVR) does not
// overflow between calls -- or enlarge it.
// ============================================================
#define FEED_LINE_MAX (N_FEATURES * 16 + 8)
static char feed_line[FEED_LINE_MAX];
static int feed_len = 0;
static bool feed_overflow = false;
unsigned long feed_accepted = 0, feed_rejected = 0;

static void feed_parse(void) {
    feed_line[feed_len] = '\0';
    char *star = strrchr(feed_line, '*');
    if (!star || strlen(star + 1) != 2) { feed_rejected++; return; }
    uint8_t sum = 0;
    for (char *p = feed_line; p < star; p++) sum ^= (uint8_t)*p;
    char *end;
    long cs = strtol(star + 1, &end, 16);
    if (*end != '\0' || (uint8_t)cs != sum) { feed_rejected++; return; }
    *star = '\0';
    double tmp[N_FEATURES];
    int k = 0;
    char *tok = strtok(feed_line, ",");
    while (tok && k < N_FEATURES) { tmp[k++] = atof(tok); tok = strtok(NULL, ","); }
    if (k != N_FEATURES || tok) { feed_rejected++; return; }
    for (int i = 0; i < N_FEATURES; i++) features[i] = tmp[i];
    feed_accepted++;
}

void read_features(void) {
    while (Serial.available() > 0) {
        char c = (char)Serial.read();
        if (c == '\r') continue;
        if (c == '\n') {
            if (feed_overflow) feed_rejected++;
            else feed_parse();
            feed_len = 0;
            feed_overflow = false;
        } else if (feed_len < FEED_LINE_MAX - 1) {
            feed_line[feed_len++] = c;
        } else {
            feed_overflow = true;
        }
    }
}
'''


# Function to turn the production .ino into the wired sketch.
def wire_sketch(ino_text: str, read_features_code: str, sample_ms: int) -> str:
    """
    Replace the placeholder read_features() and the sampling period.

    Args:
        ino_text (str): Production sketch from export_to_arduino_ino().
        read_features_code (str): New read_features() (ADC or serial).
        sample_ms (int): New SAMPLE_INTERVAL_MS.

    Returns:
        str: Sketch source.
    """
    pattern = re.compile(r'void read_features\(void\) \{.*?\n\}\n', re.S)
    if not pattern.search(ino_text):
        raise SystemExit('[!] read_features() placeholder not found -- generate the sketch '
                         'with include_sensor_placeholders=True.')
    text = pattern.sub(lambda _: read_features_code, ino_text, count=1)
    text, n = re.subn(r'#define SAMPLE_INTERVAL_MS \d+',
                      f'#define SAMPLE_INTERVAL_MS {sample_ms}', text)
    if n != 1:
        raise SystemExit('[!] SAMPLE_INTERVAL_MS not found in the sketch.')
    if '#include <math.h>' not in text:
        text = '#include <math.h>\n#include <stdlib.h>\n#include <string.h>\n' + text
    return text


# ==============================================================================
# 4. SIGNALS
# ==============================================================================

# Function to draw the length of one regime/episode.
def draw_dwell(rng: np.random.Generator, dwell_s: float, min_dwell_s: float) -> float:
    """
    Minimum dwell plus an exponential tail with the requested mean.

    Args:
        rng (np.random.Generator): Random generator.
        dwell_s (float): Mean dwell, seconds.
        min_dwell_s (float): Minimum dwell, seconds.

    Returns:
        float: Dwell time in seconds.
    """
    return min_dwell_s + rng.exponential(max(dwell_s - min_dwell_s, 1e-9))


# Function to generate regime-switching continuous sensor signals.
def synthetic_signal(model, X_train: np.ndarray, sensors: SensorMap, args,
                     rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate sensors that drift between operating regimes, one per class.

    Each regime picks a training sample that the model assigns to that class
    (an anchor); every feature follows an Ornstein-Uhlenbeck process pulled
    toward it, in the channel's warped scale (log for log channels).

    Args:
        model: Trained PyRuleAnalyzer model (used to choose anchors).
        X_train (np.ndarray): Training features.
        sensors (SensorMap): Channel scales.
        args (argparse.Namespace): Signal options.
        rng (np.random.Generator): Random generator.

    Returns:
        tuple: signal (n_steps, n_features) in physical units, label per step.
    """
    if len(X_train) > 50000:
        X_train = X_train[rng.choice(len(X_train), 50000, replace=False)]
    pred = np.asarray(model.predict(X_train)).astype(int).ravel()
    anchors = {c: sensors.warp(X_train[pred == c]) for c in np.unique(pred)}
    classes = sorted(anchors)
    std = sensors.warp(X_train).std(axis=0)
    std = np.where(std > 0, std, 1.0)

    n_steps = int(round(args.duration * 1000 / args.signal_ms))
    dt = args.signal_ms / 1000.0
    a = np.exp(-dt / args.tau)
    drive = args.wander * std * np.sqrt(1 - a * a)

    u_all = np.empty((n_steps, X_train.shape[1]))
    label = np.empty(n_steps, dtype=int)
    cls = int(rng.choice(classes))
    target = anchors[cls][rng.integers(len(anchors[cls]))]
    u = target.copy()
    next_change = draw_dwell(rng, args.dwell, args.min_dwell)
    for k in range(n_steps):
        t = k * dt
        if t >= next_change and len(classes) > 1:
            cls = int(rng.choice([c for c in classes if c != cls]))
            target = anchors[cls][rng.integers(len(anchors[cls]))]
            next_change = t + draw_dwell(rng, args.dwell, args.min_dwell)
        u = target + a * (u - target) + drive * rng.standard_normal(u.shape)
        u_all[k] = u + args.noise * std * rng.standard_normal(u.shape)
        label[k] = cls
    return sensors.unwarp(u_all), label


# Function to replay real test rows as a piecewise-constant signal.
def replay_signal(X_test: np.ndarray, y_test: np.ndarray, test_time: Optional[np.ndarray],
                  sensors: SensorMap, args, rng: np.random.Generator):
    """
    Hold real test rows for --row-ms each, in time order or in class episodes.

    Args:
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels (codes).
        test_time (np.ndarray): Time column of the test rows, or None.
        sensors (SensorMap): Channel scales (for the optional noise).
        args (argparse.Namespace): Signal options.
        rng (np.random.Generator): Random generator.

    Returns:
        tuple: signal (n_steps, n_features), label per step.
    """
    n_rows = int(args.duration * 1000 // args.row_ms)
    if test_time is not None:
        order = np.argsort(test_time, kind='stable')[:n_rows]
        print(f'  replay: {len(order):,} rows in --time-col order')
    else:
        pools = {c: rng.permutation(np.where(y_test == c)[0]) for c in np.unique(y_test)}
        used = {c: 0 for c in pools}
        classes = sorted(pools)
        order, cls = [], None
        while len(order) < n_rows:
            choices = [c for c in classes if c != cls] or classes
            cls = int(rng.choice(choices))
            n = max(1, int(round(draw_dwell(rng, args.dwell, args.min_dwell)
                                 * 1000 / args.row_ms)))
            for _ in range(min(n, n_rows - len(order))):
                pool = pools[cls]
                order.append(pool[used[cls] % len(pool)])
                used[cls] += 1
        order = np.array(order)
        print(f'  replay: {len(order):,} rows in class episodes '
              f'(mean {args.dwell:.0f} s, min {args.min_dwell:.0f} s)')

    per_row = args.row_ms // args.signal_ms
    rows = np.repeat(order, per_row)
    signal = X_test[rows].astype(float)
    if args.noise > 0:
        std = sensors.warp(X_test).std(axis=0)
        std = np.where(std > 0, std, 1.0)
        u = sensors.warp(signal) + args.noise * std * rng.standard_normal(signal.shape)
        signal = sensors.unwarp(u)
    return signal, y_test[rows]


# Function to schedule the gateway's serial lines over the UART.
def serial_feed(signal: np.ndarray, signal_ms: int, send_ms: Optional[int], baud: int):
    """
    Turn the signal into checksummed text lines with UART timing.

    A line is sent at every send_ms tick (every signal step when None) whose
    row differs from the last line sent. Lines queue on the UART: each one
    starts when it is sent or when the previous one has left, whichever is
    later, and takes 10 bits per byte.

    Args:
        signal (np.ndarray): (n_steps, n_features).
        signal_ms (int): Signal step, ms.
        send_ms (int): Gateway period, ms, or None for "on every change".
        baud (int): UART speed.

    Returns:
        dict: lines, values sent, start/end times (us), byte time, load.
    """
    byte_us = 10e6 / baud
    every = 1 if not send_ms else max(1, send_ms // signal_ms)
    lines, values, send_us, last = [], [], [], None
    for k in range(0, len(signal), every):
        row = signal[k].astype(np.float32)
        if last is not None and np.array_equal(row, last):
            continue
        last = row
        payload = ','.join('%.9g' % float(v) for v in row)
        cs = 0
        for ch in payload.encode('ascii'):
            cs ^= ch
        lines.append(f'{payload}*{cs:02X}')
        values.append([float(v) for v in payload.split(',')])
        send_us.append(k * signal_ms * 1000.0)

    start, end, free = [], [], 0.0
    for s, line in zip(send_us, lines):
        st = max(s, free)
        free = st + (len(line) + 1) * byte_us
        start.append(st)
        end.append(free)
    duration_us = len(signal) * signal_ms * 1000.0
    busy = sum((len(line) + 1) * byte_us for line in lines)
    return {'lines': lines, 'values': np.array(values), 'start_us': np.array(start),
            'end_us': np.array(end), 'byte_us': byte_us,
            'load_percent': 100.0 * busy / duration_us,
            'max_line_bytes': max((len(x) + 1 for x in lines), default=0)}


# ==============================================================================
# 5. BUILD AND RUN ON THE PC
# ==============================================================================

# Function to write the feed file the shim reads.
def write_feed(path: str, adc_counts: Optional[np.ndarray], signal_ms: int,
               serial: Optional[Dict], rx_buffer: int, duration_us: float) -> None:
    """
    ADC: "ADC steps channels signal_ms" + one row of counts per step.
    SERIAL: "SERIAL n byte_us rx_cap duration_us" + "start_us line" per line.

    Args:
        path (str): Output file.
        adc_counts (np.ndarray): Counts, or None in serial mode.
        signal_ms (int): Signal step, ms.
        serial (dict): serial_feed() output, or None in ADC mode.
        rx_buffer (int): RX buffer size in bytes (0 = unlimited).
        duration_us (float): Simulated time, us.
    """
    with open(path, 'w', encoding='ascii') as f:
        if serial is None:
            f.write(f'ADC {adc_counts.shape[0]} {adc_counts.shape[1]} {signal_ms}\n')
            np.savetxt(f, adc_counts, fmt='%d')
        else:
            f.write(f'SERIAL {len(serial["lines"])} {serial["byte_us"]!r} {rx_buffer} '
                    f'{int(duration_us)}\n')
            for st, line in zip(serial['start_us'], serial['lines']):
                f.write(f'{st!r} {line}\n')


# Function to compile the wired sketch with the shim and run it.
def run_sketch(sketch_dir: str, sketch_file: str, feed_path: str, serial_mode: bool):
    """
    Compile <sketch_file> + shim with g++ and run it on the feed.

    Args:
        sketch_dir (str): Directory with the .ino.
        sketch_file (str): .ino file name.
        feed_path (str): Feed file written by write_feed().
        serial_mode (bool): Whether the sketch reads the serial feed.

    Returns:
        tuple: (times_ms, sketch classes, feature buffers, serial log, feed stats).
    """
    gpp = shutil.which('g++') or shutil.which('clang++')
    if not gpp:
        raise SystemExit('[!] g++ not found in PATH.')

    with open(os.path.join(sketch_dir, 'arduino_shim.h'), 'w', encoding='utf-8') as f:
        f.write(ARDUINO_SHIM)
    main_src = os.path.join(sketch_dir, 'sim_main.cpp')
    with open(main_src, 'w', encoding='utf-8') as f:
        f.write(SIM_MAIN % {'sketch': sketch_file,
                            'epilogue': SERIAL_EPILOGUE if serial_mode else ''})
    exe = os.path.join(sketch_dir, 'sim.exe' if os.name == 'nt' else 'sim')

    # -ffp-contract=off: no fused multiply-add, so lo + counts*scale rounds
    # exactly like numpy; -w: infinite thresholds are emitted as 1e500
    cp = subprocess.run([gpp, '-O2', '-w', '-ffp-contract=off', '-o', exe, main_src],
                        capture_output=True, text=True)
    if cp.returncode != 0:
        print(cp.stderr[-3000:])
        raise SystemExit('[!] g++ could not compile the sketch.')

    run = subprocess.run([exe, feed_path], capture_output=True, text=True)
    if run.returncode != 0:
        raise SystemExit(f'[!] simulation exited with {run.returncode}: {run.stderr[-500:]}')

    times, classes, feats, log, stats = [], [], [], [], None
    cur_t, cur_c = None, None
    for line in run.stdout.splitlines():
        if line.startswith('#T '):
            cur_t, cur_c = int(line[3:]), None
        elif line.startswith('#X '):
            times.append(cur_t)
            classes.append(cur_c if cur_c is not None else -1)
            feats.append([float(v) for v in line[3:].split(',')])
        elif line.startswith('#S '):
            a, r, d = (int(v) for v in line[3:].split())
            stats = {'lines_accepted': a, 'lines_rejected': r, 'rx_bytes_dropped': d}
        else:
            log.append(line)
            m = re.match(r'\{"class":(-?\d+)', line)
            if m:
                cur_c = int(m.group(1))
    return np.array(times), np.array(classes), np.array(feats), log, stats


# ==============================================================================
# 6. METRICS AND PLOT
# ==============================================================================

# Function to compute the timeline metrics.
def timeline_metrics(t_s, y_true, y_sketch, y_py, settle_s):
    """
    Parity, accuracy and temporal behaviour of the classification.

    Args:
        t_s (np.ndarray): Time of each loop(), seconds.
        y_true (np.ndarray): True label at each loop().
        y_sketch (np.ndarray): Class printed by the sketch.
        y_py (np.ndarray): Python prediction on the same features.
        settle_s (float): Window after a label change excluded from settled accuracy.

    Returns:
        dict: Metrics.
    """
    change_idx = np.concatenate([[0], np.where(y_true[1:] != y_true[:-1])[0] + 1])
    change_t = t_s[change_idx]
    since = t_s - change_t[np.searchsorted(change_t, t_s, side='right') - 1]
    settled = since >= settle_s

    latencies, missed = [], 0
    for j, i0 in enumerate(change_idx[1:], start=1):
        i1 = change_idx[j + 1] if j + 1 < len(change_idx) else len(t_s)
        hit = np.where(y_sketch[i0:i1] == y_true[i0])[0]
        if len(hit):
            latencies.append(float(t_s[i0 + hit[0]] - t_s[i0]))
        else:
            missed += 1

    flips = int(np.sum(y_sketch[1:] != y_sketch[:-1]))
    changes = len(change_idx) - 1
    return {
        'loop_iterations': int(len(t_s)),
        'parity_percent': float(100.0 * np.mean(y_sketch == y_py)),
        'parity_mismatches': int(np.sum(y_sketch != y_py)),
        'accuracy_percent': float(100.0 * np.mean(y_sketch == y_true)),
        'settled_accuracy_percent': (float(100.0 * np.mean(y_sketch[settled] == y_true[settled]))
                                     if settled.any() else None),
        'label_changes': changes,
        'class_flips': flips,
        'flips_beyond_label_changes': max(0, flips - changes),
        'detection_latency_s': ({'mean': float(np.mean(latencies)),
                                 'max': float(np.max(latencies))} if latencies else None),
        'segments_never_detected': missed,
    }


# Function to rank features by how many rule conditions test them.
def most_used_features(model, n_features: int, k: int) -> List[int]:
    """
    Pick the k features that appear most often in the rule conditions.

    Args:
        model: PyRuleAnalyzer model.
        n_features (int): Number of features.
        k (int): How many to return.

    Returns:
        list: Feature indices.
    """
    counts = np.zeros(n_features)
    try:
        clf = model.classifier
        names = list(clf._array_feature_names)
        for rule in clf.final_rules or clf.initial_rules:
            for cond in rule.parsed_conditions or []:
                if cond[0] in names:
                    counts[names.index(cond[0])] += 1
    except Exception:
        pass
    return [int(i) for i in np.argsort(-counts, kind='stable')[:k]]


# Function to draw the timeline figure.
def plot_timeline(path, sig_t, signal_values, t_s, feats, y_true, y_sketch, y_py,
                  feature_names, plot_idx, class_names, kinds, title):
    """
    Signals of the most used features over the true label, and the classes.

    Args:
        path (str): Output PNG.
        sig_t (np.ndarray): Signal time axis, seconds.
        signal_values (np.ndarray): Signal as the board sees it (physical units).
        t_s (np.ndarray): loop() times, seconds.
        feats (np.ndarray): Features read by the sketch at each loop().
        y_true, y_sketch, y_py (np.ndarray): Classes per loop().
        feature_names (list): Feature names.
        plot_idx (list): Features to draw.
        class_names (list): Class names by code.
        kinds (np.ndarray): Channel kind per feature (log ones are tagged).
        title (str): Figure title.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = plt.get_cmap('tab10' if len(class_names) <= 10 else 'tab20')
    idx = np.concatenate([[0], np.where(y_true[1:] != y_true[:-1])[0] + 1, [len(t_s)]])
    spans = [(t_s[a], t_s[b] if b < len(t_s) else t_s[-1], int(y_true[a]))
             for a, b in zip(idx[:-1], idx[1:])]

    n = len(plot_idx)
    fig, axes = plt.subplots(n + 1, 1, figsize=(13, 1.7 * n + 2.8), sharex=True,
                             gridspec_kw={'height_ratios': [1] * n + [1.5]})
    axes = np.atleast_1d(axes)
    for ax, j in zip(axes[:-1], plot_idx):
        for a, b, c in spans:
            ax.axvspan(a, b, color=colors(c % colors.N), alpha=0.12, lw=0)
        ax.plot(sig_t, signal_values[:, j], lw=0.7, color='0.35')
        ax.plot(t_s, feats[:, j], '.', ms=2.5, color='black')
        label = feature_names[j][:24] + ('\n(log ADC)' if kinds[j] == KIND_LOG else '')
        ax.set_ylabel(label, fontsize=8, rotation=0, ha='right', va='center')
        ax.tick_params(labelsize=7)

    ax = axes[-1]
    for a, b, c in spans:
        ax.axvspan(a, b, color=colors(c % colors.N), alpha=0.12, lw=0)
    ax.step(t_s, y_true, where='post', color='0.5', lw=3, alpha=0.6, label='true label')
    ax.step(t_s, y_py, where='post', color='tab:blue', lw=1.2, label='Python model.predict')
    ax.plot(t_s, y_sketch, '.', ms=3, color='black', label='sketch on the PC')
    bad = y_sketch != y_py
    if bad.any():
        ax.plot(t_s[bad], y_sketch[bad], 'x', ms=8, color='red', label='sketch != Python')
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels([c[:14] for c in class_names], fontsize=7)
    ax.set_xlabel('time (s)')
    ax.legend(fontsize=7, loc='upper right', ncol=4)
    ax.tick_params(labelsize=7)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ==============================================================================
# 7. MAIN
# ==============================================================================

# Function to parse the command line.
def parse_args() -> argparse.Namespace:
    """
    Build and parse the command-line options.

    Returns:
        argparse.Namespace: Options.
    """
    p = argparse.ArgumentParser(
        description='Run a pyruleanalyzer Arduino sketch on simulated continuous inputs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Read the docstring at the top of this file for the details.')

    g = p.add_argument_group('data')
    g.add_argument('--dataset', default='iris',
                   choices=['iris', 'wine', 'breast_cancer', 'digits'],
                   help='built-in dataset (when no CSV is given)')
    g.add_argument('--csv', default=None, help='one CSV, split 70/30')
    g.add_argument('--train-csv', default=None, help='training CSV (with --test-csv)')
    g.add_argument('--test-csv', default=None, help='test CSV (replayed by --signal replay)')
    g.add_argument('--target', default=None,
                   help='target column: name or index (default: last column)')
    g.add_argument('--header', default='auto', choices=['auto', 'yes', 'no'],
                   help='CSV header row (auto: absent when the first line is all numbers)')
    g.add_argument('--drop-cols', default=None, help='comma-separated columns to ignore')
    g.add_argument('--time-col', default=None,
                   help='test column that orders the replay (numeric or date)')
    g.add_argument('--max-train-rows', type=int, default=0,
                   help='subsample the training set (0 = all)')

    g = p.add_argument_group('model')
    g.add_argument('--model', default='Decision Tree',
                   choices=['Decision Tree', 'Random Forest',
                            'Gradient Boosting Decision Trees'])
    g.add_argument('--n-estimators', type=int, default=None)
    g.add_argument('--max-depth', type=int, default=6)
    g.add_argument('--board', default='uno', help='board for the memory check')
    g.add_argument('--seed', type=int, default=42)

    g = p.add_argument_group('signal')
    g.add_argument('--signal', default='synthetic', choices=['synthetic', 'replay'])
    g.add_argument('--duration', type=float, default=120.0, help='seconds simulated')
    g.add_argument('--signal-ms', type=int, default=10, help='signal time step (ms)')
    g.add_argument('--sample-ms', type=int, default=250,
                   help='SAMPLE_INTERVAL_MS of the sketch (ms)')
    g.add_argument('--dwell', type=float, default=15.0,
                   help='mean seconds per regime / class episode')
    g.add_argument('--min-dwell', type=float, default=5.0,
                   help='minimum seconds per regime / class episode')
    g.add_argument('--row-ms', type=int, default=1000, help='replay: ms each row is held')
    g.add_argument('--tau', type=float, default=1.5,
                   help='synthetic: transition time constant (s)')
    g.add_argument('--wander', type=float, default=0.15,
                   help='synthetic: process std around the regime, in feature stds')
    g.add_argument('--noise', type=float, default=None,
                   help='white noise std in feature stds (default 0.03 synthetic, 0 replay)')

    g = p.add_argument_group('input')
    g.add_argument('--input', default='adc', choices=['adc', 'serial'])
    g.add_argument('--adc', default='auto', choices=['auto', 'linear', 'log'],
                   help='ADC transfer function (auto: log for heavy-tailed features)')
    g.add_argument('--adc-bits', type=int, default=10)
    g.add_argument('--baud', type=int, default=115200)
    g.add_argument('--rx-buffer', type=int, default=64,
                   help='serial RX buffer of the board, bytes (64 AVR, 256 ESP32, 0 = unlimited)')
    g.add_argument('--send-ms', type=int, default=None,
                   help='serial: gateway period in ms (default: replay on every row '
                        'change, synthetic every --sample-ms)')

    g = p.add_argument_group('output')
    g.add_argument('--output-dir', default='files/sim')
    g.add_argument('--name', default='sim_model')
    g.add_argument('--plot-features', type=int, default=4)
    g.add_argument('--min-parity', type=float, default=100.0)
    args = p.parse_args()

    if args.noise is None:
        args.noise = 0.03 if args.signal == 'synthetic' else 0.0
    if args.row_ms % args.signal_ms:
        p.error('--row-ms must be a multiple of --signal-ms')
    if args.input == 'serial' and args.send_ms is None and args.signal == 'synthetic':
        args.send_ms = args.sample_ms
    return args


# Function to run the whole simulation from the command line.
def main() -> int:
    """
    Prepare data, train, export, wire, simulate, compare and plot.

    Returns:
        int: Process exit code (0 pass, 1 fail).
    """
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    serial_mode = args.input == 'serial'

    print('=' * 62)
    print('CONTINUOUS-INPUT SIMULATION -- pyruleanalyzer Arduino sketch')
    print('=' * 62)

    # ---------- 1. data ----------
    print('\n[1/6] Data')
    data = prepare_data(args, args.output_dir)
    feature_names, class_names = data['features'], data['classes']

    # ---------- 2. train + export the production sketch ----------
    print('\n[2/6] Training and export')
    from pyruleanalyzer import full_pipeline
    params = dict(train_csv=data['train_csv'], test_csv=data['test_csv'],
                  target_feature=data['target'], model_type=args.model,
                  max_depth=args.max_depth, output_dir=args.output_dir,
                  output_name=args.name, generate_arduino_sketch=True,
                  board_model=args.board, serial_baud=args.baud,
                  memory_report=True, random_seed=args.seed)
    if args.n_estimators is not None:
        params['n_estimators'] = args.n_estimators
    results = full_pipeline(**params)
    model = results['model']
    ino_path = results['generated_files'].get('arduino')
    if not ino_path or not os.path.exists(ino_path):
        raise SystemExit('[!] full_pipeline produced no .ino.')

    # ---------- 3. signal ----------
    print('\n[3/6] Signal')
    sensors = SensorMap(data['X_train'], adc_bits=args.adc_bits, mode=args.adc)
    if args.signal == 'synthetic':
        signal, label = synthetic_signal(model, data['X_train'], sensors, args, rng)
    else:
        signal, label = replay_signal(data['X_test'], data['y_test'], data['test_time'],
                                      sensors, args, rng)
    duration_us = len(signal) * args.signal_ms * 1000.0
    n_changes = int(np.sum(label[1:] != label[:-1]))
    print(f'  {duration_us / 1e6:.0f} s, {len(signal):,} steps of {args.signal_ms} ms, '
          f'{n_changes} label changes')

    # ---------- 4. wire the sketch ----------
    print('\n[4/6] Input wiring')
    serial, counts = None, None
    if serial_mode:
        serial = serial_feed(signal, args.signal_ms, args.send_ms, args.baud)
        rf_code = SERIAL_READ_FEATURES
        print(f'  serial feed: {len(serial["lines"]):,} lines, up to '
              f'{serial["max_line_bytes"]} bytes each, UART load '
              f'{serial["load_percent"]:.1f}% at {args.baud} baud, RX buffer '
              f'{args.rx_buffer or "unlimited"}')
        if args.rx_buffer and serial['max_line_bytes'] > args.rx_buffer:
            print(f'  [!] lines are longer than the RX buffer: with delay({args.sample_ms}) '
                  'in loop() the board will lose bytes. Expect rejected lines.')
    else:
        counts = sensors.to_counts(signal)
        rf_code = sensors.c_code(feature_names)
        kinds = np.bincount(sensors.kind, minlength=3)
        print(f'  ADC {args.adc_bits}-bit: {kinds[KIND_LINEAR]} linear, {kinds[KIND_LOG]} log, '
              f'{kinds[KIND_DIGITAL]} digital channels')

    sketch_name = f'{args.name}_sim'
    sketch_dir = os.path.join(args.output_dir, sketch_name)
    os.makedirs(sketch_dir, exist_ok=True)
    sketch_file = f'{sketch_name}.ino'
    with open(ino_path, 'r', encoding='utf-8') as f:
        wired = wire_sketch(f.read(), rf_code, args.sample_ms)
    with open(os.path.join(sketch_dir, sketch_file), 'w', encoding='utf-8') as f:
        f.write(wired)
    print(f'  sketch: {os.path.join(sketch_dir, sketch_file)}')

    feed_path = os.path.join(args.output_dir, f'{args.name}_feed.txt')
    write_feed(feed_path, counts, args.signal_ms, serial, args.rx_buffer, duration_us)

    # ---------- 5. run the sketch on the PC ----------
    print('\n[5/6] Running setup()/loop() with g++ + Arduino shim')
    t_ms, y_sketch, feats, log, feed_stats = run_sketch(
        sketch_dir, sketch_file, os.path.abspath(feed_path), serial_mode)
    print(f'  {len(t_ms)} loop() iterations; serial banner: '
          f'{" / ".join(log[:2]) if log else "(none)"}')

    step = np.minimum(t_ms // args.signal_ms, len(signal) - 1).astype(int)
    y_true = label[step]
    t_s = t_ms / 1000.0
    y_py = np.asarray(model.predict(feats)).astype(int).ravel()

    # ---------- 6. compare ----------
    print('\n[6/6] Comparison')
    check: Dict = {}
    if serial_mode:
        # a buffer is valid when it is one of the lines sent (or the initial zeros)
        sent = {tuple(v) for v in serial['values']}
        sent.add(tuple([0.0] * len(feature_names)))
        valid = np.array([tuple(r) in sent for r in feats])
        # stale: the buffer differs from the newest line fully arrived by then
        arrived = np.searchsorted(serial['end_us'], t_ms * 1000.0, side='right') - 1
        newest = serial['values'][np.maximum(arrived, 0)]
        stale = (arrived >= 0) & np.any(feats != newest, axis=1)
        accepted = (feed_stats or {}).get('lines_accepted', 0)
        check = {'buffers_equal_to_a_sent_line_percent': float(100.0 * valid.mean()),
                 'stale_loops': int(stale.sum()),
                 'lines_sent': len(serial['lines']),
                 'lines_lost': len(serial['lines']) - accepted,
                 'serial_load_percent': serial['load_percent'],
                 'max_line_bytes': serial['max_line_bytes'],
                 **(feed_stats or {})}
        input_ok = bool(valid.all()) and accepted > 0
        sig_view = signal
    else:
        expected = sensors.to_values(counts[step])
        exact = sensors.kind != KIND_LOG
        bit_exact = bool(np.array_equal(expected[:, exact], feats[:, exact]))
        rel = (np.abs(expected[:, ~exact] - feats[:, ~exact])
               / np.maximum(np.abs(expected[:, ~exact]), 1e-300))
        max_rel = float(rel.max()) if rel.size else 0.0
        check = {'linear_digital_bit_exact': bit_exact, 'log_max_rel_error': max_rel}
        input_ok = bit_exact and max_rel <= 1e-12
        sig_view = sensors.to_values(counts)

    settle_s = 3 * args.tau if args.signal == 'synthetic' else 2 * args.sample_ms / 1000.0
    metrics = timeline_metrics(t_s, y_true, y_sketch, y_py, settle_s)
    passed = input_ok and metrics['parity_percent'] >= args.min_parity

    import pandas as pd
    timeline_path = os.path.join(args.output_dir, f'{args.name}_timeline.csv')
    pd.DataFrame({'time_s': t_s, 'true_label': y_true, 'sketch': y_sketch, 'python': y_py,
                  **{name: feats[:, i] for i, name in enumerate(feature_names)}}
                 ).to_csv(timeline_path, index=False)

    plot_path = os.path.join(args.output_dir, f'{args.name}_timeline.png')
    plot_idx = most_used_features(model, len(feature_names),
                                  min(args.plot_features, len(feature_names)))
    plot_timeline(plot_path, np.arange(len(signal)) * args.signal_ms / 1000.0, sig_view,
                  t_s, feats, y_true, y_sketch, y_py, feature_names, plot_idx, class_names,
                  sensors.kind,
                  f'{args.model} -- {args.signal} signal via {args.input} '
                  f'({args.sample_ms} ms sampling)')

    report = {
        'model': args.model,
        'data': {'train': args.train_csv or args.csv or args.dataset,
                 'test': args.test_csv, 'target': data['target'],
                 'features': feature_names, 'classes': class_names},
        'sketch': os.path.join(sketch_dir, sketch_file),
        'options': {k: v for k, v in vars(args).items()
                    if k not in ('csv', 'train_csv', 'test_csv', 'dataset')},
        'input_check': check,
        **metrics,
        'passed': passed,
    }
    report_path = os.path.join(args.output_dir, f'{args.name}_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)

    if serial_mode:
        print(f'  buffers equal to a sent line : '
              f'{check["buffers_equal_to_a_sent_line_percent"]:.2f}%')
        print(f'  stale loops (newer line had arrived): {check["stale_loops"]}')
        if feed_stats:
            print(f'  lines sent / accepted / lost : {check["lines_sent"]} / '
                  f'{feed_stats["lines_accepted"]} / {check["lines_lost"]} '
                  f'({feed_stats["lines_rejected"]} rejected by the checksum or length, '
                  f'{feed_stats["rx_bytes_dropped"]:,} bytes dropped by the RX buffer)')
        if not accepted:
            print('  [!] the sketch never received a complete line: it classified zeros.')
    else:
        print(f'  ADC values sketch == Python  : linear/digital bit-exact '
              f'{check["linear_digital_bit_exact"]}, log max rel. error '
              f'{check["log_max_rel_error"]:.1e}')
    print(f'  parity sketch vs Python      : {metrics["parity_percent"]:.2f}% '
          f'({metrics["parity_mismatches"]} mismatches)')
    settled = metrics['settled_accuracy_percent']
    print(f'  accuracy vs true label       : {metrics["accuracy_percent"]:.2f}%'
          + (f' (settled, >= {settle_s:.1f} s after a change: {settled:.2f}%)'
             if settled is not None else ''))
    print(f'  class flips                  : {metrics["class_flips"]} '
          f'({metrics["label_changes"]} label changes, '
          f'{metrics["flips_beyond_label_changes"]} beyond them)')
    lat = metrics['detection_latency_s']
    if lat:
        print(f'  detection latency            : mean {lat["mean"]:.2f} s, '
              f'max {lat["max"]:.2f} s ({metrics["segments_never_detected"]} '
              'segments never detected)')
    print(f'\n  timeline: {timeline_path}\n  plot    : {plot_path}\n  report  : {report_path}')
    print(f'\nRESULT: {"PASS" if passed else "FAIL"}')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
