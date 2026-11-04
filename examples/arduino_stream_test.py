"""
Streaming anomaly-detection test on a real Arduino/ESP32 -- before vs after
rule refinement.

`arduino_hardware_test.py` checks the exported engine one vector at a time and
`arduino_sensor_simulation.py` runs the sketch on the PC. This script closes the
loop on real hardware: the PC plays the sensors, streaming REAL rows of the
dataset to the board at a fixed rate, and the board classifies every sample and
raises an alarm when it detects an anomaly. The same scenario is played twice,
once on the model BEFORE refinement (initial rules) and once AFTER (final
rules), so the two can be compared on the board itself.

It is split in three commands so that the person who owns the board only needs
the package folder, not the training data:

    prepare  (you, on your PC)    train/load, export the two sketches, build the
                                  scenario from real test rows, compute the
                                  expected board behaviour, write a package.
    run      (whoever has the     flash each sketch, stream the scenario, log
              board)              every sample, write results/<run>/.
    report   (anyone)             compare before vs after (also run at the end
                                  of `run`).

===============================================================================
 THE SCENARIO
===============================================================================

An episode is "the plant runs normally, then an anomaly starts":

    normal rows (warm-up)            anomaly rows
    |-----------------------------|========================|
                                  ^ onset                 ^ end of rows
                                        ^ alarm -> episode stops

* rows are real test rows that were never used for training or refinement;
* with --time-col the episodes are cut around the real normal->anomaly
  transitions of the recording (SWaT, gas pipeline...); otherwise normal rows
  are followed by rows of one anomaly class, cycling through the classes;
* the board raises the alarm after --confirm consecutive non-normal
  classifications (debounce, as a real detector would); the alarm also lights
  LED_BUILTIN;
* an alarm before the onset is a FALSE ALARM: it is logged, the alarm is reset
  and the normal phase continues;
* an episode with no alarm until its last anomaly row is a MISS.

Because the alarm logic is deterministic, `prepare` computes from the Python
model exactly which sample each variant must alarm on ("oracle"). The run checks
the board against it: class parity per sample AND alarm parity per episode.

===============================================================================
 SERIAL PROTOCOL (text lines, '\\n')
===============================================================================

    PC -> board : H
    board -> PC : H,<variant>,<model_id>,<n_feat>,<n_classes>,<n_trees>,<normal>,<confirm>
    PC -> board : X                      (reset alarm and streak)
    board -> PC : X,ok
    PC -> board : D,<seq>,f0,...,fN-1*CS (CS = XOR of the bytes before '*', 2 hex)
    board -> PC : C,<seq>,<class>,<inference_us>,<streak>,<alarm 0/1>
    board -> PC : E,<reason>             (bad checksum, wrong field count...)

The PC sends one sample per --period-ms tick and waits for its answer, so the
64-byte RX buffer of an Uno never overflows. A reply that arrives after the
next tick is a deadline miss (inference slower than the sensor rate).

===============================================================================
 USAGE
===============================================================================

    # 1) you: build the package (real dataset, temporal transitions)
    python examples/arduino_stream_test.py prepare \\
        --train-csv data/swat/train.csv --test-csv data/swat/test.csv \\
        --target "Normal/Attack" --time-col Timestamp --drop-cols Timestamp \\
        --normal-class Normal --model "Random Forest" --n-estimators 10 \\
        --max-depth 6 --board mega --name swat_rf

    #    or reuse a model you already trained (pickled PyRuleAnalyzer)
    python examples/arduino_stream_test.py prepare --model-pkl files/model.pkl \\
        --train-csv ... --test-csv ... --target ... --name my_model

    # 2) check the package on your PC, no board (loopback: tests the protocol
    #    and the runner, not the C code)
    python examples/arduino_stream_test.py run --package files/stream/swat_rf --port sim

    # 3) your friend: flash + stream both variants (arduino-cli), or flash by
    #    hand in the Arduino IDE when asked (--manual-upload)
    python examples/arduino_stream_test.py run --package swat_rf --port COM5
    python examples/arduino_stream_test.py run --package swat_rf --port COM5 --manual-upload

    # 4) re-print / re-plot the comparison
    python examples/arduino_stream_test.py report --package swat_rf

The package folder also gets a LEIAME.md with these steps for the board owner.

Requirements: prepare -> the project (sklearn, pandas, numpy);
run/report -> numpy, pandas, pyserial (+ arduino-cli to flash, matplotlib for
the plot).
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Allows running straight from examples/ without installing the package
PROJECT_ROOT = str(Path(__file__).parent.parent.resolve())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
EXAMPLES_DIR = str(Path(__file__).parent.resolve())
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

from arduino_hardware_test import (BOARD_FQBN, compile_sketch, detect_port,  # noqa: E402
                                   extract_model_section, upload_sketch)

VARIANTS = ('before', 'after')
# Labels taken as normal operation when --normal-class is not given.
NORMAL_NAMES = {'normal', 'benign', 'legitimate', 'good'}
SCENARIO_FILE = 'scenario.csv'
MANIFEST_FILE = 'manifest.json'


# ==============================================================================
# 1. STREAM SKETCH (model section of the production .ino + this harness)
# ==============================================================================

STREAM_HARNESS = r'''
// ============================================================
// STREAM TEST HARNESS (generated by examples/arduino_stream_test.py)
//   H                       -> H,<variant>,<model_id>,<nf>,<nc>,<nt>,<normal>,<confirm>
//   X                       -> X,ok   (reset alarm)
//   D,<seq>,f0,...,fN*CS    -> C,<seq>,<class>,<us>,<streak>,<alarm>
//   anything else           -> E,<reason>
// pyra_predict() above is exactly the one from the production sketch.
// Alarm: ALARM_CONFIRM consecutive classes != NORMAL_CLASS.
// ============================================================

#define SERIAL_BAUD @@BAUD@@
#define NORMAL_CLASS @@NORMAL@@
#define ALARM_CONFIRM @@CONFIRM@@
#define STREAM_LINE_MAX (N_FEATURES * 16 + 32)

double features[N_FEATURES];
static char stream_line[STREAM_LINE_MAX];
static int stream_len = 0;
static bool stream_overflow = false;
static uint8_t alarm_streak = 0;
static bool alarm_on = false;

static void stream_led(bool on) {
#ifdef LED_BUILTIN
    digitalWrite(LED_BUILTIN, on ? HIGH : LOW);
#endif
}

static void stream_hello(void) {
    Serial.print(F("H,"));
    Serial.print(F(PYRA_VARIANT));  Serial.print(',');
    Serial.print(F(PYRA_MODEL_ID)); Serial.print(',');
    Serial.print(N_FEATURES);       Serial.print(',');
    Serial.print(N_CLASSES);        Serial.print(',');
    Serial.print(N_TREES);          Serial.print(',');
    Serial.print(NORMAL_CLASS);     Serial.print(',');
    Serial.println(ALARM_CONFIRM);
}

static void stream_reset(void) {
    alarm_streak = 0;
    alarm_on = false;
    stream_led(false);
    Serial.println(F("X,ok"));
}

static void stream_data(char *s) {
    char *star = strrchr(s, '*');
    if (!star || strlen(star + 1) != 2) { Serial.println(F("E,no_checksum")); return; }
    uint8_t sum = 0;
    for (char *p = s; p < star; p++) sum ^= (uint8_t)*p;
    char *end;
    long cs = strtol(star + 1, &end, 16);
    if (*end != '\0' || (uint8_t)cs != sum) { Serial.println(F("E,bad_checksum")); return; }
    *star = '\0';

    char *tok = strtok(s + 2, ",");          // skip "D,"
    if (!tok) { Serial.println(F("E,missing_seq")); return; }
    long seq = atol(tok);
    int k = 0;
    while ((tok = strtok(NULL, ",")) != NULL) {
        if (k < N_FEATURES) features[k] = atof(tok);
        k++;
    }
    if (k != N_FEATURES) { Serial.print(F("E,bad_feature_count,")); Serial.println(k); return; }

    unsigned long t0 = micros();
    int32_t cls = pyra_predict(features);
    unsigned long dt = micros() - t0;

    if (cls != NORMAL_CLASS) { if (alarm_streak < 255) alarm_streak++; }
    else alarm_streak = 0;
    bool fired = false;
    if (!alarm_on && alarm_streak >= ALARM_CONFIRM) {
        alarm_on = true;
        fired = true;
        stream_led(true);
    }

    Serial.print(F("C,"));  Serial.print(seq);
    Serial.print(',');      Serial.print((long)cls);
    Serial.print(',');      Serial.print(dt);
    Serial.print(',');      Serial.print((int)alarm_streak);
    Serial.print(',');      Serial.println(fired ? 1 : 0);
}

static void stream_handle(char *s) {
    if (s[0] == 'H' && s[1] == '\0') { stream_hello(); return; }
    if (s[0] == 'X' && s[1] == '\0') { stream_reset(); return; }
    if (s[0] == 'D' && s[1] == ',')  { stream_data(s); return; }
    Serial.println(F("E,unknown_command"));
}

void setup(void) {
    Serial.begin(SERIAL_BAUD);
    while (!Serial) delay(10);     // boards with native USB (Leonardo/Micro)
#ifdef LED_BUILTIN
    pinMode(LED_BUILTIN, OUTPUT);
#endif
    stream_led(false);
    Serial.println(F("PYRA-STREAM-READY"));
}

void loop(void) {
    while (Serial.available() > 0) {
        char c = (char)Serial.read();
        if (c == '\r') continue;
        if (c == '\n') {
            stream_line[stream_len] = '\0';
            if (stream_overflow) Serial.println(F("E,line_too_long"));
            else if (stream_len > 0) stream_handle(stream_line);
            stream_len = 0;
            stream_overflow = false;
        } else if (stream_len < STREAM_LINE_MAX - 1) {
            stream_line[stream_len++] = c;
        } else {
            stream_overflow = true;
        }
    }
}
'''


# Function to assemble the stream sketch of one variant.
def build_stream_sketch(ino_text: str, sketch_dir: str, sketch_name: str, variant: str,
                        algorithm: str, baud: int, normal: int, confirm: int) -> Dict:
    """
    Cut the model section of a production .ino and append the stream harness.

    Args:
        ino_text (str): Sketch produced by export_to_arduino_ino().
        sketch_dir (str): Folder of the sketch (same name as the .ino).
        sketch_name (str): Sketch name.
        variant (str): 'before' or 'after'.
        algorithm (str): Algorithm name, reported by the handshake.
        baud (int): Serial speed.
        normal (int): Class code of normal operation.
        confirm (int): Consecutive anomalous classes that raise the alarm.

    Returns:
        dict: Sketch path and model id (hash of the model section).
    """
    model_section = extract_model_section(ino_text)
    model_id = hashlib.sha1(model_section.encode('utf-8')).hexdigest()[:10]
    os.makedirs(sketch_dir, exist_ok=True)
    path = os.path.join(sketch_dir, f'{sketch_name}.ino')
    header = (
        '// ============================================================\n'
        f'// Stream test sketch -- model {variant.upper()} refinement\n'
        f'// Generated at {datetime.now().isoformat(timespec="seconds")}\n'
        '// DO NOT edit by hand: regenerate with examples/arduino_stream_test.py\n'
        '// ============================================================\n'
        '#include <Arduino.h>\n'
        '#include <stdint.h>\n'
        '#include <stdlib.h>\n'
        '#include <string.h>\n'
        f'#define PYRA_ALGORITHM "{algorithm}"\n'
        f'#define PYRA_VARIANT "{variant}"\n'
        f'#define PYRA_MODEL_ID "{model_id}"\n\n'
    )
    harness = (STREAM_HARNESS.replace('@@BAUD@@', str(baud))
               .replace('@@NORMAL@@', str(normal))
               .replace('@@CONFIRM@@', str(confirm)))
    with open(path, 'w', encoding='utf-8') as f:
        f.write(header + model_section + '\n' + harness)
    return {'path': path, 'model_id': model_id}


# ==============================================================================
# 2. SCENARIO
# ==============================================================================

# Function to format one feature row the way the gateway sends it.
def format_row(row: np.ndarray) -> str:
    """
    Values as float32 with 9 significant digits (exact float32 round-trip).

    Args:
        row (np.ndarray): Feature values.

    Returns:
        str: Comma-separated values.
    """
    return ','.join('%.9g' % float(v) for v in np.asarray(row, dtype=np.float32))


# Function to cut episodes around the real transitions of a recording.
def temporal_episodes(y: np.ndarray, order: np.ndarray, normal: int, n_episodes: int,
                      normal_rows: int, anomaly_rows: int) -> List[Dict]:
    """
    One episode per normal->anomaly transition of the time-ordered test set.

    Args:
        y (np.ndarray): Labels of the test rows.
        order (np.ndarray): Row indices in time order.
        normal (int): Normal class code.
        n_episodes (int): Maximum number of episodes.
        normal_rows (int): Rows before the onset (fewer if the run is shorter).
        anomaly_rows (int): Rows from the onset on.

    Returns:
        list: Episodes, each {'rows': indices, 'onset': position of the onset}.
    """
    ys = y[order]
    episodes, prev_end = [], 0
    for i in range(1, len(ys)):
        if ys[i - 1] != normal or ys[i] == normal or i < prev_end:
            continue
        start = max(prev_end, i - normal_rows)
        end = min(len(ys), i + anomaly_rows)
        if i - start < 1:
            continue
        episodes.append({'rows': order[start:end], 'onset': i - start})
        prev_end = end
        if len(episodes) >= n_episodes:
            break
    return episodes


# Function to splice normal rows and rows of one anomaly class.
def spliced_episodes(y: np.ndarray, normal: int, n_episodes: int, normal_rows: int,
                     anomaly_rows: int, rng: np.random.Generator) -> List[Dict]:
    """
    Normal rows followed by rows of one anomaly class, cycling through classes.

    Rows are drawn without replacement while each pool lasts.

    Args:
        y (np.ndarray): Labels of the test rows.
        normal (int): Normal class code.
        n_episodes (int): Number of episodes.
        normal_rows (int): Rows before the onset.
        anomaly_rows (int): Rows after the onset.
        rng (np.random.Generator): Random generator.

    Returns:
        list: Episodes, each {'rows': indices, 'onset': normal_rows}.
    """
    pools = {int(c): rng.permutation(np.where(y == c)[0]) for c in np.unique(y)}
    if normal not in pools:
        raise SystemExit('[!] The test set has no row of the normal class.')
    anomalies = sorted(c for c in pools if c != normal)
    if not anomalies:
        raise SystemExit('[!] The test set has no anomaly rows.')
    used = {c: 0 for c in pools}

    def take(c, n):
        pool = pools[c]
        idx = [pool[(used[c] + j) % len(pool)] for j in range(n)]
        used[c] += n
        return idx

    episodes = []
    for e in range(n_episodes):
        cls = anomalies[e % len(anomalies)]
        rows = take(normal, normal_rows) + take(cls, anomaly_rows)
        episodes.append({'rows': np.array(rows), 'onset': normal_rows})
    return episodes


# Function to replay the board's alarm logic on the expected classes.
def alarm_oracle(classes: np.ndarray, onset: int, normal: int, confirm: int) -> Dict:
    """
    What the board must do on one episode, given the classes it will compute.

    Mirrors the sketch and the runner: the streak counts consecutive non-normal
    classes; an alarm before the onset is a false alarm and is reset (streak 0);
    the first alarm at or after the onset stops the episode.

    Args:
        classes (np.ndarray): Expected class of every row of the episode.
        onset (int): Position of the first anomaly row.
        normal (int): Normal class code.
        confirm (int): Alarm threshold.

    Returns:
        dict: false_alarm_steps, detect_step (None = miss), samples_sent.
    """
    streak, false_alarms = 0, []
    for step, cls in enumerate(classes):
        streak = streak + 1 if cls != normal else 0
        if streak >= confirm:
            if step >= onset:
                return {'false_alarm_steps': false_alarms, 'detect_step': step,
                        'samples_sent': step + 1}
            false_alarms.append(step)
            streak = 0
    return {'false_alarm_steps': false_alarms, 'detect_step': None,
            'samples_sent': len(classes)}


# ==============================================================================
# 3. PREPARE
# ==============================================================================

LEIAME = '''# Teste de streaming no Arduino -- pacote `{name}`

Este pacote compara, **na placa**, o modelo **antes** e **depois** do refinamento
de regras do pyruleanalyzer. O PC faz o papel dos sensores: envia linhas reais da
base de dados, uma a cada {period} ms, e o Arduino classifica cada amostra e
dispara um **alarme** (LED da placa acende) depois de {confirm} classificacoes
anormais seguidas.

- Modelo: {model} | {n_features} features | classe normal: `{normal_name}`
- Cenario: {n_episodes} episodios, {n_rows} amostras reais no total
- Regras: {rules_before} antes -> {rules_after} depois do refinamento
- Placa alvo: `{board}` ({fqbn})

## O que voce precisa

1. Python 3.9+ com `pip install numpy pandas pyserial matplotlib`
2. O repositorio pyruleanalyzer (so a pasta `examples/` e usada) **ou** pelo menos
   `examples/arduino_stream_test.py` e `examples/arduino_hardware_test.py`
3. Uma das opcoes para gravar a placa:
   - **arduino-cli** (automatico): https://arduino.github.io/arduino-cli/
     depois `arduino-cli core update-index` e `arduino-cli core install arduino:avr`
     (ou `esp32:esp32` para ESP32)
   - **Arduino IDE** (manual): use `--manual-upload`, o script pede para voce
     gravar cada sketch e apertar Enter.

Feche o Monitor Serial da IDE antes de rodar: a porta e exclusiva.

## Passo a passo

Descubra a porta (`arduino-cli board list`, ou Ferramentas > Porta na IDE). Depois,
da raiz do repositorio:

```
python examples/arduino_stream_test.py run --package CAMINHO/DO/{name} --port COM5
```

com IDE em vez de arduino-cli:

```
python examples/arduino_stream_test.py run --package CAMINHO/DO/{name} --port COM5 --manual-upload
```

Sketches (se for gravar manualmente):

- antes:  `sketches/{name}_before/{name}_before.ino`
- depois: `sketches/{name}_after/{name}_after.ino`

O script grava o sketch **before**, roda o cenario inteiro, grava o **after**,
roda o mesmo cenario e escreve `results/<placa>_<data>/` com:

- `before_samples.csv`, `after_samples.csv` -- cada amostra enviada e a resposta
- `before.json`, `after.json` -- metricas de cada variante
- `comparison.md`, `comparison.json`, `comparison.png` -- antes x depois

**Mande a pasta `results/` inteira de volta.** Nao edite os sketches nem o
`scenario.csv`: o handshake confere o id do modelo gravado e o teste aborta se o
sketch errado estiver na placa.

## Testar sem placa

```
python examples/arduino_stream_test.py run --package CAMINHO/DO/{name} --port sim
```

Isso so testa o protocolo e o script (a "placa" virtual responde a classe esperada).
'''


# Function to build the test package.
def cmd_prepare(args) -> int:
    """
    Train or load the model, export both sketches, build the scenario.

    Args:
        args (argparse.Namespace): Options of the prepare command.

    Returns:
        int: Exit code.
    """
    import pandas as pd
    from arduino_sensor_simulation import prepare_data

    rng = np.random.default_rng(args.seed)
    pkg = os.path.join(args.output_dir, args.name)
    work = os.path.join(pkg, 'work')
    os.makedirs(work, exist_ok=True)

    print('=' * 62)
    print('STREAM TEST -- prepare package')
    print('=' * 62)

    # ---------- data ----------
    print('\n[1/5] Data')
    data = prepare_data(args, work)
    feature_names, class_names = data['features'], data['classes']
    y_train = pd.read_csv(data['train_csv'])[data['target']].to_numpy(int)

    named = [i for i, c in enumerate(class_names)
             if str(c).strip().lower() in NORMAL_NAMES]
    try:
        binary01 = sorted(float(c) for c in class_names) == [0.0, 1.0]
    except ValueError:
        binary01 = False
    if args.normal_class is None and len(named) == 1:
        normal = named[0]
    elif args.normal_class is None and binary01:
        normal = [float(c) for c in class_names].index(0.0)
        print('  labels are 0/1: taking 0 as normal operation (pass --normal-class 1 '
              'if it is the other way around)')
    elif args.normal_class is None:
        normal = int(np.bincount(y_train).argmax())
        print(f'  [!] no class named like {sorted(NORMAL_NAMES)}: using the most frequent '
              'one. Pass --normal-class if that is not normal operation.')
    elif args.normal_class.strip().lower() in [str(c).strip().lower() for c in class_names]:
        normal = [str(c).strip().lower() for c in class_names].index(
            args.normal_class.strip().lower())
    else:
        try:
            normal = int(args.normal_class)
            if not 0 <= normal < len(class_names):
                raise ValueError
        except ValueError:
            raise SystemExit(f'[!] --normal-class "{args.normal_class}" is not one of '
                             f'{class_names}')
    print(f'  normal class: {normal} ("{class_names[normal]}")')

    # ---------- model ----------
    print('\n[2/5] Model')
    if args.model_pkl:
        import pickle
        with open(args.model_pkl, 'rb') as f:
            model = pickle.load(f)
        print(f'  loaded {args.model_pkl}')
    else:
        from pyruleanalyzer import full_pipeline
        params = dict(train_csv=data['train_csv'], test_csv=data['test_csv'],
                      target_feature=data['target'], model_type=args.model,
                      max_depth=args.max_depth, output_dir=work, output_name=args.name,
                      generate_arduino_sketch=False, memory_report=False,
                      random_seed=args.seed)
        if args.n_estimators is not None:
            params['n_estimators'] = args.n_estimators
        model = full_pipeline(**params)['model']
    clf = model.classifier
    model_features = list(getattr(model, 'feature_names', None) or [])
    if model_features and len(model_features) != len(feature_names):
        raise SystemExit(f'[!] The model has {len(model_features)} features, the data '
                         f'{len(feature_names)}. Use the same columns (and --drop-cols).')
    if not clf.final_rules:
        print('  [!] the model has no refined rules: "after" equals "before".')
    rules = {'before': len(clf.initial_rules),
             'after': len(clf.final_rules or clf.initial_rules)}
    print(f'  rules: before {rules["before"]} -> after {rules["after"]}')

    # ---------- sketches ----------
    print('\n[3/5] Sketches')
    algorithm = getattr(clf, 'algorithm_type', args.model)
    sketches, memory_est = {}, {}
    for variant in VARIANTS:
        rule_set = clf.initial_rules if variant == 'before' else (clf.final_rules
                                                                  or clf.initial_rules)
        clf.compile_tree_arrays(rules=rule_set, feature_names=model_features or None)
        ino_tmp = os.path.join(work, f'{args.name}_{variant}_production.ino')
        exported = clf.export_to_arduino_ino(filepath=ino_tmp, board_model=args.board,
                                             serial_baud=args.baud)
        memory_est[variant] = exported.get('memory_check', {})
        with open(ino_tmp, 'r', encoding='utf-8') as f:
            ino_text = f.read()
        name = f'{args.name}_{variant}'
        sk = build_stream_sketch(ino_text, os.path.join(pkg, 'sketches', name), name,
                                 variant, algorithm, args.baud, normal, args.confirm)
        sketches[variant] = {'dir': os.path.relpath(os.path.dirname(sk['path']), pkg),
                             'model_id': sk['model_id']}
        print(f'  {variant:<6}: {sk["path"]} (model id {sk["model_id"]})')

    # ---------- scenario ----------
    print('\n[4/5] Scenario')
    X_test, y_test = data['X_test'], data['y_test']
    episodes = []
    if data['test_time'] is not None:
        order = np.argsort(data['test_time'], kind='stable')
        episodes = temporal_episodes(y_test, order, normal, args.episodes,
                                     args.normal_rows, args.anomaly_rows)
        print(f'  {len(episodes)} real normal->anomaly transitions found in time order')
        if not episodes:
            print('  [!] no transition found: falling back to spliced episodes')
    if not episodes:
        episodes = spliced_episodes(y_test, normal, args.episodes, args.normal_rows,
                                    args.anomaly_rows, rng)
        print(f'  {len(episodes)} spliced episodes ({args.normal_rows} normal rows + '
              f'{args.anomaly_rows} rows of one anomaly class, cycling)')

    rows = np.concatenate([ep['rows'] for ep in episodes])
    lines = [format_row(X_test[i]) for i in rows]
    X_sent = np.array([[float(v) for v in line.split(',')] for line in lines])
    expected = {}
    for variant in VARIANTS:
        pred = clf.predict_batch(X_sent, use_final=(variant == 'after'))
        expected[variant] = np.asarray(pred).astype(int).ravel()

    table, ep_meta, pos = [], [], 0
    for e, ep in enumerate(episodes):
        n = len(ep['rows'])
        oracle = {}
        for variant in VARIANTS:
            oracle[variant] = alarm_oracle(expected[variant][pos:pos + n], ep['onset'],
                                           normal, args.confirm)
        anomaly_labels = y_test[ep['rows'][ep['onset']:]]
        ep_meta.append({'episode': e, 'first_seq': pos, 'n_rows': n, 'onset': ep['onset'],
                        'anomaly_class': int(np.bincount(anomaly_labels).argmax()),
                        'oracle': oracle})
        for step in range(n):
            table.append({'seq': pos + step, 'episode': e, 'step': step,
                          'phase': 'normal' if step < ep['onset'] else 'anomaly',
                          'test_row': int(ep['rows'][step]),
                          'true_label': int(y_test[ep['rows'][step]]),
                          'expected_before': int(expected['before'][pos + step]),
                          'expected_after': int(expected['after'][pos + step]),
                          'features': lines[pos + step]})
        pos += n
    pd.DataFrame(table).to_csv(os.path.join(pkg, SCENARIO_FILE), index=False)
    print(f'  {len(rows):,} samples -> {os.path.join(pkg, SCENARIO_FILE)}')
    max_line = max(len(f'D,{len(rows)},{line}*00') + 1 for line in lines)

    # ---------- manifest ----------
    print('\n[5/5] Expected behaviour on the board (oracle)')
    manifest = {
        'name': args.name,
        'created': datetime.now().isoformat(timespec='seconds'),
        'model': args.model if not args.model_pkl else f'pkl:{os.path.basename(args.model_pkl)}',
        'algorithm': algorithm,
        'features': feature_names,
        'classes': class_names,
        'normal_class': normal,
        'confirm': args.confirm,
        'period_ms': args.period_ms,
        'baud': args.baud,
        'board': args.board,
        'fqbn': BOARD_FQBN.get(args.board),
        'rules': rules,
        'sketches': sketches,
        'memory_estimated': memory_est,
        'max_line_bytes': max_line,
        'episodes': ep_meta,
        'data': {'train': args.train_csv or args.csv or args.dataset, 'test': args.test_csv,
                 'target': data['target'], 'time_col': args.time_col,
                 'drop_cols': args.drop_cols, 'seed': args.seed},
    }
    with open(os.path.join(pkg, MANIFEST_FILE), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, default=str)

    for variant in VARIANTS:
        s = summarize_oracle(ep_meta, variant)
        print(f'  {variant:<6}: detects {s["detected"]}/{len(ep_meta)} episodes, '
              f'{s["false_alarms"]} false alarms, mean delay '
              f'{_fmt(s["mean_delay_samples"])} samples')

    with open(os.path.join(pkg, 'LEIAME.md'), 'w', encoding='utf-8') as f:
        f.write(LEIAME.format(name=args.name, period=args.period_ms, confirm=args.confirm,
                              model=manifest['model'], n_features=len(feature_names),
                              normal_name=class_names[normal], n_episodes=len(ep_meta),
                              n_rows=len(rows), rules_before=rules['before'],
                              rules_after=rules['after'], board=args.board,
                              fqbn=manifest['fqbn']))
    if args.board in ('uno', 'nano', 'leonardo') and max_line > 600:
        print(f'  [!] lines up to {max_line} bytes: the line buffer alone takes that much '
              'of the 2 KB SRAM. Prefer a Mega or an ESP32.')
    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    print(f'\n  package: {pkg}\n  send this folder to whoever has the board (LEIAME.md).')
    return 0


# Function to aggregate the oracle of every episode.
def summarize_oracle(ep_meta: List[Dict], variant: str) -> Dict:
    """
    Detected episodes, false alarms and mean delay the board should reproduce.

    Args:
        ep_meta (list): Episodes from the manifest.
        variant (str): 'before' or 'after'.

    Returns:
        dict: Aggregated oracle.
    """
    delays = [ep['oracle'][variant]['detect_step'] - ep['onset'] + 1 for ep in ep_meta
              if ep['oracle'][variant]['detect_step'] is not None]
    return {'detected': len(delays),
            'false_alarms': sum(len(ep['oracle'][variant]['false_alarm_steps'])
                                for ep in ep_meta),
            'mean_delay_samples': statistics.mean(delays) if delays else None}


# ==============================================================================
# 4. TRANSPORTS (real serial port, or a loopback board for dry runs)
# ==============================================================================

class SerialLink:
    """
    Line-oriented link over a real serial port (pyserial).

    Args:
        port (str): Serial port (COM5, /dev/ttyUSB0).
        baud (int): Speed.
        boot_delay (float): Seconds to wait after opening (AVR boards reset).
    """

    # Method to open the port and drain the boot banner.
    def __init__(self, port: str, baud: int, boot_delay: float):
        """
        Open the port and wait for the board to boot.

        Args:
            port (str): Serial port.
            baud (int): Speed.
            boot_delay (float): Seconds to wait after opening.
        """
        try:
            import serial  # pyserial
        except ImportError:
            raise SystemExit('[!] pyserial is not installed. Run: pip install pyserial')
        self.ser = serial.Serial(port, baud, timeout=0.05)
        time.sleep(boot_delay)
        self.ser.reset_input_buffer()

    # Method to send one line.
    def send(self, line: str) -> None:
        """
        Write one line terminated by '\\n'.

        Args:
            line (str): Line without terminator.
        """
        self.ser.write((line + '\n').encode('ascii'))
        self.ser.flush()

    # Method to read one line.
    def recv(self, timeout: float) -> Optional[str]:
        """
        Read one non-empty line, or None after timeout seconds.

        Args:
            timeout (float): Seconds.

        Returns:
            str: Line without terminator, or None.
        """
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            line = self.ser.readline().decode('utf-8', 'replace').strip()
            if line:
                return line
        return None

    # Method to close the port.
    def close(self) -> None:
        """Close the serial port."""
        self.ser.close()


class LoopbackLink:
    """
    A virtual board for dry runs: speaks the protocol and answers the class
    the scenario expects for each seq. It tests the runner, the protocol and
    the package, not the exported C code.

    Args:
        manifest (dict): Package manifest.
        variant (str): Variant this "board" was flashed with.
        expected (np.ndarray): Expected class per seq for that variant.
    """

    # Method to set up the virtual board.
    def __init__(self, manifest: Dict, variant: str, expected: np.ndarray):
        """
        Keep what the virtual sketch needs.

        Args:
            manifest (dict): Package manifest.
            variant (str): 'before' or 'after'.
            expected (np.ndarray): Expected class per seq.
        """
        self.m, self.variant, self.expected = manifest, variant, expected
        self.streak, self.alarm, self.out = 0, False, []

    # Method to process one line sent by the PC.
    def send(self, line: str) -> None:
        """
        Handle H, X and D lines like the sketch does.

        Args:
            line (str): Line without terminator.
        """
        m = self.m
        if line == 'H':
            self.out.append(f'H,{self.variant},{m["sketches"][self.variant]["model_id"]},'
                            f'{len(m["features"])},{len(m["classes"])},0,'
                            f'{m["normal_class"]},{m["confirm"]}')
        elif line == 'X':
            self.streak, self.alarm = 0, False
            self.out.append('X,ok')
        elif line.startswith('D,'):
            body, _, cs = line.rpartition('*')
            if checksum(body) != cs:
                self.out.append('E,bad_checksum')
                return
            fields = body.split(',')
            seq = int(fields[1])
            if len(fields) - 2 != len(m['features']):
                self.out.append(f'E,bad_feature_count,{len(fields) - 2}')
                return
            cls = int(self.expected[seq])
            self.streak = self.streak + 1 if cls != m['normal_class'] else 0
            fired = not self.alarm and self.streak >= m['confirm']
            self.alarm = self.alarm or fired
            self.out.append(f'C,{seq},{cls},0,{self.streak},{int(fired)}')
        else:
            self.out.append('E,unknown_command')

    # Method to read one reply.
    def recv(self, timeout: float) -> Optional[str]:
        """
        Pop the next reply.

        Args:
            timeout (float): Ignored.

        Returns:
            str: Reply or None.
        """
        return self.out.pop(0) if self.out else None

    # Method to close the link.
    def close(self) -> None:
        """Nothing to close."""


# Function to compute the NMEA-style checksum of a line.
def checksum(body: str) -> str:
    """
    XOR of the bytes, as two upper-case hex digits.

    Args:
        body (str): Bytes before '*'.

    Returns:
        str: Checksum.
    """
    cs = 0
    for ch in body.encode('ascii'):
        cs ^= ch
    return f'{cs:02X}'


# ==============================================================================
# 5. RUN
# ==============================================================================

# Function to send one command and wait for the reply with a given prefix.
def request(link, line: str, prefix: str, timeout: float) -> Optional[str]:
    """
    Send a line and return the first reply starting with prefix or 'E,'.

    Unrelated lines (boot banner, debug prints) are skipped.

    Args:
        link: SerialLink or LoopbackLink.
        line (str): Line to send.
        prefix (str): Expected reply prefix.
        timeout (float): Seconds.

    Returns:
        str: Reply, or None on timeout.
    """
    link.send(line)
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        reply = link.recv(max(0.0, deadline - time.perf_counter()))
        if reply is None:
            return None
        if reply.startswith(prefix) or reply.startswith('E,'):
            return reply
    return None


# Function to stream the scenario to a board running one variant.
def stream_variant(link, variant: str, manifest: Dict, scenario, args) -> Dict:
    """
    Handshake, then play every episode until its alarm (or its last row).

    Args:
        link: SerialLink or LoopbackLink.
        variant (str): 'before' or 'after'.
        manifest (dict): Package manifest.
        scenario (pd.DataFrame): Scenario rows.
        args (argparse.Namespace): Run options.

    Returns:
        dict: Per-sample log and per-episode outcome.
    """
    hello = request(link, 'H', 'H,', args.timeout + 2)
    if not hello or not hello.startswith('H,'):
        raise SystemExit(f'[!] No handshake from the board ({hello!r}). Is the stream '
                         'sketch flashed, the port right and the Serial Monitor closed?')
    parts = hello.split(',')
    want = manifest['sketches'][variant]['model_id']
    if parts[1] != variant or parts[2] != want:
        raise SystemExit(f'[!] The board runs "{parts[1]}" model {parts[2]}, expected '
                         f'"{variant}" model {want}. Flash sketches/'
                         f'{manifest["name"]}_{variant}.')
    print(f'  handshake: {hello}')

    period = (args.period_ms if args.period_ms is not None else manifest['period_ms']) / 1000.0
    col_exp = f'expected_{variant}'
    samples, outcomes, errors = [], [], []
    t_start = time.perf_counter()

    for ep in manifest['episodes']:
        if request(link, 'X', 'X,', args.timeout) != 'X,ok':
            raise SystemExit('[!] The board did not acknowledge the alarm reset.')
        rows = scenario.iloc[ep['first_seq']:ep['first_seq'] + ep['n_rows']]
        onset, false_alarms, detect_step, onset_t = ep['onset'], [], None, None
        next_tick = time.perf_counter()

        for step, r in enumerate(rows.itertuples(index=False)):
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(next_tick - now)
            body = f'D,{r.seq},{r.features}'
            line = f'{body}*{checksum(body)}'

            reply, tries, t_send = None, 0, time.perf_counter()
            while tries <= args.retries:
                tries += 1
                t_send = time.perf_counter()
                reply = request(link, line, 'C,', args.timeout)
                if reply and reply.startswith('C,'):
                    break
                errors.append({'seq': int(r.seq), 'try': tries, 'reply': reply or 'timeout'})
            t_reply = time.perf_counter()
            if step == onset:
                onset_t = t_send

            rec = {'seq': int(r.seq), 'episode': ep['episode'], 'step': step,
                   'phase': r.phase, 'true_label': int(r.true_label),
                   'expected': int(getattr(r, col_exp)), 'board': None,
                   'inference_us': None, 'streak': None, 'alarm': 0,
                   'rtt_ms': (t_reply - t_send) * 1000.0,
                   'deadline_miss': int(t_reply > next_tick + period),
                   't_s': t_send - t_start}
            if reply and reply.startswith('C,'):
                p = reply.split(',')
                if int(p[1]) != int(r.seq):
                    errors.append({'seq': int(r.seq), 'try': tries, 'reply': reply})
                rec.update(board=int(p[2]), inference_us=int(p[3]), streak=int(p[4]),
                           alarm=int(p[5]))
            samples.append(rec)
            next_tick += period

            if rec['alarm']:
                if step < onset:
                    false_alarms.append(step)
                    if request(link, 'X', 'X,', args.timeout) != 'X,ok':
                        raise SystemExit('[!] The board did not acknowledge the alarm reset.')
                else:
                    detect_step = step
                    break

        delay_ms = ((samples[-1]['t_s'] + t_start - onset_t) * 1000.0
                    + samples[-1]['rtt_ms']) if detect_step is not None else None
        outcomes.append({'episode': ep['episode'], 'onset': onset,
                         'anomaly_class': ep['anomaly_class'],
                         'false_alarm_steps': false_alarms, 'detect_step': detect_step,
                         'delay_samples': (detect_step - onset + 1
                                           if detect_step is not None else None),
                         'delay_ms': delay_ms,
                         'oracle': ep['oracle'][variant],
                         'alarm_parity': (false_alarms
                                          == ep['oracle'][variant]['false_alarm_steps']
                                          and detect_step
                                          == ep['oracle'][variant]['detect_step'])})
        tag = (f'alarm at +{detect_step - onset + 1} samples' if detect_step is not None
               else 'MISSED')
        print(f'    episode {ep["episode"]:>3}: {tag}'
              + (f', {len(false_alarms)} false alarm(s)' if false_alarms else '')
              + ('' if outcomes[-1]['alarm_parity'] else '  [!] differs from the oracle'))

    return {'samples': samples, 'episodes': outcomes, 'errors': errors,
            'board_info': hello, 'wall_seconds': time.perf_counter() - t_start}


# Function to compute the metrics of one variant.
def variant_metrics(variant: str, run: Dict, manifest: Dict, memory: Optional[Dict]) -> Dict:
    """
    Parity, accuracy, detection, false alarms, latency and memory.

    Args:
        variant (str): 'before' or 'after'.
        run (dict): stream_variant() output.
        manifest (dict): Package manifest.
        memory (dict): Flash/SRAM measured by arduino-cli, or None.

    Returns:
        dict: Metrics.
    """
    s = [x for x in run['samples'] if x['board'] is not None]
    eps = run['episodes']
    normal_samples = sum(1 for x in run['samples'] if x['phase'] == 'normal')
    us = sorted(x['inference_us'] for x in s)
    rtt = sorted(x['rtt_ms'] for x in s)
    delays = [e['delay_samples'] for e in eps if e['delay_samples'] is not None]
    delays_ms = [e['delay_ms'] for e in eps if e['delay_ms'] is not None]
    n_fa = sum(len(e['false_alarm_steps']) for e in eps)

    def pct(v, q):
        return v[min(len(v) - 1, int(q * len(v)))] if v else None

    return {
        'variant': variant,
        'model_id': manifest['sketches'][variant]['model_id'],
        'rules': manifest['rules'][variant],
        'samples_sent': len(run['samples']),
        'samples_answered': len(s),
        'class_parity_percent': (100.0 * sum(x['board'] == x['expected'] for x in s) / len(s)
                                 if s else None),
        'class_mismatches': [x for x in s if x['board'] != x['expected']][:20],
        'accuracy_percent': (100.0 * sum(x['board'] == x['true_label'] for x in s) / len(s)
                             if s else None),
        'episodes': len(eps),
        'detected': len(delays),
        'detection_rate_percent': 100.0 * len(delays) / len(eps) if eps else None,
        'false_alarms': n_fa,
        'false_alarms_per_1000_normal': 1000.0 * n_fa / normal_samples if normal_samples else None,
        'delay_samples_mean': statistics.mean(delays) if delays else None,
        'delay_samples_max': max(delays) if delays else None,
        'delay_ms_mean': statistics.mean(delays_ms) if delays_ms else None,
        'alarm_parity_episodes': sum(e['alarm_parity'] for e in eps),
        'inference_us': ({'mean': statistics.mean(us), 'p50': pct(us, 0.5),
                          'p95': pct(us, 0.95), 'max': us[-1]} if us else None),
        'rtt_ms': ({'mean': statistics.mean(rtt), 'p95': pct(rtt, 0.95), 'max': rtt[-1]}
                   if rtt else None),
        'deadline_misses': sum(x['deadline_miss'] for x in run['samples']),
        'communication_errors': len(run['errors']),
        'memory_measured': memory,
        'memory_estimated': manifest['memory_estimated'].get(variant),
        'board_info': run['board_info'],
        'wall_seconds': run['wall_seconds'],
    }


# Function to flash one variant (arduino-cli, by hand, or nothing).
def flash_variant(variant: str, pkg: str, manifest: Dict, args, fqbn: str) -> Optional[Dict]:
    """
    Compile and upload the variant's sketch, or ask the user to do it.

    Args:
        variant (str): 'before' or 'after'.
        pkg (str): Package folder.
        manifest (dict): Package manifest.
        args (argparse.Namespace): Run options.
        fqbn (str): Board FQBN.

    Returns:
        dict: Memory measured by arduino-cli, or None.
    """
    sketch_dir = os.path.join(pkg, manifest['sketches'][variant]['dir'])
    if args.port == 'sim' or args.skip_upload:
        return None
    if args.manual_upload:
        ino = os.path.join(sketch_dir, os.path.basename(sketch_dir) + '.ino')
        input(f'\n  >>> Flash {ino}\n      with the Arduino IDE (Serial Monitor CLOSED), '
              'then press Enter... ')
        return None
    cli = shutil.which(args.arduino_cli)
    if not cli:
        raise SystemExit('[!] arduino-cli not found. Install it, or use --manual-upload '
                         '(flash with the Arduino IDE when asked).')
    memory = compile_sketch(cli, fqbn, sketch_dir)
    if memory is None:
        raise SystemExit(f'[!] Compilation of the "{variant}" sketch failed (see above). '
                         'If it does not fit, use a bigger board.')
    if not upload_sketch(cli, fqbn, args.port, sketch_dir):
        raise SystemExit('[!] Upload failed. Is the Serial Monitor closed?')
    return memory


# Function to run the whole package on a board.
def cmd_run(args) -> int:
    """
    Flash and stream both variants, then write results and the comparison.

    Args:
        args (argparse.Namespace): Options of the run command.

    Returns:
        int: Exit code (0 when every check passed).
    """
    import pandas as pd

    pkg = args.package
    with open(os.path.join(pkg, MANIFEST_FILE), 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    scenario = pd.read_csv(os.path.join(pkg, SCENARIO_FILE), dtype={'features': str})
    board = args.board or manifest['board']
    fqbn = args.fqbn or BOARD_FQBN.get(board) or manifest['fqbn']
    baud = args.baud or manifest['baud']

    if args.port is None:
        cli = shutil.which(args.arduino_cli)
        args.port = detect_port(cli, fqbn) if cli else None
        if not args.port:
            raise SystemExit('[!] Pass --port (COM5, /dev/ttyACM0) or --port sim.')

    run_name = f'{"sim" if args.port == "sim" else board}_{datetime.now():%Y%m%d_%H%M%S}'
    out = os.path.join(pkg, 'results', run_name)
    os.makedirs(out, exist_ok=True)

    print('=' * 62)
    print(f'STREAM TEST -- {manifest["name"]} on {board} ({args.port})')
    print('=' * 62)

    metrics = {}
    for k, variant in enumerate(args.variants):
        print(f'\n[{k + 1}/{len(args.variants)}] Variant "{variant}"')
        memory = flash_variant(variant, pkg, manifest, args, fqbn)
        if args.port == 'sim':
            link = LoopbackLink(manifest, variant,
                                scenario[f'expected_{variant}'].to_numpy(int))
        else:
            link = SerialLink(args.port, baud, args.boot_delay)
        try:
            run = stream_variant(link, variant, manifest, scenario, args)
        finally:
            link.close()
        pd.DataFrame(run['samples']).to_csv(os.path.join(out, f'{variant}_samples.csv'),
                                            index=False)
        metrics[variant] = variant_metrics(variant, run, manifest, memory)
        with open(os.path.join(out, f'{variant}.json'), 'w', encoding='utf-8') as f:
            json.dump({'metrics': metrics[variant], 'episodes': run['episodes'],
                       'errors': run['errors']}, f, indent=2, default=str)

    meta = {'run': run_name, 'board': board, 'fqbn': fqbn, 'port': args.port,
            'host': platform.platform(), 'period_ms': args.period_ms or manifest['period_ms'],
            'loopback': args.port == 'sim'}
    with open(os.path.join(out, 'run.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    return 0 if write_comparison(out, manifest, metrics, meta) else 1


# ==============================================================================
# 6. REPORT
# ==============================================================================

ROWS = [
    ('Rules', lambda m: m['rules'], '{}'),
    ('Flash measured (bytes)', lambda m: (m['memory_measured'] or {}).get('flash_bytes'), '{:,}'),
    ('SRAM measured (bytes)', lambda m: (m['memory_measured'] or {}).get('ram_bytes'), '{:,}'),
    ('Inference mean (us)', lambda m: (m['inference_us'] or {}).get('mean'), '{:.1f}'),
    ('Inference p95 (us)', lambda m: (m['inference_us'] or {}).get('p95'), '{:.1f}'),
    ('Class parity board vs Python (%)', lambda m: m['class_parity_percent'], '{:.2f}'),
    ('Accuracy vs true label (%)', lambda m: m['accuracy_percent'], '{:.2f}'),
    ('Episodes detected', lambda m: f'{m["detected"]}/{m["episodes"]}', '{}'),
    ('Detection delay mean (samples)', lambda m: m['delay_samples_mean'], '{:.2f}'),
    ('Detection delay max (samples)', lambda m: m['delay_samples_max'], '{}'),
    ('Detection delay mean (ms)', lambda m: m['delay_ms_mean'], '{:.0f}'),
    ('False alarms', lambda m: m['false_alarms'], '{}'),
    ('False alarms / 1000 normal samples', lambda m: m['false_alarms_per_1000_normal'], '{:.2f}'),
    ('Alarm parity with oracle (episodes)',
     lambda m: f'{m["alarm_parity_episodes"]}/{m["episodes"]}', '{}'),
    ('Deadline misses', lambda m: m['deadline_misses'], '{}'),
    ('Communication errors', lambda m: m['communication_errors'], '{}'),
]


# Function to format a value or n/a.
def _fmt(v, spec: str = '{:.2f}') -> str:
    """
    Format v with spec, or 'n/a' when it is None.

    Args:
        v: Value.
        spec (str): Format spec.

    Returns:
        str: Text.
    """
    return 'n/a' if v is None else spec.format(v)


# Function to write the before/after comparison of one run.
def write_comparison(out: str, manifest: Dict, metrics: Dict, meta: Dict) -> bool:
    """
    Markdown + JSON (+ PNG when matplotlib is there); print the table.

    The run passes when, for every variant, the board answered every sample,
    agrees with Python on every class and alarms exactly where the oracle says.

    Args:
        out (str): Results folder.
        manifest (dict): Package manifest.
        metrics (dict): variant -> variant_metrics().
        meta (dict): Run metadata.

    Returns:
        bool: Whether the run passed.
    """
    variants = [v for v in VARIANTS if v in metrics]
    checks = {v: {'all_answered': metrics[v]['samples_answered'] == metrics[v]['samples_sent'],
                  'class_parity_100': metrics[v]['class_parity_percent'] == 100.0,
                  'alarm_parity_all': (metrics[v]['alarm_parity_episodes']
                                       == metrics[v]['episodes'])}
              for v in variants}
    passed = all(all(c.values()) for c in checks.values())

    lines = [f'# Stream test -- {manifest["name"]} ({meta["board"]}, run {meta["run"]})', '',
             f'Model {manifest["model"]} ({manifest["algorithm"]}), '
             f'{len(manifest["features"])} features, normal class '
             f'`{manifest["classes"][manifest["normal_class"]]}`, alarm after '
             f'{manifest["confirm"]} consecutive anomalous samples, one sample every '
             f'{meta["period_ms"]} ms.', '']
    if meta.get('loopback'):
        lines += ['> **Loopback dry run**: no board, no C code was executed. Only the '
                  'protocol and the runner were tested.', '']
    lines += ['| Metric | ' + ' | '.join(v.capitalize() for v in variants) + ' |',
              '|---|' + '---:|' * len(variants)]
    for label, get, spec in ROWS:
        lines.append(f'| {label} | ' + ' | '.join(_fmt(get(metrics[v]), spec)
                                                  for v in variants) + ' |')
    lines += ['', f'**Result: {"PASS" if passed else "FAIL"}** '
              '(every sample answered, 100% class parity, alarms where the oracle says)']
    for v in variants:
        bad = [k for k, ok in checks[v].items() if not ok]
        if bad:
            lines.append(f'- {v}: failed {", ".join(bad)}')
    text = '\n'.join(lines) + '\n'

    with open(os.path.join(out, 'comparison.md'), 'w', encoding='utf-8') as f:
        f.write(text)
    with open(os.path.join(out, 'comparison.json'), 'w', encoding='utf-8') as f:
        json.dump({'meta': meta, 'checks': checks, 'passed': passed, 'metrics': metrics},
                  f, indent=2, default=str)
    plot_comparison(out, metrics, variants)
    print('\n' + text)
    print(f'  results: {out}')
    return passed


# Function to plot the per-episode detection delay of each variant.
def plot_comparison(out: str, metrics: Dict, variants: List[str]) -> None:
    """
    Bar chart of the detection delay per episode (misses drawn as 'x').

    Args:
        out (str): Results folder (reads <variant>.json).
        metrics (dict): variant -> metrics (unused, kept for symmetry).
        variants (list): Variants present.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    eps = {}
    for v in variants:
        with open(os.path.join(out, f'{v}.json'), 'r', encoding='utf-8') as f:
            eps[v] = json.load(f)['episodes']
    n = len(next(iter(eps.values())))
    width = 0.8 / len(variants)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.45), 3.4))
    for k, v in enumerate(variants):
        x = np.arange(n) + (k - (len(variants) - 1) / 2) * width
        d = [e['delay_samples'] for e in eps[v]]
        ax.bar(x, [y or 0 for y in d], width, label=v)
        miss = [i for i, y in enumerate(d) if y is None]
        ax.plot(x[miss], [0.3] * len(miss), 'x', color='red', ms=8,
                label='missed' if k == 0 and miss else None)
    ax.set_xlabel('episode')
    ax.set_ylabel('samples from onset to alarm')
    ax.set_xticks(range(n))
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'comparison.png'), dpi=130)
    plt.close(fig)


# Function to rebuild the comparison of a finished run.
def cmd_report(args) -> int:
    """
    Re-print the comparison of a run (the latest one by default).

    Args:
        args (argparse.Namespace): Options of the report command.

    Returns:
        int: Exit code (0 when the run passed).
    """
    with open(os.path.join(args.package, MANIFEST_FILE), 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    root = os.path.join(args.package, 'results')
    runs = sorted(os.listdir(root)) if os.path.isdir(root) else []
    run = args.run or (runs[-1] if runs else None)
    if not run:
        raise SystemExit('[!] No run in results/ yet.')
    out = os.path.join(root, run)
    metrics = {}
    for v in VARIANTS:
        path = os.path.join(out, f'{v}.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                metrics[v] = json.load(f)['metrics']
    with open(os.path.join(out, 'run.json'), 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return 0 if write_comparison(out, manifest, metrics, meta) else 1


# ==============================================================================
# 7. MAIN
# ==============================================================================

# Function to parse the command line.
def parse_args() -> argparse.Namespace:
    """
    Build and parse the prepare / run / report commands.

    Returns:
        argparse.Namespace: Options.
    """
    p = argparse.ArgumentParser(
        description='Streaming anomaly-detection test on Arduino/ESP32, before vs after '
                    'refinement.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Read the docstring at the top of this file for the details.')
    sub = p.add_subparsers(dest='command', required=True)

    g = sub.add_parser('prepare', help='build the test package (no board needed)')
    g.add_argument('--dataset', default='iris',
                   choices=['iris', 'wine', 'breast_cancer', 'digits'],
                   help='built-in dataset (when no CSV is given)')
    g.add_argument('--csv', default=None, help='one CSV, split 70/30')
    g.add_argument('--train-csv', default=None)
    g.add_argument('--test-csv', default=None, help='the scenario uses only these rows')
    g.add_argument('--target', default=None, help='target column (default: last column)')
    g.add_argument('--header', default='auto', choices=['auto', 'yes', 'no'])
    g.add_argument('--drop-cols', default=None, help='comma-separated columns to ignore')
    g.add_argument('--time-col', default=None,
                   help='test column with the time: episodes follow the real transitions')
    g.add_argument('--max-train-rows', type=int, default=0)
    g.add_argument('--model', default='Decision Tree',
                   choices=['Decision Tree', 'Random Forest',
                            'Gradient Boosting Decision Trees'])
    g.add_argument('--n-estimators', type=int, default=None)
    g.add_argument('--max-depth', type=int, default=6)
    g.add_argument('--model-pkl', default=None,
                   help='pickled PyRuleAnalyzer already trained and refined')
    g.add_argument('--normal-class', default=None,
                   help='label (or code) of normal operation; default: the class named '
                        'normal/benign/legitimate/good, else 0 for 0/1 labels, else the '
                        'most frequent')
    g.add_argument('--episodes', type=int, default=20)
    g.add_argument('--normal-rows', type=int, default=30, help='rows before the onset')
    g.add_argument('--anomaly-rows', type=int, default=30, help='rows from the onset on')
    g.add_argument('--confirm', type=int, default=3,
                   help='consecutive anomalous classes that raise the alarm')
    g.add_argument('--period-ms', type=int, default=100, help='one sample every N ms')
    g.add_argument('--board', default='uno', choices=sorted(BOARD_FQBN))
    g.add_argument('--baud', type=int, default=115200)
    g.add_argument('--seed', type=int, default=42)
    g.add_argument('--output-dir', default='files/stream')
    g.add_argument('--name', default='stream_model')
    g.add_argument('--keep-work', action='store_true', help='keep the intermediate files')

    g = sub.add_parser('run', help='flash and stream both variants')
    g.add_argument('--package', required=True)
    g.add_argument('--port', default=None, help='COM5, /dev/ttyACM0, or "sim" (no board)')
    g.add_argument('--board', default=None, choices=sorted(BOARD_FQBN),
                   help='override the board of the package')
    g.add_argument('--fqbn', default=None)
    g.add_argument('--baud', type=int, default=None)
    g.add_argument('--arduino-cli', default='arduino-cli')
    g.add_argument('--manual-upload', action='store_true',
                   help='pause so you can flash each sketch with the Arduino IDE')
    g.add_argument('--skip-upload', action='store_true',
                   help='do not flash (only to repeat a single --variants run)')
    g.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=VARIANTS)
    g.add_argument('--period-ms', type=int, default=None, help='override the package rate')
    g.add_argument('--boot-delay', type=float, default=2.5)
    g.add_argument('--timeout', type=float, default=3.0, help='per-reply timeout (s)')
    g.add_argument('--retries', type=int, default=2, help='resends after a bad reply')

    g = sub.add_parser('report', help='re-print the comparison of a run')
    g.add_argument('--package', required=True)
    g.add_argument('--run', default=None, help='results/<run> folder (default: latest)')

    return p.parse_args()


# Function to dispatch the command.
def main() -> int:
    """
    Run prepare, run or report.

    Returns:
        int: Exit code.
    """
    args = parse_args()
    return {'prepare': cmd_prepare, 'run': cmd_run, 'report': cmd_report}[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
