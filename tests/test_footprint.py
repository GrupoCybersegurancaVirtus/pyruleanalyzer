"""
Tests for ``pyruleanalyzer.footprint``: the size of a model before training.

* The node and tree counts of the formula are those of the compiled arrays
  that the C/Arduino exports write.
* The bound of a depth is never below the size of a model trained with it.
* The deepest depth for a board is the last one whose bound fits flash and RAM,
  and the closed form agrees with the search.
"""

import os
import sys

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyruleanalyzer import PyRuleAnalyzer
from pyruleanalyzer.footprint import (
    BOARDS, Board, estimate_footprint, fit_to_board, make_board,
    max_depth_for_budget, max_leaves_for_budget,
    model_bytes, n_trees, node_bytes, probe_leaves, suggest_complexity,
)

FEATS = [f"f{i}" for i in range(6)]


def _data(n_classes=3):
    X, y = make_classification(n_samples=800, n_features=6, n_informative=4,
                               n_redundant=0, n_classes=n_classes, random_state=0)
    return X, y


def _compiled(model, X, y):
    """(trees, nodes) of the arrays the C export writes for a fitted model."""
    model.fit(X, y)
    clf = PyRuleAnalyzer.from_sklearn(model, FEATS).classifier
    clf.compile_tree_arrays(clf.initial_rules, feature_names=FEATS)
    trees = [t for t in clf._tree_arrays if 'init_score' not in t]
    return len(trees), sum(len(t['feature_idx']) for t in trees)


@pytest.mark.parametrize("algo,model,n_classes", [
    ('gbdt', GradientBoostingClassifier(n_estimators=6, max_depth=3, random_state=0), 3),
    ('gbdt', GradientBoostingClassifier(n_estimators=6, max_depth=3, random_state=0), 2),
    ('rf', RandomForestClassifier(n_estimators=5, max_depth=4, random_state=0), 3),
    ('dt', DecisionTreeClassifier(max_depth=5, random_state=0), 3),
])
def test_counts_match_the_compiled_arrays(algo, model, n_classes):
    X, y = _data(n_classes)
    trees, nodes = _compiled(model, X, y)
    n_est = getattr(model, 'n_estimators', 1)
    assert trees == n_trees(algo, n_est, n_classes)
    leaves = np.mean([t.tree_.n_leaves for t in np.ravel(getattr(model, 'estimators_', [model]))])
    est = estimate_footprint(algo, n_est, n_classes, model.max_depth, leaves_per_tree=leaves)
    assert est['nodes'] == nodes
    bound = estimate_footprint(algo, n_est, n_classes, model.max_depth)
    assert bound['bound'] and bound['model_bytes'] >= est['model_bytes']
    assert bound['model_bytes'] >= model_bytes(trees, nodes, algo, n_classes)


def test_node_bytes_follow_the_exported_types():
    assert node_bytes('gbdt') == 28           # int32 x3 + double x2
    # scikit-learn DT and RF always record a side for missing values, and the
    # exports always write it (one byte per node) unless told otherwise.
    assert node_bytes('dt') == 25             # int32 value + missing_left
    assert node_bytes('dt', missing_routing=False) == 24
    assert node_bytes('rf', n_classes=4) == 20 + 32 + 1
    assert node_bytes('rf', n_classes=4, missing_routing=False) == 52
    assert node_bytes('gbdt', missing_routing=True) == 28  # GBDT never routes NaN
    assert node_bytes('gbdt', double_bytes=4) == 20        # AVR
    with pytest.raises(ValueError):
        node_bytes('svm')


def test_bound_is_capped_by_leaf_limit_and_rows():
    a = estimate_footprint('gbdt', 10, 2, max_depth=8, max_leaf_nodes=16)
    assert a['leaves'] == 10 * 16
    b = estimate_footprint('gbdt', 10, 2, max_depth=8, n_samples=40)
    assert b['leaves'] == 10 * 40
    with pytest.raises(ValueError):
        estimate_footprint('gbdt', 10, 2)


def test_without_board_only_sizes_are_reported():
    r = suggest_complexity(n_classes=3, n_estimators=10, depths=range(1, 5))
    assert r.max_depth is None and r.board is None
    assert [row['max_depth'] for row in r.rows] == [1, 2, 3, 4]
    assert all('fits' not in row for row in r.rows)
    sizes = [row['max_bytes'] for row in r.rows]
    assert sizes == sorted(sizes)
    assert 'max_bytes' in str(r)


@pytest.mark.parametrize("n_classes", [2, 8, 15, 19])
def test_deepest_depth_fits_and_next_does_not(n_classes):
    r = suggest_complexity(n_classes=n_classes, n_estimators=50, n_features=40,
                           board='esp32_huge_app', depths=range(1, 14))
    by_depth = {row['max_depth']: row for row in r.rows}
    d = r.max_depth
    assert by_depth[d]['fits'] and not by_depth[d + 1]['fits']
    assert by_depth[d]['flash'] <= BOARDS['esp32_huge_app'].flash_bytes
    assert d == max_depth_for_budget(r.flash_budget, 'gbdt', 50, n_classes)


def test_custom_memory_and_overrides():
    small = suggest_complexity(n_classes=2, n_estimators=10, n_features=4,
                               flash_bytes=60_000, ram_bytes=10_000, depths=range(1, 10))
    large = suggest_complexity(n_classes=2, n_estimators=10, n_features=4,
                               flash_bytes=600_000, ram_bytes=10_000, depths=range(1, 10))
    assert small.max_depth < large.max_depth
    b = make_board('esp32', ram_bytes=1_000)
    assert b.flash_bytes == BOARDS['esp32'].flash_bytes and b.ram_bytes == 1_000
    r = suggest_complexity(n_classes=2, n_estimators=10, n_features=4, board=b,
                           depths=range(1, 4))
    assert r.max_depth is None  # the firmware alone exceeds 1,000 bytes of RAM
    assert make_board() is None
    with pytest.raises(ValueError):
        make_board('pdp11')


def test_avr_keeps_the_model_in_flash():
    """PROGMEM: on the Uno the model takes flash, not RAM; the RAM is the
    firmware and the feature vector whatever the model size."""
    uno = suggest_complexity(n_classes=2, n_estimators=1, n_features=4,
                             algorithm='dt', board='uno', depths=range(1, 14))
    rows = {r['max_depth']: r for r in uno.rows}
    assert rows[1]['ram'] == rows[9]['ram']
    assert rows[1]['flash'] == BOARDS['uno'].firmware_flash + rows[1]['max_bytes']
    # 20 bytes per node: depth 9 (1,023 nodes, ~20 KB) fits the 32 KB of
    # flash, depth 10 (2,047 nodes, ~41 KB) does not
    assert rows[9]['fits'] and not rows[10]['fits']
    assert uno.max_depth == 9
    # Users can still describe a toolchain that copies const data to RAM
    ram_uno = make_board('uno', constants_in_ram=True)
    r = suggest_complexity(n_classes=2, n_estimators=1, n_features=4,
                           algorithm='dt', board=ram_uno, depths=range(1, 12))
    assert r.max_depth is not None and r.max_depth < uno.max_depth


def test_avr_caps_the_nodes_of_one_tree():
    """avr-gcc refuses an array over 32,767 bytes: on the Mega one decision
    tree may not exceed 4,096 leaves (8,191 nodes of 4-byte fields) even
    though the flash would hold more."""
    from pyruleanalyzer.footprint import max_leaves_per_array
    assert max_leaves_per_array(32_767, 'dt', 2, 4) == 4096
    assert max_leaves_per_array(32_767, 'rf', 4, 4) == 1024
    assert max_leaves_per_array(None, 'dt') is None
    from pyruleanalyzer.footprint import max_leaves_for_budget
    mega = BOARDS['mega2560']
    by_flash = max_leaves_for_budget(mega.flash_bytes - mega.firmware_flash, 'dt', 1, 2,
                                     double_bytes=4)
    assert by_flash > 5_000  # the flash would hold ~6,000 leaves
    X, y = _data()
    fit = fit_to_board(X, y, board='mega2560', algorithm='dt', sample_rows=None)
    assert fit.params['max_leaf_nodes'] == 4096


def test_probe_estimates_leaves_within_the_bound():
    X, y = _data(3)
    leaves = probe_leaves(X, y, 'gbdt', max_depth=4, n_estimators=3)
    assert 1 <= leaves <= 16
    r = suggest_complexity(X, y, n_estimators=20, depths=[2, 4],
                           board=Board(flash_bytes=200_000, ram_bytes=100_000))
    for row in r.rows:
        assert row['expected_bytes'] <= row['max_bytes']
        assert 'fits_expected' in row
    assert r.max_depth_expected >= r.max_depth


def test_grid_over_trees_and_leaves():
    r = suggest_complexity(n_classes=8, n_estimators=[25, 50, 100], max_leaf_nodes=[None, 16],
                           n_features=40, board='esp32_huge_app', depths=range(3, 9))
    assert len(r.rows) == 3 * 2 * 6
    assert {row['n_estimators'] for row in r.rows} == {25, 50, 100}
    best = r.largest
    assert best['fits'] and all(best['max_bytes'] >= row['max_bytes'] for row in r.fitting())
    # fewer trees leave room for deeper ones
    deepest = {n: max(row['max_depth'] for row in r.fitting()
                      if row['n_estimators'] == n and row['max_leaf_nodes'] is None)
               for n in (25, 100)}
    assert deepest[25] > deepest[100]
    assert 'n_estimators' in str(r)


def test_max_leaves_fills_the_budget():
    budget = 3_000_000
    for n, k in [(50, 2), (50, 15), (200, 8)]:
        leaves = max_leaves_for_budget(budget, 'gbdt', n, k)
        fit = estimate_footprint('gbdt', n, k, max_leaf_nodes=leaves)['model_bytes']
        over = estimate_footprint('gbdt', n, k, max_leaf_nodes=leaves + 1)['model_bytes']
        assert fit <= budget < over
        # never fewer leaves than the depth bound allows
        assert leaves >= 2 ** max_depth_for_budget(budget, 'gbdt', n, k)
    assert max_leaves_for_budget(1_000, 'gbdt', 50, 2) is None


def test_fit_to_board_picks_a_fitting_candidate():
    X, y = _data(3)
    board = Board(flash_bytes=120_000, ram_bytes=100_000, firmware_flash=20_000)
    r = fit_to_board(X, y, board=board, n_estimators=(5, 10, 20), sample_rows=None,
                     scoring='accuracy')
    assert len(r.candidates) == 3
    assert all(c['flash'] <= board.flash_bytes for c in r.candidates)
    best = max(r.candidates, key=lambda c: c['score'])
    assert r.params['n_estimators'] == best['n_estimators']
    assert r.params['max_depth'] is None and r.params['max_leaf_nodes'] >= 2
    # the chosen parameters really give a model within the guaranteed size
    m = GradientBoostingClassifier(random_state=0, **r.params).fit(X, y)
    trees = np.ravel(m.estimators_)
    nodes = sum(t.tree_.node_count for t in trees)
    assert model_bytes(len(trees), nodes) <= r.model_budget
    assert 'best' in str(r)


def test_fit_to_board_on_avr_bounds_ram_and_needs_a_board():
    X, y = _data(2)
    r = fit_to_board(X, y, board='uno', algorithm='dt', sample_rows=None)
    (c,) = r.candidates
    assert c['ram'] <= BOARDS['uno'].ram_bytes and c['flash'] <= BOARDS['uno'].flash_bytes
    with pytest.raises(ValueError):
        fit_to_board(X, y)


def test_board_presets_follow_the_arduino_core_limits():
    # Sketch limits of each core's boards.txt (app partition for the ESP32s).
    assert (BOARDS['pico'].flash_bytes, BOARDS['pico'].ram_bytes) == (2_093_056, 262_144)
    assert (BOARDS['esp32s3'].flash_bytes, BOARDS['esp32s3'].ram_bytes) == (1_310_720, 327_680)
    assert BOARDS['esp32s3_16mb'].flash_bytes == 3_145_728
    assert BOARDS['esp32s3_32mb'].flash_bytes == 13_107_200
    # A progression of flash from the Uno to the 32 MB ESP32-S3.
    ladder = ['uno', 'mega2560', 'pico', 'esp32s3_16mb', 'esp32s3_32mb']
    sizes = [BOARDS[b].flash_bytes for b in ladder]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)
    assert make_board('pico', firmware_flash=10_000).firmware_flash == 10_000


@pytest.mark.parametrize("board", ["uno", "mega", "pico", "esp32s3_32mb", "leonardo"])
def test_arduino_export_checks_memory_against_the_board(board, tmp_path):
    X, y = _data()
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        a = PyRuleAnalyzer.from_sklearn(model, feature_names=FEATS)
        a.classifier.compile_tree_arrays(feature_names=FEATS)
        out = a.classifier.export_to_arduino_ino(filepath=str(tmp_path / "m.ino"),
                                                 board_model=board)
    expected = {"mega": (253_952, 8_192), "leonardo": (28_672, 2_560)}.get(
        board, (BOARDS[board].flash_bytes, BOARDS[board].ram_bytes) if board in BOARDS else None)
    text = open(tmp_path / "m.ino", encoding="utf-8").read()
    assert f"Target Board: {board.upper()}" in text
    mc = out["memory_check"]
    assert (mc["max_flash"], mc["max_sram"]) == expected


def test_arduino_export_refuses_an_unknown_board(tmp_path):
    X, y = _data()
    model = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        a = PyRuleAnalyzer.from_sklearn(model, feature_names=FEATS)
        a.classifier.compile_tree_arrays(feature_names=FEATS)
    with pytest.raises(ValueError, match="unknown board_model"):
        a.classifier.export_to_arduino_ino(filepath=str(tmp_path / "m.ino"), board_model="nucleo")



def _arduino_cli_with(core):
    """arduino-cli when it is on PATH with `core` installed, else skip."""
    import shutil, subprocess
    cli = shutil.which("arduino-cli")
    if cli is None:
        pytest.skip("arduino-cli not on PATH")
    cores = subprocess.run([cli, "core", "list"], capture_output=True, text=True).stdout
    if core not in cores:
        pytest.skip(f"{core} core not installed")
    return cli


def _compile_sketch(cli, fqbn, model, board_model, tmp_path, feats=FEATS):
    """Export `model` as a sketch for `board_model`, compile it for `fqbn`
    and return (sketch source, flash bytes, RAM bytes)."""
    import contextlib, io, re, subprocess
    sk = tmp_path / "pyra_model"
    sk.mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        a = PyRuleAnalyzer.from_sklearn(model, feature_names=feats)
        a.classifier.compile_tree_arrays(feature_names=feats)
        a.classifier.export_to_arduino_ino(filepath=str(sk / "pyra_model.ino"),
                                           board_model=board_model)
    cp = subprocess.run([cli, "compile", "--fqbn", fqbn, str(sk)],
                        capture_output=True, text=True)
    out = cp.stdout + cp.stderr
    assert cp.returncode == 0, out[-3000:]
    flash = int(re.search(r"Sketch uses (\d+) bytes", out).group(1))
    ram = int(re.search(r"Global variables use (\d+) bytes", out).group(1))
    return (sk / "pyra_model.ino").read_text(encoding="utf-8"), flash, ram


@pytest.mark.parametrize("kind", ["dt", "rf", "gbdt"])
@pytest.mark.parametrize("board,fqbn", [("uno", "arduino:avr:uno"),
                                        ("mega", "arduino:avr:mega:cpu=atmega2560")])
def test_arduino_sketch_compiles_for_avr(kind, board, fqbn, tmp_path):
    """avr-libc has no HUGE_VAL, and the arrays go to flash (PROGMEM): the
    sketch compiles and its RAM does not grow with the model."""
    cli = _arduino_cli_with("arduino:avr")
    X, y = _data()
    model = {"dt": DecisionTreeClassifier(max_depth=5, random_state=0),
             "rf": RandomForestClassifier(n_estimators=4, max_depth=4, random_state=0),
             "gbdt": GradientBoostingClassifier(n_estimators=4, max_depth=3, random_state=0),
             }[kind].fit(X, y)
    src, flash, ram = _compile_sketch(cli, fqbn, model, board, tmp_path)
    assert "#ifndef HUGE_VAL" in src and "PYRA_ROM =" in src
    # the arrays alone are larger than the RAM the whole sketch uses
    assert ram < BOARDS["uno"].ram_bytes
    assert flash > BOARDS["uno"].firmware_flash


def test_mega_sketch_reads_a_model_past_64_kb(tmp_path):
    """A model larger than 64 KB lies partly beyond a 16-bit pointer; the
    Mega export reads it with far addresses and still compiles."""
    cli = _arduino_cli_with("arduino:avr")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12_000, 6))
    y = rng.integers(0, 3, size=12_000)
    model = DecisionTreeClassifier(max_leaf_nodes=4000, random_state=0).fit(X, y)
    assert model.get_n_leaves() == 4000
    src, flash, ram = _compile_sketch(cli, "arduino:avr:mega:cpu=atmega2560",
                                      model, "mega", tmp_path)
    assert flash > 70_000 and ram < 2_000


def test_esp32s3_custom_partition_preset(tmp_path):
    """The 16 MB ESP32-S3 preset with one 15.9 MB app: the export writes the
    partitions.csv the core picks up, and the core accepts it."""
    import subprocess
    X, y = _data()
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    cli = _arduino_cli_with("esp32:esp32")
    src, flash, ram = _compile_sketch(
        cli, "esp32:esp32:esp32s3:FlashSize=16M,PartitionScheme=custom",
        model, "esp32s3_16mb_custom", tmp_path)
    part = (tmp_path / "pyra_model" / "partitions.csv").read_text()
    assert "0xFF0000" in part and "PartitionScheme=custom" in src
    assert BOARDS["esp32s3_16mb_custom"].flash_bytes == 0xFF0000
    assert flash < BOARDS["esp32s3_16mb_custom"].flash_bytes


def test_arduino_sketch_includes_tinyusb_when_the_core_uses_it(tmp_path):
    """The Adafruit nRF52 core links Serial only with Adafruit_TinyUSB.h."""
    import contextlib, io
    X, y = _data()
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    with contextlib.redirect_stdout(io.StringIO()):
        a = PyRuleAnalyzer.from_sklearn(model, feature_names=FEATS)
        a.classifier.compile_tree_arrays(feature_names=FEATS)
        a.classifier.export_to_arduino_ino(filepath=str(tmp_path / "m.ino"), board_model="nrf52840")
    src = (tmp_path / "m.ino").read_text(encoding="utf-8")
    assert "#if defined(USE_TINYUSB)\n#include <Adafruit_TinyUSB.h>\n#endif" in src


def test_nrf52840_sketch_compiles(tmp_path):
    cli = _arduino_cli_with("adafruit:nrf52")
    X, y = _data()
    model = GradientBoostingClassifier(n_estimators=4, max_depth=3, random_state=0).fit(X, y)
    src, flash, ram = _compile_sketch(cli, "adafruit:nrf52:feather52840", model, "nrf52840", tmp_path)
    assert flash < BOARDS["nrf52840"].flash_bytes


def _avr_bin(name):
    """An AVR binutils program of the arduino:avr core, or skip."""
    import glob
    from pathlib import Path
    hits = glob.glob(str(Path.home() / "**" / "packages" / "arduino" / "tools" / "avr-gcc"
                         / "*" / "bin" / f"{name}*"), recursive=True)
    if not hits:
        pytest.skip(f"{name} not found")
    return hits[0]


@pytest.mark.parametrize("kind", ["gbdt", "rf"])
def test_avr_tree_table_points_at_the_arrays(kind, tmp_path):
    """Read the linked Mega binary: every entry of the tree table is the
    address of the array it names (some above 64 KB), and every threshold is
    the largest float32 not above the model's threshold."""
    import contextlib, io, re, struct, subprocess
    cli = _arduino_cli_with("arduino:avr")
    nm, objcopy = _avr_bin("avr-nm"), _avr_bin("avr-objcopy")
    Xb, yb = make_classification(n_samples=3000, n_features=8, n_informative=6,
                                 n_classes=3, random_state=0)
    feats = [f"f{i}" for i in range(8)]
    model = (GradientBoostingClassifier(n_estimators=40, max_depth=4, random_state=0) if kind == "gbdt"
             else RandomForestClassifier(n_estimators=20, max_depth=6, random_state=0, n_jobs=1))
    sk, build = tmp_path / "m", tmp_path / "b"
    sk.mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        a = PyRuleAnalyzer.from_sklearn(model.fit(Xb, yb), feature_names=feats)
        clf = a.classifier
        clf.compile_tree_arrays(feature_names=feats)
        clf.export_to_arduino_ino(filepath=str(sk / "m.ino"), board_model="mega")
    cp = subprocess.run([cli, "compile", "--fqbn", "arduino:avr:mega:cpu=atmega2560",
                         "--build-path", str(build), str(sk)], capture_output=True, text=True)
    assert cp.returncode == 0, cp.stdout[-2000:] + cp.stderr[-2000:]
    sym = {}
    for line in subprocess.run([nm, str(build / "m.ino.elf")], capture_output=True,
                               text=True).stdout.splitlines():
        parts = line.split()
        if len(parts) == 3:
            sym[parts[2]] = int(parts[0], 16)
    subprocess.run([objcopy, "-O", "binary", "-j", ".text", "-j", ".data",
                    str(build / "m.ino.elf"), str(tmp_path / "f.bin")], check=True)
    img = (tmp_path / "f.bin").read_bytes()
    entries = clf._c_table_trees()
    keys = ["f", "t", "l", "r"] + (["m"] if f"pyra_t{entries[0][0]}_m" in sym else []) + ["v"]
    base = sym["pyra_tree_table"]
    for i, (t, _) in enumerate(entries):
        for j, k in enumerate(keys):
            assert struct.unpack_from("<I", img, base + 4 * (i * len(keys) + j))[0] == sym[f"pyra_t{t}_{k}"]
        for n, v in enumerate(clf._tree_arrays[t]["threshold"]):
            got = np.float32(struct.unpack_from("<f", img, sym[f"pyra_t{t}_t"] + 4 * n)[0])
            if np.isfinite(v):
                assert float(got) <= v and float(np.nextafter(got, np.float32(np.inf))) > v
    if kind == "gbdt":  # the model reaches past a 16-bit pointer
        assert max(sym[f"pyra_t{t}_v"] for t, _ in entries) > 0xFFFF
