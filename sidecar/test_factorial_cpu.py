import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

import factorial

WAVE2 = [{"name": "base", "delta": -0.2788, "best_delta": -0.2888, "steps": 4400},
         {"name": "err", "delta": -0.2288, "best_delta": -0.2388, "steps": 4100},
         {"name": "small", "delta": -0.2500, "best_delta": -0.2600, "steps": 4600},
         {"name": "data", "delta": -0.3000, "best_delta": -0.3100, "steps": 3000}]

SHIFTED = [dict(r, delta=r["delta"] - 0.0500, best_delta=r["best_delta"] - 0.0500) for r in WAVE2]


def write(root, variant, rows):
    path = os.path.join(root, variant)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "result.json"), "w", encoding="utf8") as f:
        json.dump(rows, f)
    return path


def run(root, *variants):
    buf = io.StringIO()
    argv = sys.argv
    sys.argv = ["factorial.py", "--root", root, *variants]
    try:
        with contextlib.redirect_stdout(buf):
            factorial.main()
    finally:
        sys.argv = argv
    return buf.getvalue()


def test_two_backbones_replicate():
    root = tempfile.mkdtemp()
    try:
        write(root, "v2sweep", WAVE2)
        write(root, "v25sweep", SHIFTED)
        out = run(root, "v2sweep", "v25sweep")
        rows = {l.split()[0]: l for l in out.split("== best_delta")[0].splitlines()
                if l[:8].strip() in ("base", "err", "small", "data")}
        for want in ("-0.2788", "-0.3288", "-0.3038", "+0.0000", "+0.0500"):
            assert want in rows["base"], (want, rows["base"])
        for want in ("-0.3000", "-0.3500", "-0.3250", "-0.0212"):
            assert want in rows["data"], (want, rows["data"])
        assert "backbone effect +0.0500 mean, worst config deviates 0.0000 -> replicates" in out, out
        assert "steps [3000, 4100, 4400, 4600]" in out, out
        print("factorial: two backbones, constant effect -> replicates, mean and vs-base columns")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_missing_config_is_nan_not_zero():
    root = tempfile.mkdtemp()
    try:
        write(root, "v2sweep", WAVE2)
        write(root, "v25sweep", [r for r in SHIFTED if r["name"] != "data"])
        out = run(root, "v2sweep", "v25sweep")
        row = [line for line in out.splitlines() if line.startswith("data")]
        assert row and "n/a" in row[0], row
        assert row[0].strip().endswith("n/a"), row
        assert "backbone effect" in out and "replicates" in out, out
        print("factorial: a config missing on one backbone reads n/a, not 0, and the mean still lands")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_single_variant_has_no_backbone_effect():
    root = tempfile.mkdtemp()
    try:
        write(root, "v2sweep", WAVE2)
        out = run(root, "v2sweep")
        assert "backbone effect" not in out, out
        assert "-0.2788" in out and "-0.3000" in out
        print("factorial: one variant prints the table without a backbone comparison")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_dict_result_json():
    root = tempfile.mkdtemp()
    try:
        write(root, "v2base", {"name": "sidecar", "delta": -0.1000, "best_delta": -0.1100, "steps": 900})
        out = run(root, "v2base")
        assert "sidecar " in out and "-0.1000" in out and "-0.1100" in out, out
        print("factorial: a single-config result.json dict is accepted as one row")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def window_rows(per_config):
    return [dict(r, base_window_ce=[2.0] * 8, val_window_ce=[2.0 - d for d in deltas])
            for r, deltas in zip(WAVE2, per_config)]


def test_paired_per_window():
    root = tempfile.mkdtemp()
    try:
        jitter = [0.08 if i % 2 == 0 else 0.12 for i in range(8)]
        write(root, "v2sweep", window_rows([[0.0] * 8, jitter, [0.05] * 8, [0.05] * 8]))
        write(root, "v25sweep", window_rows([[0.05] * 8] * 4))
        out = run(root, "v2sweep", "v25sweep")
        section = out.split("== paired per-window")[1]
        within = [l for l in section.splitlines() if " - base" in l and " err " in l]
        assert "+0.1000 +- 0.0076" in within[0] and "t  +13.23" in within[0] and "n 8" in within[0], within
        assert "+0.0000 +- 0.0000" in within[1], within
        cross = [l for l in section.splitlines() if "v2sweep - v25sweep" in l and " err " in l]
        assert "+0.0500 +- 0.0076" in cross[0] and "n 8" in cross[0], cross
        print("factorial: paired per-window deltas give mean +- SE and t over the shared val windows")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_paired_missing_is_nan_free():
    root = tempfile.mkdtemp()
    try:
        write(root, "v2sweep", WAVE2)
        write(root, "v25sweep", SHIFTED)
        out = run(root, "v2sweep", "v25sweep")
        section = out.split("== paired per-window")[1]
        assert section.count("n/a (no per-window ce)") == 10, section
        assert "nan" not in section.lower(), section
        print("factorial: configs without per-window ce read n/a in the paired section too")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_paired_skipped_when_val_windows_differ():
    root = tempfile.mkdtemp()
    try:
        write(root, "v2sweep", [dict(r, val_fingerprint="aa") for r in window_rows([[0.05] * 8] * 4)])
        write(root, "v25sweep", [dict(r, val_fingerprint="bb") for r in window_rows([[0.05] * 8] * 4)])
        out = run(root, "v2sweep", "v25sweep")
        section = out.split("== paired per-window")[1]
        assert "val windows differ, no cross-backbone pairing" in section, section
        assert "v2sweep - v25sweep  err" not in section, section
        assert " - base" in section, section
        print("factorial: a differing val fingerprint drops cross-backbone pairing, keeps within-variant")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_two_backbones_replicate()
    test_missing_config_is_nan_not_zero()
    test_single_variant_has_no_backbone_effect()
    test_dict_result_json()
    test_paired_per_window()
    test_paired_missing_is_nan_free()
    test_paired_skipped_when_val_windows_differ()
