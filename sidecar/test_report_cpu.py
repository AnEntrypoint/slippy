import contextlib
import io
import json
import os
import shutil
import tempfile

import report

SWEEP_LOG = """[19:51:55Z] gpu Tesla T4 cc (7, 5) tflops fp16 21.01 bf16 2.17 -> compute torch.float16
[19:51:56Z] attn probe at cache_len 8192: default 0.42s 12.57GB -> default
[19:52:00Z] host ram 15.60GB -> 768 of 768 windows
[19:52:01Z] cache train 768x8192 val 64x8192 hidden_bytes 12.87GB
[19:53:00Z] train cache done 573s base_ce 2.6934
[19:53:01Z] training 4 config(s) on a 12000s budget = 3000s each
[19:53:02Z] sidecar params 17.73M use_err 0 read [16]
[19:54:40Z] [base] step 250 lr 2.51e-04 loss 2.7192 elapsed 169s steps/s 1.49 mem 12.57GB
[19:54:41Z] [base] eval step 400: base_ce 2.6788 sidecar_ce 2.5000 delta -0.1788
[19:55:00Z] [base] refilled 38 of 768 windows in 85s
[19:55:01Z] [base] eval step 800: base_ce 2.6788 sidecar_ce 2.4000 delta -0.2788
[19:55:02Z] [base] trained 4400 steps in 3000s plus refill 210s
[19:55:03Z] [base] final: base_ce 2.6788 sidecar_ce 2.4000 delta -0.2788 best_val_ce 2.3900 best_delta -0.2888
[20:45:40Z] [err] step 250 lr 2.51e-04 loss 2.7000 elapsed 169s steps/s 1.40 mem 12.60GB
[20:45:41Z] [err] eval step 400: base_ce 2.6788 sidecar_ce 2.4500 delta -0.2288
[20:45:42Z] [err] final: base_ce 2.6788 sidecar_ce 2.4500 delta -0.2288 best_val_ce 2.4400 best_delta -0.2388
"""

SOLO_LOG = """[19:51:55Z] gpu Tesla T4 cc (7, 5) tflops fp16 21.01 bf16 2.17 -> compute torch.float16
[19:54:40Z] step 250 lr 2.51e-04 loss 2.7192 elapsed 169s steps/s 1.49 mem 12.57GB
[19:55:00Z] eval step 0: base_ce 2.6788 sidecar_ce 2.6788 (must match base: correction starts at zero)
[19:55:01Z] eval step 1000: base_ce 2.6788 sidecar_ce 2.6000 delta -0.0788
[19:55:02Z] final: base_ce 2.6788 sidecar_ce 2.6000 delta -0.0788 best_val_ce 2.5900 best_delta -0.0888
"""

SWEEP_RESULTS = [{"name": "base", "steps": 4400, "delta": -0.2788, "best_delta": -0.2888,
                  "params_M": 17.73, "val_fingerprint": "0123456789abcdef" * 2 + "01234567"},
                 {"name": "err", "steps": 4100, "delta": -0.2288, "best_delta": -0.2388, "params_M": 17.73}]


def run(root, variant):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report.digest(root, variant)
    return buf.getvalue()


def write(root, variant, text, results=None):
    path = os.path.join(root, variant)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "train.log"), "w", encoding="utf8") as f:
        f.write(text)
    if results is not None:
        with open(os.path.join(path, "result.json"), "w", encoding="utf8") as f:
            json.dump(results, f)
    return path


def test_sweep_curves():
    root = tempfile.mkdtemp()
    try:
        write(root, "v2sweep", SWEEP_LOG, SWEEP_RESULTS)
        out = run(root, "v2sweep")
        assert "base     best -0.2788 last -0.2788 | 400:-0.179 800:-0.279" in out, out
        assert "err      best -0.2288 last -0.2288 | 400:-0.229" in out, out
        assert "base     best_delta -0.2888" in out, out
        assert "1.49 steps/s 12.6GB" in out, out
        assert "1.40 steps/s 12.6GB" in out, out
        assert "result base     steps 4400" in out and "result err      steps 4100" in out, out
        assert "fp 01234567" in out and "fp -" in out, out
        assert "attn probe" in out and "host ram" in out and "refilled" in out, out
        print("report: sweep log yields one curve per config with rate, best_delta, result and fp rows")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_solo_run_without_name_prefix():
    root = tempfile.mkdtemp()
    try:
        write(root, "v2base", SOLO_LOG)
        out = run(root, "v2base")
        assert "sidecar  best -0.0788 last -0.0788 | 1000:-0.079" in out, out
        assert "sidecar  best_delta -0.0888" in out, out
        assert "1.49 steps/s 12.6GB" in out, out
        print("report: a single-config log parses as sidecar despite no [name] tag")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_missing_variant():
    root = tempfile.mkdtemp()
    try:
        out = run(root, "nope")
        assert out.startswith("nope: no "), out
        print("report: a missing variant is named, not crashed on")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_sweep_curves()
    test_solo_run_without_name_prefix()
    test_missing_variant()
