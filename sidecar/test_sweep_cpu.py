import argparse
import json
import math
import os
import random
import shutil
import tempfile

import torch

import train2

DEFAULTS = {"train": "t.jsonl", "val": "v.jsonl", "out": "out", "steps": 1000,
            "peak_lr": 3e-4, "floor_lr": 1e-5, "warmup": 300, "d": 512, "layers": 4,
            "rank": 256, "read_layers": "", "use_err": 0, "loss_mask": "all",
            "cache_len": 8192, "cache_tokens": 2097152, "cache_ram_frac": 0.6,
            "cache_reserve_gb": 4.0, "refill_frac": 0.05, "pack": 1, "val_windows": 16,
            "loss_chunk": 4096, "eval_every": 1000, "wall_budget_s": 18000.0,
            "save_margin_s": 900.0, "seed": 0, "backbone": "LiquidAI/LFM2-350M",
            "sweep": ""}

WAVE2 = [{"name": "base"}, {"name": "err", "use_err": 1},
         {"name": "small", "d": 256}, {"name": "data", "refill_frac": 0.25}]


def fake_args(sweep):
    return argparse.Namespace(**dict(DEFAULTS, sweep=sweep))


def rejects(sweep, needle):
    try:
        train2.build_configs(fake_args(sweep))
    except ValueError as exc:
        assert needle in str(exc), (needle, str(exc))
        return str(exc)
    raise AssertionError(f"sweep {sweep} should have been rejected")


def test_build_configs():
    assert train2.build_configs(fake_args("")) == [("sidecar", {})]
    got = train2.build_configs(fake_args(WAVE2))
    assert got == [("base", {}), ("err", {"use_err": 1}), ("small", {"d": 256}),
                   ("data", {"refill_frac": 0.25})], got
    unnamed = train2.build_configs(fake_args([{"use_err": 1}, {"d": 64}]))
    assert [n for n, _ in unnamed] == ["cfg0", "cfg1"], unnamed
    print(f"build_configs: {len(got)} named configs, unnamed fallback cfg0/cfg1")


def raising_fill(exc):
    def fake(cache, backbone, seqs, idxs, *rest, **kwargs):
        raise exc
    return fake


def test_fill_oom_fallback_shrinks(monkeypatch):
    cache = train2.alloc_cache(32, 8, 4, 1, torch.float32)
    seen = []

    def fake(cache_, backbone, seqs, idxs, *rest, **kwargs):
        n = len(list(idxs))
        seen.append(n)
        if n > 16:
            raise RuntimeError("Resource exhausted: attempting to allocate 4.00GiB")
        for i in range(n):
            cache_["base_ce"][i] = 1.0

    monkeypatch.setattr(train2, "fill_cache", fake)
    got = train2.fill_windows_with_oom_fallback(cache, None, None, 8, random.Random(0), "all",
                                                1, torch.device("cpu"), [1])
    assert seen == [32, 16], seen
    assert got == 16 and cache["windows"] == 16, (got, cache["windows"])
    assert len(cache["base_ce"]) == 16 and sum(cache["base_ce"]) == 16.0, cache["base_ce"]
    print(f"fill oom fallback: attempts {seen} -> {got} windows")


def test_fill_oom_fallback_reraises_other_errors(monkeypatch):
    cache = train2.alloc_cache(32, 8, 4, 1, torch.float32)
    monkeypatch.setattr(train2, "fill_cache", raising_fill(ValueError("tokenizer broke")))
    try:
        train2.fill_windows_with_oom_fallback(cache, None, None, 8, random.Random(0), "all",
                                              1, torch.device("cpu"), [1])
    except ValueError as exc:
        assert "tokenizer broke" in str(exc), exc
    else:
        raise AssertionError("a non-oom failure during the initial fill must not be swallowed")
    assert cache["windows"] == 32, cache["windows"]


def test_config_args_applies_overrides():
    args = train2.config_args(fake_args(""), {"use_err": 1, "d": 256})
    assert (args.use_err, args.d) == (1, 256), (args.use_err, args.d)
    assert args.backbone == DEFAULTS["backbone"], args.backbone
    assert args.refill_frac == DEFAULTS["refill_frac"], args.refill_frac
    print("config_args: an override of an existing key wins, untouched defaults survive")


def test_fixed_and_unknown_keys():
    for key in ("val_windows", "backbone", "cache_len", "loss_mask", "seed", "pack"):
        message = rejects([{"name": "x", key: 1}], "fixed keys")
        assert key in message, (key, message)
    print("sweep rejects overrides of " + ", ".join(
        k for k in ("val_windows", "backbone", "cache_len", "loss_mask", "seed", "pack")))
    message = rejects([{"name": "x", "nope": 1}], "unknown keys")
    assert "nope" in message
    for bad in ("{}", "[1]", "[]"):
        try:
            train2.build_configs(fake_args(bad))
        except ValueError:
            continue
        raise AssertionError(f"sweep {bad} should have been rejected")
    print("sweep rejects unknown keys and non-object sweep specs")


def test_lr_schedule():
    peak, floor, warmup = 3e-4, 1e-5, 300
    assert train2.lr_at(0, 0.0, peak, floor, warmup) == peak / warmup
    assert abs(train2.lr_at(warmup - 1, 0.0, peak, floor, warmup) - peak) < 1e-12
    assert abs(train2.lr_at(warmup, 0.0, peak, floor, warmup) - peak) < 1e-12
    assert abs(train2.lr_at(warmup, 1.0, peak, floor, warmup) - floor) < 1e-12
    clock_only = train2.lr_at(warmup + 10, 1.0, peak, floor, warmup)
    assert abs(clock_only - floor) < 1e-12, clock_only
    ramping = train2.lr_at(10, 1.0, peak, floor, warmup)
    assert abs(ramping - peak * 11 / warmup) < 1e-12, ramping
    curve = [train2.lr_at(1000, p / 10, peak, floor, warmup) for p in range(11)]
    assert all(b <= a + 1e-12 for a, b in zip(curve, curve[1:])), curve
    assert abs(curve[0] - peak) < 1e-12 and abs(curve[-1] - floor) < 1e-12
    print(f"lr_at: warmup ramps to {peak:.1e}, cosine {curve[0]:.1e} -> {curve[-1]:.1e}, "
          f"clock-only progress 1.0 lands on floor {clock_only:.1e}, "
          f"warmup outranks a finished clock ({ramping:.1e} at step 10)")


def test_crash_rescue_keeps_earlier_configs():
    out = tempfile.mkdtemp()
    try:
        train2.RESULTS.clear()
        train2.RESULTS.append({"name": "base", "delta": -0.10})
        train2.RESULTS.append({"name": "err", "delta": -0.20})
        train2.CRASH_STATE.clear()
        train2.CRASH_STATE.update(args=argparse.Namespace(out=out), step=900, prefix="data")
        train2.rescue_after_crash()
        doc = json.load(open(os.path.join(out, "result.json")))
        assert [r["name"] for r in doc] == ["base", "err"], doc
        print(f"crash rescue: {len(doc)} configs survive a config-3 crash")

        train2.RESULTS.clear()
        train2.RESULTS.append({"name": "base", "delta": -0.10})
        train2.CRASH_STATE.clear()
        train2.CRASH_STATE.update(args=argparse.Namespace(out=out))
        train2.rescue_after_crash()
        solo = json.load(open(os.path.join(out, "result.json")))
        assert isinstance(solo, dict) and solo["name"] == "base", solo
        print("crash rescue: a single config is written as a dict, not a 1-list")
    finally:
        shutil.rmtree(out, ignore_errors=True)
        train2.RESULTS.clear()
        train2.CRASH_STATE.clear()


def test_budget_split():
    args = fake_args(WAVE2)
    budget = max(args.wall_budget_s - args.save_margin_s, 1.0) / len(WAVE2)
    assert abs(budget - (18000 - 900) / 4) < 1e-9, budget
    print(f"budget split: {args.wall_budget_s:.0f}s minus {args.save_margin_s:.0f}s "
          f"over {len(WAVE2)} configs = {budget:.0f}s each")


if __name__ == "__main__":
    test_build_configs()
    test_config_args_applies_overrides()
    test_fixed_and_unknown_keys()
    test_lr_schedule()
    test_crash_rescue_keeps_earlier_configs()
    test_budget_split()
