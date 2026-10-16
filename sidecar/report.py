import argparse
import glob
import json
import os
import re
import sys

PATTERNS = (
    ("gpu", "gpu "),
    ("dtype", "-> compute"),
    ("attn", "attn probe"),
    ("encode", "encoded train"),
    ("ram", "host ram"),
    ("cache", "cache train"),
    ("fill", "cache done"),
    ("budget", "training "),
    ("params", "sidecar params"),
    ("refill", "refilled"),
    ("oom", "out of memory"),
    ("fatal", "FATAL"),
    ("sample", "sample sidecar"),
)

LOG_PREFIX = r"(?:\[\d\d:\d\d:\d\dZ\]\s+)?(?:\[(\w+)\]\s+)?"
EVAL = re.compile(r"^" + LOG_PREFIX + r"eval step (\d+): base_ce ([\d.]+) sidecar_ce ([\d.]+) delta ([+-][\d.]+)")
FINAL = re.compile(r"^" + LOG_PREFIX + r"final: base_ce ([\d.]+) sidecar_ce ([\d.]+) delta ([+-][\d.]+) "
                   r"best_val_ce ([\d.]+) best_delta ([+-][\d.]+)")
RATE = re.compile(r"^" + LOG_PREFIX + r"step (\d+).*steps/s ([\d.]+) mem ([\d.]+)GB")


def digest(root, variant):
    log_path = os.path.join(root, variant, "train.log")
    if not os.path.exists(log_path):
        print(f"{variant}: no {log_path}")
        return
    oneoffs, curves, rates = {}, {}, {}
    for line in open(log_path, encoding="utf8", errors="replace"):
        line = line.rstrip()
        for key, needle in PATTERNS:
            if needle in line and key not in oneoffs:
                oneoffs[key] = line
        m = EVAL.match(line)
        if m:
            name, step, base, ce = m.group(1) or "sidecar", int(m.group(2)), float(m.group(3)), float(m.group(4))
            delta = float(m.group(5))
            curves.setdefault(name, []).append((step, ce - base))
            continue
        m = FINAL.match(line)
        if m:
            curves.setdefault((m.group(1) or "sidecar") + "!", []).append((0, float(m.group(6))))
            continue
        m = RATE.match(line)
        if m:
            rates[m.group(1) or "sidecar"] = (float(m.group(3)), float(m.group(4)))

    print(f"== {variant}")
    printed = set()
    for key, _ in PATTERNS:
        line = oneoffs.get(key)
        if line and line not in printed:
            printed.add(line)
            print(f"   {line[:150]}")
    result_path = os.path.join(root, variant, "result.json")
    if os.path.exists(result_path):
        doc = json.load(open(result_path, encoding="utf8"))
        for row in doc if isinstance(doc, list) else [doc]:
            print("   result %-8s steps %-6s delta %+.4f best_delta %+.4f params %.2fM fp %s"
                  % (row.get("name", ""), row.get("steps"), row.get("delta", float("nan")),
                     row.get("best_delta", float("nan")), (row.get("params_M") or float("nan")),
                     (row.get("val_fingerprint") or "-")[:8]))
    for name, points in curves.items():
        if name.endswith("!"):
            continue
        deltas = [d for _, d in points]
        rate = rates.get(name)
        tail = " ".join("%d:%+.3f" % (s, d) for s, d in points)
        print(f"   {name:<8} best {min(deltas):+.4f} last {deltas[-1]:+.4f} "
              f"| {tail}" + (f" | {rate[0]:.2f} steps/s {rate[1]:.1f}GB" if rate else ""))
    for name, points in curves.items():
        if name.endswith("!"):
            print(f"   {name[:-1]:<8} best_delta {points[-1][1]:+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="kaggle_out")
    ap.add_argument("variants", nargs="*")
    args = ap.parse_args()
    names = args.variants or sorted(os.path.basename(os.path.dirname(p))
                                    for p in glob.glob(os.path.join(args.root, "*", "train.log")))
    for variant in names:
        digest(args.root, variant)


if __name__ == "__main__":
    main()
