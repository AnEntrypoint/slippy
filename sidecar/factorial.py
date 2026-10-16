import argparse
import glob
import json
import os

METRICS = ("delta", "best_delta")


def load(root, variant):
    path = os.path.join(root, variant, "result.json")
    doc = json.load(open(path, encoding="utf8"))
    rows = doc if isinstance(doc, list) else [doc]
    return {r.get("name", variant): r for r in rows}


def cell(row, metric):
    value = row.get(metric)
    return float("nan") if value is None else float(value)


def fmt(value):
    return "    n/a" if value != value else "%+7.4f" % value


def window_deltas(row):
    return [b - c for b, c in zip(row.get("base_window_ce") or (), row.get("val_window_ce") or ())]


def paired_stats(left, right):
    n = min(len(left), len(right))
    if n < 2:
        return None
    diffs = [x - y for x, y in zip(left[:n], right[:n])]
    m = sum(diffs) / n
    se = (sum((d - m) ** 2 for d in diffs) / (n - 1) / n) ** 0.5
    return m, se, n


def print_paired(tables, names):
    print("== paired per-window")
    for variant, table in tables:
        baseline = window_deltas(table.get("base", {}))
        for name in names:
            if name == "base":
                continue
            stats = paired_stats(window_deltas(table.get(name, {})), baseline)
            if stats is None:
                print(f"   {variant:<10} {name:<7} - base    n/a (no per-window ce)")
                continue
            m, se, n = stats
            print(f"   {variant:<10} {name:<7} - base    {m:+7.4f} +- {se:.4f}"
                  f"  t {m / se if se else float('inf'):+7.2f}  n {n}")
    for (left_variant, left), (right_variant, right) in zip(tables, tables[1:]):
        if {r.get("val_fingerprint") for r in left.values()} != {r.get("val_fingerprint") for r in right.values()}:
            print(f"   {left_variant} - {right_variant}: val windows differ, no cross-backbone pairing")
            continue
        for name in names:
            stats = paired_stats(window_deltas(left.get(name, {})), window_deltas(right.get(name, {})))
            if stats is None:
                print(f"   {left_variant} - {right_variant}  {name:<7} n/a (no per-window ce)")
                continue
            m, se, n = stats
            print(f"   {left_variant} - {right_variant}  {name:<7} {m:+7.4f} +- {se:.4f}"
                  f"  t {m / se if se else float('inf'):+7.2f}  n {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="kaggle_out")
    ap.add_argument("variants", nargs="+", help="two sweep variants, one per backbone")
    args = ap.parse_args()
    tables = [(v, load(args.root, v)) for v in args.variants]
    names = []
    for _, table in tables:
        for name in table:
            if name not in names:
                names.append(name)
    for metric in METRICS:
        print(f"== {metric}")
        header = f"{'config':<8}" + "".join(f"{v:>12}" for v, _ in tables)
        print(header + f"{'mean':>12}{'vs base':>12}{'spread':>10}")
        base_mean = {}
        for name in names:
            values = [cell(t.get(name, {}), metric) for _, t in tables]
            good = [v for v in values if v == v]
            mean = sum(good) / len(good) if good else float("nan")
            base_mean[name] = mean
        baseline = base_mean.get("base", float("nan"))
        for name in names:
            values = [cell(t.get(name, {}), metric) for _, t in tables]
            mean = base_mean[name]
            spread = max(values) - min(values) if all(v == v for v in values) else float("nan")
            print(f"{name:<8}" + "".join(f"{fmt(v):>12}" for v in values)
                  + f"{fmt(mean):>12}{fmt(mean - baseline):>12}{fmt(spread):>10}")
        backbone_effect = []
        for (_, a), (_, b) in zip(tables, tables[1:]):
            for name in names:
                va, vb = cell(a.get(name, {}), metric), cell(b.get(name, {}), metric)
                if va == va and vb == vb:
                    backbone_effect.append(va - vb)
        if backbone_effect:
            mean_effect = sum(backbone_effect) / len(backbone_effect)
            deviation = max(abs(e - mean_effect) for e in backbone_effect)
            print(f"   backbone effect {mean_effect:+.4f} mean, worst config deviates {deviation:.4f}"
                  f" -> {'replicates' if deviation < abs(mean_effect) or deviation < 0.02 else 'does NOT replicate'}")
        steps = [t.get(name, {}).get("steps") for _, t in tables for name in names]
        print(f"   steps {sorted({s for s in steps if s})}")
    print_paired(tables, names)
    print("== available")
    for path in sorted(glob.glob(os.path.join(args.root, "*", "result.json"))):
        print("   " + os.path.dirname(path))


if __name__ == "__main__":
    main()
