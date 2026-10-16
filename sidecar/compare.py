import argparse
import glob
import json
import os

COLUMNS = (
    ("variant", "{:<14}", "variant"),
    ("config", "{:<9}", "name"),
    ("backbone", "{:<12}", None),
    ("err", "{:>4}", None),
    ("d", "{:>5}", None),
    ("refill", "{:>7}", None),
    ("steps", "{:>7}", "steps"),
    ("base_ce", "{:>8.4f}", "base_ce"),
    ("delta", "{:>9.4f}", "delta"),
    ("best_delta", "{:>11.4f}", "best_delta"),
    ("params_M", "{:>9.2f}", "params_M"),
)


def load_results(path):
    doc = json.load(open(path, encoding="utf8"))
    return doc if isinstance(doc, list) else [doc]


def backbone_of(args):
    return str(args.get("backbone") or "").rstrip("/").split("/")[-1]


def to_row(variant, result):
    args = result.get("args") or {}
    values = {
        "variant": variant,
        "config": result.get("name") or "",
        "backbone": backbone_of(args),
        "err": args.get("use_err"),
        "d": args.get("d"),
        "refill": args.get("refill_frac"),
        "best_delta": result.get("best_delta", result.get("delta")),
    }
    for key, _, source in COLUMNS:
        if source is not None and key not in values:
            values[key] = result.get(source)
    return values


def render(rows):
    header = "  ".join(name.rjust(width_of(spec)) for name, spec, _ in COLUMNS)
    lines = [header]
    for row in rows:
        cells = []
        for key, spec, _ in COLUMNS:
            value = row.get(key)
            width = width_of(spec)
            cells.append(fmt(value, spec) if value is not None else "n/a".rjust(width))
        lines.append("  ".join(cells))
    return "\n".join(lines)


def width_of(spec):
    digits = ""
    for char in spec[2:-1]:
        if char.isdigit():
            digits += char
        elif digits:
            break
    return int(digits)


def fmt(value, spec):
    try:
        return spec.format(value)
    except (ValueError, TypeError):
        return spec.format(str(value))


def print_curve(root, variant, needle):
    path = os.path.join(root, variant, "train.log")
    if not os.path.exists(path):
        print(f"no {path}")
        return
    for line in open(path, encoding="utf8", errors="replace"):
        if needle in line:
            print(line.rstrip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="kaggle_out")
    ap.add_argument("--curve", default="")
    ap.add_argument("--needle", default="eval step")
    args = ap.parse_args()

    if args.curve:
        print_curve(args.root, args.curve, args.needle)
        return

    rows = []
    for path in sorted(glob.glob(os.path.join(args.root, "*", "result.json"))):
        variant = os.path.basename(os.path.dirname(path))
        for result in load_results(path):
            rows.append(to_row(variant, result))
    rows.sort(key=lambda r: (r.get("best_delta") is None, r.get("best_delta")))
    print(render(rows))


if __name__ == "__main__":
    main()
