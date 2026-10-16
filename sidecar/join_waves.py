import argparse
import json
import os


def rows_of(root, variant):
    path = os.path.join(root, variant, "result.json")
    doc = json.load(open(path, encoding="utf8"))
    return doc if isinstance(doc, list) else [doc]


def rename_spec(spec):
    variant, _, renames = spec.partition(":")
    mapping = {}
    for pair in renames.split(","):
        if pair:
            old, _, new = pair.partition("=")
            mapping[old] = new
    return variant, mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="kaggle_out")
    ap.add_argument("--out", required=True, help="new variant directory to write under --root")
    ap.add_argument("specs", nargs="+", help="variant[:old=new,old=new] per run to merge")
    args = ap.parse_args()
    joined = []
    for spec in args.specs:
        variant, mapping = rename_spec(spec)
        for row in rows_of(args.root, variant):
            row = dict(row)
            name = row.get("name", variant)
            row["name"] = mapping.get(name, name)
            row["source_variant"] = variant
            joined.append(row)
    outdir = os.path.join(args.root, args.out)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "result.json"), "w", encoding="utf8") as fh:
        json.dump(joined, fh, indent=2)
    for row in joined:
        print(f"   {row['source_variant']:<10} -> {row['name']}")
    print(f"wrote {os.path.join(outdir, 'result.json')} ({len(joined)} rows)")


if __name__ == "__main__":
    main()
