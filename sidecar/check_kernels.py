import ast
import glob
import os

KEYS = ("use_err", "read_layers", "cache_tokens", "refill_frac", "backbone", "wall_budget_s", "val_windows")

for path in sorted(glob.glob(os.path.join("kaggle_pkg", "kernel_*", "train_all.py"))):
    src = open(path, encoding="utf8").read()
    ast.parse(src)
    variant = os.path.basename(os.path.dirname(path))
    hit = []
    for line in src.splitlines():
        if "add_argument" in line and any('"--%s"' % k in line for k in KEYS):
            hit.append(line.strip()[len("ap.add_argument("):])
    print(variant, " ".join(hit))
    meta = open(os.path.join(os.path.dirname(path), "kernel-metadata.json"), encoding="utf8").read()
    print("   ", meta.replace("\n", " ").replace("  ", ""))
