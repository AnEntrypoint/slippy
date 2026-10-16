import argparse
import ast
import glob
import json
import os

KEYS = ("backbone", "cache_len", "cache_tokens", "eval_every", "val_windows", "wall_budget_s",
        "save_margin_s", "pack", "sweep", "d", "layers", "rank", "refill_frac", "seed",
        "loss_mask", "read_layers", "cache_ram_frac", "cache_reserve_gb", "steps")


def defaults_of(path):
    src = open(path, encoding="utf8").read()
    ast.parse(src)
    tree = ast.parse(src)
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        name = str(node.args[0].value).lstrip("-")
        for kw in node.keywords:
            if kw.arg == "default":
                try:
                    out[name] = ast.literal_eval(kw.value)
                except ValueError:
                    out[name] = ast.unparse(kw.value)
    return out, src, tree


def sweep_of(src, tree, defaults):
    parts = [ast.get_source_segment(src, n) for n in tree.body
             if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "SWEEP_FIXED" for t in n.targets)]
    parts += [ast.get_source_segment(src, n) for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "build_configs"]
    ns = {"json": json, "argparse": argparse}
    parts = [p for p in parts if p]
    if not parts:
        return None
    exec("\n".join(parts), ns)
    return ns["build_configs"](argparse.Namespace(**defaults))


def main():
    for path in sorted(glob.glob(os.path.join("kaggle_pkg", "kernel_*", "train_all.py"))):
        variant = os.path.basename(os.path.dirname(path))
        defaults, src, tree = defaults_of(path)
        meta = json.load(open(os.path.join(os.path.dirname(path), "kernel-metadata.json"), encoding="utf8"))
        configs = sweep_of(src, tree, defaults)
        if configs is None:
            print(f"{variant}  {meta['id']}  legacy (no sweep support)")
            continue
        budget = max(defaults["wall_budget_s"] - defaults["save_margin_s"], 1.0) / len(configs)
        wanted = max(8, int(defaults["cache_tokens"] / (defaults["cache_len"] * 1)))
        print(f"{variant}  {meta['id']}")
        print(f"   backbone {defaults['backbone']}  pack {defaults['pack']}  mask {defaults['loss_mask']}")
        print(f"   budget {budget:.0f}s x {len(configs)}  eval_every {defaults['eval_every']} "
              f"val_windows {defaults['val_windows']}  cache {wanted}x{defaults['cache_len']}")
        print(f"   configs {[(n, o) for n, o in configs]}")
        for key in KEYS:
            if key not in defaults:
                print(f"   MISSING default {key}")
        for name, overrides in configs:
            clash = sorted(set(overrides) & {"pack", "backbone", "cache_len", "cache_tokens", "read_layers"})
            if clash:
                print(f"   {name} overrides fixed {clash}")


if __name__ == "__main__":
    main()
