import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OWNER = os.environ.get("KAGGLE_OWNER", "admincoas")
DATASET = f"{OWNER}/lfm-sidecar-data"


def build(variant, overrides, out_root=None):
    side = open(os.path.join(HERE, "sidecar.py"), encoding="utf8").read()
    train = open(os.path.join(HERE, "train_kaggle.py"), encoding="utf8").read()
    train = train.replace("from sidecar import Sidecar, count_params, sidecar_loss\n", "")
    side = side.replace("import math\n\nimport torch", "import torch", 1)
    lines = []
    for line in train.splitlines():
        for key, val in overrides.items():
            if f'"--{key}"' in line:
                line = re.sub(r"default=[^,)]+", f"default={val!r}", line, count=1)
        lines.append(line)
    merged = side.rstrip() + "\n\n\n" + "\n".join(lines).lstrip() + "\n"
    out = os.path.join(out_root or os.path.join(HERE, "kaggle_pkg"), f"kernel_{variant}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "train_all.py"), "w", encoding="utf8") as f:
        f.write(merged)
    meta = {
        "id": f"{OWNER}/lfm-sidecar-{variant}",
        "title": f"lfm-sidecar-{variant}",
        "code_file": "train_all.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [DATASET],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    with open(os.path.join(out, "kernel-metadata.json"), "w", encoding="utf8") as f:
        json.dump(meta, f, indent=2)
    return out


if __name__ == "__main__":
    argv = sys.argv[1:]
    out_root = None
    if "--out" in argv:
        at = argv.index("--out")
        out_root = argv[at + 1]
        del argv[at:at + 2]
    print(build(argv[0], json.loads(argv[1]) if len(argv) > 1 else {}, out_root))
