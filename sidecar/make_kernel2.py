import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OWNER = os.environ.get("KAGGLE_OWNER", "admincoas")
DATASET = f"{OWNER}/lfm-sidecar-data"
ENV_PATH = os.path.join(HERE, "..", ".env")
TOKEN_PLACEHOLDER = 'EMBEDDED_HF_TOKEN = ""'


def env_hf_token(path=ENV_PATH):
    try:
        with open(path, encoding="utf8") as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    return line.strip().split("=", 1)[1]
    except FileNotFoundError:
        pass
    return ""


def build(variant, overrides, out_root=None, accelerator="gpu"):
    side = open(os.path.join(HERE, "sidecar2.py"), encoding="utf8").read()
    accel = open(os.path.join(HERE, "accel.py"), encoding="utf8").read()
    train = open(os.path.join(HERE, "train2.py"), encoding="utf8").read()
    train = train.replace("from sidecar2 import Sidecar, count_params, sidecar_loss\n", "")
    token = env_hf_token()
    if token:
        if train.count(TOKEN_PLACEHOLDER) != 1:
            raise RuntimeError("train2.py must contain the token placeholder exactly once")
        train = train.replace(TOKEN_PLACEHOLDER, f"EMBEDDED_HF_TOKEN = {token!r}")
    train = re.sub(r"^from accel import [^\n]*\n", "", train, flags=re.M)
    side = side.replace("import torch\nimport torch.nn as nn", "import torch\nimport torch.nn as nn")
    if accelerator not in ("gpu", "tpu"):
        raise ValueError(f"accelerator must be gpu or tpu, got {accelerator!r}")
    overrides = dict(overrides)
    overrides.setdefault("accelerator", accelerator)
    lines = []
    for line in train.splitlines():
        for key, val in overrides.items():
            if f'"--{key}"' in line:
                line = re.sub(r"default=[^,)]+", f"default={val!r}", line, count=1)
        lines.append(line)
    train = "\n".join(lines)
    train = train.replace('if __name__ == "__main__":', "if True:")
    merged = side.rstrip() + "\n\n\n" + accel.lstrip() + "\n\n\n" + train.lstrip() + "\n"
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
        "enable_gpu": accelerator == "gpu",
        "enable_tpu": accelerator == "tpu",
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
    accelerator = "gpu"
    if "--out" in argv:
        at = argv.index("--out")
        out_root = argv[at + 1]
        del argv[at:at + 2]
    if "--tpu" in argv:
        argv.remove("--tpu")
        accelerator = "tpu"
    print(build(argv[0], json.loads(argv[1]) if len(argv) > 1 else {}, out_root, accelerator))
