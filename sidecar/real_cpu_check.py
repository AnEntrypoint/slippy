import argparse
import json
import os
import random
import sys
import time
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train2
from sidecar2 import Sidecar, sidecar_loss


class RealBackbone:
    """transformers 4.57.3's LFM2 has no output_hidden_states; read the post-norm final layer."""

    def __init__(self, model):
        self.model = self
        self.lm_head = model.lm_head
        self.emb = model.get_input_embeddings()
        self.config = model.config
        self.inner = model

    def __call__(self, x, output_hidden_states=False):
        last = self.inner.model(x).last_hidden_state
        return types.SimpleNamespace(hidden_states=[last] * (self.config.num_hidden_layers + 1))


def load_rows(path, n, offset=0):
    rows = []
    with open(path, encoding="utf8") as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if len(rows) >= n:
                break
            rows.append(json.loads(line)["messages"])
    return rows


def read_env_token(env_path):
    if not os.path.exists(env_path):
        return
    for line in open(env_path, encoding="utf8"):
        if line.startswith("HF_TOKEN="):
            os.environ["HF_TOKEN"] = line.strip().split("=", 1)[1]


def build_cache(shim, seqs, windows, length, seed, device):
    cache = train2.alloc_cache(windows, length, shim.config.hidden_size, 1, torch.float32)
    train2.fill_cache(cache, shim, seqs, range(windows), length, random.Random(seed), "all", True,
                      device, [shim.config.num_hidden_layers])
    return cache


def val_ce(sidecar, cache, shim, n_read, loss_chunk):
    return train2.mean(train2.evaluate_windows(sidecar, cache, "cpu", shim.emb.weight, 0, n_read, loss_chunk))


def run_config(name, cap, accum, refill, micro, shim, train_seqs, train_cache, val_cache, args):
    base_val = train2.mean(val_cache["base_ce"])
    base_train = train2.mean(train_cache["base_ce"])
    sidecar = Sidecar(d_model=shim.config.hidden_size, d=args.d, layers=args.layers, rank=args.rank,
                      n_read=1, use_err=False, corr_max_rel=cap)
    opt = torch.optim.AdamW(sidecar.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)
    n_train = train_cache["windows"]
    order = list(range(n_train))
    rng = random.Random(args.seed)
    rng.shuffle(order)
    pos = 0
    t0 = time.time()
    curve = []
    start = val_ce(sidecar, val_cache, shim, 1, args.loss_chunk)
    print("[%s] step 0 val_delta %+.4f (base %.4f) base_train %.4f"
          % (name, start - base_val, base_val, base_train), flush=True)
    done = 0
    refills = 0
    while done < micro:
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accum):
            if done >= micro:
                break
            if pos >= n_train:
                rng.shuffle(order)
                pos = 0
                if refill > 0:
                    pick = rng.sample(range(n_train), max(1, int(n_train * refill)))
                    train2.fill_cache(train_cache, shim, train_seqs, pick, args.length, rng, "all", True,
                                      "cpu", [shim.config.num_hidden_layers])
                    refills += 1
            i = order[pos]
            pos += 1
            hs, tgt, w, err = train2.window_tensors(train_cache, i, "cpu", shim.emb.weight, 0, 1)
            loss = sidecar_loss(sidecar, hs, shim.lm_head, tgt, w, err, args.loss_chunk)
            (loss / accum).backward()
            total += float(loss)
            done += 1
        torch.nn.utils.clip_grad_norm_(sidecar.parameters(), 1.0)
        opt.step()
        if done % args.eval_every == 0 or done == micro:
            v = val_ce(sidecar, val_cache, shim, 1, args.loss_chunk)
            curve.append((done, v - base_val))
            print("[%s] micro %d (%.0f exposures, %d refills) train_loss %.4f val_delta %+.4f  %.0fs"
                  % (name, done, done / n_train, refills, total / accum, v - base_val, time.time() - t0), flush=True)
    best = min(c[1] for c in curve)
    print("[%s] best val_delta %+.4f  curve %s" % (name, best, curve), flush=True)
    return best, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LiquidAI/LFM2-350M")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trainingdata", "train_messages.jsonl"))
    ap.add_argument("--env", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
    ap.add_argument("--rows", type=int, default=60)
    ap.add_argument("--val_offset", type=int, default=200)
    ap.add_argument("--train_windows", type=int, default=32)
    ap.add_argument("--val_windows", type=int, default=12)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--micro", type=int, default=640)
    ap.add_argument("--eval_every", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--loss_chunk", type=int, default=4096)
    ap.add_argument("--refill", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs", default="0,1,0;0,1,1")
    args = ap.parse_args()

    read_env_token(args.env)
    torch.set_num_threads(os.cpu_count())
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    shim = RealBackbone(model)
    train2.LM_HEAD = model.lm_head
    tok = AutoTokenizer.from_pretrained(args.model)
    train_seqs = train2.encode_rows(tok, load_rows(args.data, args.rows, 0))
    val_seqs = train2.encode_rows(tok, load_rows(args.data, args.rows, args.val_offset))
    print("train rows %d  val rows %d (offset %d)  train %d x %d  val %d x %d  threads %d"
          % (len(train_seqs), len(val_seqs), args.val_offset, args.train_windows, args.length,
             args.val_windows, args.length, torch.get_num_threads()), flush=True)

    t0 = time.time()
    train_cache = build_cache(shim, train_seqs, args.train_windows, args.length, args.seed, "cpu")
    print("train cache %.0fs base_ce %.4f" % (time.time() - t0, train2.mean(train_cache["base_ce"])), flush=True)
    t0 = time.time()
    val_cache = build_cache(shim, val_seqs, args.val_windows, args.length, args.seed + 1000, "cpu")
    print("val cache %.0fs base_ce %.4f" % (time.time() - t0, train2.mean(val_cache["base_ce"])), flush=True)

    summary = []
    for spec in args.configs.split(";"):
        cap, accum, refill = spec.split(",")
        best, curve = run_config("cap%s accum%s refill%s" % (cap, accum, refill), float(cap), int(accum),
                                 float(refill), args.micro, shim, train_seqs, train_cache, val_cache, args)
        summary.append((spec, best, curve[-1][1]))
    print("\n%12s %10s %10s" % ("config", "best", "last"))
    for spec, best, last in summary:
        print("%12s %+10.4f %+10.4f" % (spec, best, last))


if __name__ == "__main__":
    main()
