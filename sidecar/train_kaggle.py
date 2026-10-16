import argparse
import builtins
import json
import math
import os
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sidecar import Sidecar, count_params, sidecar_loss

BACKBONE = "LiquidAI/LFM2-350M"
LOG_FILE = None


def log(msg):
    line = f"[{time.strftime('%H:%M:%S', time.gmtime())}Z] {msg}"
    builtins.print(line, flush=True)
    if LOG_FILE is not None:
        LOG_FILE.write(line + "\n")
        LOG_FILE.flush()


def load_hf_token():
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
    return os.environ.get("HF_TOKEN")


def resolve_input(path):
    if os.path.exists(path):
        return path
    name = os.path.basename(path)
    for root, _, files in os.walk("/kaggle/input"):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(f"{name} not found under /kaggle/input")


def load_rows(path):
    with open(resolve_input(path), encoding="utf8") as f:
        return [json.loads(line)["messages"] for line in f]


def encode_rows(tok, rows):
    out = []
    for messages in rows:
        ids, token_is_assistant = [], []
        for m in messages:
            t = tok(f"<|{m['role']}|>\n{m['content']}\n", add_special_tokens=False)["input_ids"]
            ids += t
            token_is_assistant += [m["role"] == "assistant"] * len(t)
        out.append((ids, token_is_assistant))
    return out


def sample_window(seqs, length, rng, mask):
    """Return (ids, weights): ids has length+1 tokens; weights[i] weights target ids[i+1]."""
    ids, token_is_assistant = rng.choice(seqs)
    n = len(ids)
    if n <= length:
        ids = ids + [ids[-1]] * (length + 1 - n)
        is_real_token = [1] * n + [0] * (length + 1 - n)
        token_is_assistant = token_is_assistant + [False] * (length + 1 - n)
    else:
        s = rng.randrange(0, n - length)
        ids = ids[s:s + length + 1]
        is_real_token = [1] * (length + 1)
        token_is_assistant = token_is_assistant[s:s + length + 1]
    if mask == "assistant":
        weights = [int(is_assistant and is_real) for is_assistant, is_real in zip(token_is_assistant, is_real_token)]
    else:
        weights = is_real_token
    return ids, weights[1:]


def phase_length(step, total, phases):
    cumulative_fraction = 0.0
    for length, frac in phases:
        cumulative_fraction += frac
        if step < cumulative_fraction * total:
            return length
    return phases[-1][0]


def lr_at(step, total, peak, floor, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * t))


def window_tensors(win, device):
    ids, weights = win
    x = torch.tensor([ids[:-1]], device=device)
    y = torch.tensor([ids[1:]], device=device)
    w = torch.tensor([weights], dtype=torch.float32, device=device)
    return x, y, w


@torch.no_grad()
def evaluate(backbone, sidecar, val_windows, device):
    base_ce_sum, corrected_ce_sum, window_count = 0.0, 0.0, 0
    for win in val_windows:
        x, y, w = window_tensors(win, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hid = backbone.model(x).last_hidden_state
        base_ce_sum += sidecar_loss(None, hid, backbone.lm_head, y, w).item()
        corrected_ce_sum += sidecar_loss(sidecar, hid, backbone.lm_head, y, w).item()
        window_count += 1
    return base_ce_sum / window_count, corrected_ce_sum / window_count


def save(sidecar, out_dir, step, args):
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"step": step, "args": vars(args), "state_dict": sidecar.state_dict()}, os.path.join(out_dir, "sidecar.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/kaggle/input/lfm-sidecar-data/train_messages.jsonl")
    ap.add_argument("--val", default="/kaggle/input/lfm-sidecar-data/val_messages.jsonl")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--peak_lr", type=float, default=3e-4)
    ap.add_argument("--floor_lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--loss_mask", choices=["all", "assistant"], default="all")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--val_windows", type=int, default=16)
    ap.add_argument("--wall_budget_s", type=float, default=2.5 * 3600)
    ap.add_argument("--save_margin_s", type=float, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    global LOG_FILE
    sys.stdout.reconfigure(line_buffering=True)
    os.makedirs(args.out, exist_ok=True)
    LOG_FILE = open(os.path.join(args.out, "train.log"), "a", encoding="utf8")
    log("args " + json.dumps(vars(args)))
    log("hf token: " + ("set" if load_hf_token() else "absent (anonymous, slower downloads)"))
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    backbone = AutoModelForCausalLM.from_pretrained(BACKBONE, torch_dtype=torch.bfloat16).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    train_seqs = encode_rows(tok, load_rows(args.train))
    val_seqs = encode_rows(tok, load_rows(args.val))
    val_windows = [sample_window(val_seqs, 8192, random.Random(1000 + i), args.loss_mask) for i in range(args.val_windows)]

    sidecar = Sidecar(d_model=backbone.config.hidden_size, d=args.d, layers=args.layers, rank=args.rank).to(device)
    log(f"sidecar params: {count_params(sidecar) / 1e6:.2f}M loss_mask={args.loss_mask}")
    opt = torch.optim.AdamW(sidecar.parameters(), lr=args.peak_lr, betas=(0.9, 0.95), weight_decay=0.05)

    length_curriculum = [(2048, 0.15), (8192, 0.35), (16384, 0.50)]
    deadline = time.time() + args.wall_budget_s - args.save_margin_s
    start = time.time()
    last_saved = 0
    step = -1

    for step in range(args.steps):
        if time.time() > deadline:
            log(f"wall budget reached at step {step}, stopping")
            break
        length = phase_length(step, args.steps, length_curriculum)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.peak_lr, args.floor_lr, args.warmup)

        sidecar.train()
        x, y, w = window_tensors(sample_window(train_seqs, length, rng, args.loss_mask), device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hid = backbone.model(x).last_hidden_state
        loss = sidecar_loss(sidecar, hid, backbone.lm_head, y, w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sidecar.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % 10 == 0:
            elapsed = time.time() - start
            mem = torch.cuda.max_memory_allocated() / 1e9
            log(f"step {step} len {length} lr {opt.param_groups[0]['lr']:.2e} loss {loss.item():.4f} "
                f"elapsed {elapsed:.0f}s steps/s {(step + 1) / max(elapsed, 1e-6):.2f} peak_mem {mem:.2f}GB")
        if (step + 1) % args.eval_every == 0:
            sidecar.eval()
            base, corr = evaluate(backbone, sidecar, val_windows, device)
            log(f"eval step {step + 1}: base_ce {base:.4f} sidecar_ce {corr:.4f} delta {corr - base:+.4f}")
            save(sidecar, args.out, step + 1, args)
            last_saved = step + 1

    sidecar.eval()
    base, corr = evaluate(backbone, sidecar, val_windows, device)
    log(f"final: base_ce {base:.4f} sidecar_ce {corr:.4f} delta {corr - base:+.4f}")
    save(sidecar, args.out, step + 1, args)
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump({"loss_mask": args.loss_mask, "base_ce": base, "sidecar_ce": corr, "delta": corr - base,
                   "last_eval_step": last_saved, "params_M": count_params(sidecar) / 1e6, "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()
