import argparse
import builtins
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from sidecar2 import Sidecar, count_params, sidecar_loss
from accel import attn_implementation, device_dtype, empty_cache, is_cuda, is_oom, is_xla, mark_step
from accel import memory_stats, model_hidden_states, optimizer_step, pick_device, reset_peak
from accel import device_label, sync, to_device

BACKBONE = "LiquidAI/LFM2-350M"
LOG_FILE = None
LM_HEAD = None
ATTN_SDPA = None
CRASH_STATE = {}
RESULTS = []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S', time.gmtime())}Z] {msg}"
    builtins.print(line, flush=True)
    if LOG_FILE is not None:
        LOG_FILE.write(line + "\n")
        LOG_FILE.flush()


def matmul_tflops(device, dtype, n=4096, reps=20):
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    for _ in range(3):
        torch.mm(a, b)
    sync(device)
    started = time.time()
    for _ in range(reps):
        torch.mm(a, b)
    sync(device)
    elapsed = time.time() - started
    del a, b
    return 2 * n * n * n * reps / max(elapsed, 1e-6) / 1e12


def pick_compute_dtype(device):
    if not is_cuda(device):
        return device_dtype(device), {"fp16": 0.0, "bf16": 0.0}
    scores = {}
    for name, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        try:
            scores[name] = matmul_tflops(device, dtype)
        except Exception:
            scores[name] = 0.0
    return (torch.bfloat16 if scores["bf16"] > scores["fp16"] else torch.float16), scores


EMBEDDED_HF_TOKEN = ""


def load_hf_token():
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        if EMBEDDED_HF_TOKEN:
            os.environ.setdefault("HF_TOKEN", EMBEDDED_HF_TOKEN)
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
    bos = tok.bos_token_id
    out = []
    for messages in rows:
        ids = [] if bos is None else [bos]
        token_is_assistant = [] if bos is None else [False]
        for m in messages:
            role = m["role"]
            head = tok(f"<|im_start|>{role}\n", add_special_tokens=False)["input_ids"]
            body = tok(m["content"], add_special_tokens=False)["input_ids"]
            tail = tok("<|im_end|>\n", add_special_tokens=False)["input_ids"]
            is_assistant = role == "assistant"
            ids += head + body + tail
            token_is_assistant += [False] * len(head) + [is_assistant] * len(body) + [is_assistant] * len(tail)
        out.append((ids, token_is_assistant))
    return out


def pack_window(seqs, length, rng, mask):
    ids, token_is_assistant = [], []
    while len(ids) < length + 1:
        row_ids, row_assistant = rng.choice(seqs)
        need = length + 1 - len(ids)
        if len(row_ids) > need:
            s = rng.randrange(0, len(row_ids) - need + 1)
            row_ids = row_ids[s:s + need]
            row_assistant = row_assistant[s:s + need]
        ids += row_ids
        token_is_assistant += row_assistant
    weights = [int(a) for a in token_is_assistant] if mask == "assistant" else [1] * len(ids)
    return ids, weights[1:]


def sample_window(seqs, length, rng, mask, pack=True):
    """Return (ids, weights): ids has length+1 tokens; weights[i] weights target ids[i+1]."""
    if pack:
        return pack_window(seqs, length, rng, mask)
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


def parse_read_layers(spec, n_layers):
    if not spec or spec.strip() == "":
        return [n_layers]
    vals = sorted({max(1, min(n_layers, int(v))) for v in spec.split(",") if v.strip()})
    if n_layers not in vals:
        vals.append(n_layers)
    return sorted(vals)


def probe_attention_backends(backbone, device, length, vocab, ce_chunk=4096):
    if not is_cuda(device):
        log(f"attn probe skipped on {device}")
        return None
    attn = getattr(torch.nn, "attention", None)
    sdpa_kernel = getattr(attn, "sdpa_kernel", None)
    backends = getattr(attn, "SDPBackend", None)
    log(f"attn probe impl {getattr(backbone.config, '_attn_implementation', 'unknown')}")
    if sdpa_kernel is None or backends is None:
        log("attn probe: torch.nn.attention unavailable")
        return None
    generator = torch.Generator(device=device).manual_seed(0)
    x = torch.randint(0, vocab, (1, length), generator=generator, device=device)
    y = torch.randint(0, vocab, (length,), generator=generator, device=device)
    lm_head = backbone.lm_head
    logit_dtype = lm_head.weight.dtype
    measured = {}

    def run(label, ctx):
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        try:
            with ctx, torch.no_grad():
                final_hidden = backbone.model(x, output_hidden_states=True).hidden_states[-1][0]
                argmax = torch.empty(length, dtype=torch.int64, device=device)
                ce_sum = 0.0
                for s in range(0, length, ce_chunk):
                    lg = lm_head(final_hidden[s:s + ce_chunk].to(logit_dtype)).float()
                    argmax[s:s + ce_chunk] = lg.argmax(-1)
                    ce_sum += F.cross_entropy(lg, y[s:s + ce_chunk], reduction="sum").item()
            torch.cuda.synchronize()
            secs = time.time() - started
            peak = torch.cuda.max_memory_allocated() / 1e9
            reference = measured.get("default  ")
            agree = 1.0 if reference is None else (argmax == reference["argmax"]).float().mean().item()
            measured[label] = {"ok": True, "secs": secs, "peak": peak,
                               "ce": ce_sum / length, "argmax": argmax}
            log(f"attn probe {label} {secs:.2f}s peak {peak:.2f}GB ce {ce_sum / length:.4f} "
                f"argmax_agree {agree:.4f}")
        except Exception as exc:
            measured[label] = {"ok": False}
            log(f"attn probe {label} failed {type(exc).__name__} {exc}")
        torch.cuda.empty_cache()

    run("default  ", contextlib.nullcontext())
    candidates = {"efficient": backends.EFFICIENT_ATTENTION, "math     ": backends.MATH}
    for label, backend in candidates.items():
        run(label, sdpa_kernel([backend]))
    del x, y
    base = measured.get("default  ")
    if base is None or not base["ok"]:
        return None
    chosen = None
    for label in candidates:
        got = measured.get(label)
        if got is None or not got["ok"]:
            continue
        if got["peak"] > base["peak"] * 0.9 or got["secs"] > base["secs"] * 1.5:
            continue
        if abs(got["ce"] - base["ce"]) > 5e-3:
            continue
        if (got["argmax"] == base["argmax"]).float().mean().item() < 0.999:
            continue
        if chosen is None or got["peak"] < measured[chosen]["peak"]:
            chosen = label
    log(f"attn probe chosen {chosen or 'default'}")
    return candidates[chosen] if chosen else None


def fill_ctx():
    attn = getattr(torch.nn, "attention", None)
    if ATTN_SDPA is None or attn is None:
        return contextlib.nullcontext()
    return attn.sdpa_kernel([ATTN_SDPA])


def total_ram_bytes():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


def clamp_cache_windows(wanted, extra_windows, length, n_read, d_model, ram_frac, reserve_bytes):
    per_window = length * n_read * d_model * 2 + length * 11
    total = total_ram_bytes()
    if total <= 0:
        return wanted, 0, per_window
    usable = int(total * ram_frac) - reserve_bytes - extra_windows * per_window
    fits = int(max(usable, per_window * 8) // per_window)
    return max(8, min(wanted, fits)), total, per_window


def alloc_cache(windows, length, d_model, n_read, dtype):
    return {
        "windows": windows,
        "hidden": [torch.empty(windows, length, d_model, dtype=dtype) for _ in range(n_read)],
        "targets": torch.empty(windows, length, dtype=torch.int32),
        "weights": torch.empty(windows, length, dtype=torch.uint8),
        "argmax": torch.empty(windows, length, dtype=torch.int32),
        "base_ce": [0.0] * windows,
        "base_token_ce": torch.empty(windows, length, dtype=torch.float16),
    }


class FillDeadline(Exception):
    def __init__(self, filled):
        super().__init__(f"fill deadline reached after {filled} windows")
        self.filled = filled


def fill_cache(cache, backbone, seqs, idxs, length, rng, mask, pack, device, read_layers, ce_chunk=4096,
               deadline=None, progress_every=64):
    lm_head = backbone.lm_head
    logit_dtype = lm_head.weight.dtype
    started = time.time()
    for done, i in enumerate(idxs):
        if deadline is not None and time.time() > deadline:
            raise FillDeadline(i)
        if progress_every and done and done % progress_every == 0:
            log(f"filled {done} windows in {time.time() - started:.0f}s ({(time.time() - started) / done:.2f}s/window)")
        ids, w = sample_window(seqs, length, rng, mask, pack)
        x = to_device(torch.tensor([ids[:-1]]), device)
        y = to_device(torch.tensor(ids[1:], dtype=torch.int64), device)
        wt = to_device(torch.tensor(w, dtype=torch.float32), device)
        with torch.no_grad(), fill_ctx():
            hs = model_hidden_states(backbone.model, x, backbone.lm_head, log)
        final_hidden = hs[-1][0].detach()
        argmax_pred = torch.empty(length, dtype=torch.int64, device=device)
        ce_chunks = []
        for s in range(0, length, ce_chunk):
            lg = lm_head(final_hidden[s:s + ce_chunk].to(logit_dtype)).float()
            ce_chunks.append(F.cross_entropy(lg, y[s:s + ce_chunk], reduction="none"))
            argmax_pred[s:s + ce_chunk] = lg.argmax(-1)
        token_ce = torch.cat(ce_chunks)
        cache["base_token_ce"][i] = token_ce.to(torch.float16).cpu()
        cache["targets"][i] = y.to(torch.int32).cpu()
        cache["weights"][i] = torch.tensor(w, dtype=torch.uint8)
        cache["argmax"][i] = argmax_pred.to(torch.int32).cpu()
        cache["base_ce"][i] = ((token_ce * wt).sum() / wt.sum().clamp(min=1.0)).item()
        cache.setdefault("block_id", {})[i] = hashlib.blake2b(
            torch.tensor(ids, dtype=torch.int32).numpy().tobytes(), digest_size=8).hexdigest()
        for read_index, layer_index in enumerate(read_layers):
            cache["hidden"][read_index][i] = hs[layer_index][0].detach().to(cache["hidden"][read_index].dtype).cpu()
    del x, y, wt, hs, final_hidden, argmax_pred


def fill_windows_with_oom_fallback(cache, backbone, seqs, length, rng, mask, pack, device,
                                   read_layers, min_windows=8, deadline=None):
    n = cache["windows"]
    while True:
        try:
            fill_cache(cache, backbone, seqs, range(n), length, rng, mask, pack, device, read_layers,
                       deadline=deadline)
            return n
        except FillDeadline as stop:
            if stop.filled < min_windows:
                raise
            n = stop.filled
            cache["windows"] = n
            cache["base_ce"] = cache["base_ce"][:n]
            empty_cache(device)
            log(f"fill deadline: training on the {n} windows filled")
            return n
        except Exception as exc:
            if not is_oom(exc) or n <= min_windows:
                raise
            n = max(min_windows, n // 2)
            empty_cache(device)
            cache["windows"] = n
            cache["base_ce"] = cache["base_ce"][:n]
            log(f"initial fill out of memory, retrying with {n} windows: {exc}")


def make_err(emb, target, argmax_pred):
    target = target.long()
    argmax_pred = argmax_pred.long()
    previous_target = torch.empty_like(target)
    previous_argmax = torch.empty_like(argmax_pred)
    previous_target[0] = target[0]
    previous_target[1:] = target[:-1]
    previous_argmax[0] = argmax_pred[0]
    previous_argmax[1:] = argmax_pred[:-1]
    err = emb[previous_target] - emb[previous_argmax]
    err[0] = 0
    return err.unsqueeze(0)


def window_tensors(cache, i, device, emb, use_err, n_read, hard_frac=1.0):
    hidden_states = [to_device(cache["hidden"][j][i], device).unsqueeze(0) for j in range(n_read)]
    targets = to_device(cache["targets"][i], device)
    weights = cache["weights"][i].float()
    if hard_frac < 1.0:
        base_ce = cache["base_token_ce"][i].float()
        k = max(1, int(hard_frac * base_ce.numel()))
        weights = weights * (base_ce >= torch.topk(base_ce, k).values.min()).float()
    weights = to_device(weights, device)
    err = make_err(emb, targets, to_device(cache["argmax"][i], device)) if use_err else None
    return hidden_states, targets, weights, err


@torch.no_grad()
def evaluate_windows(sidecar, cache, device, emb, use_err, n_read, loss_chunk):
    per_window = []
    for i in range(cache["windows"]):
        hidden_states, targets, weights, err = window_tensors(cache, i, device, emb, use_err, n_read)
        per_window.append(sidecar_loss(sidecar, hidden_states, LM_HEAD, targets, weights, err, loss_chunk))
        mark_step(device)
    return torch.stack(per_window).tolist()


def load_backbone(name, dtype, attn):
    errors = []
    for kwargs in ({"dtype": dtype, "attn_implementation": attn},
                   {"torch_dtype": dtype, "attn_implementation": attn},
                   {"dtype": dtype},
                   {"torch_dtype": dtype}):
        try:
            return AutoModelForCausalLM.from_pretrained(name, **kwargs)
        except Exception as exc:
            errors.append(f"{type(exc).__name__} {exc}")
    raise RuntimeError("backbone load failed: " + " | ".join(errors))


def mean(values):
    return sum(values) / max(len(values), 1)


def lr_at(step, progress, peak, floor, warmup):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * min(progress, 1.0)))


def save(sidecar, out_dir, step, args, name="sidecar.pt"):
    os.makedirs(out_dir, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in sidecar.state_dict().items()}
    torch.save({"step": step, "args": vars(args), "state_dict": state},
               os.path.join(out_dir, name))


def generate(backbone, sidecar, read_layers, prompt_ids, n_new, device, use_err, emb, eos_id):
    def decode(with_sidecar):
        ids = list(prompt_ids)
        logit_dtype = backbone.lm_head.weight.dtype
        for _ in range(n_new):
            x = torch.tensor([ids], device=device)
            with torch.no_grad():
                hidden = model_hidden_states(backbone.model, x, backbone.lm_head, log)
            final_hidden = hidden[-1]
            logits = backbone.lm_head(final_hidden[:, -1:].to(logit_dtype)).float()
            if with_sidecar:
                err = None
                if use_err:
                    full = backbone.lm_head(final_hidden.to(logit_dtype))
                    prediction = full.argmax(-1)[0]
                    shifted = torch.cat([prediction[:1], prediction[:-1]])
                    err = (emb[torch.tensor(ids, device=device)] - emb[shifted]).unsqueeze(0)
                    err[:, 0] = 0
                correction = sidecar([hidden[i] for i in read_layers], err)
                logits = backbone.lm_head((final_hidden[:, -1:] + correction[:, -1:]).to(logit_dtype)).float()
            nxt = int(logits[0, -1].argmax())
            ids.append(nxt)
            if nxt == eos_id:
                break
        return ids

    return decode(False), decode(True)


SWEEP_FIXED = {"train", "val", "out", "steps", "read_layers", "loss_mask", "cache_len",
               "cache_tokens", "cache_ram_frac", "cache_reserve_gb", "backbone", "seed",
               "accelerator", "sweep", "wall_budget_s", "save_margin_s", "pack", "val_windows",
               "fill_budget_s"}


def build_configs(args):
    entries = args.sweep if isinstance(args.sweep, (list, tuple)) else []
    if not entries:
        spec = (args.sweep or "").strip() if isinstance(args.sweep, str) else ""
        if not spec:
            return [("sidecar", {})]
        entries = json.loads(spec)
    if not isinstance(entries, list) or not entries:
        raise ValueError("sweep must be a non-empty JSON list of objects")
    known = set(vars(args))
    out = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"sweep entry {index} must be an object")
        entry = dict(entry)
        name = str(entry.pop("name", f"cfg{index}"))
        clash = sorted(set(entry) & SWEEP_FIXED)
        if clash:
            raise ValueError(f"sweep {name} overrides fixed keys {clash}")
        unknown = sorted(set(entry) - known)
        if unknown:
            raise ValueError(f"sweep {name} overrides unknown keys {unknown}")
        out.append((name, entry))
    return out


def config_args(base_args, overrides):
    return argparse.Namespace(**{**vars(base_args), **overrides})


def release_memory(device):
    try:
        empty_cache(device)
    except Exception as exc:
        if not is_oom(exc):
            raise
        log(f"cache release hit out of memory, continuing: {exc}")


def train_one(base_args, name, prefix, overrides, ctx, budget):
    args = config_args(base_args, overrides)
    device = ctx["device"]
    backbone = ctx["backbone"]
    emb = ctx["emb"]
    train_cache = ctx["train_cache"]
    val_cache = ctx["val_cache"]
    val_base = ctx["val_base"]
    train_seqs = ctx["train_seqs"]
    length = ctx["length"]
    n_read = ctx["n_read"]
    read_layers = ctx["read_layers"]
    n_train = train_cache["windows"]
    use_err = bool(args.use_err)
    rng = random.Random(args.seed)
    ctx["rng"] = rng
    ctx["order"] = list(range(n_train))
    rng.shuffle(ctx["order"])
    ctx["order_pos"] = 0

    CRASH_STATE.clear()
    reset_peak(device)
    sidecar = Sidecar(d_model=ctx["d_model"], d=args.d, layers=args.layers, rank=args.rank,
                      n_read=n_read, use_err=use_err, corr_max_rel=args.corr_max_rel).to(device)
    log(f"[{name}] sidecar params {count_params(sidecar) / 1e6:.2f}M use_err {args.use_err} "
        f"read {read_layers} overrides {overrides}")
    CRASH_STATE.update(sidecar=sidecar, args=args, step=0, name=name, prefix=prefix)
    opt = torch.optim.AdamW(sidecar.parameters(), lr=args.peak_lr, betas=(0.9, 0.95), weight_decay=0.05)

    deadline = time.time() + budget
    start = time.time()
    fill_overhead = 0.0
    step = -1
    steps_done = 0
    loss = None
    best_val_ce = float("inf")
    sidecar.eval()
    log(f"[{name}] eval step 0: base_ce {val_base:.4f} sidecar_ce "
        f"{mean(evaluate_windows(sidecar, val_cache, device, emb, use_err, n_read, args.loss_chunk)):.4f} "
        f"(must match base: correction starts at zero)")

    for step in range(args.steps):
        if time.time() > deadline:
            log(f"[{name}] budget reached at step {step}, stopping")
            break
        steps_done += 1
        elapsed = time.time() - start - fill_overhead
        progress = max((step + 1) / args.steps, elapsed / budget)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, progress, args.peak_lr, args.floor_lr, args.warmup)

        loss = None
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            if ctx["order_pos"] >= n_train:
                rng.shuffle(ctx["order"])
                ctx["order_pos"] = 0
                if args.refill_frac > 0:
                    m = max(1, int(n_train * args.refill_frac))
                    pick = rng.sample(range(n_train), m)
                    release_memory(device)
                    t1 = time.time()
                    try:
                        fill_cache(train_cache, backbone, train_seqs, pick, length, rng,
                                   args.loss_mask, args.pack, device, read_layers)
                        log(f"[{name}] refilled {m} windows in {time.time() - t1:.0f}s")
                    except Exception as exc:
                        if not is_oom(exc):
                            raise
                        log(f"[{name}] refill of {m} windows out of memory, keeping existing windows: {exc}")
                    release_memory(device)
                    spent = time.time() - t1
                    if fill_overhead < budget:
                        deadline += spent
                    fill_overhead += spent

            i = ctx["order"][ctx["order_pos"]]
            ctx["order_pos"] += 1
            sidecar.train()
            hidden_states, targets, weights, err = window_tensors(train_cache, i, device, emb, use_err, n_read, args.hard_frac)
            loss = sidecar_loss(sidecar, hidden_states, LM_HEAD, targets, weights, err, args.loss_chunk)
            (loss / args.accum).backward()
        optimizer_step(opt, device, sidecar.parameters(), 1.0)
        opt.zero_grad(set_to_none=True)
        mark_step(device)

        if step % 50 == 0:
            elapsed = time.time() - start - fill_overhead
            mem = memory_stats(device)
            log(f"[{name}] step {step} lr {opt.param_groups[0]['lr']:.2e} loss {loss.item():.4f} "
                f"elapsed {elapsed:.0f}s steps/s {(step + 1) / max(elapsed, 1e-6):.2f} "
                + " ".join(f"{k} {v:.2f}GB" for k, v in mem.items()))
        if (step + 1) % args.eval_every == 0:
            sidecar.eval()
            corr = mean(evaluate_windows(sidecar, val_cache, device, emb, use_err, n_read, args.loss_chunk))
            log(f"[{name}] eval step {step + 1}: base_ce {val_base:.4f} sidecar_ce {corr:.4f} "
                f"delta {corr - val_base:+.4f}")
            CRASH_STATE["step"] = step + 1
            if corr < best_val_ce:
                best_val_ce = corr
                save(sidecar, args.out, step + 1, args, name=prefix + ".pt")

    sidecar.eval()
    val_window_ce = evaluate_windows(sidecar, val_cache, device, emb, use_err, n_read, args.loss_chunk)
    corr = mean(val_window_ce)
    log(f"[{name}] final: base_ce {val_base:.4f} sidecar_ce {corr:.4f} delta {corr - val_base:+.4f} "
        f"best_val_ce {best_val_ce:.4f} best_delta {best_val_ce - val_base:+.4f}")
    log(f"[{name}] trained {steps_done} steps in {time.time() - start - fill_overhead:.0f}s "
        f"plus refill {fill_overhead:.0f}s")
    save(sidecar, args.out, steps_done, args, name=prefix + "_final.pt")
    if is_xla(device):
        log(f"[{name}] sample generation skipped on xla: one token per graph recompiles every step")
    else:
        try:
            tok = ctx["tok"]
            eos_id = tok.eos_token_id if tok.eos_token_id is not None else 7
            prompt = ctx["val_seqs"][0][0][:512]
            plain, corrected_tokens = generate(backbone, sidecar, read_layers, prompt, 200,
                                               device, use_err, emb, eos_id)
            log(f"[{name}] sample prompt tail " + repr(tok.decode(prompt[-60:])))
            log(f"[{name}] sample backbone    " + repr(tok.decode(plain[len(prompt):])))
            log(f"[{name}] sample sidecar     " + repr(tok.decode(corrected_tokens[len(prompt):])))
        except Exception:
            log(f"[{name}] sample generation failed\n" + traceback.format_exc())

    result = {"name": name, "loss_mask": args.loss_mask, "use_err": args.use_err, "read_layers": read_layers,
              "base_ce": val_base, "sidecar_ce": corr, "delta": corr - val_base,
              "best_val_ce": best_val_ce, "best_delta": best_val_ce - val_base,
              "val_window_ce": val_window_ce, "base_window_ce": list(val_cache["base_ce"]),
              "val_block_id": [val_cache["block_id"][i] for i in range(len(val_cache["base_ce"]))],
              "val_fingerprint": ctx["val_fingerprint"],
              "steps": steps_done, "params_M": count_params(sidecar) / 1e6,
              "refill_s": round(fill_overhead, 1),
              "cache": {"n": n_train, "len": length, "k": n_read},
              "args": vars(args)}
    del loss, opt, sidecar
    empty_cache(device)
    return result


def main():
    global LOG_FILE, LM_HEAD, ATTN_SDPA
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/kaggle/input/lfm-sidecar-data/train_messages.jsonl")
    ap.add_argument("--val", default="/kaggle/input/lfm-sidecar-data/val_messages.jsonl")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--steps", type=int, default=1000000)
    ap.add_argument("--peak_lr", type=float, default=3e-4)
    ap.add_argument("--floor_lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--corr_max_rel", type=float, default=0.0)
    ap.add_argument("--read_layers", default="")
    ap.add_argument("--use_err", type=int, default=0)
    ap.add_argument("--loss_mask", choices=["all", "assistant"], default="all")
    ap.add_argument("--cache_len", type=int, default=8192)
    ap.add_argument("--cache_tokens", type=int, default=2097152)
    ap.add_argument("--fill_budget_s", type=float, default=0.0)
    ap.add_argument("--cache_ram_frac", type=float, default=0.6)
    ap.add_argument("--cache_reserve_gb", type=float, default=4.0)
    ap.add_argument("--refill_frac", type=float, default=0.05)
    ap.add_argument("--pack", type=int, default=1)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--hard_frac", type=float, default=1.0)
    ap.add_argument("--val_windows", type=int, default=16)
    ap.add_argument("--loss_chunk", type=int, default=4096)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--wall_budget_s", type=float, default=18000.0)
    ap.add_argument("--save_margin_s", type=float, default=900.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone", default="LiquidAI/LFM2-350M")
    ap.add_argument("--accelerator", default="gpu", choices=("gpu", "tpu", "cpu", "auto"))
    ap.add_argument("--sweep", default="")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    sys.stdout.reconfigure(line_buffering=True)
    os.makedirs(args.out, exist_ok=True)
    LOG_FILE = open(os.path.join(args.out, "train.log"), "a", encoding="utf8")
    log("args " + json.dumps(vars(args)))
    log("hf token: " + ("set" if load_hf_token() else "absent (anonymous, slower downloads)"))
    configs = build_configs(args)
    log(f"sweep {len(configs)} config(s): {[name for name, _ in configs]}")

    device = pick_device(args.accelerator)
    compute_dtype, dtype_scores = pick_compute_dtype(device)
    log(f"{device_label(device)} tflops fp16 {dtype_scores['fp16']:.2f} bf16 {dtype_scores['bf16']:.2f}"
        f" -> compute {compute_dtype}")

    try:
        tok = AutoTokenizer.from_pretrained(args.backbone)
    except Exception:
        tok = AutoTokenizer.from_pretrained(BACKBONE)
        log(f"tokenizer for {args.backbone} unavailable, using {BACKBONE}")
    attn = attn_implementation(device)
    backbone = load_backbone(args.backbone, compute_dtype, attn).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    LM_HEAD = backbone.lm_head
    emb = backbone.get_input_embeddings().weight

    with torch.no_grad():
        states = model_hidden_states(backbone.model, to_device(torch.tensor([[1]]), device), LM_HEAD, log)
    n_layers = len(states) - 1
    read_layers = parse_read_layers(args.read_layers, n_layers)
    log(f"layers {n_layers} read_layers {read_layers} vocab {backbone.config.vocab_size}")
    try:
        ATTN_SDPA = probe_attention_backends(backbone, device, args.cache_len, backbone.config.vocab_size)
    except Exception:
        log("attn probe crashed\n" + traceback.format_exc())
    empty_cache(device)

    t0 = time.time()
    train_seqs = encode_rows(tok, load_rows(args.train))
    val_seqs = encode_rows(tok, load_rows(args.val))
    n_tok = sum(len(s[0]) for s in train_seqs)
    log(f"encoded train {len(train_seqs)} rows {n_tok} tokens val {len(val_seqs)} rows in {time.time() - t0:.0f}s")

    n_read = len(read_layers)
    length = args.cache_len
    d_model = backbone.config.hidden_size
    wanted = max(8, int(args.cache_tokens / (length * n_read)))
    n_val = max(1, args.val_windows)
    n_train, total_ram, per_window = clamp_cache_windows(
        wanted, n_val, length, n_read, d_model, args.cache_ram_frac, int(args.cache_reserve_gb * 1e9))
    if total_ram:
        log(f"host ram {total_ram / 1e9:.1f}GB frac {args.cache_ram_frac} reserve {args.cache_reserve_gb:.1f}GB"
            f" -> {n_train} of {wanted} windows ({n_train * per_window / 1e9:.2f}GB)")
    else:
        log(f"host ram unknown -> {n_train} of {wanted} windows")
    log(f"cache train {n_train}x{length} val {n_val}x{length} hidden_bytes {n_train * length * n_read * d_model * 2 / 1e9:.2f}GB")

    train_cache = alloc_cache(n_train, length, d_model, n_read, compute_dtype)
    val_cache = alloc_cache(n_val, length, d_model, n_read, compute_dtype)
    t0 = time.time()
    fill_deadline = time.time() + args.fill_budget_s if args.fill_budget_s > 0 else None
    n_train = fill_windows_with_oom_fallback(train_cache, backbone, train_seqs, length, rng,
                                             args.loss_mask, args.pack, device, read_layers,
                                             deadline=fill_deadline)
    log(f"train cache done {time.time() - t0:.0f}s base_ce {sum(train_cache['base_ce']) / n_train:.4f}")
    fill_cache(val_cache, backbone, val_seqs, range(n_val), length, random.Random(1000),
               args.loss_mask, args.pack, device, read_layers)
    val_base = sum(val_cache["base_ce"]) / n_val
    val_fingerprint = hashlib.sha1(val_cache["targets"].cpu().numpy().tobytes()).hexdigest()
    log(f"val cache done base_ce {val_base:.4f} fingerprint {val_fingerprint[:12]}")
    empty_cache(device)

    order = list(range(n_train))
    rng.shuffle(order)
    ctx = {"device": device, "backbone": backbone, "emb": emb, "tok": tok,
           "train_cache": train_cache, "val_cache": val_cache, "val_base": val_base,
           "train_seqs": train_seqs, "val_seqs": val_seqs, "rng": rng,
           "val_fingerprint": val_fingerprint,
           "length": length, "n_read": n_read, "read_layers": read_layers,
           "d_model": d_model, "order": order, "order_pos": 0}
    budget = max(args.wall_budget_s - args.save_margin_s, 1.0) / len(configs)
    log(f"training {len(configs)} config(s) at {budget:.0f}s each")

    results = RESULTS
    results_path = os.path.join(args.out, "result.json")
    for name, overrides in configs:
        prefix = "sidecar" if len(configs) == 1 else name
        results.append(train_one(args, name, prefix, overrides, ctx, budget))
        with open(results_path, "w") as f:
            json.dump(results[0] if len(results) == 1 else results, f, indent=2)


def rescue_after_crash():
    args = CRASH_STATE.get("args")
    if args is None:
        return
    if RESULTS:
        try:
            with open(os.path.join(args.out, "result.json"), "w") as f:
                json.dump(RESULTS[0] if len(RESULTS) == 1 else RESULTS, f, indent=2)
            log(f"result.json rescued with {len(RESULTS)} config(s)")
        except Exception:
            log("result.json rescue failed\n" + traceback.format_exc())
    rescued = CRASH_STATE.get("sidecar")
    if rescued is not None:
        try:
            save(rescued, args.out, CRASH_STATE["step"], args,
                 name=f"{CRASH_STATE.get('prefix', 'sidecar')}_crash.pt")
            log(f"crash checkpoint saved at step {CRASH_STATE['step']}")
        except Exception:
            log("crash checkpoint failed\n" + traceback.format_exc())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FATAL\n" + traceback.format_exc())
        rescue_after_crash()
        raise
