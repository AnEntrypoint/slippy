import argparse
import collections
import json
import os
import random
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def read_env_token(env_path):
    if not os.path.exists(env_path):
        return
    for line in open(env_path, encoding="utf8"):
        if line.startswith("HF_TOKEN="):
            os.environ["HF_TOKEN"] = line.strip().split("=", 1)[1]


def encode_rows(tok, rows):
    bos = tok.bos_token_id
    out = []
    for messages in rows:
        ids = [] if bos is None else [bos]
        tags = [] if bos is None else ["bos"]
        for m in messages:
            head = tok(f"<|im_start|>{m['role']}\n", add_special_tokens=False)["input_ids"]
            body = tok(m["content"], add_special_tokens=False)["input_ids"]
            tail = tok("<|im_end|>\n", add_special_tokens=False)["input_ids"]
            ids += head + body + tail
            tags += ["head:" + m["role"]] * len(head) + [m["role"]] * len(body) + ["tail:" + m["role"]] * len(tail)
        out.append((ids, tags))
    return out


def pack_window(seqs, length, rng):
    ids, tags = [], []
    while len(ids) < length + 1:
        row_ids, row_tags = rng.choice(seqs)
        need = length + 1 - len(ids)
        if len(row_ids) > need:
            s = rng.randrange(0, len(row_ids) - need + 1)
            row_ids = row_ids[s:s + need]
            row_tags = row_tags[s:s + need]
        ids += row_ids
        tags += row_tags
    return ids, tags


def window_ce(model, ids, length, chunk):
    x = torch.tensor([ids[:-1]])
    y = torch.tensor(ids[1:])
    with torch.no_grad():
        hidden = model.model(x).last_hidden_state[0]
        parts = []
        for s in range(0, length, chunk):
            logits = model.lm_head(hidden[s:s + chunk]).float()
            parts.append(F.cross_entropy(logits, y[s:s + chunk], reduction="none"))
    return torch.cat(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LiquidAI/LFM2-350M")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trainingdata", "train_messages.jsonl"))
    ap.add_argument("--env", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
    ap.add_argument("--rows", type=int, default=60)
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    read_env_token(args.env)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    rows = []
    with open(args.data, encoding="utf8") as f:
        for i, line in enumerate(f):
            if i >= args.rows:
                break
            rows.append(json.loads(line)["messages"])
    roles = collections.Counter(m["role"] for r in rows for m in r)
    seqs = encode_rows(tok, rows)
    print("model %s  rows %d  roles %s" % (args.model, len(rows), dict(roles)))

    rng = random.Random(args.seed)
    all_ce, all_tags = [], []
    for w in range(args.windows):
        ids, tags = pack_window(seqs, args.length, rng)
        ce = window_ce(model, ids, args.length, args.chunk)
        all_ce.append(ce)
        all_tags += tags[1:]
        print("window %d mean_ce %.4f" % (w, float(ce.mean())), flush=True)

    ce = torch.cat(all_ce)
    n = ce.numel()
    total = float(ce.sum())
    q = torch.quantile(ce, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9, 0.99]))
    print("\ntokens %d  mean_ce %.4f" % (n, float(ce.mean())))
    print("quantiles p10 %.3f p25 %.3f p50 %.3f p75 %.3f p90 %.3f p99 %.3f" % tuple(float(v) for v in q))
    for frac in (0.25, 0.5):
        top = torch.topk(ce, int(frac * n)).values
        print("hardest %2.0f%%: mean %.3f  share of total ce %.3f"
              % (frac * 100, float(top.mean()), float(top.sum()) / total))
    easy = torch.topk(ce, n // 2, largest=False).values
    print("easiest 50%%: mean %.3f  share of total ce %.3f" % (float(easy.mean()), float(easy.sum()) / total))
    print("ce < 0.05: %.3f of tokens | ce > 5: %.3f of tokens"
          % (float((ce < 0.05).float().mean()), float((ce > 5).float().mean())))

    buckets = collections.defaultdict(lambda: [0.0, 0])
    for j, tag in enumerate(all_tags):
        buckets[tag][0] += float(ce[j])
        buckets[tag][1] += 1
    print("\n%20s %7s %7s %8s %8s" % ("bucket", "share", "ce_shr", "mean_ce", "n"))
    for tag, (c, k) in sorted(buckets.items(), key=lambda kv: -kv[1][0]):
        print("%20s %7.3f %7.3f %8.3f %8d" % (tag, k / n, c / total, c / max(k, 1), k))


if __name__ == "__main__":
    main()
