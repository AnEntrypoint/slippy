import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

BACKBONE = "LiquidAI/LFM2-350M"
T = 1024


def enc_v2(tok, rows):
    bos = tok.bos_token_id
    out = []
    for messages in rows:
        ids = [bos]
        is_asst = [False]
        for m in messages:
            head = tok(f"<|im_start|>{m['role']}\n", add_special_tokens=False)["input_ids"]
            body = tok(m["content"], add_special_tokens=False)["input_ids"]
            tail = tok("<|im_end|>\n", add_special_tokens=False)["input_ids"]
            a = m["role"] == "assistant"
            ids += head + body + tail
            is_asst += [False] * len(head) + [a] * len(body) + [a] * len(tail)
        out.append((ids, is_asst))
    return out


def enc_v1(tok, rows):
    out = []
    for messages in rows:
        ids, is_asst = [], []
        for m in messages:
            t = tok(f"<|{m['role']}|>\n{m['content']}\n", add_special_tokens=False)["input_ids"]
            ids += t
            is_asst += [m["role"] == "assistant"] * len(t)
        out.append((ids, is_asst))
    return out


def enc_tpl(tok, rows):
    return [(tok.apply_chat_template(ms, tokenize=True, add_generation_prompt=False),
             [False] * len(tok.apply_chat_template(ms, tokenize=True, add_generation_prompt=False)))
            for ms in rows]


def ce_of(model, ids, t):
    x = torch.tensor([ids[:t]])
    y = torch.tensor(ids[1:t + 1])
    with torch.no_grad():
        lg = model(x).logits[0].float()
    return F.cross_entropy(lg, y).item()


def main():
    print("v2 == apply_chat_template:", v2[0][0] == tp[0][0], "len", len(v2[0][0]), len(tp[0][0]))
    print("v1 len", len(v1[0][0]))
    print("head v2:", repr(tok.decode(v2[0][0][:24])))
    print("head v1:", repr(tok.decode(v1[0][0][:24])))
    if "--ce" not in sys.argv:
        return
    tok = tok
    model = AutoModelForCausalLM.from_pretrained(BACKBONE, dtype=torch.float32).eval()
    for name, seqs in (("v2", v2), ("v1", v1)):
        ids = seqs[0][0]
        if len(ids) <= T + 1:
            print(name, "too short", len(ids))
            continue
        print(name, "ce@1024", round(ce_of(model, ids, T), 4))


if __name__ == "__main__":
    main()
