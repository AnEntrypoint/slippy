import json
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

BACKBONE = "LiquidAI/LFM2-350M"


def encode(tok, messages):
    bos = tok.bos_token_id
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
    return ids, token_is_assistant


def main():
    path = sys.argv[1]
    rows = [json.loads(line)["messages"] for line in open(path, encoding="utf8")]
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    ids, is_a = encode(tok, rows[0])
    tpl = tok.apply_chat_template(rows[0], tokenize=True, add_generation_prompt=False)
    print("row0 tokens", len(ids), "chars", sum(len(m["content"]) for m in rows[0]))
    print("matches apply_chat_template", ids == tpl)
    print("first ids", ids[:14])
    print("HEAD", repr(tok.decode(ids[:110])))
    print("roles", [m["role"] for m in rows[0]])
    print("asst frac", round(sum(is_a) / max(len(is_a), 1), 3))
    if "--ce" not in sys.argv:
        return
    length = 512
    for arg in sys.argv[2:]:
        if arg.isdigit():
            length = int(arg)
    model = AutoModelForCausalLM.from_pretrained(BACKBONE, dtype=torch.float32).eval()
    x = torch.tensor([ids[:length]])
    y = torch.tensor(ids[1:length + 1])
    with torch.no_grad():
        lg = model(x).logits[0].float()
    print("ce@%d" % length, round(F.cross_entropy(lg, y).item(), 4))
    print("perplexity", round(float(torch.exp(F.cross_entropy(lg, y)).item()), 2))


if __name__ == "__main__":
    main()
