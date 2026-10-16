import glob
import io
import json
import os
import random
import re
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, 'trainingdata', 'train_messages.jsonl')
OUT = os.path.join(ROOT, 'sidecar', 'tokstats.json')

BLOCK = re.compile(r'\[tool_(call|result):\s*([A-Za-z_][A-Za-z0-9_.\-]*)\]')

NROWS = 7017
SEED = 1234
SAMPLE = int(os.environ.get('SAMPLE', '') or NROWS)
FULL = SAMPLE >= NROWS


def load_tok():
    snap = glob.glob(os.path.expanduser(
        '~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots/*/tokenizer.json'))[0]
    from tokenizers import Tokenizer
    return Tokenizer.from_file(snap)


def ntok(tok, text):
    return len(tok.encode(text, add_special_tokens=False).ids)


def result_spans(content):
    marks = [(m.start(), m.group(1)) for m in BLOCK.finditer(content)]
    out = []
    for i, (pos, kind) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(content)
        if kind == 'result':
            out.append(content[pos:end])
    return out


def pct(v, p):
    if not v:
        return 0
    s = sorted(v)
    return s[min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))]


def main():
    tok = load_tok()
    want = None if FULL else set(random.Random(SEED).sample(range(NROWS), SAMPLE))
    rows = []
    t0 = time.time()
    chars_done = 0
    with io.open(TRAIN, 'r', encoding='utf-8', errors='replace', newline='') as f:
        idx = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            i = idx
            idx += 1
            if want is not None and i not in want:
                continue
            d = json.loads(line)
            msgs = d['messages']
            role_len = {}
            offsets = []
            cursor = 0
            for m in msgs:
                role = m.get('role')
                content = m.get('content')
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False)
                text = '<|' + role + '|>\n' + content + '\n'
                n = ntok(tok, text)
                offsets.append((role, cursor, cursor + n, content))
                role_len[role] = role_len.get(role, 0) + n
                cursor += n
                chars_done += len(content)
            total = cursor
            res_tokens = 0
            asst_res_tokens = 0
            for role, s, e, content in offsets:
                r = sum(ntok(tok, sp) for sp in result_spans(content))
                res_tokens += r
                if role == 'assistant':
                    asst_res_tokens += r
            first_user_start = None
            last_asst_end = None
            for role, s, e, c in offsets:
                if role == 'user' and first_user_start is None:
                    first_user_start = s
                if role == 'assistant':
                    last_asst_end = e
            dist = None
            if first_user_start is not None and last_asst_end:
                dist = last_asst_end - 1 - first_user_start
            rows.append({
                'i': i,
                'total': total,
                'user': role_len.get('user', 0),
                'asst': role_len.get('assistant', 0),
                'res': res_tokens,
                'asst_res': asst_res_tokens,
                'dist': dist,
                'nmsg': len(msgs),
            })
    elapsed = time.time() - t0
    totals = [r['total'] for r in rows]
    dists = [r['dist'] for r in rows if r['dist'] is not None]
    sum_total = sum(totals)
    sum_asst = sum(r['asst'] for r in rows)
    sum_user = sum(r['user'] for r in rows)
    sum_res = sum(r['res'] for r in rows)
    sum_asst_res = sum(r['asst_res'] for r in rows)
    n = len(rows)
    scale = NROWS / n
    res = {
        'sampled': not FULL,
        'rows_measured': n,
        'seed': SEED,
        'seconds': round(elapsed, 1),
        'chars_per_token': round(chars_done / max(1, sum_total), 3),
        'mean_tokens': round(sum_total / n, 1),
        'p50': pct(totals, 50),
        'p90': pct(totals, 90),
        'p95': pct(totals, 95),
        'p99': pct(totals, 99),
        'max': max(totals),
        'min': min(totals),
        'total_tokens_estimate': int(round(sum_total * scale)),
        'assistant_token_frac': round(sum_asst / sum_total, 4),
        'user_token_frac': round(sum_user / sum_total, 4),
        'result_token_frac_of_all': round(sum_res / sum_total, 4),
        'result_token_frac_of_assistant': round(sum_asst_res / max(1, sum_asst), 4),
        'dist_mean': round(sum(dists) / len(dists), 1),
        'dist_p50': pct(dists, 50),
        'dist_p90': pct(dists, 90),
        'dist_max': max(dists),
        'dist_frac_ge_2048': round(sum(1 for x in dists if x >= 2048) / len(dists), 4),
        'dist_frac_ge_8192': round(sum(1 for x in dists if x >= 8192) / len(dists), 4),
        'dist_frac_ge_16384': round(sum(1 for x in dists if x >= 16384) / len(dists), 4),
        'row_tokens_gt_8192_frac': round(sum(1 for x in totals if x > 8192) / n, 4),
        'row_tokens_gt_16384_frac': round(sum(1 for x in totals if x > 16384) / n, 4),
    }
    with io.open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'summary': res, 'rows': rows}, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
