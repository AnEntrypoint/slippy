import hashlib
import io
import json
import os
import re
import sys
from typing import NamedTuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "trainingdata", "train_messages.jsonl")
VAL = os.path.join(ROOT, "trainingdata", "val_messages.jsonl")
OUT = os.path.join(ROOT, "sidecar", "scan.json")

TOOL_CALL = re.compile(r"\[tool_call:\s*([A-Za-z_][A-Za-z0-9_.\-]*)\]")
BLOCK = re.compile(r"\[tool_(call|result):\s*([A-Za-z_][A-Za-z0-9_.\-]*)\]")


class BlockSpan(NamedTuple):
    kind: str
    name: str
    start: int
    end: int


def block_spans(content):
    marks = [(m.start(), m.group(1), m.group(2)) for m in BLOCK.finditer(content)]
    spans = []
    for i, (start, kind, name) in enumerate(marks):
        next_mark_start = marks[i + 1][0] if i + 1 < len(marks) else len(content)
        spans.append(BlockSpan(kind=kind, name=name, start=start, end=next_mark_start))
    return spans


def scan(path, prefix_len=200):
    rows = 0
    bad_json_lines = 0
    message_count = 0
    user_chars = 0
    asst_chars = 0
    other_chars = 0
    asst_in_result_chars = 0
    all_in_result_chars = 0
    calls = {}
    prefixes = {}
    row_char_lens = []
    row_asst_char_lens = []
    nmsg_hist = {}
    with io.open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                d = json.loads(line)
            except Exception:
                bad_json_lines += 1
                continue
            messages = d.get("messages") or []
            message_count += len(messages)
            nmsg_hist[len(messages)] = nmsg_hist.get(len(messages), 0) + 1
            row_chars = 0
            row_asst = 0
            first_user = None
            for m in messages:
                role = m.get("role")
                content = m.get("content")
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False)
                n = len(content)
                row_chars += n
                if role == "assistant":
                    asst_chars += n
                    row_asst += n
                elif role == "user":
                    user_chars += n
                    if first_user is None:
                        first_user = content
                else:
                    other_chars += n
                result_block_chars = sum(end - start for kind, _name, start, end in block_spans(content) if kind == "result")
                all_in_result_chars += result_block_chars
                if role == "assistant":
                    asst_in_result_chars += result_block_chars
                for tool_name in TOOL_CALL.findall(content):
                    calls[tool_name] = calls.get(tool_name, 0) + 1
            row_char_lens.append(row_chars)
            row_asst_char_lens.append(row_asst)
            if first_user is not None:
                key = hashlib.blake2b(first_user[:prefix_len].encode("utf-8", "replace"), digest_size=16).hexdigest()
                prefixes[key] = prefixes.get(key, 0) + 1
    row_char_lens.sort()
    row_asst_char_lens.sort()
    total_prefix = sum(prefixes.values())
    dup_rows = sum(v - 1 for v in prefixes.values() if v > 1)
    return {
        "path": path,
        "rows": rows,
        "bad_json": bad_json_lines,
        "messages": message_count,
        "user_chars": user_chars,
        "asst_chars": asst_chars,
        "other_chars": other_chars,
        "asst_in_result_chars": asst_in_result_chars,
        "all_in_result_chars": all_in_result_chars,
        "tool_calls": calls,
        "row_char_lens_sorted": row_char_lens,
        "row_asst_char_lens_sorted": row_asst_char_lens,
        "nmsg_hist": nmsg_hist,
        "unique_prefixes": len(prefixes),
        "dup_rows": dup_rows,
        "prefix_collisions": sum(v for v in prefixes.values() if v > 1),
        "max_prefix_mult": max(prefixes.values()) if prefixes else 0,
        "prefixes": prefixes,
        "total_prefixed_rows": total_prefix,
    }


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, int(round((p / 100.0) * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def summarise(report):
    row_char_lens = report["row_char_lens_sorted"]
    row_asst_char_lens = report["row_asst_char_lens_sorted"]
    all_chars = max(1, report["user_chars"] + report["asst_chars"] + report["other_chars"])
    return {
        "rows": report["rows"],
        "bad_json": report["bad_json"],
        "messages": report["messages"],
        "mean_chars": round(sum(row_char_lens) / len(row_char_lens), 1) if row_char_lens else 0,
        "p50_chars": percentile(row_char_lens, 50),
        "p90_chars": percentile(row_char_lens, 90),
        "p99_chars": percentile(row_char_lens, 99),
        "max_chars": row_char_lens[-1] if row_char_lens else 0,
        "mean_asst_chars": round(sum(row_asst_char_lens) / len(row_asst_char_lens), 1) if row_asst_char_lens else 0,
        "asst_char_frac": round(report["asst_chars"] / all_chars, 4),
        "asst_in_result_frac_of_asst": round(report["asst_in_result_chars"] / max(1, report["asst_chars"]), 4),
        "result_frac_of_all": round(report["all_in_result_chars"] / all_chars, 4),
        "nmsg_hist": report["nmsg_hist"],
        "unique_prefixes": report["unique_prefixes"],
        "dup_rows": report["dup_rows"],
        "prefix_collision_rows": report["prefix_collisions"],
        "max_prefix_mult": report["max_prefix_mult"],
        "top_tool_calls": sorted(report["tool_calls"].items(), key=lambda kv: -kv[1])[:30],
        "distinct_tool_calls": len(report["tool_calls"]),
    }


def main():
    train = scan(TRAIN)
    val = scan(VAL)
    train_prefixes = set(train["prefixes"])
    val_prefixes = set(val["prefixes"])
    shared_prefixes = train_prefixes & val_prefixes
    result = {
        "train": summarise(train),
        "val": summarise(val),
        "val_rows_in_train_prefix": sum(train["prefixes"].get(k, 0) for k in shared_prefixes),
        "contaminated_val_prefixes": len(shared_prefixes),
        "contaminated_val_rows": sum(val["prefixes"][k] for k in shared_prefixes),
    }
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in result.items() if k != "train" and k != "val"}, indent=1))
    print("TRAIN", json.dumps(result["train"], indent=1)[:4000])
    print("VAL", json.dumps(result["val"], indent=1)[:4000])


if __name__ == "__main__":
    main()
