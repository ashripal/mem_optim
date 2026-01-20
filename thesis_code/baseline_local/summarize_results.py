"""
Summarize baseline results written to Tier 2 (disk) as JSONL
"""

from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from tqdm import tqdm

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Path to baseline_longbench_*.jsonl file")
    args = ap.parse_args()

    path = Path(args.jsonl).resolve()
    by_task = defaultdict(list)

    with path.open("r", encodings="utf-8") as f:
        for line in tqdm(f, desc="Reading results"):
            record = json.loads(line)
            dataset = record.get("dataset_name") or record.get("task_file")
            by_task[dataset].append(record)

    print(f"\nLoaded {sum(len(v) for v in by_task.values())} records from {len(by_task)} tasks.\n")

    for task, rows in sorted(by_task.items(), key=lambda x: len(x[1]), reverse=True):
        lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
        tps = [r["tokens_per_sec"] for r in rows if r.get("tokens_per_sec") is not None]
        in_tok = [r["input_tokens"] for r in rows if r.get("input_tokens") is not None]
        out_tok = [r["output_tokens"] for r in rows if r.get("output_tokens") is not None]

        print(f"== Task: {task} ==")
        print(f"n={len(rows)}")
        if lat:
            print(f"avg latency (s): {mean(lat):.3f}")
        if tps:
            print(f"avg tokens/sec: {mean(tps):.2f}")
        if in_tok and out_tok:
            print(f"avg in/out tok: {mean(in_tok):.1f} / {mean(out_tok):.1f}")
        print("")

if __name__ == "__main__":
    main()