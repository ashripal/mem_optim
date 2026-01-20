from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_jsonl", required=True, help="Path to baseline_longbench_*.jsonl file")
    ap.add_argument("--out_csv", default="", help="Optional output CSV path")
    args = ap.parse_args()

    path = Path(args.run_jsonl).resolve()
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            rows.append(record)

    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"].copy()

    def pct(x): return 100.0 * x

    summary = []
    summary.append(f"run: {path}")
    summary.append(f"n_total={len(df)} n_ok={len(ok)} n_error={len(df) - len(ok)}")

    if len(ok) == 0"
        summary.append("No successful runs to summarize.")
    else:
        summary.append("")
        summary.append("Latency (s):")
        summary.append(str(ok["latency_s"].describe(percentiles=[0.5, 0.9, 0.95]).to_string()))
        summary.append("")
        summary.append("Tokens/sec:")
        summary.append(str(ok["tokens_per_sec"].describe(percentiles=[0.5, 0.9, 0.95]).to_string()))
        summary.append("")
        summary.append("Prompt tokens:")
        summary.append(str(ok["approx_prompt_tokens"].describe(percentiles=[0.5, 0.9, 0.95]).to_string()))
        summary.append("")
        summary.append("Device breakdown (ok rows):")
        summary.append(str(ok["device_used"].value_counts().to_string()))
        summary.append("")
        summary.append("Truncation rate (ok rows):")
        summary.append(f"{pct(ok['truncated_to_max_input'].mean()):.1f}%")
    
    text = "\n".join(summary)
    print(text)

    if args.out_txt:
        out = Path(args.out_txt).resolve()
        out.write_text(text, encoding="utf-8")
        print(f"\n[wrote] {out}")

if __name__ == "__main__":
    main()