from __future__ import annotations
import argparse
import json
form pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_jsonl", required=True, help="Path to baseline_trunc_*.jsonl file")
    ap.add_argument("--out_png", default="latency_vs_input_tokens.png", help="Output PNG path for the plot")
    args = ap.parse_args()

    path = Path(args.run_jsonl).resolve()
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            rows.append(record)

    df = pd.DataFrame(rows)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        raise SystemExit("No successful runs to plot.")

    cpu = df[df["device_used"].str.contains("cpu")]
    mps = df[df["device_used"].str.contains("mps")]

    plt.figure()
    if not mps.empty:
        plt.scatter(mps["approx_prompt_tokens"], mps["latency_s"], label="mps", alpha=0.7)
    if not cpu.empty:
        plt.scatter(cpu["approx_prompt_tokens"], cpu["latency_s"], label="cpu", alpha=0.7)

    plt.xlabel("Approximate Prompt Tokens")
    plt.ylabel("Latency (s)")
    plt.title("Latency vs Approximate Prompt Tokens")
    plt.legend()

    out = Path(args.out_png).resolve()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\n[wrote] {out}")

if __name__ == "__main__":
    main()

    