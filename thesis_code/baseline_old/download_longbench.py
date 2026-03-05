"""
Tier 2 (Disk) setup: download LongBench dataset repo files locally

LongBench data format: input, context, answers, length, dataset, etc.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo_id",
        default="zai-org/LongBench", 
        help="HF dataset repo ID to download from",
    )
    ap.add_argument(
        "--out_dir",
        default="tier2_disk/longbench_repo",
        help="local folder (Tier 2)"
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"p[download] Repo: {args.repo_id}")
    print(f"p[download] Out dir: {out_dir}")

    local_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(out_dir),
        local_dir_use_symlinks=False
    )

    print(f"p[download] Completed. Local path: {local_path}")

if __name__ == "__main__":
    main()
