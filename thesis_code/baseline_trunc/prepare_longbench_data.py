from __future__ import annotations
import argparse
import zipfile
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_dir", default="tier2_disk/longbench_repo", help="Path to LongBench dataset repo")
    ap.add_argument("--out_dir", default="", help="Optional extraction folder (defaults to repo_dir)")
    args = ap.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    zip_path = repo_dir / "data.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Expected {zip_path} to exist.")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else repo_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"p[extract] Zip path: {zip_path}")
    print(f"p[extract] Out dir: {out_dir}")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(out_dir)

    # LongBench zip usually creates out_dir/data/*.jsonl
    candidate = out_dir / "data"
    jsonl_dir = candidate if candidate.exists() else out_dir

    n_jsonl = len(list(jsonl_dir.glob("*.jsonl")))
    print(f"[tier2] jsonl_dir: {jsonl_dir} (found {n_jsonl} .jsonl files)")

    if n_jsonl == 0:
        raise FileNotFoundError(f"No JSONL files found in {jsonl_dir}.")

    print(str(jsonl_dir))

if __name__ == "__main__":
    main()