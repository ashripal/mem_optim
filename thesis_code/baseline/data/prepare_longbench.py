# data/prepare_longbench.py
"""
Prepare (extract) the LongBench dataset into a usable Tier-2 (Disk) directory.

This script is a safer + more robust version of the earlier `prepare_longbench_data.py`:
- Verifies `data.zip` exists inside the repo directory (default: tier2_disk/longbench_repo)
- Safely extracts (prevents Zip Slip path traversal)
- Ensures JSONL files are actually present after extraction
- Locates the directory containing `*.jsonl` (commonly repo_dir/data/*.jsonl)
- Optionally prints (and can write) the resolved JSONL directory for downstream scripts

Expected structure (typical):
  tier2_disk/longbench_repo/
    data.zip
    data/
      *.jsonl

If the zip extracts into a nested folder, we search for `*.jsonl` recursively.
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def _is_within_directory(base_dir: Path, target_path: Path) -> bool:
    """
    Returns True if target_path is within base_dir after resolving symlinks/.. segments.
    """
    base = base_dir.resolve()
    target = target_path.resolve()
    try:
        target.relative_to(base)
        return True
    except ValueError:
        return False


def _safe_extract_zip(zip_path: Path, out_dir: Path) -> None:
    """
    Safely extract a zip file into out_dir, preventing Zip Slip attacks.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()

        for m in members:
            # Skip directory entries; they'll be created as needed.
            # Some zips have entries like "data/".
            member_name = m.filename

            # Zip files may contain weird absolute paths or ".."
            dest_path = (out_dir / member_name)

            if not _is_within_directory(out_dir, dest_path):
                raise RuntimeError(
                    f"Unsafe zip entry detected (path traversal): {member_name}"
                )

        zf.extractall(out_dir)


def _find_jsonl_dirs(root: Path) -> List[Path]:
    """
    Return candidate directories that contain at least one .jsonl file.
    """
    dirs = set()
    for p in root.rglob("*.jsonl"):
        if p.is_file():
            dirs.add(p.parent)
    return sorted(dirs)


def _pick_best_jsonl_dir(candidates: List[Path]) -> Path:
    """
    Prefer a directory literally named 'data' if present; otherwise pick the one
    with the most JSONL files.
    """
    if not candidates:
        raise FileNotFoundError("No directories containing .jsonl files were found.")

    # Prefer ".../data" directory if it exists
    data_dirs = [d for d in candidates if d.name.lower() == "data"]
    if data_dirs:
        # If multiple, choose the one with the most jsonl files
        data_dirs.sort(key=lambda d: len(list(d.glob("*.jsonl"))), reverse=True)
        return data_dirs[0]

    # Otherwise choose the directory with the most jsonl files
    candidates.sort(key=lambda d: len(list(d.glob("*.jsonl"))), reverse=True)
    return candidates[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo_dir",
        default="tier2_disk/longbench_repo",
        help="Path to the local LongBench dataset repo directory (contains data.zip).",
    )
    ap.add_argument(
        "--zip_name",
        default="data.zip",
        help="Zip file name inside repo_dir to extract (default: data.zip).",
    )
    ap.add_argument(
        "--out_dir",
        default="",
        help="Optional extraction directory. If omitted, extracts into repo_dir.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if JSONL files already appear to be present.",
    )
    ap.add_argument(
        "--print_jsonl_dir_only",
        action="store_true",
        help="Print only the resolved JSONL directory path (useful for scripting).",
    )
    ap.add_argument(
        "--write_jsonl_dir_to",
        default="",
        help="Optional file path to write the resolved JSONL directory (plain text).",
    )
    args = ap.parse_args()

    repo_dir = Path(args.repo_dir).expanduser().resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(f"repo_dir does not exist: {repo_dir}")

    zip_path = repo_dir / args.zip_name
    if not zip_path.exists():
        raise FileNotFoundError(f"Expected zip to exist at: {zip_path}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else repo_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # If already extracted, allow skip unless --force
    pre_candidates = _find_jsonl_dirs(out_dir)
    if pre_candidates and not args.force:
        jsonl_dir = _pick_best_jsonl_dir(pre_candidates)
        n_jsonl = len(list(jsonl_dir.glob("*.jsonl")))

        if args.print_jsonl_dir_only:
            print(str(jsonl_dir))
        else:
            print(f"[prepare] repo_dir: {repo_dir}")
            print(f"[prepare] zip_path: {zip_path}")
            print(f"[prepare] out_dir : {out_dir}")
            print(f"[prepare] Detected existing extraction.")
            print(f"[tier2] jsonl_dir: {jsonl_dir} (found {n_jsonl} .jsonl files)")

        if args.write_jsonl_dir_to:
            dst = Path(args.write_jsonl_dir_to).expanduser().resolve()
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(str(jsonl_dir), encoding="utf-8")

        return

    if not args.print_jsonl_dir_only:
        print(f"[prepare] repo_dir: {repo_dir}")
        print(f"[prepare] zip_path: {zip_path}")
        print(f"[prepare] out_dir : {out_dir}")
        print("[prepare] Extracting...")

    _safe_extract_zip(zip_path, out_dir)

    # Post-check: find where the jsonl files ended up
    candidates = _find_jsonl_dirs(out_dir)
    if not candidates:
        # Helpful error message: show top-level listing
        top = sorted([p.name for p in out_dir.iterdir()]) if out_dir.exists() else []
        raise FileNotFoundError(
            f"No .jsonl files found after extraction in: {out_dir}\n"
            f"Top-level contents: {top}\n"
            f"Check whether {zip_path.name} is the correct archive."
        )

    jsonl_dir = _pick_best_jsonl_dir(candidates)
    n_jsonl = len(list(jsonl_dir.glob("*.jsonl")))
    if n_jsonl == 0:
        # Should not happen if candidates are correct, but be defensive.
        raise FileNotFoundError(f"No JSONL files found in selected dir: {jsonl_dir}")

    if args.print_jsonl_dir_only:
        print(str(jsonl_dir))
    else:
        print(f"[prepare] Extraction complete.")
        print(f"[tier2] jsonl_dir: {jsonl_dir} (found {n_jsonl} .jsonl files)")

    if args.write_jsonl_dir_to:
        dst = Path(args.write_jsonl_dir_to).expanduser().resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(str(jsonl_dir), encoding="utf-8")


if __name__ == "__main__":
    main()