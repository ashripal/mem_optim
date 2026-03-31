# validate_all_workloads.py
import json
from pathlib import Path
from collections import defaultdict, Counter

FILES = [
    "data/squad_clean/workload_exact.jsonl",
    "data/squad_clean/workload_paraphrase.jsonl",
    "data/squad_clean/workload_family_clustered.jsonl",
]

def validate_file(path):
    print(f"\n=== Checking {path} ===")

    with open(path) as f:
        data = [json.loads(line) for line in f]

    print(f"Total rows: {len(data)}")

    # ------------------------
    # Required fields
    # ------------------------
    required = ["context", "question", "answer", "family_id", "variant"]
    for i, row in enumerate(data[:100]):
        for k in required:
            if k not in row:
                raise ValueError(f"[FAIL] Missing {k} in row {i}")

    print("✔ Required fields OK")

    # ------------------------
    # Family consistency
    # ------------------------
    families = defaultdict(list)
    for row in data:
        families[row["family_id"]].append(row)

    bad = 0
    for fid, rows in families.items():
        contexts = {r["context"] for r in rows}
        answers = {r["answer"] for r in rows}

        if len(contexts) != 1 or len(answers) != 1:
            print(f"[FAIL] inconsistent family: {fid}")
            bad += 1

    if bad == 0:
        print("✔ Family consistency OK")
    else:
        print(f"[FAIL] {bad} bad families")

    # ------------------------
    # Variant structure
    # ------------------------
    variant_counts = Counter(r["variant"] for r in data)
    print("Variant counts:", dict(variant_counts))

    # ------------------------
    # Check ordering for clustered
    # ------------------------
    if "family_clustered" in path:
        prev_fid = None
        seen = set()

        for row in data:
            fid = row["family_id"]
            if fid != prev_fid:
                if fid in seen:
                    raise ValueError(f"[FAIL] family {fid} is not contiguous")
                seen.add(fid)
                prev_fid = fid

        print("✔ Family clustering contiguous")

    # ------------------------
    # Check exact duplicates
    # ------------------------
    if "exact" in path:
        seen = set()
        duplicates = 0

        for row in data:
            key = (row["context"], row["question"])
            if key in seen:
                duplicates += 1
            else:
                seen.add(key)

        print(f"Exact duplicates: {duplicates}")

    print("=== DONE ===")


def main():
    for f in FILES:
        validate_file(f)


if __name__ == "__main__":
    main()