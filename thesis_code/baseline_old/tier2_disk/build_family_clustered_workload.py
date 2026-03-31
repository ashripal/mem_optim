import json
from pathlib import Path
from collections import defaultdict

INPUT = Path("data/squad_clean/squad_train_paraphrased.jsonl")
OUTPUT = Path("data/squad_clean/workload_family_clustered.jsonl")

def variant_key(v):
    if v == "original":
        return -1
    if v.startswith("para_"):
        return int(v.split("_")[1])
    return 999

def main():
    with INPUT.open() as f:
        data = [json.loads(line) for line in f]

    families = defaultdict(list)
    for row in data:
        families[row["family_id"]].append(row)

    out = []

    for fid in sorted(families.keys()):
        family = sorted(families[fid], key=lambda x: variant_key(x["variant"]))
        out.extend(family)

    with OUTPUT.open("w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")

    print(f"[done] wrote {len(out)} rows")

if __name__ == "__main__":
    main()