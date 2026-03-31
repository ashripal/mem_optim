import json
from pathlib import Path
import random
from collections import defaultdict

INPUT = Path("data/squad_clean/squad_train_paraphrased.jsonl")
OUTPUT = Path("data/squad_clean/workload_paraphrase.jsonl")

def main():
    with INPUT.open() as f:
        data = [json.loads(line) for line in f]

    families = defaultdict(list)
    for row in data:
        families[row["family_id"]].append(row)

    out = []

    for fid, rows in families.items():
        # ensure original first exposure
        original = [r for r in rows if r["variant"] == "original"][0]
        others = [r for r in rows if r["variant"] != "original"]

        out.append(original)
        random.shuffle(others)
        out.extend(others)

    random.shuffle(out)  # global shuffle AFTER structuring

    with OUTPUT.open("w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")

    print(f"[done] wrote {len(out)} rows")

if __name__ == "__main__":
    main()