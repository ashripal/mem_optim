import json
from pathlib import Path
import random
import copy

INPUT = Path("data/squad_clean/squad_train_paraphrased.jsonl")
OUTPUT = Path("data/squad_clean/workload_exact.jsonl")

REPEATS = 3

def main():
    with INPUT.open() as f:
        data = [json.loads(line) for line in f]

    originals = [x for x in data if x["variant"] == "original"]

    out = []
    for ex in originals:
        for _ in range(REPEATS):
            out.append(copy.deepcopy(ex))  # FIX

    random.shuffle(out)

    with OUTPUT.open("w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")

    print(f"[done] wrote {len(out)} rows")

if __name__ == "__main__":
    main()