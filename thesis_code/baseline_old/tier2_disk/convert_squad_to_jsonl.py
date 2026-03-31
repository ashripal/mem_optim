import json
from pathlib import Path

RAW_DIR = Path("squad_raw")
OUT_DIR = Path("data/squad_clean")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def convert_raw_squad_file(in_path: Path, out_path: Path, split_name: str, max_examples: int | None = None) -> int:
    print(f"[info] Reading: {in_path.resolve()}")
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path.resolve()}")

    with in_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    data = payload.get("data", [])
    print(f"[info] Top-level articles: {len(data)}")

    n_written = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for article in data:
            title = article.get("title", "")
            for paragraph_block in article.get("paragraphs", []):
                context = paragraph_block.get("context", "")
                if not context:
                    continue

                for qa in paragraph_block.get("qas", []):
                    answers = qa.get("answers", []) or []
                    answer_texts = [a.get("text", "") for a in answers if a.get("text", "")]
                    if not answer_texts:
                        continue

                    family_id = f"squad_{n_written:06d}"

                    record = {
                        "context": context,
                        "question": qa.get("question", ""),
                        "answer": answer_texts[0],
                        "answers": answer_texts,
                        "task": "squad",
                        "example_id": family_id,
                        "family_id": family_id,
                        "variant": "original",
                        "source_split": split_name,
                        "title": title,
                        "dataset_id": qa.get("id", ""),
                    }

                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_written += 1

                    if max_examples is not None and n_written >= max_examples:
                        print(f"[info] Reached max_examples={max_examples}")
                        print(f"[info] Wrote: {out_path.resolve()}")
                        return n_written

    print(f"[info] Wrote: {out_path.resolve()}")
    return n_written


def main() -> None:
    train_in = RAW_DIR / "train-v1.1.json"
    dev_in = RAW_DIR / "dev-v1.1.json"

    train_out = OUT_DIR / "squad_train.jsonl"
    dev_out = OUT_DIR / "squad_validation.jsonl"

    n_train = convert_raw_squad_file(train_in, train_out, "train", max_examples=1000)
    n_dev = convert_raw_squad_file(dev_in, dev_out, "validation", max_examples=250)

    print("[done]")
    print(f"train examples written      : {n_train}")
    print(f"validation examples written : {n_dev}")


if __name__ == "__main__":
    main()