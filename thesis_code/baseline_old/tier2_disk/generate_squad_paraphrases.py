import json
import os
from pathlib import Path
from openai import OpenAI

# CONFIG
INPUT_PATH = Path("data/squad_clean/squad_train.jsonl")
OUTPUT_PATH = Path("data/squad_clean/squad_train_paraphrased.jsonl")

PARAPHRASES_PER_QUESTION = 3
MAX_EXAMPLES = 500  # start small, increase later

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def paraphrase_question(question: str, context: str, answer: str):
    prompt = f"""
    You are generating STRICT paraphrases for a controlled question answering evaluation.

    This is NOT a creative task. It is a constrained transformation task.

    Your goal:
    Rewrite the question in different wording while preserving EXACT meaning and answerability.

    HARD CONSTRAINTS (must all be satisfied):
    1. The answer MUST remain EXACTLY: "{answer}"
    2. The paraphrased question MUST be answerable from the SAME context
    3. DO NOT introduce any new information
    4. DO NOT remove key constraints from the question
    5. DO NOT make the question more vague or more general
    6. DO NOT change named entities, numbers, dates, or relationships
    7. DO NOT turn the question into yes/no unless the original was yes/no
    8. DO NOT change question type (who/what/when/where/which/how)
    9. The paraphrase MUST require the SAME span of text as the answer
    10. Keep the difficulty level the SAME

    STRICT LINGUISTIC VARIATION RULES:
    - Only change phrasing, structure, or wording
    - You MAY:
    • reorder clauses
    • replace words with close synonyms
    • change passive/active voice
    • slightly restructure sentence form
    - You MUST NOT:
    • simplify meaning
    • generalize the question
    • make it broader or narrower

    VALIDATION CHECK (before outputting each paraphrase):
    Ask yourself:
    "If I only change the question wording and keep the same context, would the exact same answer string still be correct?"

    If not, DO NOT include that paraphrase.

    INPUT:
    Original Question:
    {question}

    Context:
    {context}

    Required Answer:
    {answer}

    TASK:
    Generate EXACTLY {PARAPHRASES_PER_QUESTION} paraphrases that satisfy ALL constraints.

    OUTPUT FORMAT:
    Return ONLY a valid JSON list of strings.
    No explanations.
    No extra text.
    """

    response = client.responses.create(
        model="gpt-5.2",
        input=prompt,
        max_output_tokens=200,
        temperature=0.2
    )

    text = response.output_text.strip()

    try:
        paras = json.loads(text)
        return paras
    except Exception:
        print("[warn] failed to parse JSON, raw output:")
        print(text)
        return []

def is_valid_paraphrase(p, original_question, answer):
    if not p.strip():
        return False
    if p.strip() == original_question.strip():
        return False
    if answer.lower() not in answer.lower():
        return True  # placeholder, extend later
    return True


def main():
    print(f"[info] reading {INPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    total_written = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as out_f:
        for i, line in enumerate(lines):
            if MAX_EXAMPLES and i >= MAX_EXAMPLES:
                break

            record = json.loads(line)

            context = record["context"]
            question = record["question"]
            answer = record["answer"]
            family_id = record["family_id"]

            # Write original
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_written += 1

            paraphrases = paraphrase_question(question, context, answer)

            for j, p in enumerate(paraphrases):
                new_record = dict(record)
                new_record["question"] = p
                new_record["example_id"] = f"{family_id}_para_{j}"
                new_record["variant"] = f"para_{j}"

                out_f.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                total_written += 1

            if i % 50 == 0:
                print(f"[progress] processed {i} examples")

    print(f"[done] wrote {total_written} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()