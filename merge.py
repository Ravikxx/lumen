import json
import random
import sys
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    examples = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Skipping malformed line {i} in {path}: {e}")
    return examples


def normalize(example: dict) -> dict | None:
    """Convert common foreign formats into {"messages": [...]} format."""
    # already correct format
    if "messages" in example:
        msgs = example["messages"]
        if isinstance(msgs, list) and all("role" in m and "content" in m for m in msgs):
            return example

    # {"instruction": "...", "output": "..."} — alpaca style
    if "instruction" in example and "output" in example:
        user_content = example["instruction"]
        if example.get("input"):
            user_content += "\n\n" + example["input"]
        return {"messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]},
        ]}

    # {"prompt": "...", "completion": "..."}
    if "prompt" in example and "completion" in example:
        return {"messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]}

    # {"question": "...", "answer": "..."}
    if "question" in example and "answer" in example:
        return {"messages": [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]}

    return None


def merge(inputs: list[str], output: str, shuffle: bool = True):
    all_examples = []

    for path in inputs:
        raw = load_jsonl(path)
        normalized = []
        skipped = 0
        for ex in raw:
            result = normalize(ex)
            if result:
                normalized.append(result)
            else:
                skipped += 1
        print(f"{path}: {len(normalized)} loaded, {skipped} skipped")
        all_examples.extend(normalized)

    if shuffle:
        random.shuffle(all_examples)

    with open(output, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nMerged {len(all_examples)} total examples -> {output}")


if __name__ == "__main__":
    # usage: python merge.py output.jsonl file1.jsonl file2.jsonl ...
    if len(sys.argv) < 3:
        print("Usage: python merge.py <output.jsonl> <input1.jsonl> [input2.jsonl ...]")
        sys.exit(1)

    output_path = sys.argv[1]
    input_paths = sys.argv[2:]
    merge(input_paths, output_path)
