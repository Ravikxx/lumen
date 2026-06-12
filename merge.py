import json
import random
import sys
from pathlib import Path


def load_jsonl(path: str, limit: int | None = None) -> list[dict]:
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
            if limit and len(examples) >= limit:
                break
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
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": example["output"]},
        ]}

    # {"prompt": "...", "completion": "..."}
    if "prompt" in example and "completion" in example:
        return {"messages": [
            {"role": "user",      "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]}

    # {"question": "...", "answer": "..."}
    if "question" in example and "answer" in example:
        return {"messages": [
            {"role": "user",      "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]}

    # {"query": "...", "response": "..."} — MetaMathQA
    if "query" in example and "response" in example:
        return {"messages": [
            {"role": "user",      "content": example["query"]},
            {"role": "assistant", "content": example["response"]},
        ]}

    # {"conversations": [{"from": "human"|"gpt", "value": "..."}]} — OpenHermes / ShareGPT
    if "conversations" in example:
        convs = example["conversations"]
        messages = []
        for turn in convs:
            role_raw = turn.get("from", "")
            content  = turn.get("value", "").strip()
            if not content:
                continue
            if role_raw in ("human", "user"):
                messages.append({"role": "user",      "content": content})
            elif role_raw in ("gpt", "assistant"):
                messages.append({"role": "assistant", "content": content})
            elif role_raw == "system":
                messages.append({"role": "system",    "content": content})
        if len(messages) >= 2:
            return {"messages": messages}

    return None


def merge(inputs: list[str], output: str, shuffle: bool = True,
          limits: dict[str, int] | None = None):
    all_examples = []
    limits = limits or {}

    for path in inputs:
        limit = limits.get(path)
        raw = load_jsonl(path, limit=limit)
        normalized, skipped = [], 0
        for ex in raw:
            result = normalize(ex)
            if result:
                normalized.append(result)
            else:
                skipped += 1
        lbl = f" (capped at {limit})" if limit else ""
        print(f"{path}: {len(normalized)} loaded{lbl}, {skipped} skipped")
        all_examples.extend(normalized)

    if shuffle:
        random.shuffle(all_examples)

    with open(output, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nMerged {len(all_examples)} total examples -> {output}")


if __name__ == "__main__":
    # usage: python merge.py output.jsonl file1.jsonl [file2.jsonl ...]
    # optional per-file cap:  file.jsonl:10000
    if len(sys.argv) < 3:
        print("Usage: python merge.py <output.jsonl> <input1.jsonl[:limit]> [input2.jsonl[:limit]] ...")
        print("Example: python merge.py merged_121.jsonl clean.jsonl metamath_15k.jsonl openhermes_6k.jsonl")
        sys.exit(1)

    output_path = sys.argv[1]
    input_paths = []
    limits = {}
    for arg in sys.argv[2:]:
        if ":" in arg and not arg.startswith("/"):
            path, lim = arg.rsplit(":", 1)
            input_paths.append(path)
            limits[path] = int(lim)
        else:
            input_paths.append(arg)

    merge(input_paths, output_path, limits=limits)
