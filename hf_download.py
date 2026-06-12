import json
import sys
from datasets import load_dataset

# datasets to pull and how many examples to take (None = all)
SOURCES = [
    {
        "id": "sahil2801/CodeAlpaca-20k",
        "split": "train",
        "limit": None,
        "format": "alpaca",
        "filter": lambda ex: "java" in (ex.get("input") or "").lower()
                          or "java" in (ex.get("instruction") or "").lower()
                          or "java" in (ex.get("output") or "").lower(),
    },
    {
        "id": "nickrosh/Evol-Instruct-Code-80k-v1",
        "split": "train",
        "limit": 30000,
        "format": "alpaca",
        "filter": lambda ex: "java" in (ex.get("instruction") or "").lower()
                          or "java" in (ex.get("output") or "").lower(),
    },
    {
        "id": "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "split": "train",
        "limit": 30000,
        "format": "alpaca",
        "filter": lambda ex: (ex.get("lang") or "").lower() == "java",
    },
]


def to_messages(example: dict, fmt: str) -> dict | None:
    if fmt == "alpaca":
        instruction = (example.get("instruction") or example.get("problem") or "").strip()
        inp = (example.get("input") or "").strip()
        output = (example.get("output") or example.get("solution") or "").strip()
        if not instruction or not output:
            return None
        user = f"{instruction}\n\n{inp}" if inp else instruction
        return {"messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": output},
        ]}

    if fmt == "messages":
        msgs = example.get("messages") or example.get("conversations")
        if not msgs:
            return None
        normalized = []
        for m in msgs:
            role = m.get("role") or m.get("from")
            content = m.get("content") or m.get("value")
            if role in ("human", "user"):
                role = "user"
            elif role in ("gpt", "assistant", "bot"):
                role = "assistant"
            else:
                continue
            if content:
                normalized.append({"role": role, "content": content})
        if len(normalized) < 2:
            return None
        return {"messages": normalized}

    return None


def download(source: dict, output_path: str):
    ds_id = source["id"]
    split = source["split"]
    limit = source["limit"]
    fmt = source["format"]
    filter_fn = source.get("filter")

    print(f"Loading {ds_id} ({split})...")
    ds = load_dataset(ds_id, split=split)

    converted = []
    skipped = 0

    for example in ds:
        if filter_fn and not filter_fn(example):
            continue
        result = to_messages(example, fmt)
        if result:
            converted.append(result)
        else:
            skipped += 1
        if limit and len(converted) >= limit:
            break

    print(f"  {len(converted)} converted, {skipped} skipped -> {output_path}")

    with open(output_path, "w") as f:
        for ex in converted:
            f.write(json.dumps(ex) + "\n")


if __name__ == "__main__":
    # optionally pass a single dataset id to only download that one
    target = sys.argv[1] if len(sys.argv) > 1 else None

    for source in SOURCES:
        if target and target not in source["id"]:
            continue
        safe_name = source["id"].replace("/", "_")
        download(source, f"{safe_name}.jsonl")

    print("\nDone. Now run merge.py to combine everything.")
