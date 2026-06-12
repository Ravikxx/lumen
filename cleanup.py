import json
import re
import sys
from pathlib import Path

AI_PHRASES = [
    "as an ai",
    "as a language model",
    "as an artificial intelligence",
    "i'm an ai",
    "i am an ai",
    "i don't have access to",
    "i don't have the ability",
    "i cannot browse",
    "i can't browse",
    "i cannot access the internet",
    "as chatgpt",
    "as gpt",
    "openai",
    "i was trained",
    "my training data",
    "my knowledge cutoff",
    "as of my last update",
    "i must clarify that i",
    "i should note that i",
]

PLACEHOLDER_PHRASES = [
    "insert code here",
    "your code here",
    "add your",
    "// ...",
    "/* ... */",
    "todo:",
    "fixme:",
    "put your",
]

MIN_ANSWER_LEN = 80
MIN_QUESTION_LEN = 15


def count_code_fences(text: str) -> int:
    return text.count("```")


def is_clean(example: dict) -> tuple[bool, str]:
    msgs = example.get("messages", [])
    if not msgs or len(msgs) < 2:
        return False, "missing messages"

    user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
    assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

    if not user_msg or not assistant_msg:
        return False, "empty role content"

    if len(user_msg.strip()) < MIN_QUESTION_LEN:
        return False, f"question too short ({len(user_msg.strip())} chars)"

    if len(assistant_msg.strip()) < MIN_ANSWER_LEN:
        return False, f"answer too short ({len(assistant_msg.strip())} chars)"

    lower_answer = assistant_msg.lower()
    lower_question = user_msg.lower()

    for phrase in AI_PHRASES:
        if phrase in lower_answer:
            return False, f"AI identity phrase: '{phrase}'"

    for phrase in PLACEHOLDER_PHRASES:
        if phrase in lower_answer or phrase in lower_question:
            return False, f"placeholder phrase: '{phrase}'"

    fences = count_code_fences(assistant_msg)
    if fences % 2 != 0:
        return False, f"truncated code block (odd fence count: {fences})"

    # flag suspiciously long answers that are likely hallucinated walls of text
    if len(assistant_msg) > 8000:
        return False, f"answer suspiciously long ({len(assistant_msg)} chars)"

    return True, ""


def dedup_key(example: dict) -> str:
    msgs = example.get("messages", [])
    user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
    return user_msg.strip().lower()[:200]


def cleanup(input_path: str, output_path: str, verbose: bool = False):
    examples = []
    with open(input_path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"  Skipping malformed line {i}: {e}")

    print(f"Loaded {len(examples)} examples from {input_path}")

    kept = []
    removed_reasons: dict[str, int] = {}
    seen_keys: set[str] = set()

    for ex in examples:
        ok, reason = is_clean(ex)
        if not ok:
            removed_reasons[reason] = removed_reasons.get(reason, 0) + 1
            continue

        key = dedup_key(ex)
        if key in seen_keys:
            removed_reasons["duplicate"] = removed_reasons.get("duplicate", 0) + 1
            continue
        seen_keys.add(key)
        kept.append(ex)

    removed = len(examples) - len(kept)
    print(f"Removed {removed} examples ({removed/len(examples)*100:.1f}%)")

    if removed_reasons:
        print("Removal breakdown:")
        for reason, count in sorted(removed_reasons.items(), key=lambda x: -x[1]):
            print(f"  {count:>6}  {reason}")

    with open(output_path, "w") as f:
        for ex in kept:
            f.write(json.dumps(ex) + "\n")

    print(f"Kept {len(kept)} examples -> {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cleanup.py <input.jsonl> [output.jsonl]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace(".jsonl", "_clean.jsonl")
    cleanup(input_path, output_path)
