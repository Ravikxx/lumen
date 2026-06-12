#!/usr/bin/env python3
"""
Download MetaMathQA and OpenHermes for Lumen 1.2.1 retraining mix.
Saves subsampled JSONL files alongside the existing coding datasets.

Usage: python3 download_mix.py
"""
import json
import random
from pathlib import Path
from datasets import load_dataset

SEED = 42
random.seed(SEED)

# ── config ───────────────────────────────────────────────────────────────────
DOWNLOADS = [
    {
        "name":    "MetaMathQA",
        "hf_id":   "meta-math/MetaMathQA",
        "split":   "train",
        "limit":   15_000,
        "out":     "metamath_15k.jsonl",
        # filter: prefer GSM8K-origin problems (cleaner, less synthetic noise)
        "filter":  lambda ex: "GSM" in ex.get("type", "") or "MATH" in ex.get("type", ""),
    },
    {
        "name":    "OpenHermes-2.5",
        "hf_id":   "teknium/OpenHermes-2.5",
        "split":   "train",
        "limit":   6_000,
        "out":     "openhermes_6k.jsonl",
        "filter":  None,
    },
]

# ── normalizers ───────────────────────────────────────────────────────────────
def norm_metamath(ex: dict) -> dict | None:
    q = ex.get("query", "").strip()
    a = ex.get("response", "").strip()
    if not q or not a:
        return None
    return {"messages": [
        {"role": "user",      "content": q},
        {"role": "assistant", "content": a},
    ]}

def norm_openhermes(ex: dict) -> dict | None:
    convs = ex.get("conversations", [])
    if not convs:
        return None
    messages = []
    for turn in convs:
        role_raw = turn.get("from", "")
        content  = turn.get("value", "").strip()
        if not content:
            continue
        if role_raw in ("human", "user"):
            role = "user"
        elif role_raw in ("gpt", "assistant"):
            role = "assistant"
        elif role_raw == "system":
            role = "system"
        else:
            continue
        messages.append({"role": role, "content": content})
    if len(messages) < 2:
        return None
    return {"messages": messages}

NORMALIZERS = {
    "MetaMathQA":     norm_metamath,
    "OpenHermes-2.5": norm_openhermes,
}

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    for cfg in DOWNLOADS:
        name   = cfg["name"]
        out    = Path(cfg["out"])
        limit  = cfg["limit"]
        norm   = NORMALIZERS[name]
        filt   = cfg["filter"]

        if out.exists():
            existing = sum(1 for _ in out.open())
            print(f"{name}: {out} already exists ({existing} lines) — skipping. Delete to re-download.")
            continue

        print(f"\n{name}: loading from HuggingFace…")
        ds = load_dataset(cfg["hf_id"], split=cfg["split"])
        examples = list(ds)

        if filt:
            filtered = [ex for ex in examples if filt(ex)]
            print(f"  {len(examples)} total → {len(filtered)} after filter")
            examples = filtered

        random.shuffle(examples)
        examples = examples[:limit]

        normalized, skipped = [], 0
        for ex in examples:
            result = norm(ex)
            if result:
                normalized.append(result)
            else:
                skipped += 1

        print(f"  {len(normalized)} normalized, {skipped} skipped")

        with open(out, "w") as f:
            for ex in normalized:
                f.write(json.dumps(ex) + "\n")

        print(f"  Saved → {out}")

    print("\nDone. Now run:")
    print("  python3 merge.py merged_121.jsonl clean.jsonl metamath_15k.jsonl openhermes_6k.jsonl")

if __name__ == "__main__":
    main()
