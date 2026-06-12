import os
import json
import time
import random
import requests
from pathlib import Path

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

GROQ_MODEL = "llama-3.3-70b-versatile"
OPENROUTER_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
    "qwen/qwen3-coder:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "openai/gpt-oss-120b:free",
]
_openrouter_idx = 0

OUTPUT_FILE = "dataset.jsonl"
TARGET_COUNT = 50000
BATCH_SIZE = 10

TOPICS = [
    # Paper API
    "Paper API event system and listeners",
    "Paper API custom commands and tab completion",
    "Paper API player data and persistence",
    "Paper API world manipulation and chunk loading",
    "Paper API entity AI and pathfinding",
    "Paper API inventory and item management",
    "Paper API scheduling and async tasks",
    "Paper API block manipulation and custom structures",
    "Paper API plugin messaging and BungeeCord",
    "Paper API NBT and custom item metadata",
    "Paper API permissions and rank systems",
    "Paper API scoreboard and teams",
    "Paper API particle effects and sounds",
    "Paper API GUIs and custom menus",
    "Paper API packet handling",
    # General Java
    "Java collections and generics",
    "Java concurrency and threading",
    "Java streams and lambdas",
    "Java design patterns",
    "Java I/O and NIO file handling",
    "Java networking and sockets",
    "Java reflection and annotations",
    "Java memory management and GC tuning",
    "Java functional interfaces and method references",
    "Java records, sealed classes, and modern features",
    "Java exception handling best practices",
    "Java serialization and deserialization",
    "Java build tools (Maven, Gradle)",
    "Java unit testing with JUnit",
    "Java performance optimization",
    # Minecraft mechanics
    "Minecraft redstone mechanics",
    "Minecraft mob behavior and spawning rules",
    "Minecraft world generation and biomes",
    "Minecraft enchanting and anvil mechanics",
    "Minecraft farming and crop growth",
    "Minecraft combat mechanics and damage calculations",
    "Minecraft villager trading and curing",
    "Minecraft potion brewing",
    "Minecraft NBT data and data packs",
    "Minecraft server optimization and performance",
]

SYSTEM_PROMPT = """You generate training data for a Minecraft Paper plugin development assistant.
Generate a realistic question a developer might ask, followed by a precise, helpful answer.
For code examples use Java. Be concise but complete.

Respond using EXACTLY this format with these two markers and nothing else:

QUESTION:
<the question here>

ANSWER:
<the answer here>"""


def make_example_prompt(topic: str) -> str:
    return f"Generate one question and answer pair about: {topic}"


def call_groq(prompt: str) -> str | None:
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.9,
            "max_tokens": 1024,
        },
        timeout=30,
    )
    if resp.status_code == 429:
        print("  Groq rate limited, falling back...")
        return None
    if not resp.ok:
        print(f"  Groq {resp.status_code}: {resp.text[:200]}")
        resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_openrouter(prompt: str) -> str | None:
    global _openrouter_idx
    model = OPENROUTER_MODELS[_openrouter_idx % len(OPENROUTER_MODELS)]
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.9,
            "max_tokens": 1024,
        },
        timeout=60,
    )
    if resp.status_code == 429:
        print(f"  OpenRouter 429 on {model}, rotating model...")
        _openrouter_idx += 1
        raise requests.HTTPError("429", response=resp)
    if not resp.ok:
        print(f"  OpenRouter {resp.status_code} ({model}): {resp.text[:200]}")
        _openrouter_idx += 1
        resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_response(raw: str) -> dict | None:
    try:
        raw = raw.strip()
        if "QUESTION:" not in raw or "ANSWER:" not in raw:
            return None
        q_start = raw.index("QUESTION:") + len("QUESTION:")
        a_start = raw.index("ANSWER:") + len("ANSWER:")
        question = raw[q_start:raw.index("ANSWER:")].strip()
        answer = raw[a_start:].strip()
        if not question or not answer:
            return None
        return {"user": question, "assistant": answer}
    except Exception:
        return None


def generate_example(topic: str) -> dict | None:
    prompt = make_example_prompt(topic)
    raw = None

    # try Groq first
    try:
        raw = call_groq(prompt)
    except Exception as e:
        print(f"  Groq error: {e}")

    # fallback to OpenRouter
    if raw is None:
        try:
            print("  Falling back to OpenRouter...")
            raw = call_openrouter(prompt)
            time.sleep(0.5)
        except Exception as e:
            print(f"  OpenRouter error: {e}")
            return None

    parsed = parse_response(raw)
    if not parsed or "user" not in parsed or "assistant" not in parsed:
        print(f"  Parse failed. Raw response: {repr(raw[:300]) if raw else None}")
        return None

    return {
        "messages": [
            {"role": "user", "content": parsed["user"]},
            {"role": "assistant", "content": parsed["assistant"]},
        ]
    }


def count_existing(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    return sum(1 for _ in p.open())


def main():
    existing = count_existing(OUTPUT_FILE)
    print(f"Resuming from {existing} existing examples, target {TARGET_COUNT}")

    with open(OUTPUT_FILE, "a") as f:
        generated = existing
        errors = 0

        while generated < TARGET_COUNT:
            topic = random.choice(TOPICS)
            example = generate_example(topic)

            if example:
                f.write(json.dumps(example) + "\n")
                f.flush()
                generated += 1
                errors = 0
                if generated % 100 == 0:
                    print(f"Progress: {generated}/{TARGET_COUNT}")
            else:
                errors += 1
                print(f"  Failed to generate example (consecutive errors: {errors})")
                if errors >= 10:
                    print("Too many consecutive errors, sleeping 60s...")
                    time.sleep(60)
                    errors = 0
                else:
                    time.sleep(2)

    print(f"Done! Generated {TARGET_COUNT} examples in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
