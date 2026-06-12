# ✦ Lumen

**Lumen** is Axion Labs' own model — a QLoRA fine-tune of **Llama 3.1 8B Instruct**, trained on coding-instruction datasets and deployed free with no API key.

- **Try it**: [huggingface.co/spaces/RavikxxBGamin/Lumen](https://huggingface.co/spaces/RavikxxBGamin/Lumen) — Gradio chat UI
- **API**: OpenAI-compatible `POST https://ravikxxbgamin-lumen.hf.space/v1/chat/completions`
- **In the Axion CLI**: `/model lumen` (no key needed)

## How it was built

1. **Datasets** — CodeAlpaca-20k, Evol-Instruct-Code-80k, Magicoder-OSS-Instruct-75K, plus custom identity data (`generate.py`), cleaned and merged (`cleanup.py`, `merge.py`). Dataset `.jsonl` files are gitignored; the scripts regenerate them.
2. **Training** — QLoRA fine-tune of Llama 3.1 8B Instruct on Google Colab (`train_lumen.ipynb`).
3. **Conversion** — the LoRA adapter (~168 MB) is converted to GGUF with llama.cpp's `convert_lora_to_gguf.py` (`convert_adapter.ipynb`) and uploaded to [RavikxxBGamin/Lumen](https://huggingface.co/RavikxxBGamin/Lumen).
4. **Serving** — `space/` contains the HF Space (Docker SDK): FastAPI + llama-cpp-python loads the Q4_K_M base model with the LoRA adapter applied at runtime, mounts a Gradio UI at `/` and an OpenAI-compatible streaming API at `/v1/chat/completions`.

## Repo layout

```
space/                  HF Space — app.py, Dockerfile, requirements.txt
train_lumen.ipynb       QLoRA training notebook (Colab)
convert_adapter.ipynb   LoRA → GGUF conversion notebook (Colab)
generate.py             custom identity dataset generator
cleanup.py / merge.py   dataset cleaning + merging
hf_download.py          dataset download helper
PLAN.md                 build plan / notes
```

---

MIT — made by [Axion Labs](https://github.com/Ravikxx)
