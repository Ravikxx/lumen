# Lumen Deployment Plan

## Strategy
Skip the full model merge entirely. Convert only the LoRA adapter (~200MB) to GGUF format.
The Space downloads a public Llama 3.1 8B Q4 GGUF + the adapter and loads both.
No A100, no 16GB files, no numpy conflicts.

---

## Step 1 — Convert LoRA adapter to GGUF (Colab, CPU runtime is fine)

Notebook: `convert_adapter.ipynb`

### Cell 1: Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Clone llama.cpp
```python
!git clone https://github.com/ggerganov/llama.cpp /content/llama.cpp
```

### Cell 3: Download base model config only (no weights — just architecture info)
```python
from huggingface_hub import hf_hub_download
import os

HF_TOKEN = ''  # paste HF token
BASE_DIR = '/content/base-config'
os.makedirs(BASE_DIR, exist_ok=True)

for f in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']:
    path = hf_hub_download('meta-llama/Meta-Llama-3.1-8B-Instruct', f, token=HF_TOKEN, local_dir=BASE_DIR)
    print(f'Downloaded {f}')
```

### Cell 4: Convert adapter to GGUF
```python
ADAPTER = '/content/drive/MyDrive/Lumen/lumen-checkpoints/checkpoint-2643'
OUT     = '/content/lumen-adapter.gguf'

!pip install -q gguf

!python /content/llama.cpp/convert_lora_to_gguf.py \
    {ADAPTER} \
    --base {BASE_DIR} \
    --outfile {OUT}

import os
if os.path.exists(OUT) and os.path.getsize(OUT) > 1e6:
    print('Adapter GGUF done. Size:', round(os.path.getsize(OUT)/1e6, 1), 'MB')
else:
    print('ERROR: adapter GGUF not created.')
```

### Cell 5: Upload adapter to HF
```python
from huggingface_hub import HfApi

HF_TOKEN = ''  # paste HF token
api = HfApi(token=HF_TOKEN)
api.upload_file(
    path_or_fileobj='/content/lumen-adapter.gguf',
    path_in_repo='lumen-adapter.gguf',
    repo_id='RavikxxBGamin/Lumen',
    repo_type='model',
)
print('Uploaded.')
```

**CHECKPOINT:** If cell 4 fails saying it needs model weights, we fall back to Option A
(merge on A100 with reordered cells — merge first, then llama.cpp).

---

## Step 2 — Update Space

### app.py
- On startup: download public `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` Q4_K_M + `lumen-adapter.gguf`
- Load with `llama-cpp-python` using `lora_path`
- Gradio ChatInterface with streaming

### requirements.txt
```
llama-cpp-python
gradio
huggingface_hub
```

---

## Status
- [ ] Step 1: Convert + upload adapter
- [ ] Step 2: Update Space
