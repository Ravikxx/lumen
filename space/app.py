import os
import json
import time
import uuid
import asyncio
import threading
import queue as queue_mod
import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

HF_TOKEN   = os.environ.get("HF_TOKEN")
BASE_PATH  = "/tmp/base.gguf"
LORA_PATH  = "/tmp/lumen-adapter.gguf"

SYSTEM_PROMPT = (
    "You are Lumen, an AI assistant made by Axion Labs. "
    "Give precise, helpful answers to any question the user asks."
)

llm = None  # loaded in background on startup
# llama.cpp contexts are not safe for concurrent generation — one request at a time
infer_lock = threading.Lock()

def _load_model():
    global llm
    if not os.path.exists(BASE_PATH):
        print("Downloading base model...")
        hf_hub_download(
            repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            local_dir="/tmp",
        )
        os.rename("/tmp/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", BASE_PATH)

    if not os.path.exists(LORA_PATH):
        print("Downloading adapter...")
        hf_hub_download(
            repo_id="RavikxxBGamin/Lumen",
            filename="lumen-adapter.gguf",
            token=HF_TOKEN,
            local_dir="/tmp",
        )

    print("Loading model...")
    llm = Llama(
        model_path=BASE_PATH,
        lora_path=LORA_PATH,
        n_ctx=8192,
        n_threads=2,
        verbose=False,
    )
    print("Model ready.")

# ── FastAPI app ───────────────────────────────────────────────────────────────

fastapi_app = FastAPI()

@fastapi_app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _load_model)

@fastapi_app.get("/health")
def health():
    return {"status": "ready" if llm is not None else "loading"}

@fastapi_app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if llm is None:
        return JSONResponse({"error": "Model is still loading, try again in a moment."}, status_code=503)

    body = await request.json()
    messages = body.get("messages", [])
    max_tokens = int(body.get("max_tokens", 512))
    temperature = float(body.get("temperature", 0.7))
    stream = body.get("stream", False)
    model_id = body.get("model", "lumen")

    if not any(m.get("role") == "system" for m in messages):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    if stream:
        async def event_stream():
            resp_id = "chatcmpl-" + uuid.uuid4().hex
            created = int(time.time())
            # Run generation in a worker thread; pass chunks back through a
            # queue so the event loop (health checks, UI) never blocks.
            q = queue_mod.Queue(maxsize=64)
            DONE = object()

            def produce():
                try:
                    with infer_lock:
                        for chunk in llm.create_chat_completion(
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            stream=True,
                        ):
                            q.put(chunk)
                except Exception as e:
                    q.put(e)
                finally:
                    q.put(DONE)

            threading.Thread(target=produce, daemon=True).start()
            while True:
                chunk = await asyncio.to_thread(q.get)
                if chunk is DONE:
                    break
                if isinstance(chunk, Exception):
                    yield f"data: {json.dumps({'error': str(chunk)})}\n\n"
                    break
                delta = chunk["choices"][0]["delta"]
                finish = chunk["choices"][0].get("finish_reason")
                data = {
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    def generate():
        with infer_lock:
            return llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )

    result = await asyncio.to_thread(generate)
    return JSONResponse(result)

# ── Gradio UI ─────────────────────────────────────────────────────────────────

def respond(message, history, temperature, max_tokens):
    if llm is None:
        yield "Model is still loading, please wait a moment and try again."
        return
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history:
        if isinstance(item, dict):
            content = item["content"]
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            messages.append({"role": item["role"], "content": content})
        else:
            user_msg, assistant_msg = item
            messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    response = ""
    with infer_lock:
        for chunk in llm.create_chat_completion(
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=temperature,
            stream=True,
        ):
            delta = chunk["choices"][0]["delta"].get("content", "")
            response += delta
            yield response

def model_status():
    if llm is not None:
        return "<div class='lumen-status ready'>&#9679; Model ready</div>"
    return "<div class='lumen-status loading'>&#9679; Loading model&hellip; (first boot takes a few minutes)</div>"

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="*neutral_950",
    body_background_fill_dark="*neutral_950",
    block_background_fill="*neutral_900",
    block_background_fill_dark="*neutral_900",
    block_border_width="0px",
    button_primary_background_fill="linear-gradient(90deg, #7c3aed, #4f46e5)",
    button_primary_background_fill_hover="linear-gradient(90deg, #8b5cf6, #6366f1)",
)

CSS = """
.gradio-container { max-width: 880px !important; margin: 0 auto !important; }
#lumen-header { text-align: center; padding: 18px 0 4px; }
#lumen-header h1 {
    font-size: 2.4em; font-weight: 800; margin: 0; letter-spacing: -0.02em;
    background: linear-gradient(90deg, #a78bfa, #818cf8, #60a5fa);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
#lumen-header p { color: #94a3b8; margin: 4px 0 0; font-size: 0.95em; }
#lumen-header .badges { margin-top: 8px; }
#lumen-header .badge {
    display: inline-block; font-size: 0.72em; padding: 3px 10px; margin: 0 3px;
    border-radius: 999px; border: 1px solid #3f3f56; color: #c4b5fd; background: rgba(124,58,237,.08);
}
.lumen-status { text-align: center; font-size: 0.8em; padding: 2px 0 8px; }
.lumen-status.ready { color: #4ade80; }
.lumen-status.loading { color: #fbbf24; }
#lumen-footer { text-align: center; color: #64748b; font-size: 0.75em; padding: 10px 0 16px; }
#lumen-footer code { background: #1e1e2e; padding: 2px 6px; border-radius: 6px; color: #a5b4fc; }
footer { display: none !important; }
"""

HEADER_HTML = """
<div id="lumen-header">
    <h1>&#10022; Lumen</h1>
    <p>A fine-tuned Llama 3.1 8B by Axion Labs</p>
    <div class="badges">
        <span class="badge">8B &middot; Q4_K_M</span>
        <span class="badge">LoRA fine-tune</span>
        <span class="badge">Free &middot; no key needed</span>
    </div>
</div>
"""

FOOTER_HTML = """
<div id="lumen-footer">
    OpenAI-compatible API: <code>POST /v1/chat/completions</code>
    &nbsp;&middot;&nbsp; works with the Axion CLI via <code>/model lumen</code>
</div>
"""

with gr.Blocks(theme=THEME, css=CSS, title="Lumen — Axion Labs") as demo:
    gr.HTML(HEADER_HTML)
    status = gr.HTML(model_status)
    gr.ChatInterface(
        fn=respond,
        type="messages",
        examples=[
            ["Who are you?"],
            ["Explain how a binary search works"],
            ["Write a haiku about terminals"],
            ["What's the difference between TCP and UDP?"],
        ],
        additional_inputs=[
            gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature"),
            gr.Slider(64, 1024, value=512, step=64, label="Max tokens"),
        ],
        additional_inputs_accordion=gr.Accordion("⚙️ Settings", open=False),
    )
    gr.HTML(FOOTER_HTML)
    demo.load(model_status, outputs=status)

app = gr.mount_gradio_app(fastapi_app, demo, path="/")
