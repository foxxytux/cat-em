#!/usr/bin/env python3
"""CodeAgent-RWKV FastAPI server — optimized for speed."""

import os, sys, json, time, logging
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("codeagent")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_MODEL = os.environ.get("BASE_MODEL", "SmerkyG/RWKV7-Goose-0.4B-Pile-HF")
CHECKPOINT = os.environ.get("CHECKPOINT", str(Path(__file__).parent.parent.parent / "checkpoints"))
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8080"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))

# ---------------------------------------------------------------------------
# Optimized model holder
# ---------------------------------------------------------------------------
@dataclass
class Engine:
    model: torch.nn.Module
    tokenizer: AutoTokenizer
    device: str

engine: Engine | None = None


def load_engine():
    global engine
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    log.info(f"Loading tokenizer from {BASE_MODEL}")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ckpt = Path(CHECKPOINT)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if (ckpt / "model.safetensors").exists():
        log.info(f"Loading checkpoint {CHECKPOINT}")
        model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, torch_dtype=dtype, trust_remote_code=True)
    else:
        log.info(f"Loading base model {BASE_MODEL}")
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype, trust_remote_code=True)

    model.to(device)
    model.eval()

    # Compile for speed (first inference will be slow, subsequent fast)
    if device == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            log.info("torch.compile enabled")
        except Exception as e:
            log.warning(f"torch.compile failed: {e}")

    params = sum(p.numel() for p in model.parameters())
    log.info(f"Ready: {params/1e6:.0f}M params on {device.upper()}")
    engine = Engine(model=model, tokenizer=tok, device=device)


# ---------------------------------------------------------------------------
# Fast generate
# ---------------------------------------------------------------------------
@torch.inference_mode()
def generate(prompt: str, max_tokens: int, temp: float, top_p: float) -> tuple[str, int, float]:
    t0 = time.perf_counter()
    tok = engine.tokenizer
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(engine.device)
    pad_id = tok.pad_token_id or tok.eos_token_id

    output = engine.model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=temp > 0,
        temperature=temp if temp > 0 else 1.0,
        top_p=top_p,
        pad_token_id=pad_id,
        eos_token_id=pad_id,
        use_cache=True,
        num_beams=1,
    )
    text = tok.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    elapsed = time.perf_counter() - t0
    new_tokens = output.shape[1] - inputs.input_ids.shape[1]
    return text, new_tokens, elapsed


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_engine()
    yield


app = FastAPI(title="CAT-EM API", version="1.0.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    prompt: str
    system: str = "You are a helpful coding assistant. Think step by step."
    max_tokens: int = MAX_TOKENS
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    stream: bool = False


@app.get("/health")
async def health():
    return {"status": "ok", "device": engine.device}


@app.post("/v1/chat")
async def chat(req: ChatRequest):
    full = f"System: {req.system}\n\nUser: {req.prompt}\n\nThinking..."
    text, tokens, elapsed = generate(full, req.max_tokens, req.temperature, req.top_p)
    return {"text": text, "usage": {"tokens": tokens, "time": round(elapsed, 3)}}


@app.post("/v1/chat/stream")
async def chat_stream(req: ChatRequest):
    full = f"System: {req.system}\n\nUser: {req.prompt}\n\nThinking..."
    text, tokens, elapsed = generate(full, req.max_tokens, req.temperature, req.top_p)

    async def events():
        yield f"data: {json.dumps({'text': text, 'tokens': tokens, 'time': elapsed})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/v1/complete")
async def complete(req: ChatRequest):
    text, tokens, elapsed = generate(req.prompt, req.max_tokens, req.temperature, req.top_p)
    return {"text": text, "usage": {"tokens": tokens, "time": round(elapsed, 3)}}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
