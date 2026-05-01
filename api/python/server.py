#!/usr/bin/env python3
"""CodeAgent-RWKV REST API server. Run with: python3 server.py"""

import os, sys, json, time, logging
from pathlib import Path
from contextlib import asynccontextmanager

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
# Model
# ---------------------------------------------------------------------------
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    global model, tokenizer
    log.info(f"Loading tokenizer from {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ckpt_path = Path(CHECKPOINT)
    if (ckpt_path / "model.safetensors").exists():
        log.info(f"Loading checkpoint from {CHECKPOINT}")
        model = AutoModelForCausalLM.from_pretrained(
            CHECKPOINT, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
    else:
        log.info(f"Loading base model from {BASE_MODEL}")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        )

    model.to(device)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    log.info(f"Model loaded: {params/1e6:.0f}M params on {device}")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="CodeAgent-RWKV API",
    version="1.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User prompt text")
    system: str = Field(default="You are a helpful coding assistant. Think step by step.", description="System prompt")
    max_tokens: int = Field(default=MAX_TOKENS, le=2048)
    temperature: float = Field(default=TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=TOP_P, ge=0.0, le=1.0)
    stream: bool = Field(default=False)


class ChatResponse(BaseModel):
    text: str
    usage: dict


def generate(prompt: str, max_tokens: int, temperature: float, top_p: float):
    start = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    elapsed = time.time() - start
    tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    return text, tokens, elapsed


@app.get("/health")
async def health():
    return {"status": "ok", "device": device, "model": "CodeAgent-RWKV7-0.4B"}


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    full_prompt = f"System: {req.system}\n\nUser: {req.prompt}\n\nThinking..."
    try:
        text, tokens, elapsed = generate(full_prompt, req.max_tokens, req.temperature, req.top_p)
        return ChatResponse(text=text, usage={"tokens": tokens, "time": round(elapsed, 2)})
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/v1/chat/stream")
async def chat_stream(req: ChatRequest):
    full_prompt = f"System: {req.system}\n\nUser: {req.prompt}\n\nThinking..."
    req.stream = True

    async def event_stream():
        try:
            text, tokens, elapsed = generate(full_prompt, req.max_tokens, req.temperature, req.top_p)
            yield f"data: {json.dumps({'text': text, 'tokens': tokens, 'time': elapsed})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/complete")
async def complete(req: ChatRequest):
    try:
        text, tokens, elapsed = generate(req.prompt, req.max_tokens, req.temperature, req.top_p)
        return ChatResponse(text=text, usage={"tokens": tokens, "time": round(elapsed, 2)})
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
