#!/usr/bin/env python3
"""CAT-EM CLI chatbot — colored streaming with optimized inference."""

import os, sys, readline
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = os.environ.get("BASE_MODEL", "SmerkyG/RWKV7-Goose-0.4B-Pile-HF")
CHECKPOINT = os.environ.get("CHECKPOINT", str(Path(__file__).parent.parent / "checkpoints"))
SYSTEM = "You are a helpful coding assistant. Think step by step."

C = {"r": "\033[0m", "b": "\033[1m", "d": "\033[2m",
     "g": "\033[32m", "y": "\033[33m", "c": "\033[36m"}


def clr(text, mode):
    if mode == "thinking": return f"{C['c']}{C['d']}{text}{C['r']}"
    if mode == "answer":   return f"{C['g']}{text}{C['r']}"
    if mode == "user":     return f"{C['y']}{C['b']}{text}{C['r']}"
    if mode == "dim":      return f"{C['d']}{text}{C['r']}"
    if mode == "green":    return f"{C['g']}{text}{C['r']}"
    return text


@torch.inference_mode()
def stream_generate(tok, model, prompt, max_tokens=256, temp=0.7, top_p=0.9):
    device = model.device
    pad_id = tok.pad_token_id or tok.eos_token_id
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    input_len = inputs.input_ids.shape[1]
    ids = inputs.input_ids

    print(f"\n{clr('Thinking...', 'thinking')}")
    in_answer = False
    new_tokens = 0

    for _ in range(max_tokens):
        with torch.inference_mode():
            out = model(ids, use_cache=True)
            logits = out.logits if hasattr(out, 'logits') else out[0]

        next_logits = logits[:, -1, :] / (temp if temp > 0 else 1.0)

        if temp > 0 and top_p < 1.0:
            s, si = torch.sort(next_logits, descending=True)
            cp = torch.cumsum(torch.softmax(s, dim=-1), dim=-1)
            mask = cp > top_p
            mask[:, 1:] = mask[:, :-1].clone()
            mask[:, 0] = False
            idx = mask.scatter(1, si, mask)
            next_logits[idx] = float('-inf')
            probs = torch.softmax(next_logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
        elif temp > 0:
            nxt = torch.multinomial(torch.softmax(next_logits, dim=-1), 1)
        else:
            nxt = torch.argmax(next_logits, dim=-1, keepdim=True)

        tid = nxt.item()
        if tid == pad_id:
            break

        ids = torch.cat([ids, nxt], dim=-1)
        new_tokens += 1
        word = tok.decode([tid])

        if "Answer:" in word and not in_answer:
            in_answer = True
            parts = word.split("Answer:", 1)
            sys.stdout.write(clr(parts[0], "thinking"))
            sys.stdout.write(clr("Answer:" + parts[1], "answer"))
        elif in_answer:
            sys.stdout.write(clr(word, "answer"))
        else:
            sys.stdout.write(clr(word, "thinking"))
        sys.stdout.flush()

    sys.stdout.write(C['r'] + "\n")
    return tok.decode(ids[0, input_len:], skip_special_tokens=True).strip(), new_tokens


def main():
    print(clr("CAT-EM · RWKV-7 0.4B — loading...", "dim"))

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tok.pad_token = tok.eos_token or tok.pad_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    ckpt = Path(CHECKPOINT)
    if (ckpt / "model.safetensors").exists():
        model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, torch_dtype=dtype, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype, trust_remote_code=True)

    model.to(device)
    model.eval()
    if device == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except:
            pass

    p = sum(x.numel() for x in model.parameters())
    print(clr(f"Ready — {p/1e6:.0f}M params · {device.upper()}", "green"))
    if device == "cpu":
        print(clr("  CPU mode — expect ~30s per response", "dim"))
    print(clr("  /help /tokens N /temp F /clear /quit", "dim"))

    history = []
    mt, tp, tpp = 256, 0.7, 0.9

    while True:
        try:
            user = input(f"\n{C['y']}{C['b']}»{C['r']} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(clr("\nbye", "dim"))
            break

        if not user: continue
        if user in ("/quit", "/exit"):
            print(clr("bye", "dim")); break
        if user == "/help":
            print(clr("/tokens N /temp F /clear /quit", "dim")); continue
        if user == "/clear":
            history.clear(); print(clr("cleared", "dim")); continue
        if user.startswith("/tokens "):
            mt = int(user.split()[1]); print(clr(f"max_tokens={mt}", "dim")); continue
        if user.startswith("/temp "):
            tp = float(user.split()[1]); print(clr(f"temp={tp}", "dim")); continue

        prompt = f"System: {SYSTEM}"
        for h in history[-6:]:
            prompt += f"\n\nUser: {h['u']}\n\nThinking...\n\nAnswer: {h['a']}"
        prompt += f"\n\nUser: {user}\n\nThinking..."

        text, nt = stream_generate(tok, model, prompt, mt, tp, tpp)
        history.append({"u": user, "a": text})


if __name__ == "__main__":
    main()
