#!/usr/bin/env python3
"""CLI chatbot for CodeAgent-RWKV with streaming output and colored thinking blocks."""

import os, sys, readline
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

BASE_MODEL = os.environ.get("BASE_MODEL", "SmerkyG/RWKV7-Goose-0.4B-Pile-HF")
CHECKPOINT = os.environ.get("CHECKPOINT", str(Path(__file__).parent.parent / "checkpoints"))

SYSTEM = "You are a helpful coding assistant. Think step by step before answering."

# ANSI colors
C = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
     "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
     "blue": "\033[34m", "cyan": "\033[36m", "white": "\033[37m",
     "magenta": "\033[35m"}


def colorize(text, mode="thinking"):
    """Colorize thinking vs answer blocks."""
    if mode == "thinking":
        return f"{C['cyan']}{C['dim']}{text}{C['reset']}"
    elif mode == "answer":
        return f"{C['green']}{text}{C['reset']}"
    elif mode == "user":
        return f"{C['yellow']}{C['bold']}{text}{C['reset']}"
    elif mode == "info":
        return f"{C['dim']}{text}{C['reset']}"
    return text


class ColoredStreamer(TextStreamer):
    """Stream tokens with color based on thinking/answer state."""
    def __init__(self, tokenizer, skip_prompt=True):
        super().__init__(tokenizer, skip_prompt=skip_prompt)
        self.in_thinking = True
        self.buffer = ""

    def on_finalized_text(self, text, stream_end=False):
        self.buffer += text
        # Detect mode changes
        displayed = ""
        if "Answer:" in self.buffer and self.in_thinking:
            idx = self.buffer.index("Answer:")
            thinking_part = self.buffer[:idx]
            answer_part = self.buffer[idx:]
            displayed = colorize(thinking_part, "thinking") + colorize(answer_part, "answer")
            self.in_thinking = False
        elif self.in_thinking:
            displayed = colorize(self.buffer, "thinking")
        else:
            displayed = colorize(text, "answer")

        sys.stdout.write(displayed)
        sys.stdout.flush()
        if not self.in_thinking or "Answer:" in self.buffer:
            self.buffer = ""


def generate_stream(tok, model, prompt, max_tokens=256, temp=0.7, top_p=0.9):
    """Generate with streaming output."""
    device = model.device
    inputs = tok(prompt, return_tensors="pt").to(device)

    generated = inputs.input_ids.clone()
    past_state = None
    pad_id = tok.pad_token_id or tok.eos_token_id

    # Print user prompt
    print(f"\n{colorize('Thinking...', 'thinking')}")
    in_answer = False

    for _ in range(max_tokens):
        with torch.no_grad():
            try:
                outputs = model(generated, use_cache=True)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            except:
                outputs = model(generated)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

        next_logits = logits[:, -1, :]

        if temp > 0:
            next_logits = next_logits / temp
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=-1)
        token_id = next_token.item()

        if token_id == pad_id:
            break

        # Decode just this token
        word = tok.decode([token_id])

        # Detect "Answer:" marker for color switch
        if "Answer:" in word and not in_answer:
            in_answer = True
            # Find the Answer: position and split
            parts = word.split("Answer:", 1)
            if len(parts) == 2:
                sys.stdout.write(colorize(parts[0], "thinking"))
                sys.stdout.write(colorize("Answer:", "answer"))
                sys.stdout.write(colorize(parts[1], "answer"))
            else:
                sys.stdout.write(colorize(word, "answer"))
        elif in_answer:
            sys.stdout.write(colorize(word, "answer"))
        else:
            sys.stdout.write(colorize(word, "thinking"))

        sys.stdout.flush()

    print(colorize("\n", "reset"))

    full_text = tok.decode(generated[0], skip_special_tokens=True)
    return full_text[len(prompt):].strip()


def main():
    print(colorize("Loading CodeAgent-RWKV-7 0.4B...", "info"))

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = Path(CHECKPOINT)
    if (ckpt / "model.safetensors").exists():
        print(colorize(f"Loading checkpoint ({device})...", "info"))
        model = AutoModelForCausalLM.from_pretrained(
            CHECKPOINT,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
    else:
        print(colorize(f"Checkpoint not found, loading base model ({device})...", "info"))
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        )

    model.to(device)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(colorize(f"Ready — {params/1e6:.0f}M params on {device.upper()}", "green"))
    print(colorize("Type /help for commands, /quit to exit.", "info"))

    if device == "cpu":
        print(colorize("  ⚠ No GPU detected. Inference will be slow (~30s per response).", "yellow"))

    history = []
    max_tokens = 256
    temp = 0.7
    top_p = 0.9

    while True:
        try:
            user = input(f"\n{C['yellow']}>>> {C['reset']}").strip()
        except (EOFError, KeyboardInterrupt):
            print(colorize("\nGoodbye!", "green"))
            break

        if not user:
            continue
        if user in ("/quit", "/exit"):
            print(colorize("Goodbye!", "green"))
            break
        if user == "/help":
            print(colorize("Commands: /clear /tokens N /temp F /cpu /quit", "info"))
            continue
        if user == "/clear":
            history.clear()
            print(colorize("History cleared.", "info"))
            continue
        if user == "/cpu":
            device = "cpu"
            model.to(device)
            print(colorize(f"Switched to CPU", "info"))
            continue
        if user.startswith("/tokens "):
            max_tokens = int(user.split()[1])
            print(colorize(f"max_tokens = {max_tokens}", "info"))
            continue
        if user.startswith("/temp "):
            temp = float(user.split()[1])
            print(colorize(f"temperature = {temp}", "info"))
            continue

        prompt = f"System: {SYSTEM}"
        for h in history[-6:]:
            prompt += f"\n\nUser: {h['user']}\n\nThinking...\n\nAnswer: {h['assistant']}"
        prompt += f"\n\nUser: {user}\n\nThinking..."

        try:
            response = generate_stream(tok, model, prompt, max_tokens, temp, top_p)
            history.append({"user": user, "assistant": response})
        except Exception as e:
            print(colorize(f"Error: {e}", "red"))


if __name__ == "__main__":
    main()
