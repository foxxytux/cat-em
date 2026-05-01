#!/usr/bin/env python3
"""
Robust single-file training script for CodeAgent-RWKV
Designed to run on Vast.ai with limited VRAM.
"""

import os
import sys
import json
import math
import time
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

# Configure logging
os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("./logs/training.log"),
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    phase: int
    ctx_len: int
    batch_size: int
    grad_accum: int
    lr: float
    min_lr: float
    weight_decay: float
    max_grad_norm: float
    num_epochs: int
    warmup_ratio: float
    output_dir: str
    base_model: str
    datasets: List[str]
    
    @classmethod
    def from_phase(cls, phase: int):
        configs = {
            # Phase 0: Real continued pre-training (no ctx extension, heavy data)
            0: cls(phase=0, ctx_len=4096, batch_size=1, grad_accum=4,
                   lr=5e-5, min_lr=5e-7, weight_decay=0.01, max_grad_norm=1.0,
                   num_epochs=10, warmup_ratio=0.02,
                   output_dir="./checkpoints/real",
                   base_model="SmerkyG/RWKV7-Goose-0.4B-Pile-HF",
                   datasets=[]),
            # Context extension phases (quick fine-tune)
            1: cls(phase=1, ctx_len=4096, batch_size=1, grad_accum=4,
                   lr=5e-5, min_lr=5e-6, weight_decay=0.01, max_grad_norm=1.0,
                   num_epochs=1, warmup_ratio=0.05,
                   output_dir="./checkpoints/phase1",
                   base_model="SmerkyG/RWKV7-Goose-0.4B-Pile-HF",
                   datasets=[]),
            2: cls(phase=2, ctx_len=16384, batch_size=1, grad_accum=8,
                   lr=2e-5, min_lr=2e-6, weight_decay=0.01, max_grad_norm=1.0,
                   num_epochs=1, warmup_ratio=0.03,
                   output_dir="./checkpoints/phase2",
                   base_model="./checkpoints/phase1/final",
                   datasets=[]),
            3: cls(phase=3, ctx_len=65536, batch_size=1, grad_accum=16,
                   lr=1e-5, min_lr=1e-6, weight_decay=0.01, max_grad_norm=1.0,
                   num_epochs=1, warmup_ratio=0.02,
                   output_dir="./checkpoints/phase3",
                   base_model="./checkpoints/phase2/final",
                   datasets=[]),
            4: cls(phase=4, ctx_len=131072, batch_size=1, grad_accum=32,
                   lr=5e-6, min_lr=5e-7, weight_decay=0.01, max_grad_norm=1.0,
                   num_epochs=1, warmup_ratio=0.01,
                   output_dir="./checkpoints/phase4",
                   base_model="./checkpoints/phase3/final",
                   datasets=[]),
        }
        return configs[phase]

# ---------------------------------------------------------------------------
# Thinking Format
# ---------------------------------------------------------------------------

THINKING_TOKEN = "Thinking..."
ANSWER_TOKEN = "Answer:"
USER_TOKEN = "User:"
SYSTEM_PROMPT = "You are a helpful coding assistant. Think step by step before providing your answer."

def apply_thinking_format(text: str, is_code: bool = False) -> str:
    """Apply thinking format to text data."""
    if THINKING_TOKEN in text or ANSWER_TOKEN in text:
        return text  # Already formatted
    
    if is_code:
        return f"{USER_TOKEN} Write code.\n\n{THINKING_TOKEN}\nLet me think about the implementation...\n\n{ANSWER_TOKEN}\n{text}"
    else:
        return f"{USER_TOKEN} {text[:200]}\n\n{THINKING_TOKEN}\nLet me analyze this carefully...\n\n{ANSWER_TOKEN}\n{text}"

# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

DATASET_SPECS = {
    # Code
    "m-a-p/CodeFeedback-Filtered-Instruction": {"category": "code", "format": "instruction"},
    "princeton-nlp/SWE-bench_Verified": {"category": "code", "format": "swe"},
    "HuggingFaceTB/smollm-corpus": {"category": "code", "format": "code", "streaming": True},
    "iamtarun/python_code_instructions_18k_alpaca": {"category": "code", "format": "instruction"},
    # Agentic
    "THUDM/AgentInstruct": {"category": "agentic", "format": "instruction"},
    "tuandunghcmut/toolbench-v1": {"category": "agentic", "format": "tool"},
    "open-thoughts/OpenThoughts-Agent-v1-SFT": {"category": "agentic", "format": "reasoning"},
    "princeton-nlp/SWE-bench": {"category": "agentic", "format": "swe"},
    "Lite-Coder/LiteCoder-SFT-Terminal-preview": {"category": "agentic", "format": "tool"},
    "TIGER-Lab/SWE-Next-SFT-Trajectories": {"category": "agentic", "format": "swe"},
    # Reasoning
    "open-thoughts/OpenThoughts-114k": {"category": "reasoning", "format": "reasoning", "streaming": True},
    # Conversation
    "HuggingFaceTB/smoltalk": {"category": "conversation", "format": "conversation"},
}

CATEGORY_WEIGHTS = {"code": 0.30, "agentic": 0.30, "reasoning": 0.25, "conversation": 0.15}

class MixedDataset(IterableDataset):
    """Mixed dataset with category weighting for streaming."""
    
    def __init__(self, dataset_names: List[str], tokenizer, max_length: int, seed: int = 42):
        self.dataset_names = dataset_names
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed
        self.rng = None
        self.hf_token = os.environ.get("HF_TOKEN", None)
        
        # Initialize datasets
        self.iterators = []
        self.weights = []
        
        for name in dataset_names:
            spec = DATASET_SPECS.get(name)
            if not spec:
                continue
            
            try:
                ds = self._load_dataset(name, spec)
                if ds:
                    self.iterators.append(iter(ds))
                    self.weights.append(CATEGORY_WEIGHTS.get(spec["category"], 0.1))
                    logger.info(f"Loaded dataset: {name}")
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")
        
        # Normalize weights
        total = sum(self.weights)
        if total > 0:
            self.weights = [w / total for w in self.weights]
        
        logger.info(f"Successfully loaded {len(self.iterators)} datasets")
    
    def _load_dataset(self, name: str, spec: Dict):
        from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
        
        streaming = spec.get("streaming", False)
        
        load_kwargs = {"streaming": streaming}
        if self.hf_token:
            load_kwargs["token"] = self.hf_token
        
        # Handle specific datasets with known configs
        KNOWN_CONFIGS = {
            "bigcode/starcoder2data-extras": "stackoverflow",
            "nvidia/OpenCodeReasoning": "split_0",
            "HuggingFaceTB/smollm-corpus": "python-edu",
        }
        
        if name in KNOWN_CONFIGS:
            load_kwargs["name"] = KNOWN_CONFIGS[name]
        else:
            # Determine config name if needed
            configs = []
            try:
                configs = get_dataset_config_names(name, trust_remote_code=False)
            except:
                pass
            
            if configs:
                load_kwargs["name"] = configs[0]
        
        # Determine split
        try:
            cfg = load_kwargs.get("name")
            splits = list(get_dataset_split_names(name, **({"config": cfg} if cfg else {})))
        except:
            splits = []
        
        if "train" in splits:
            load_kwargs["split"] = "train"
        elif splits:
            load_kwargs["split"] = splits[0]
        else:
            load_kwargs["split"] = "train"
        
        # Try loading without trust_remote_code first
        for trust in [False, True]:
            try:
                load_kwargs["trust_remote_code"] = trust
                ds = load_dataset(name, **load_kwargs)
                return self._format_dataset(ds, spec["format"])
            except Exception as e:
                err = str(e)
                if "trust_remote_code" in err and not trust:
                    continue
                if "Config name is missing" in err:
                    continue
                logger.warning(f"Could not load {name}: {err[:200]}")
                return None
        
        return None
    
    def _format_dataset(self, ds, fmt: str):
        """Format dataset examples with thinking format."""
        
        def format_fn(example):
            try:
                text = self._extract_text(example, fmt)
                if text and len(text) > 50:
                    return {"text": text}
            except:
                pass
            return None
        
        if not hasattr(ds, 'map'):
            return ds
        
        # Streaming datasets: lazy map (no remove_columns)
        is_streaming = getattr(ds, '_ex_iterable', False) or getattr(ds, '_ex_checked', False)
        
        if is_streaming:
            ds = ds.map(format_fn)
        else:
            ds = ds.map(format_fn, remove_columns=ds.column_names)
        
        ds = ds.filter(lambda x: x is not None and len(x.get("text", "")) > 50)
        return ds
    
    def _extract_text(self, example: Dict, fmt: str) -> str:
        if fmt == "code":
            text = example.get("content", example.get("code", example.get("text", "")))
            return apply_thinking_format(text, is_code=True)
        
        elif fmt == "instruction":
            inst = example.get("instruction", example.get("prompt", ""))
            inp = example.get("input", "")
            out = example.get("output", example.get("response", ""))
            reasoning = example.get("reasoning", "")
            
            user = inst
            if inp:
                user += f"\n{inp}"
            
            if reasoning:
                return f"{USER_TOKEN} {user}\n\n{THINKING_TOKEN}\n{reasoning}\n\n{ANSWER_TOKEN} {out}"
            else:
                return f"{USER_TOKEN} {user}\n\n{THINKING_TOKEN}\nLet me think about this...\n\n{ANSWER_TOKEN} {out}"
        
        elif fmt == "reasoning":
            q = example.get("question", example.get("problem", example.get("instruction", "")))
            r = example.get("reasoning", example.get("chain_of_thought", example.get("thought", "")))
            a = example.get("answer", example.get("solution", example.get("output", "")))
            
            if r:
                return f"{USER_TOKEN} {q}\n\n{THINKING_TOKEN}\n{r}\n\n{ANSWER_TOKEN} {a}"
            else:
                return apply_thinking_format(q + "\n" + a)
        
        elif fmt == "swe":
            problem = example.get("problem_statement", example.get("text", ""))
            patch = example.get("patch", example.get("solution", ""))
            return f"{USER_TOKEN} Fix this issue:\n{problem}\n\n{THINKING_TOKEN}\nLet me analyze the codebase and develop a fix...\n\n{ANSWER_TOKEN} {patch}"
        
        elif fmt == "agentic":
            task = example.get("query", example.get("task", example.get("instruction", "")))
            result = example.get("result", example.get("output", example.get("solution", "")))
            return f"{USER_TOKEN} {task}\n\n{THINKING_TOKEN}\nI'll solve this step by step using available tools...\n\n{ANSWER_TOKEN} {result}"
        
        elif fmt == "tool":
            query = example.get("query", example.get("instruction", ""))
            answer = example.get("final_answer", example.get("output", ""))
            return f"{USER_TOKEN} {query}\n\n{THINKING_TOKEN}\nLet me use the appropriate tools...\n\n{ANSWER_TOKEN} {answer}"
        
        elif fmt == "conversation":
            data = example.get("messages", example.get("data", example))
            if isinstance(data, list):
                parts = []
                for turn in data:
                    role = turn.get("role", turn.get("from", "user"))
                    content = turn.get("content", turn.get("value", ""))
                    if role in ["user", "human"]:
                        parts.append(f"{USER_TOKEN} {content}")
                    elif role in ["assistant", "gpt"]:
                        parts.append(f"\n{THINKING_TOKEN}\nProcessing...\n\n{ANSWER_TOKEN} {content}")
                return "\n".join(parts)
            elif isinstance(data, dict) and "messages" in data:
                return self._extract_text(data, "conversation")
            else:
                return apply_thinking_format(str(data))
        
        return ""
    
    def __iter__(self):
        if not self.iterators:
            return
        
        import random
        rng = random.Random(self.seed)
        iters = [iter(it) for it in self.iterators]
        weights = list(self.weights)
        
        while iters:
            idx = rng.choices(range(len(iters)), weights=weights, k=1)[0]
            
            try:
                item = next(iters[idx])
                if isinstance(item, dict) and "text" in item:
                    yield item
            except StopIteration:
                iters.pop(idx)
                weights.pop(idx)
                if weights:
                    total = sum(weights)
                    weights = [w / total for w in weights]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_phase(config: TrainingConfig):
    """Train a single phase."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    
    logger.info(f"=== Phase {config.phase}: ctx_len={config.ctx_len} ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load tokenizer
    base = "SmerkyG/RWKV7-Goose-0.4B-Pile-HF" if not Path(config.base_model).exists() else config.base_model
    logger.info(f"Loading tokenizer from {base}")
    
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens for thinking format
    special_tokens = {"additional_special_tokens": [THINKING_TOKEN, ANSWER_TOKEN, USER_TOKEN]}
    num_added = tokenizer.add_special_tokens(special_tokens)
    
    # Load model
    logger.info(f"Loading model from {config.base_model}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model if Path(config.base_model).exists() else base,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Trying to load base model instead...")
        model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    model.to(device)
    
    # Setup dataset
    all_datasets = list(DATASET_SPECS.keys())
    logger.info(f"Loading datasets: {all_datasets}")
    
    dataset = MixedDataset(all_datasets, tokenizer, config.ctx_len)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=config.weight_decay,
    )
    
    # Real training: 20K steps (~327M tokens, ~8h on RTX 5070 Ti)
    # Context extension: targeted step counts
    STEP_COUNTS = {0: 20000, 1: 1800, 2: 150, 3: 15, 4: 5}
    total_steps = STEP_COUNTS[config.phase]
    warmup_steps = max(5, int(total_steps * config.warmup_ratio))
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    logger.info(f"Total steps: {total_steps}, Warmup: {warmup_steps}")
    
    # Training loop
    model.train()
    global_step = 0
    accum_step = 0
    optimizer.zero_grad()
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(config.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        
        epoch_loss = 0.0
        step_loss = 0.0
        
        # Manual batching for streaming dataset
        batch_texts = []
        
        for item in dataset:
            batch_texts.append(item["text"])
            
            if len(batch_texts) < config.batch_size:
                continue
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                max_length=config.ctx_len,
                padding=True,
                return_tensors="pt",
            )
            
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Forward
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
                loss = outputs.loss / config.grad_accum
            
            # Backward
            loss.backward()
            
            step_loss += loss.item()
            batch_texts = []
            accum_step += 1
            
            # Gradient accumulation
            if accum_step % config.grad_accum == 0:
                # Clip
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                # Step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Clear cache to reduce fragmentation
                if global_step % 50 == 0:
                    torch.cuda.empty_cache()
                
                # Log
                avg_loss = step_loss
                current_lr = scheduler.get_last_lr()[0]
                
                if global_step % 10 == 0:
                    logger.info(f"Step {global_step} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
                    sys.stdout.flush()
                
                epoch_loss += avg_loss
                step_loss = 0.0
                global_step += 1
                accum_step = 0
                
                # Save checkpoint
                if global_step % 500 == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(exist_ok=True)
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    logger.info(f"Saved checkpoint to {ckpt_dir}")
                    sys.stdout.flush()
                
                # Break if we've reached estimated steps
                if global_step >= total_steps:
                    break
        
        logger.info(f"Epoch {epoch+1} complete. Avg loss: {epoch_loss / max(global_step, 1):.4f}")
        sys.stdout.flush()
        
        if global_step >= total_steps:
            break
    
    # Save final
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    logger.info(f"Phase {config.phase} complete! Model saved to {final_dir}")
    
    # Cleanup
    del model
    del optimizer
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--base-model", type=str, default=None)
    args = parser.parse_args()
    
    config = TrainingConfig.from_phase(args.phase)
    if args.base_model:
        config.base_model = args.base_model
    
    train_phase(config)

if __name__ == "__main__":
    main()
