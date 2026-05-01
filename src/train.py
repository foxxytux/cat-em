"""
Main training script for CodeAgent-RWKV
Supports multi-phase training with context extension.
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    default_data_collator,
)
from datasets import load_dataset
import wandb
from tqdm import tqdm

# Local imports
from dataset import (
    ThinkingFormat,
    get_tokenized_dataset,
    DATASET_CONFIGS,
    CATEGORY_WEIGHTS,
)
from context_extension import prepare_model_for_context, patch_model_for_long_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RWKVTrainer:
    """Trainer for CodeAgent-RWKV with thinking format."""
    
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.phase = self.config["training"]["phase"]
        self.ctx_len = self.config["training"]["ctx_len"]
        
        # Setup device
        self.device = torch.device(self.config["hardware"]["device"])
        
        # Setup output dirs
        self.output_dir = Path(self.config["logging"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logging_dir = Path(self.config["logging"]["logging_dir"])
        self.logging_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer and model
        self._setup_model()
        self._setup_tokenizer()
        self._setup_optimizer()
        self._setup_data()
        
        # Logging
        if self.config["logging"]["report_to"] == "wandb":
            wandb.init(
                project="codeagent-rwkv7",
                name=self.config["logging"]["run_name"],
                config=self.config,
            )
        
        self.global_step = 0
        self.epoch = 0
    
    def _setup_model(self):
        """Load or initialize the RWKV model."""
        model_name = self.config["model"]["base_model"]
        
        logger.info(f"Loading model: {model_name}")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.config["optimization"]["precision"] == "bf16" else torch.float32,
            trust_remote_code=True,
        )
        
        # Patch for long context
        if self.ctx_len > 4096:
            logger.info(f"Extending context to {self.ctx_len}")
            base_ctx = 4096 if self.phase == 1 else self.config["training"].get("base_ctx", 4096)
            
            self.model = prepare_model_for_context(
                self.model,
                target_ctx=self.ctx_len,
                base_ctx=base_ctx,
                method=self.config.get("extension", {}).get("method", "ntk-aware"),
            )
            self.model = patch_model_for_long_context(self.model, self.ctx_len)
        
        # Enable gradient checkpointing if configured
        if self.config["optimization"]["gradient_checkpointing"]:
            self.model.gradient_checkpointing_enable()
        
        self.model.to(self.device)
        
        logger.info(f"Model loaded. Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")
    
    def _setup_tokenizer(self):
        """Setup tokenizer with thinking format tokens."""
        model_name = self.config["model"]["base_model"]
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        # Add special tokens for thinking format
        special_tokens = {
            "additional_special_tokens": [
                self.config["format"]["thinking_token"],
                self.config["format"]["answer_token"],
                self.config["format"]["user_token"],
            ]
        }
        
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Added {num_added} special tokens to tokenizer")
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        lr = self.config["training"]["learning_rate"]
        weight_decay = self.config["training"]["weight_decay"]
        
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if "bias" in name or "norm" in name or "ln" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            betas=(
                self.config["optimization"]["beta1"],
                self.config["optimization"]["beta2"],
            ),
            eps=self.config["optimization"]["eps"],
        )
        
        # Learning rate scheduler will be setup after we know dataset length
        self.scheduler = None
        self.scaler = GradScaler() if self.config["optimization"]["precision"] == "bf16" else None
    
    def _setup_data(self):
        """Setup datasets and dataloader."""
        formatter = ThinkingFormat(
            user_token=self.config["format"]["user_token"],
            thinking_token=self.config["format"]["thinking_token"],
            answer_token=self.config["format"]["answer_token"],
            system_prompt=self.config["format"]["system_prompt"],
        )
        
        dataset_names = self.config["data"]["datasets"]
        
        # Try to load datasets
        logger.info("Loading datasets...")
        
        # For efficiency, we'll load a subset in non-streaming mode for smaller datasets
        # and streaming for very large ones
        streaming_datasets = []
        static_datasets = []
        
        for name in dataset_names:
            ds_config = DATASET_CONFIGS.get(name, {})
            is_streaming = ds_config.get("streaming", False) or "the-stack" in name.lower()
            
            try:
                if is_streaming:
                    streaming_datasets.append(name)
                else:
                    static_datasets.append(name)
            except Exception as e:
                logger.warning(f"Could not queue {name}: {e}")
        
        # For now, use a simple approach: load all as streaming if possible
        self.train_dataset = get_tokenized_dataset(
            tokenizer=self.tokenizer,
            dataset_names=dataset_names,
            max_length=self.ctx_len,
            formatter=formatter,
            streaming=True,
            num_proc=self.config["data"].get("num_proc", 4),
        )
        
        # Create dataloader
        self.batch_size = self.config["training"]["batch_size"]
        self.grad_accum_steps = self.config["training"]["gradient_accumulation_steps"]
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=self.config["hardware"]["dataloader_num_workers"],
            pin_memory=self.config["hardware"]["dataloader_pin_memory"],
        )
        
        # Setup scheduler now that we have dataloader
        total_steps = self._estimate_total_steps()
        warmup_steps = int(total_steps * self.config["training"]["warmup_ratio"])
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5,
        )
        
        logger.info(f"Total training steps: {total_steps}, Warmup: {warmup_steps}")
    
    def _estimate_total_steps(self) -> int:
        """Estimate total training steps."""
        # For streaming datasets, we use a fixed estimate
        # 500M tokens / (batch_size * ctx_len) * num_epochs
        tokens_per_batch = self.batch_size * self.ctx_len * self.grad_accum_steps
        estimated_tokens = 500_000_000  # 500M tokens per epoch
        steps_per_epoch = estimated_tokens // tokens_per_batch
        return steps_per_epoch * self.config["training"]["num_epochs"]
    
    def _collate_fn(self, batch):
        """Collate and tokenize batch on the fly."""
        texts = [item["text"] for item in batch]
        
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.ctx_len,
            padding=True,
            return_tensors="pt",
        )
        
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    def train(self):
        """Main training loop."""
        logger.info(f"Starting Phase {self.phase} training (ctx={self.ctx_len})")
        
        self.model.train()
        
        save_steps = self.config["logging"]["save_steps"]
        logging_steps = self.config["logging"]["logging_steps"]
        
        for epoch in range(self.config["training"]["num_epochs"]):
            self.epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{self.config['training']['num_epochs']}")
            
            epoch_loss = 0.0
            step_in_epoch = 0
            
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                with autocast(dtype=torch.bfloat16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / self.grad_accum_steps
                
                # Backward pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                epoch_loss += loss.item() * self.grad_accum_steps
                
                # Gradient accumulation
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    # Clip gradients
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["training"]["max_grad_norm"],
                    )
                    
                    # Optimizer step
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    
                    if self.scheduler:
                        self.scheduler.step()
                    
                    self.global_step += 1
                    step_in_epoch += 1
                    
                    # Logging
                    if self.global_step % logging_steps == 0:
                        current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else lr
                        avg_loss = epoch_loss / step_in_epoch
                        
                        metrics = {
                            "loss": avg_loss,
                            "learning_rate": current_lr,
                            "epoch": epoch + 1,
                            "step": self.global_step,
                        }
                        
                        progress_bar.set_postfix(metrics)
                        
                        if self.config["logging"]["report_to"] == "wandb":
                            wandb.log(metrics)
                    
                    # Save checkpoint
                    if self.global_step % save_steps == 0:
                        self.save_checkpoint()
            
            # End of epoch
            logger.info(f"Epoch {epoch+1} complete. Avg loss: {epoch_loss / max(step_in_epoch, 1):.4f}")
        
        # Final save
        self.save_checkpoint(is_final=True)
        logger.info("Training complete!")
    
    def save_checkpoint(self, is_final: bool = False):
        """Save model checkpoint."""
        if is_final:
            save_path = self.output_dir / "final"
        else:
            save_path = self.output_dir / f"checkpoint-{self.global_step}"
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving checkpoint to {save_path}")
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save training state
        torch.save({
            "global_step": self.global_step,
            "epoch": self.epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
        }, save_path / "training_state.pt")
        
        # Clean up old checkpoints
        if not is_final:
            self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints keeping only the most recent N."""
        max_checkpoints = self.config["logging"].get("save_total_limit", 3)
        
        checkpoints = sorted(
            self.output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]),
        )
        
        while len(checkpoints) > max_checkpoints:
            old_ckpt = checkpoints.pop(0)
            logger.info(f"Removing old checkpoint: {old_ckpt}")
            import shutil
            shutil.rmtree(old_ckpt)


def main():
    parser = argparse.ArgumentParser(description="Train CodeAgent-RWKV")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    trainer = RWKVTrainer(args.config)
    
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        # Load checkpoint logic here
    
    trainer.train()


if __name__ == "__main__":
    main()
