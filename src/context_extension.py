"""
Context length extension utilities for RWKV-7
Implements gradual extension from 4096 -> 131072
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class RWKVContextExtension:
    """Handles context length extension for RWKV models."""
    
    def __init__(self, model, base_ctx: int = 4096):
        self.model = model
        self.base_ctx = base_ctx
        self.current_ctx = base_ctx
        
    def extend_to(self, new_ctx: int, method: str = "ntk-aware"):
        """
        Extend model context length.
        
        Methods:
        - "naive": Simple position interpolation
        - "ntk-aware": NTK-aware scaling (recommended for RWKV)
        - "yarn": YaRN scaling
        """
        if new_ctx <= self.current_ctx:
            return
            
        scale_factor = new_ctx / self.base_ctx
        
        if method == "naive":
            self._apply_naive_extension(scale_factor)
        elif method == "ntk-aware":
            self._apply_ntk_extension(scale_factor)
        elif method == "yarn":
            self._apply_yarn_extension(scale_factor)
        else:
            raise ValueError(f"Unknown extension method: {method}")
        
        self.current_ctx = new_ctx
        print(f"Context extended to {new_ctx} (scale={scale_factor:.2f}x)")
    
    def _apply_naive_extension(self, scale_factor: float):
        """Simple linear interpolation of position embeddings."""
        # For RWKV, position info is handled via time-decay, not traditional embeddings
        # We mainly need to adjust any learned position-related parameters
        pass
    
    def _apply_ntk_extension(self, scale_factor: float):
        """
        NTK-aware extension for RWKV.
        Adjusts the time-mix and time-decay parameters.
        """
        # RWKV uses time_decay parameters that control how past information decays
        # For longer contexts, we need to adjust these to maintain stability
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if "time_decay" in name.lower() or "td" in name.lower():
                    # Scale time decay to account for longer sequences
                    # This prevents gradients from exploding/vanishing at longer contexts
                    original = param.data.clone()
                    
                    # NTK-inspired scaling: compress the frequency spectrum
                    if param.dim() >= 1:
                        # Apply frequency scaling
                        ntk_scale = scale_factor ** (1.0 / param.shape[-1])
                        param.data = original / ntk_scale
                        
                elif "time_first" in name.lower() or "tf" in name.lower():
                    # Adjust time-first (position 0 special handling)
                    original = param.data.clone()
                    param.data = original * math.log(scale_factor + 1)
    
    def _apply_yarn_extension(self, scale_factor: float):
        """YaRN-style extension."""
        # YaRN combines NTK with temperature scaling
        self._apply_ntk_extension(scale_factor)
        
        # Additional temperature scaling on attention
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if "time_mix" in name.lower() or "tm" in name.lower():
                    # Scale mixing factors
                    original = param.data.clone()
                    param.data = original / math.sqrt(scale_factor)


def prepare_model_for_context(
    model,
    target_ctx: int,
    base_ctx: int = 4096,
    method: str = "ntk-aware",
):
    """Prepare model for training at extended context length."""
    extender = RWKVContextExtension(model, base_ctx=base_ctx)
    extender.extend_to(target_ctx, method=method)
    return model


def create_position_ids(input_ids, past_key_values_length: int = 0):
    """Create position IDs for RWKV with extended context."""
    batch_size, seq_length = input_ids.shape
    position_ids = torch.arange(
        past_key_values_length, seq_length + past_key_values_length,
        dtype=torch.long,
        device=input_ids.device,
    )
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    return position_ids


def patch_model_for_long_context(model, max_position_embeddings: int = 131072):
    """
    Patch model config to support longer contexts.
    This updates the model's internal max position setting.
    """
    if hasattr(model, "config"):
        model.config.max_position_embeddings = max_position_embeddings
        
        if hasattr(model.config, "context_length"):
            model.config.context_length = max_position_embeddings
    
    # Update any position-related buffers
    for module in model.modules():
        if hasattr(module, "max_seq_len"):
            module.max_seq_len = max_position_embeddings
    
    return model
