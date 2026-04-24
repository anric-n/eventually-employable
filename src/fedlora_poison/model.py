"""Base model + LoRA adapter configuration with layer-region masks.

Handles:
- Loading base LLM (Llama-3.2-1B or Qwen2.5-1.5B)
- Configuring LoRA with layer-region targeting (early/middle/late/full)
- Utilities for extracting and loading LoRA state dicts
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class LayerRegion(str, Enum):
    """Which transformer blocks get LoRA adapters."""

    EARLY = "early"  # first third of layers
    MIDDLE = "middle"  # middle third
    LATE = "late"  # last third
    FULL = "full"  # all layers (baseline)


def get_layer_indices(
    num_layers: int, region: LayerRegion
) -> list[int]:
    """Return layer indices for a given region."""
    third = num_layers // 3
    if region == LayerRegion.EARLY:
        return list(range(0, third))
    elif region == LayerRegion.MIDDLE:
        return list(range(third, 2 * third))
    elif region == LayerRegion.LATE:
        return list(range(2 * third, num_layers))
    else:  # FULL
        return list(range(num_layers))


def get_target_modules(
    model_name: str,
    region: LayerRegion,
    num_layers: Optional[int] = None,
) -> list[str]:
    """Get LoRA target module names for a specific layer region.

    For a model with layers named like 'model.layers.{i}.self_attn.{q,k,v,o}_proj'
    and 'model.layers.{i}.mlp.{gate,up,down}_proj', this returns only the modules
    in the specified region.
    """
    if num_layers is None:
        # Default layer counts for supported models
        if "Llama-3.2-1B" in model_name or "llama-3.2-1b" in model_name.lower():
            num_layers = 16
        elif "Qwen2.5-1.5B" in model_name or "qwen2.5-1.5b" in model_name.lower():
            num_layers = 28
        else:
            num_layers = 16  # conservative default
            logger.warning(f"Unknown model {model_name}, assuming {num_layers} layers")

    indices = get_layer_indices(num_layers, region)
    modules = []
    for i in indices:
        modules.extend([
            f"model.layers.{i}.self_attn.q_proj",
            f"model.layers.{i}.self_attn.k_proj",
            f"model.layers.{i}.self_attn.v_proj",
            f"model.layers.{i}.self_attn.o_proj",
            f"model.layers.{i}.mlp.gate_proj",
            f"model.layers.{i}.mlp.up_proj",
            f"model.layers.{i}.mlp.down_proj",
        ])
    return modules


def create_lora_config(
    model_name: str,
    region: LayerRegion = LayerRegion.FULL,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    num_layers: Optional[int] = None,
) -> LoraConfig:
    """Create a LoRA config targeting a specific layer region."""
    target_modules = get_target_modules(model_name, region, num_layers)
    logger.info(
        f"LoRA config: region={region.value}, rank={rank}, "
        f"targeting {len(target_modules)} modules"
    )
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def load_model_and_tokenizer(
    model_name: str = "meta-llama/Llama-3.2-1B",
    region: LayerRegion = LayerRegion.FULL,
    lora_rank: int = 8,
    device: str = "auto",
) -> tuple:
    """Load base model with LoRA adapters attached to the specified layer region."""
    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    lora_config = create_lora_config(model_name, region, rank=lora_rank)
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        f"Model loaded: {total:,} total params, {trainable:,} trainable "
        f"({trainable/total:.2%})"
    )

    return model, tokenizer


def get_lora_state_dict(model) -> dict[str, torch.Tensor]:
    """Extract only the LoRA adapter parameters as a flat dict."""
    state_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            state_dict[name] = param.detach().cpu().clone()
    return state_dict


def set_lora_state_dict(model, state_dict: dict[str, torch.Tensor]) -> None:
    """Load LoRA parameters from a flat dict back into the model."""
    model_state = model.state_dict()
    for name, param in state_dict.items():
        if name in model_state:
            model_state[name].copy_(param)
        else:
            logger.warning(f"Key {name} not found in model state dict")
    model.load_state_dict(model_state, strict=False)
