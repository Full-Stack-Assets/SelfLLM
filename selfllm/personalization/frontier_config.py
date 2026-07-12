"""Frontier model configurations built on SelfLLM's advanced architecture features.

These configs extend the standard small/medium/full presets with Mixture of
Experts (MoE) routing and sliding-window attention for long-context support.
"""

from selfllm.model.config import ModelConfig
from selfllm.real_training import get_full_config, get_medium_config, get_small_config


def get_frontier_small_config() -> ModelConfig:
    """CPU-friendly frontier model (~5M params) with MoE FFN layers."""
    config = get_small_config()
    config.use_moe = True
    config.moe_num_experts = 4
    config.moe_top_k = 2
    return config


def get_frontier_medium_config() -> ModelConfig:
    """Single-GPU frontier model (~50M params) with MoE and sliding-window attention."""
    config = get_medium_config()
    config.use_moe = True
    config.moe_num_experts = 8
    config.moe_top_k = 2
    config.sliding_window = 512
    config.attention_sinks = 4
    return config


def get_frontier_full_config() -> ModelConfig:
    """Multi-GPU frontier model (~350M params) with MoE and long-context attention."""
    config = get_full_config()
    config.use_moe = True
    config.moe_num_experts = 8
    config.moe_top_k = 2
    config.sliding_window = 1024
    config.attention_sinks = 4
    return config


def get_frontier_config(scale: str) -> ModelConfig:
    """Return the frontier config for *scale* (``small``, ``medium``, or ``full``)."""
    factories = {
        "small": get_frontier_small_config,
        "medium": get_frontier_medium_config,
        "full": get_frontier_full_config,
    }
    if scale not in factories:
        raise ValueError(f"Unknown frontier scale: {scale!r}. Choose from {list(factories)}")
    return factories[scale]()
