"""KV-cache memory estimation utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelShape:
    name: str
    num_layers: int
    hidden_size: int
    num_heads: int | None = None
    num_kv_heads: int | None = None


def bytes_to_gib(n_bytes: float) -> float:
    return n_bytes / (1024**3)


def bytes_to_gb(n_bytes: float) -> float:
    return n_bytes / 1e9


def kv_cache_bytes_simplified(
    num_layers: int,
    hidden_size: int,
    context_tokens: int,
    bytes_per_value: int = 2,
    active_sequences: int = 1,
) -> int:
    """Simplified KV cache estimate.

    Formula:
        2 * layers * context_tokens * hidden_size * bytes_per_value * active_sequences

    This assumes full multi-head attention where KV hidden size equals model hidden size.
    Grouped-query attention can reduce actual KV cache because num_kv_heads < num_heads.
    """
    return int(2 * num_layers * context_tokens * hidden_size * bytes_per_value * active_sequences)


def kv_cache_bytes_with_gqa(
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    context_tokens: int,
    bytes_per_value: int = 2,
    active_sequences: int = 1,
) -> int:
    """More precise KV cache estimate for grouped-query attention.

    If num_kv_heads is smaller than num_heads, the KV cache is smaller than the
    simplified full-attention estimate.
    """
    head_dim = hidden_size // num_heads
    kv_hidden = num_kv_heads * head_dim
    return int(2 * num_layers * context_tokens * kv_hidden * bytes_per_value * active_sequences)


def weight_memory_bytes(num_parameters: float, bytes_per_parameter: float) -> float:
    return num_parameters * bytes_per_parameter


COMMON_MODEL_SHAPES = {
    "llama_7b_like": ModelShape("llama_7b_like", num_layers=32, hidden_size=4096, num_heads=32, num_kv_heads=32),
    "llama_13b_like": ModelShape("llama_13b_like", num_layers=40, hidden_size=5120, num_heads=40, num_kv_heads=40),
    "llama_70b_like_gqa": ModelShape("llama_70b_like_gqa", num_layers=80, hidden_size=8192, num_heads=64, num_kv_heads=8),
    "tinyllama_1_1b": ModelShape("tinyllama_1_1b", num_layers=22, hidden_size=2048, num_heads=32, num_kv_heads=4),
}
