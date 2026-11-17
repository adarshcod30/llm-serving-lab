"""KV-cache arithmetic: how much memory does one token of context cost?

Every transformer layer stores a key and a value vector per KV head for every
token it has seen. So per token:

    bytes = 2 (K and V) * layers * kv_heads * head_dim * bytes_per_element

Grouped-query attention (kv_heads < attention heads) is why modern models fit
far more context than the old multi-head layout. This module computes the number
from a model's config.json alone, so it can be compared with what vLLM reports.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KVSpec:
    layers: int
    attn_heads: int
    kv_heads: int
    head_dim: int
    dtype_bytes: int = 2  # fp16

    @property
    def bytes_per_token(self) -> int:
        return 2 * self.layers * self.kv_heads * self.head_dim * self.dtype_bytes

    @property
    def mha_bytes_per_token(self) -> int:
        """What it would cost with one KV head per attention head (no GQA)."""
        return 2 * self.layers * self.attn_heads * self.head_dim * self.dtype_bytes

    def tokens_that_fit(self, kv_budget_bytes: float) -> int:
        return int(kv_budget_bytes // self.bytes_per_token)


def spec_from_config(cfg: dict, dtype_bytes: int = 2) -> KVSpec:
    attn = cfg["num_attention_heads"]
    return KVSpec(
        layers=cfg["num_hidden_layers"],
        attn_heads=attn,
        kv_heads=cfg.get("num_key_value_heads", attn),
        head_dim=cfg.get("head_dim") or cfg["hidden_size"] // attn,
        dtype_bytes=dtype_bytes,
    )


def kv_budget_bytes(gpu_total_bytes: float, util: float, weights_bytes: float, overhead_bytes: float) -> float:
    """Memory vLLM can give to the KV cache: util * total - weights - activations/overhead."""
    return gpu_total_bytes * util - weights_bytes - overhead_bytes
