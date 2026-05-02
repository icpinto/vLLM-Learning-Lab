"""Small simulations that make KV-cache fragmentation and PagedAttention visible."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SequenceRequest:
    request_id: int
    actual_tokens: int
    max_reserved_tokens: int


@dataclass
class AllocationSummary:
    strategy: str
    total_reserved_tokens: int
    total_used_tokens: int
    wasted_tokens: int
    waste_ratio: float
    block_size: int | None = None


def generate_requests(
    n: int = 100,
    min_tokens: int = 32,
    max_tokens: int = 4096,
    max_reserved_tokens: int = 4096,
    seed: int = 7,
) -> List[SequenceRequest]:
    """Generate a skewed request-length distribution.

    Real LLM traffic is rarely uniform. Many prompts are short, some are medium,
    and a few are very long. A log-normal distribution is a reasonable toy model.
    """
    rng = random.Random(seed)
    requests = []
    for i in range(n):
        raw = int(rng.lognormvariate(mu=6.0, sigma=1.0))
        actual = max(min_tokens, min(raw, max_tokens))
        requests.append(SequenceRequest(i, actual, max_reserved_tokens))
    return requests


def naive_contiguous_allocation(requests: List[SequenceRequest]) -> AllocationSummary:
    """Simulate naive allocation where every request reserves max sequence length."""
    reserved = sum(r.max_reserved_tokens for r in requests)
    used = sum(r.actual_tokens for r in requests)
    wasted = reserved - used
    return AllocationSummary(
        strategy="naive_max_length_reservation",
        total_reserved_tokens=reserved,
        total_used_tokens=used,
        wasted_tokens=wasted,
        waste_ratio=wasted / reserved if reserved else 0.0,
    )


def paged_allocation(requests: List[SequenceRequest], block_size: int = 16) -> AllocationSummary:
    """Simulate paged allocation using fixed-size token blocks."""
    reserved = sum(math.ceil(r.actual_tokens / block_size) * block_size for r in requests)
    used = sum(r.actual_tokens for r in requests)
    wasted = reserved - used
    return AllocationSummary(
        strategy="paged_attention_style_blocks",
        total_reserved_tokens=reserved,
        total_used_tokens=used,
        wasted_tokens=wasted,
        waste_ratio=wasted / reserved if reserved else 0.0,
        block_size=block_size,
    )


def allocation_curve(requests: List[SequenceRequest], block_sizes: List[int]) -> List[AllocationSummary]:
    rows = [naive_contiguous_allocation(requests)]
    for block_size in block_sizes:
        rows.append(paged_allocation(requests, block_size=block_size))
    return rows


def make_memory_grid(requests: List[SequenceRequest], block_size: int = 16, max_rows: int = 32) -> List[List[int]]:
    """Return a simple 0/1 grid representing used blocks for visualisation."""
    rows = []
    for r in requests[:max_rows]:
        used_blocks = math.ceil(r.actual_tokens / block_size)
        max_blocks = math.ceil(r.max_reserved_tokens / block_size)
        rows.append([1] * used_blocks + [0] * (max_blocks - used_blocks))
    return rows
