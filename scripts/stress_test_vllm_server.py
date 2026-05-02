"""Concurrent stress test for a vLLM OpenAI-compatible server."""

from __future__ import annotations

import argparse
import asyncio
import csv
import time
from pathlib import Path

from vllm_lab.benchmark import benchmark_openai_concurrent

PROMPTS = [
    "Explain KV cache in transformer inference.",
    "Explain why batching improves GPU throughput.",
    "Explain PagedAttention using an operating-system analogy.",
    "Summarise the difference between prefill and decode in LLM serving.",
    "Explain how max_model_len affects vLLM memory pressure.",
    "Give three risks of deploying an LLM API without monitoring.",
]


async def run(args: argparse.Namespace) -> None:
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.requests)]
    started = time.perf_counter()
    rows = await benchmark_openai_concurrent(
        base_url=args.base_url,
        model=args.model,
        prompts=prompts,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        api_key=args.api_key,
    )
    wall_s = time.perf_counter() - started
    total_output_tokens = sum((r.get("output_tokens") or 0) for r in rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stress_concurrency_{args.concurrency}_{int(time.time())}.csv"

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote: {out_path}")
    print(f"requests: {len(rows)}")
    print(f"wall_time_s: {wall_s:.2f}")
    print(f"total_output_tokens: {total_output_tokens}")
    print(f"aggregate_output_tokens_per_second: {total_output_tokens / wall_s:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
