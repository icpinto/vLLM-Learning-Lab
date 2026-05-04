"""Benchmark helpers for Transformers, vLLM offline, and OpenAI-compatible APIs."""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional

import httpx


@dataclass
class GenerationResult:
    backend: str
    prompt: str
    output_text: str
    prompt_tokens: int
    output_tokens: int
    latency_s: float
    end_to_end_latency_s: Optional[float] = None
    tokenization_latency_s: Optional[float] = None
    prefill_latency_s: Optional[float] = None
    decode_latency_s: Optional[float] = None
    time_to_first_token_s: Optional[float] = None
    memory_allocated_mb: Optional[float] = None
    memory_reserved_mb: Optional[float] = None

    @property
    def output_tokens_per_second(self) -> float:
        return self.output_tokens / self.latency_s if self.latency_s > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["output_tokens_per_second"] = self.output_tokens_per_second
        return d


def summarize_results(results: Iterable[GenerationResult]) -> Dict[str, float]:
    rows = list(results)
    if not rows:
        return {}
    latencies = [r.latency_s for r in rows]
    tps = [r.output_tokens_per_second for r in rows]
    return {
        "count": len(rows),
        "latency_mean_s": statistics.mean(latencies),
        "latency_p50_s": statistics.median(latencies),
        "latency_max_s": max(latencies),
        "output_tps_mean": statistics.mean(tps),
        "output_tps_sum": sum(r.output_tokens for r in rows) / sum(latencies),
        "total_output_tokens": sum(r.output_tokens for r in rows),
    }


def benchmark_transformers(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    max_new_tokens: int = 128,
    device: Optional[str] = None,
    warmup: bool = True,
    batch_size: int = 1,
) -> List[GenerationResult]:
    """Run an improved Transformers benchmark with richer performance telemetry."""
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    is_cuda_device = str(device).startswith("cuda")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Warm-up prevents one-time initialization overhead from polluting metrics.
    if warmup and prompts:
        warmup_inputs = tokenizer(prompts[0], return_tensors="pt").to(device)
        with torch.inference_mode():
            _ = model.generate(
                **warmup_inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        if is_cuda_device:
            torch.cuda.synchronize()

    results: List[GenerationResult] = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        tokenization_start = time.perf_counter()
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
        if is_cuda_device:
            torch.cuda.synchronize()
        tokenization_latency_s = time.perf_counter() - tokenization_start

        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            prompt_lengths = attention_mask.sum(dim=1).tolist()
        else:
            prompt_lengths = [int(inputs["input_ids"].shape[-1])] * len(batch_prompts)

        generation_start = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        if is_cuda_device:
            torch.cuda.synchronize()
        generation_latency_s = time.perf_counter() - generation_start

        one_token_start = time.perf_counter()
        with torch.inference_mode():
            _ = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        if is_cuda_device:
            torch.cuda.synchronize()
        one_token_latency_s = time.perf_counter() - one_token_start

        decode_start = time.perf_counter()
        decoded_texts: List[str] = []
        output_token_counts: List[int] = []
        for row_idx, prompt_len in enumerate(prompt_lengths):
            generated = outputs[row_idx][int(prompt_len) :]
            decoded_texts.append(tokenizer.decode(generated, skip_special_tokens=True))
            output_token_counts.append(int(generated.shape[-1]))
        decode_latency_s = time.perf_counter() - decode_start

        end_to_end_latency_s = tokenization_latency_s + generation_latency_s + decode_latency_s

        mem_alloc_mb = mem_reserved_mb = None
        if is_cuda_device:
            mem_alloc_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            mem_reserved_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)

        for row_idx, prompt in enumerate(batch_prompts):
            ttft_s = min(one_token_latency_s, generation_latency_s)
            results.append(
                GenerationResult(
                    backend="transformers_batched" if batch_size > 1 else "transformers_sequential",
                    prompt=prompt,
                    output_text=decoded_texts[row_idx],
                    prompt_tokens=int(prompt_lengths[row_idx]),
                    output_tokens=output_token_counts[row_idx],
                    latency_s=generation_latency_s,
                    end_to_end_latency_s=end_to_end_latency_s,
                    tokenization_latency_s=tokenization_latency_s,
                    prefill_latency_s=ttft_s,
                    decode_latency_s=max(generation_latency_s - one_token_latency_s, 0.0),
                    time_to_first_token_s=ttft_s,
                    memory_allocated_mb=mem_alloc_mb,
                    memory_reserved_mb=mem_reserved_mb,
                )
            )
    return results


def benchmark_openai_sync(
    base_url: str,
    model: str,
    prompts: List[str],
    api_key: str = "EMPTY",
    max_tokens: int = 128,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    """Benchmark a vLLM OpenAI-compatible server sequentially."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    rows: List[Dict[str, Any]] = []
    for prompt in prompts:
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency_s = time.perf_counter() - start
        usage = response.usage
        completion_tokens = usage.completion_tokens if usage else None
        rows.append(
            {
                "backend": "vllm_openai_sync",
                "prompt": prompt,
                "output_text": response.choices[0].message.content,
                "latency_s": latency_s,
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "output_tokens": completion_tokens,
                "output_tokens_per_second": completion_tokens / latency_s if completion_tokens else None,
            }
        )
    return rows


async def _one_chat_request(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    start = time.perf_counter()
    response = await client.post("/chat/completions", json=payload)
    latency_s = time.perf_counter() - start
    data = response.json()
    usage = data.get("usage") or {}
    return {
        "status_code": response.status_code,
        "latency_s": latency_s,
        "prompt": prompt,
        "output_text": data.get("choices", [{}])[0].get("message", {}).get("content"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "output_tokens_per_second": (usage.get("completion_tokens") or 0) / latency_s if latency_s else 0,
    }


async def benchmark_openai_concurrent(
    base_url: str,
    model: str,
    prompts: List[str],
    concurrency: int,
    max_tokens: int = 128,
    temperature: float = 0.0,
    api_key: str = "EMPTY",
) -> List[Dict[str, Any]]:
    """Run concurrent requests against a vLLM OpenAI-compatible server."""
    headers = {"Authorization": f"Bearer {api_key}"}
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(
        base_url=base_url,
        headers=headers,
        timeout=None,
        limits=limits,
    ) as client:
        sem = asyncio.Semaphore(concurrency)

        async def guarded(prompt: str) -> Dict[str, Any]:
            async with sem:
                return await _one_chat_request(client, model, prompt, max_tokens, temperature)

        return await asyncio.gather(*(guarded(p) for p in prompts))
