# vLLM Concepts

## Baseline inference

The baseline experiment uses Hugging Face Transformers in a sequential `generate()` loop. This is deliberately naive. It shows what happens when requests are processed one at a time without a serving scheduler.

## Prefill

Prefill is the phase where the model processes the input prompt and creates KV-cache entries for the prompt tokens. Long prompts increase time-to-first-token and KV-cache pressure.

## Decode

Decode is the token-by-token generation phase. Each step produces the next token and appends new KV entries. Decode is usually memory-bandwidth-sensitive.

## KV cache

The KV cache stores key and value vectors from previous tokens. It avoids recomputing attention states for all previous tokens during each decode step. It is essential for speed but consumes memory.

## Fragmentation

Naive KV-cache allocation may reserve memory based on maximum sequence length. If most requests are shorter than the maximum, reserved memory remains unused. This is internal fragmentation.

## PagedAttention

PagedAttention splits KV cache into fixed-size blocks. Requests receive blocks as needed. Logical token order is mapped to physical blocks using a block table. This reduces memory waste and improves concurrency.

## Continuous batching

Continuous batching updates the active batch over time. As requests finish, new requests can enter. This avoids waiting for the longest request in a static batch.

## Chunked prefill

Chunked prefill splits long prompt processing into chunks so long prompts do not monopolise the scheduler and block decode work for existing requests.

## Prefix caching

Prefix caching reuses KV-cache entries when requests share the same prefix, such as a common system prompt or RAG instruction template.

## OpenAI-compatible API

vLLM can expose a local server with OpenAI-compatible endpoints. This makes it easier to integrate into existing applications.

## Production tuning

The most important tuning knobs are:

- `max_model_len`
- `max_num_seqs`
- `max_num_batched_tokens`
- `gpu_memory_utilization`
- quantisation settings
- tensor parallelism settings
- prefix caching settings

Tune against measured workload distributions, not assumptions.
