# vLLM Tuning Cheatsheet

## Symptom: Out-of-memory at startup

Try:

- use a smaller model
- use quantisation
- lower `max_model_len`
- reduce `gpu_memory_utilization` if startup overhead collides with cache allocation
- use tensor parallelism if multiple GPUs are available

## Symptom: Out-of-memory during serving

Try:

- reduce `max_num_seqs`
- reduce `max_num_batched_tokens`
- reduce `max_model_len`
- reduce max output tokens at the application layer
- avoid too many long-context concurrent requests

## Symptom: Low throughput

Try:

- increase concurrency in the client/orchestrator
- increase `max_num_seqs` if memory allows
- increase `max_num_batched_tokens` if memory allows
- enable prefix caching if prompts share stable prefixes
- test a quantised model
- confirm the GPU is actually used

## Symptom: High time-to-first-token

Try:

- reduce prompt length
- enable prefix caching for repeated prefixes
- use chunked prefill where appropriate
- reduce queueing delay by lowering overload
- use a smaller or quantised model

## Symptom: Good throughput but bad user experience

Throughput and latency are not the same. Aggressive batching may improve tokens per second while making individual users wait longer. Decide which metric matters.

## Bad tuning habits

- setting `max_model_len` to the largest advertised context length without workload evidence
- benchmarking one request at a time and claiming production throughput
- ignoring prompt length distribution
- ignoring output length distribution
- forgetting that KV cache grows with active sequences
- assuming quantisation solves KV-cache pressure
