#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8022}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.35}

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype auto \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
