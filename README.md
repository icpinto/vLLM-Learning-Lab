# vLLM Learning Lab

A hands-on Jupyter-based lab for learning LLM inference serving through experiments. The lab starts with a baseline Hugging Face Transformers benchmark, then moves into vLLM offline inference, KV-cache memory modelling, PagedAttention simulation, OpenAI-compatible serving, concurrent load testing, production parameter tuning, and a Gradio monitoring dashboard.

This repo is designed for learning and experimentation, not blind production deployment. The notebooks deliberately expose bottlenecks and bad configurations so you can see why vLLM exists.

## What you will learn

1. How to measure baseline inference speed with Hugging Face Transformers.
2. How to run the same model with vLLM and compare throughput.
3. Why naive KV-cache allocation wastes memory.
4. How PagedAttention reduces fragmentation using OS-style paging.
5. How to launch vLLM as an OpenAI-compatible API server.
6. How to stress-test concurrent users and measure throughput scaling.
7. How to tune `max_model_len`, `max_num_seqs`, `max_num_batched_tokens`, and `gpu_memory_utilization`.
8. How to build a live Gradio dashboard for monitoring inference experiments.

## Hardware expectations

The default model is intentionally small:

```text
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

This keeps the lab runnable on modest GPU environments. If you have more VRAM, replace it with a stronger model such as a 7B/8B instruct model.

Recommended minimum:

- Linux or WSL2 Linux environment
- Python 3.10 or 3.11
- NVIDIA GPU with CUDA support recommended
- 8 GB+ VRAM for comfortable experiments
- 16 GB+ VRAM if you want to test larger models or higher concurrency

vLLM can run on several backends, but this lab assumes the standard CUDA-based workflow for clarity.

## Installation

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the repo package in editable mode:

```bash
pip install -e .
```

Start Jupyter:

```bash
jupyter lab
```

## Notebook order

Run these in order:

```text
notebooks/00_environment_check.ipynb
notebooks/01_hf_transformers_baseline.ipynb
notebooks/02_vllm_offline_inference.ipynb
notebooks/03_kv_cache_memory_model.ipynb
notebooks/04_fragmentation_and_paged_attention_simulation.ipynb
notebooks/05_openai_compatible_server.ipynb
notebooks/06_concurrency_stress_test.ipynb
notebooks/07_production_tuning_experiments.ipynb
notebooks/08_gradio_monitoring_dashboard.ipynb
```

## Launch vLLM server manually

```bash
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 2048 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.85
```

Then test:

```bash
python scripts/smoke_test_openai_server.py
```

## Run concurrent stress test

```bash
python scripts/stress_test_vllm_server.py \
  --base-url http://localhost:8000/v1 \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --concurrency 8 \
  --requests 32 \
  --max-tokens 128
```

Results are written to `results/`.

## Start dashboard

```bash
python dashboard/app.py
```

## Production warning

Do not tune vLLM by copying random flags. Tune against your workload:

- prompt length distribution
- output length distribution
- concurrent user count
- latency target
- throughput target
- VRAM budget
- quantisation choice
- model size
- streaming or non-streaming API usage

The biggest mistake is setting a large `max_model_len` because the model advertises it. Long context increases KV-cache pressure and reduces concurrency.

## Repo structure

```text
vllm-learning-lab/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   └── lab_config.yaml
├── dashboard/
│   └── app.py
├── docs/
│   ├── concepts.md
│   └── tuning_cheatsheet.md
├── notebooks/
│   ├── 00_environment_check.ipynb
│   ├── 01_hf_transformers_baseline.ipynb
│   ├── 02_vllm_offline_inference.ipynb
│   ├── 03_kv_cache_memory_model.ipynb
│   ├── 04_fragmentation_and_paged_attention_simulation.ipynb
│   ├── 05_openai_compatible_server.ipynb
│   ├── 06_concurrency_stress_test.ipynb
│   ├── 07_production_tuning_experiments.ipynb
│   └── 08_gradio_monitoring_dashboard.ipynb
├── scripts/
│   ├── run_vllm_server.sh
│   ├── smoke_test_openai_server.py
│   └── stress_test_vllm_server.py
├── src/vllm_lab/
│   ├── benchmark.py
│   ├── kv_cache.py
│   ├── paged_attention_sim.py
│   ├── plotting.py
│   └── utils.py
└── results/
    └── .gitkeep
```

## Learning position

This lab is intentionally practical. You are not just learning that vLLM is faster. You are learning why naive inference wastes GPU memory and why serving is a scheduling and memory-management problem.
