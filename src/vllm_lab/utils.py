"""Utility helpers for the vLLM learning lab."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


def load_config(path: str | Path = "configs/lab_config.yaml") -> Dict[str, Any]:
    """Load the lab YAML config."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def now_ms() -> float:
    """Return current wall-clock time in milliseconds."""
    return time.perf_counter() * 1000


def count_tokens(tokenizer: Any, text: str) -> int:
    """Count tokens using a Hugging Face tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_gpu_snapshot() -> Dict[str, Any]:
    """Return a best-effort GPU memory snapshot.

    Uses pynvml if available. Returns a safe dictionary even on CPU-only machines.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            gpus.append(
                {
                    "index": i,
                    "name": name,
                    "memory_total_gb": mem.total / 1e9,
                    "memory_used_gb": mem.used / 1e9,
                    "memory_free_gb": mem.free / 1e9,
                }
            )
        return {"available": True, "gpus": gpus}
    except Exception as exc:  # noqa: BLE001 - notebook helper should not crash
        return {"available": False, "error": str(exc), "gpus": []}


def environment_report() -> Dict[str, Any]:
    """Collect a lightweight environment report for notebook 00."""
    report: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "gpu": get_gpu_snapshot(),
    }
    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        }
    except Exception as exc:  # noqa: BLE001
        report["torch_error"] = str(exc)
    return report


def shell(command: str) -> str:
    """Run a shell command and return combined output.

    This is for notebooks. For production code, prefer subprocess with explicit args.
    """
    result = subprocess.run(
        command,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.stdout
