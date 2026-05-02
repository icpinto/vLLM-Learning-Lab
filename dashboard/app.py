"""Gradio dashboard for vLLM learning-lab experiment results.

The dashboard reads CSV files produced by scripts/stress_test_vllm_server.py.
It does not scrape internal vLLM state. Treat it as an experiment dashboard.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Tuple

import gradio as gr
import pandas as pd

RESULTS_DIR = Path("results")


def load_results() -> pd.DataFrame:
    files = sorted(glob.glob(str(RESULTS_DIR / "stress_concurrency_*.csv")))
    frames = []
    for file in files:
        df = pd.read_csv(file)
        df["source_file"] = Path(file).name
        frames.append(df)
    if not frames:
        return pd.DataFrame(
            columns=[
                "source_file",
                "status_code",
                "latency_s",
                "output_tokens",
                "output_tokens_per_second",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def summarize() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    df = load_results()
    if df.empty:
        return df, pd.DataFrame(), "No stress-test CSV files found in results/. Run scripts/stress_test_vllm_server.py first."

    summary = (
        df.groupby("source_file")
        .agg(
            requests=("source_file", "count"),
            ok=("status_code", lambda s: int((s == 200).sum())),
            latency_mean_s=("latency_s", "mean"),
            latency_p50_s=("latency_s", "median"),
            latency_max_s=("latency_s", "max"),
            total_output_tokens=("output_tokens", "sum"),
            mean_output_tps=("output_tokens_per_second", "mean"),
        )
        .reset_index()
    )
    note = "Dashboard loaded experiment result files. Compare latency and throughput before assuming a configuration is better."
    return df, summary, note


with gr.Blocks(title="vLLM Learning Lab Dashboard") as demo:
    gr.Markdown("# vLLM Learning Lab Monitoring Dashboard")
    gr.Markdown(
        "This dashboard summarises local stress-test CSV files. It is a capstone exercise for observing throughput, latency, and failure patterns."
    )
    refresh = gr.Button("Refresh results")
    raw = gr.Dataframe(label="Raw request-level results")
    summary = gr.Dataframe(label="Run-level summary")
    note = gr.Textbox(label="Interpretation note")
    refresh.click(fn=summarize, outputs=[raw, summary, note])
    demo.load(fn=summarize, outputs=[raw, summary, note])


if __name__ == "__main__":
    demo.launch()
