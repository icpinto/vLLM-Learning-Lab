"""Smoke test for a vLLM OpenAI-compatible API server."""

from __future__ import annotations

import argparse
from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "Explain vLLM in two sentences."}],
        temperature=0.0,
        max_tokens=80,
    )
    print(response.choices[0].message.content)
    print("usage:", response.usage)


if __name__ == "__main__":
    main()
