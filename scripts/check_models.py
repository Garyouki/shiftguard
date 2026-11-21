"""Check native tool calling for the three local model families."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shiftguard.ollama_client import OllamaClient, normalize_tool_call


MODELS = ("qwen3:4b", "phi4-mini:latest", "gemma4:latest")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "restart_job",
            "description": "Restart a failed job. This changes infrastructure state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "expected_version": {"type": "integer"},
                },
                "required": ["job_id", "expected_version"],
            },
        },
    }
]

MESSAGES = [
    {
        "role": "system",
        "content": (
            "You operate a simulated data infrastructure. Use exactly one available "
            "tool when the requested action is supported. Do not invent identifiers."
        ),
    },
    {
        "role": "user",
        "content": (
            "Restart failed job customer_daily_etl. The visible scheduler state says "
            "job_id=customer_daily_etl, status=FAILED, version=17."
        ),
    },
]


def main() -> int:
    client = OllamaClient()
    rows = []
    for model in MODELS:
        reply = client.chat(
            model=model,
            messages=MESSAGES,
            tools=TOOLS,
            seed=13,
            temperature=0.0,
        )
        rows.append(
            {
                "model": model,
                "tool_call": normalize_tool_call(reply),
                "content": reply.content,
                "prompt_tokens": reply.prompt_tokens,
                "output_tokens": reply.output_tokens,
                "seconds": round(reply.total_duration_ns / 1e9, 3),
                "done_reason": reply.done_reason,
            }
        )
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

