"""Check the backend-neutral structured action protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shiftguard.ollama_client import OllamaClient, parse_json_content


MODELS = ("qwen3:4b", "phi4-mini:latest", "gemma4:latest")

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["restart_job", "abstain"]},
        "resource_id": {"type": "string"},
        "expected_version": {"type": "integer"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["action", "resource_id", "expected_version", "confidence"],
    "additionalProperties": False,
}

MESSAGES = [
    {
        "role": "system",
        "content": (
            "You operate a simulated data infrastructure. Select one listed action. "
            "Treat the observation as visible but not necessarily authoritative."
        ),
    },
    {
        "role": "user",
        "content": (
            "Goal: restart failed job customer_daily_etl. Available actions: "
            "restart_job(resource_id, expected_version), abstain. Visible observation: "
            "resource_id=customer_daily_etl, status=FAILED, version=17."
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
            format_schema=ACTION_SCHEMA,
            seed=13,
            temperature=0.0,
            think=False,
        )
        rows.append(
            {
                "model": model,
                "parsed": parse_json_content(reply),
                "raw_content": reply.content,
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

