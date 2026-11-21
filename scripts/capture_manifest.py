"""Capture model, prompt, code, and hardware metadata for an experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shiftguard.ollama_client import OllamaClient
from shiftguard.prompts import SYSTEM_PROMPT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["qwen3:4b", "phi4-mini:latest", "gemma4:latest"])
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30).strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def main() -> int:
    args = parse_args()
    client = OllamaClient(timeout=60)
    models = {}
    for model in args.models:
        details = client.show(model)
        modelfile = str(details.get("modelfile", ""))
        from_line = next((line for line in modelfile.splitlines() if line.startswith("FROM ")), "")
        base_blob = from_line.removeprefix("FROM ").strip()
        models[model] = {
            "details": details.get("details"),
            "model_info": details.get("model_info"),
            "capabilities": details.get("capabilities"),
            "modified_at": details.get("modified_at"),
            "base_blob": base_blob,
            "template_sha256": hashlib.sha256(str(details.get("template", "")).encode()).hexdigest(),
            "modelfile_sha256": hashlib.sha256(modelfile.encode()).hexdigest(),
        }
    code_files = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    manifest = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "ollama_version": command_output([r"E:\Ollama\Programs\Ollama\ollama.exe", "--version"]),
        "gpu": command_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "models": models,
        "inference": {
            "temperature": 0.6, "top_p": 0.9, "num_ctx": 4096,
            "num_predict": 128, "think": False, "seeds": [13, 42, 2026],
            "structured_output": "Ollama JSON schema",
        },
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in code_files},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
