"""Small dependency-free Ollama client used by local experiments."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelReply:
    model: str
    content: str
    tool_calls: tuple[dict[str, Any], ...]
    done_reason: str | None
    prompt_tokens: int
    output_tokens: int
    total_duration_ns: int
    raw: dict[str, Any]


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        seed: int = 13,
        temperature: float = 0.2,
        top_p: float = 0.9,
        num_ctx: int = 4096,
        num_predict: int = 256,
        keep_alive: str = "5m",
        think: bool | None = None,
        format_schema: dict[str, Any] | None = None,
    ) -> ModelReply:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {
                "seed": seed,
                "temperature": temperature,
                "top_p": top_p,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        }
        if tools:
            payload["tools"] = tools
        if think is not None:
            payload["think"] = think
        if format_schema is not None:
            payload["format"] = format_schema
        raw = self._post("/api/chat", payload)
        message = raw.get("message") or {}
        return ModelReply(
            model=raw.get("model", model),
            content=message.get("content", ""),
            tool_calls=tuple(message.get("tool_calls") or ()),
            done_reason=raw.get("done_reason"),
            prompt_tokens=int(raw.get("prompt_eval_count") or 0),
            output_tokens=int(raw.get("eval_count") or 0),
            total_duration_ns=int(raw.get("total_duration") or 0),
            raw=raw,
        )

    def unload(self, model: str) -> None:
        """Unload a resident model without deleting its local files."""

        self._post("/api/generate", {"model": model, "keep_alive": 0})

    def show(self, model: str) -> dict[str, Any]:
        """Return the local model manifest and metadata exposed by Ollama."""

        return self._post("/api/show", {"model": model})


def normalize_tool_call(reply: ModelReply) -> dict[str, Any] | None:
    """Return the first native call in a model-independent shape."""

    if not reply.tool_calls:
        return None
    call = reply.tool_calls[0]
    function = call.get("function") or {}
    return {
        "name": function.get("name"),
        "arguments": function.get("arguments") or {},
    }


def parse_json_content(reply: ModelReply) -> dict[str, Any] | None:
    """Parse a structured-output response without accepting trailing prose."""

    try:
        parsed = json.loads(reply.content)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None
