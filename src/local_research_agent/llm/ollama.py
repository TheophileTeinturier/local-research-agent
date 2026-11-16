"""`LLMClient` adapter for Ollama's native `POST /api/chat` endpoint (non-streaming).

Checked against https://docs.ollama.com/api/chat, /capabilities/tool-calling and
/capabilities/thinking, and against a live Ollama 0.40.14 running qwen3:8b (2026-11-04):

- `think` is a top-level request field; `temperature` and `num_ctx` go inside `options`.
- `message.tool_calls[i].function.arguments` is a JSON object, not a string. The live server
  also returns `tool_calls[i].id` and `function.index`, which the docs do not show.
- With think=true, the reasoning is in `message.thinking`, separate from `message.content`.
  With think=false the field is absent.
- All `*_duration` fields are in nanoseconds, including `load_duration` (model loading time).
- Several tool calls in one response are several entries of `message.tool_calls`; each result
  goes back as its own `{"role": "tool", "tool_name": ..., "content": ...}` message, in call
  order. There is no `tool_call_id` in the documented format (re-checked 2026-11-05).
- `seed` is an integer inside `options` ("Random seed used for reproducible outputs"); the
  docs state no determinism guarantee. `keep_alive` is a top-level field (default 5 min);
  it is not sent, so the server default applies (re-checked 2026-11-10).
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

import httpx

from .base import ChatOptions, LLMError, LLMMetrics, LLMResponse, Message, ToolCall, ToolSchema

# 127.0.0.1 rather than "localhost": on Windows, "localhost" tries IPv6 (::1) first, while
# Ollama listens on IPv4 only, which costs ~2 s on every new connection (measured).
DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_PORT = 11434


def normalize_host(host: str) -> str:
    """Accept the same OLLAMA_HOST forms as Ollama itself, e.g. `127.0.0.1:11434`, `localhost`."""
    host = host.strip().rstrip("/")
    if "://" in host:
        return host
    # No scheme: this is the bare host[:port] form Ollama uses. Add Ollama's port if missing.
    return f"http://{host}" if ":" in host else f"http://{host}:{DEFAULT_PORT}"


class OllamaClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        *,
        connect_timeout: float = 5.0,
        read_timeout: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.host = normalize_host(host)
        # read_timeout also applies to write/pool. It must cover model loading (several
        # seconds on a cold start) plus full generation, because stream=false sends nothing
        # until the answer is complete.
        self._http = httpx.Client(
            base_url=self.host,
            timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
            transport=transport,
        )

    @classmethod
    def from_env(cls, model: str | None = None, host: str | None = None) -> OllamaClient:
        """Explicit arguments win over environment variables, which win over defaults."""
        return cls(
            model=model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
            host=host or os.environ.get("OLLAMA_HOST", DEFAULT_HOST),
            connect_timeout=float(os.environ.get("OLLAMA_CONNECT_TIMEOUT", 5.0)),
            read_timeout=float(os.environ.get("OLLAMA_READ_TIMEOUT", 300.0)),
        )

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] | None = None,
        options: ChatOptions | None = None,
    ) -> LLMResponse:
        opts = options or ChatOptions()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "options": {"temperature": opts.temperature, "num_ctx": opts.num_ctx},
        }
        if opts.seed is not None:
            payload["options"]["seed"] = opts.seed  # a model option, like temperature
        if opts.think is not None:
            payload["think"] = opts.think
        if tools:
            payload["tools"] = list(tools)

        try:
            response = self._http.post("/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMError(f"Ollama request timed out ({self.host}): {exc!r}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Cannot reach Ollama at {self.host}: {exc!r}") from exc

        if response.status_code != 200:
            raise LLMError(f"Ollama returned HTTP {response.status_code}: {response.text[:500]}")
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError(f"Ollama returned invalid JSON: {response.text[:200]!r}") from exc
        return parse_chat_response(data)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def parse_chat_response(data: Any) -> LLMResponse:
    """Translate a non-streaming /api/chat response body into an `LLMResponse`."""
    message = data.get("message") if isinstance(data, dict) else None
    if not isinstance(message, dict):
        raise LLMError(f"Ollama response has no 'message' object: {str(data)[:200]}")

    tool_calls = [_parse_tool_call(raw) for raw in message.get("tool_calls") or []]
    metrics = LLMMetrics(
        prompt_eval_count=_int_or_none(data.get("prompt_eval_count")),
        eval_count=_int_or_none(data.get("eval_count")),
        total_duration_ns=_int_or_none(data.get("total_duration")),
        load_duration_ns=_int_or_none(data.get("load_duration")),
    )
    return LLMResponse(message=message, tool_calls=tool_calls, metrics=metrics)


def _parse_tool_call(raw: Any) -> ToolCall:
    function = raw.get("function") if isinstance(raw, dict) else None
    if not isinstance(function, dict):
        # The model asked for something malformed: keep it so the registry can report it.
        return ToolCall(name="", arguments=json.dumps(raw, default=str))

    arguments = function.get("arguments")
    if arguments is None:
        arguments = {}
    elif not isinstance(arguments, dict | str):
        # e.g. a JSON array: keep it as raw text so the registry rejects it with a clear error.
        arguments = json.dumps(arguments, default=str)
    return ToolCall(name=str(function.get("name") or ""), arguments=arguments, id=raw.get("id"))


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
