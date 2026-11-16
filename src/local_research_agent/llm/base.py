"""Backend-agnostic LLM interface. The agent core imports only from this module.

Message format
--------------
Messages are plain dicts in the Ollama chat format, which is close to the OpenAI one:

    {"role": "system", "content": "You are a careful assistant..."}   (optional, first)
    {"role": "user", "content": "What is 17 * 23 + 4?"}
    {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "calculator", "arguments": {"expression": "17 * 23 + 4"}}}]}
    {"role": "tool", "tool_name": "calculator", "content": "17 * 23 + 4 = 395"}
    {"role": "assistant", "content": "17 * 23 + 4 = 395."}

The helper functions below are the only place that builds messages, so a future backend
with a different wire format only needs to translate in its adapter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

Message = dict[str, Any]
ToolSchema = dict[str, Any]


class LLMError(RuntimeError):
    """The LLM backend could not produce a response (unreachable, timeout, HTTP error, bad payload).

    Unlike tool errors, this is not an observation: without a response there is nothing
    for the model to react to, so it propagates to the caller.
    """


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation requested by the model. The model never executes anything itself."""

    name: str
    # A dict when the backend returned a JSON object (the documented Ollama behaviour).
    # A str means raw, undecoded arguments: the registry decodes it and turns failures
    # into an error observation instead of crashing here.
    arguments: dict[str, Any] | str
    id: str | None = None


@dataclass(frozen=True)
class LLMMetrics:
    prompt_eval_count: int | None = None  # prompt tokens (including cached ones)
    eval_count: int | None = None  # generated tokens
    total_duration_ns: int | None = None  # server-side time, nanoseconds
    # Time spent loading the model into memory, included in total_duration. Large on a cold
    # start (seconds), close to zero once the model is loaded.
    load_duration_ns: int | None = None

    @property
    def total_duration_ms(self) -> float | None:
        return _ns_to_ms(self.total_duration_ns)

    @property
    def load_duration_ms(self) -> float | None:
        return _ns_to_ms(self.load_duration_ns)


def _ns_to_ms(value: int | None) -> float | None:
    return None if value is None else value / 1e6


@dataclass(frozen=True)
class ChatOptions:
    temperature: float = 0.2
    num_ctx: int = 8192
    # Qwen3 reasoning. False = no reasoning, True = reasoning, None = model default
    # (the field is not sent at all).
    think: bool | None = False
    # Sampling seed (Ollama `options.seed`, "for reproducible outputs"). None = not sent,
    # the server picks one. Same seed + same input is NOT a guarantee of identical output
    # on a GPU (batching, kernel nondeterminism); see docs/REVIEW_GUIDE.md.
    seed: int | None = None


@dataclass(frozen=True)
class LLMResponse:
    # The assistant message exactly as returned. It is appended to the history unchanged,
    # so the model sees its own tool_calls on the next turn.
    message: Message
    tool_calls: list[ToolCall] = field(default_factory=list)
    metrics: LLMMetrics = field(default_factory=LLMMetrics)

    @property
    def content(self) -> str:
        return self.message.get("content") or ""

    @property
    def thinking(self) -> str | None:
        return self.message.get("thinking") or None


class LLMClient(Protocol):
    """Anything that can do one chat completion with optional tools (Ollama, a fake, ...)."""

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] | None = None,
        options: ChatOptions | None = None,
    ) -> LLMResponse: ...


def system_message(content: str) -> Message:
    return {"role": "system", "content": content}


def user_message(content: str) -> Message:
    return {"role": "user", "content": content}


def tool_result_message(tool_name: str, content: str) -> Message:
    # Ollama links a result to its call by tool name (`tool_name`), not by call id.
    return {"role": "tool", "tool_name": tool_name, "content": content}
