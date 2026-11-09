"""Test doubles. `FakeLLM` satisfies the `LLMClient` protocol structurally (no inheritance)."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from local_research_agent.llm import (
    ChatOptions,
    LLMMetrics,
    LLMResponse,
    Message,
    ToolCall,
    ToolSchema,
)


@dataclass
class RecordedCall:
    messages: list[Message]
    tools: list[ToolSchema] | None
    options: ChatOptions | None


class FakeLLM:
    """Replays a scripted sequence of responses and records every call it receives.

    A script item that is an exception is raised instead of returned (e.g. an `LLMError`).
    """

    def __init__(self, responses: Sequence[LLMResponse | BaseException]) -> None:
        self._responses = list(responses)
        self.calls: list[RecordedCall] = []

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] | None = None,
        options: ChatOptions | None = None,
    ) -> LLMResponse:
        if len(self.calls) >= len(self._responses):
            raise AssertionError(f"FakeLLM script exhausted after {len(self._responses)} calls")
        # Deep copy: assert on what the client saw *at call time*.
        self.calls.append(
            RecordedCall(copy.deepcopy(list(messages)), copy.deepcopy(tools), options)
        )
        response = self._responses[len(self.calls) - 1]
        if isinstance(response, BaseException):
            raise response
        return response


class FakeClock:
    """Injectable monotonic clock: time only moves when a test calls `advance`."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def text_response(content: str) -> LLMResponse:
    return LLMResponse(
        message={"role": "assistant", "content": content},
        metrics=LLMMetrics(
            prompt_eval_count=10, eval_count=5, total_duration_ns=1_000_000, load_duration_ns=0
        ),
    )


def tool_call_response(*calls: tuple[str, dict[str, Any] | str]) -> LLMResponse:
    """An assistant response requesting tool calls, shaped like a real Ollama message."""
    tool_calls = [
        ToolCall(name=name, arguments=args, id=f"call_{i}") for i, (name, args) in enumerate(calls)
    ]
    raw = [
        {"id": c.id, "function": {"index": i, "name": c.name, "arguments": c.arguments}}
        for i, c in enumerate(tool_calls)
    ]
    return LLMResponse(
        message={"role": "assistant", "content": "", "tool_calls": raw},
        tool_calls=tool_calls,
        metrics=LLMMetrics(prompt_eval_count=20, eval_count=8, total_duration_ns=2_000_000),
    )
