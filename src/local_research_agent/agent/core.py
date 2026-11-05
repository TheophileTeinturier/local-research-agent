"""`run_once`: a single tool-calling round trip. The full agent loop comes in a later step.

Message sequence (the history grows in this order):

    1. user                       the question
    2. assistant (tool_calls)     the model *asks* for tool calls; nothing runs on its side
    3. tool  (one per call)       we validate + execute each call and append its result
    4. assistant                  the model reads the results and answers in text

If the first response contains no tool call, steps 2-4 collapse into one assistant answer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from ..llm import (
    ChatOptions,
    LLMClient,
    LLMMetrics,
    LLMResponse,
    Message,
    ToolCall,
    ToolSchema,
    tool_result_message,
)
from ..llm import (
    user_message as make_user_message,
)
from ..observability import log_event
from ..tools import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMStep:
    content: str
    tool_calls: list[ToolCall]
    metrics: LLMMetrics
    latency_ms: float  # wall-clock, client side (includes HTTP + model load)
    thinking: str | None = None


@dataclass(frozen=True)
class ToolStep:
    call: ToolCall
    result: ToolResult


Step = LLMStep | ToolStep


@dataclass(frozen=True)
class RunResult:
    answer: str
    trace: list[Step]
    messages: list[Message]  # the full conversation history, in order
    # False when the second response asked for *more* tool calls. run_once does not loop,
    # so they were not executed; the agent loop (next step) will handle this case.
    completed: bool


def run_once(
    llm: LLMClient,
    registry: ToolRegistry,
    user_message: str,
    *,
    options: ChatOptions | None = None,
) -> RunResult:
    messages: list[Message] = [make_user_message(user_message)]
    tools = registry.schemas()
    trace: list[Step] = []

    first = _call_llm(llm, messages, tools, options, trace)
    # The assistant message goes into the history *with* its tool_calls: the model needs to
    # see what it asked for in order to make sense of the tool messages that follow.
    messages.append(first.message)
    if not first.tool_calls:
        return RunResult(answer=first.content, trace=trace, messages=messages, completed=True)

    for call in first.tool_calls:
        # Never raises: unknown tool, bad arguments and tool crashes all come back as
        # ToolResult(ok=False), and the model receives them as observations.
        result = registry.execute(call.name, call.arguments)
        trace.append(ToolStep(call=call, result=result))
        messages.append(tool_result_message(call.name, result.content))

    # Tools are offered again, so this call behaves exactly like one iteration of the
    # future loop. If the model asks for more tools, `completed` reports it.
    final = _call_llm(llm, messages, tools, options, trace)
    messages.append(final.message)
    return RunResult(
        answer=final.content,
        trace=trace,
        messages=messages,
        completed=not final.tool_calls,
    )


def _call_llm(
    llm: LLMClient,
    messages: list[Message],
    tools: list[ToolSchema],
    options: ChatOptions | None,
    trace: list[Step],
) -> LLMResponse:
    start = time.perf_counter()
    # Pass a copy so later appends to the history can't change what the client received.
    response = llm.chat(list(messages), tools=tools, options=options)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    trace.append(
        LLMStep(
            content=response.content,
            tool_calls=response.tool_calls,
            metrics=response.metrics,
            latency_ms=latency_ms,
            thinking=response.thinking,
        )
    )
    log_event(
        logger,
        "llm_call",
        messages=len(messages),
        tool_calls=[call.name for call in response.tool_calls],
        prompt_eval_count=response.metrics.prompt_eval_count,
        eval_count=response.metrics.eval_count,
        total_duration_ms=response.metrics.total_duration_ms,
        latency_ms=latency_ms,
    )
    return response
