"""The agent loop: `run_agent`.

One iteration:

    deadline left?  --no-->  TIMEOUT
    call the LLM with the tool schemas   --LLMError / LLM timeout-->  LLM_ERROR (or TIMEOUT)
    append the assistant message (verbatim, with its tool_calls)
    no tool call?   --> FINAL_ANSWER
    last iteration? --> do not execute the tools, MAX_ITERATIONS
    for EACH tool call, in order: exactly one `tool` message
        new call              -> execute with a timeout (errors and timeouts are observations)
        repeated once         -> not executed, warning observation
        repeated again        -> REPETITION
    too many consecutive tool errors? --> TOO_MANY_TOOL_ERRORS
    repeat

Invariants (see docs/REVIEW_GUIDE.md):
- it always terminates: every iteration either makes an LLM call, which increments
  `state.iteration` (bounded by max_iterations), or returns a StopReason;
- it always sets `state.stop_reason` (only in `_finish`, the single exit);
- it never raises for expected failures: `LLMError`, timeouts and every tool failure are
  turned into a StopReason or an observation. Programming errors (a bug in an adapter, an
  invalid `AgentConfig`) still raise, on purpose;
- `AgentState` is the only mutable object, and only this module's main thread mutates it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

from ..llm import (
    ChatOptions,
    LLMClient,
    LLMError,
    LLMResponse,
    Message,
    ToolCall,
    ToolSchema,
    system_message,
    tool_result_message,
)
from ..llm import user_message as make_user_message
from ..observability import log_event
from ..tools import ToolRegistry, ToolResult
from .guards import (
    AgentLimits,
    Repetition,
    check_deadline,
    check_error_budget,
    check_iterations,
    detect_repetition,
    normalize_arguments,
)
from .prompts import SYSTEM_PROMPT
from .state import AgentMetrics, AgentResult, AgentState, LLMCallRecord, StopReason, ToolCallRecord
from .timeouts import CallTimedOut, call_with_timeout

logger = logging.getLogger(__name__)

# Seconds, monotonic. perf_counter rather than time.monotonic: on Windows the latter ticks
# every ~15 ms. Injectable so tests control time and never sleep.
Clock = Callable[[], float]


@dataclass(frozen=True)
class AgentConfig:
    """How to run the agent. Everything here is read-only during a run."""

    system_prompt: str = SYSTEM_PROMPT  # "" = no system message
    allowed_tools: tuple[str, ...] | None = None  # None = every tool of the registry
    limits: AgentLimits = field(default_factory=AgentLimits)
    llm_options: ChatOptions = field(default_factory=ChatOptions)  # num_ctx, temperature, think


@dataclass(frozen=True)
class _Run:
    """The read-only collaborators of one run, bundled to keep helper signatures short."""

    llm: LLMClient
    tools: ToolRegistry  # already restricted to allowed_tools
    schemas: list[ToolSchema]
    config: AgentConfig
    clock: Clock

    @property
    def limits(self) -> AgentLimits:
        return self.config.limits


def run_agent(
    llm: LLMClient,
    registry: ToolRegistry,
    config: AgentConfig,
    user_message: str,
    *,
    clock: Clock = time.perf_counter,
) -> AgentResult:
    """Answer `user_message`, calling tools as the model requests, within `config.limits`.

    Always returns an `AgentResult` with a `stop_reason`; `final_answer` is None unless the
    model actually answered (no extra LLM call is made to force an answer).
    Raises `ValueError` only for a configuration error (unknown name in `allowed_tools`).
    """
    tools = registry if config.allowed_tools is None else registry.subset(config.allowed_tools)
    run = _Run(llm=llm, tools=tools, schemas=tools.schemas(), config=config, clock=clock)
    state = AgentState(
        messages=_initial_messages(config.system_prompt, user_message), started_at=clock()
    )
    log_event(logger, "agent_started", tools=tools.names, limits=asdict(config.limits))

    stop: StopReason | None = None
    while stop is None:
        stop = _run_iteration(state, run)
    return _finish(state, run, stop)


# --- one iteration ------------------------------------------------------------------------


def _run_iteration(state: AgentState, run: _Run) -> StopReason | None:
    """LLM call, then the requested tools. Returns a StopReason when the run must end."""
    stop = check_deadline(state.started_at, run.clock(), run.limits)
    if stop is not None:
        return stop

    response = _call_llm(state, run)
    if isinstance(response, StopReason):
        return response
    # Verbatim, with its tool_calls: the model must see what it asked for to make sense
    # of the tool messages that follow.
    state.messages.append(response.message)

    if not response.tool_calls:
        state.final_answer = response.content
        stop = StopReason.FINAL_ANSWER
    else:
        # At the last iteration no LLM call would ever read the results: the tools are not
        # executed, and `_handle_tool_calls` only answers them with "not executed".
        stop = check_iterations(state.iteration, run.limits)
        stop = _handle_tool_calls(state, run, response.tool_calls, stop)
        stop = stop or check_error_budget(state.consecutive_tool_errors, run.limits)
    _log_iteration(state, stop)
    return stop


def _call_llm(state: AgentState, run: _Run) -> LLMResponse | StopReason:
    """One LLM call with a timeout. Failures become LLM_ERROR (or TIMEOUT), never raise."""
    # Capped by the total budget, so a slow call cannot overshoot the run's deadline.
    timeout_s = min(run.limits.llm_timeout_s, _remaining_s(state, run))
    messages = list(state.messages)  # snapshot: the worker thread never sees later appends
    start = time.perf_counter()
    try:
        response = call_with_timeout(
            lambda: run.llm.chat(messages, tools=run.schemas, options=run.config.llm_options),
            timeout_s,
            name="llm-call",
        )
    except CallTimedOut:
        cut_by_total_budget = timeout_s < run.limits.llm_timeout_s
        stop = StopReason.TIMEOUT if cut_by_total_budget else StopReason.LLM_ERROR
        return _llm_failure(state, stop, f"LLM call timed out after {timeout_s:g} s")
    except LLMError as exc:
        return _llm_failure(state, StopReason.LLM_ERROR, str(exc))

    _record_llm_call(state, response, (time.perf_counter() - start) * 1000, len(messages))
    return response


def _handle_tool_calls(
    state: AgentState, run: _Run, calls: Sequence[ToolCall], stop: StopReason | None
) -> StopReason | None:
    """Give EVERY call exactly one `tool` message, in call order.

    Once a stop reason is known (passed in, or raised by a call of this batch), the remaining
    calls are not executed; they get a short "not executed" message so the history stays
    well formed (one result per call).
    """
    for call in calls:
        stop = stop or check_deadline(state.started_at, run.clock(), run.limits)
        if stop is not None:
            message = f"Error: Not executed: the run stopped ({stop.value})."
            state.messages.append(tool_result_message(call.name, message))
            continue
        stop = _handle_tool_call(state, run, call)
    return stop


def _handle_tool_call(state: AgentState, run: _Run, call: ToolCall) -> StopReason | None:
    """Apply the repetition policy to one call, then execute it if it is new."""
    arguments = normalize_arguments(call.arguments)
    verdict = detect_repetition(state.tool_calls, call.name, arguments, run.limits)

    if verdict is Repetition.STOP:
        message = "Error: Not executed: identical call repeated after a warning. The run stops."
        state.messages.append(tool_result_message(call.name, message))
        return StopReason.REPETITION
    if verdict is Repetition.WARN:
        warning = ToolResult(
            ok=False,
            content=(
                f"Error: Repeated call: you already called {call.name!r} with exactly these "
                "arguments, and its result is in an earlier tool message. Use that result or "
                "give your final answer. Repeating this call again will stop the run."
            ),
            metadata={"error_type": "repeated_call"},
        )
        _record_tool_call(state, call, arguments, warning, repeated=True)
        return None

    _record_tool_call(state, call, arguments, _execute_tool(state, run, call), repeated=False)
    return None


def _execute_tool(state: AgentState, run: _Run, call: ToolCall) -> ToolResult:
    """Run the tool in a worker thread. A timeout becomes an error observation."""
    timeout_s = min(run.limits.tool_timeout_s, _remaining_s(state, run))
    try:
        # registry.execute never raises (except Ctrl-C / SystemExit): unknown tool, bad
        # arguments and tool crashes already come back as ToolResult(ok=False).
        return call_with_timeout(
            lambda: run.tools.execute(call.name, call.arguments),
            timeout_s,
            name=f"tool-{call.name}",
        )
    except CallTimedOut:
        # The thread may still be running: see the limitation in agent/timeouts.py.
        log_event(
            logger, "tool_timeout", level=logging.WARNING, tool=call.name, timeout_s=timeout_s
        )
        return ToolResult(
            ok=False,
            content=f"Error: Tool {call.name!r} timed out after {timeout_s:g} s.",
            metadata={"error_type": "timeout", "duration_ms": timeout_s * 1000},
        )


# --- state updates (records, counters, log events) ------------------------------------------


def _record_llm_call(
    state: AgentState, response: LLMResponse, latency_ms: float, messages_sent: int
) -> None:
    metrics = response.metrics
    state.iteration += 1
    record = LLMCallRecord(
        iteration=state.iteration,
        tool_calls=[call.name for call in response.tool_calls],
        prompt_tokens=metrics.prompt_eval_count or 0,
        completion_tokens=metrics.eval_count or 0,
        latency_ms=round(latency_ms, 1),
        load_ms=round(metrics.load_duration_ms or 0.0, 1),
        messages=messages_sent,
    )
    state.llm_calls.append(record)
    state.prompt_tokens += record.prompt_tokens
    state.completion_tokens += record.completion_tokens
    state.llm_latency_ms += record.latency_ms
    state.llm_load_ms += record.load_ms
    log_event(
        logger,
        "llm_call",
        iteration=state.iteration,
        messages=messages_sent,
        tool_calls=record.tool_calls,
        prompt_eval_count=metrics.prompt_eval_count,
        eval_count=metrics.eval_count,
        total_duration_ms=metrics.total_duration_ms,
        load_duration_ms=metrics.load_duration_ms,
        latency_ms=record.latency_ms,
    )


def _record_tool_call(
    state: AgentState, call: ToolCall, arguments: str, result: ToolResult, *, repeated: bool
) -> None:
    state.tool_calls.append(
        ToolCallRecord(
            iteration=state.iteration,
            name=call.name,
            arguments=arguments,
            ok=result.ok,
            content=result.content,
            repeated=repeated,
            error_type=result.metadata.get("error_type"),
            duration_ms=result.metadata.get("duration_ms", 0.0),
        )
    )
    state.messages.append(tool_result_message(call.name, result.content))
    # A repetition warning is neither a success nor a tool error: it leaves the counter as is.
    if not repeated:
        state.consecutive_tool_errors = 0 if result.ok else state.consecutive_tool_errors + 1


def _llm_failure(state: AgentState, stop: StopReason, error: str) -> StopReason:
    state.error = error
    log_event(logger, "llm_error", level=logging.WARNING, stop_reason=stop.value, error=error)
    return stop


def _finish(state: AgentState, run: _Run, stop: StopReason) -> AgentResult:
    """The single exit of `run_agent`: sets the stop reason and builds the result."""
    state.stop_reason = stop
    if stop is not StopReason.FINAL_ANSWER:
        state.final_answer = None
    metrics = AgentMetrics.from_state(state, run.clock())
    log_event(
        logger, "agent_finished", stop_reason=stop.value, error=state.error, **asdict(metrics)
    )
    return AgentResult(
        final_answer=state.final_answer, stop_reason=stop, state=state, metrics=metrics
    )


# --- small pure helpers ---------------------------------------------------------------------


def _initial_messages(system_prompt: str, user_message: str) -> list[Message]:
    messages = [system_message(system_prompt)] if system_prompt else []
    messages.append(make_user_message(user_message))
    return messages


def _remaining_s(state: AgentState, run: _Run) -> float:
    elapsed = run.clock() - state.started_at
    return max(run.limits.total_timeout_s - elapsed, 0.0)


def _log_iteration(state: AgentState, stop: StopReason | None) -> None:
    current = state.llm_calls[-1]
    previous = state.llm_calls[-2] if len(state.llm_calls) > 1 else None
    calls = [c for c in state.tool_calls if c.iteration == current.iteration]
    log_event(
        logger,
        "agent_iteration",
        iteration=current.iteration,
        tool_calls=[{"name": c.name, "ok": c.ok, "repeated": c.repeated} for c in calls],
        prompt_tokens=current.prompt_tokens,
        completion_tokens=current.completion_tokens,
        # How much the prompt grew since the previous iteration (assistant + tool messages).
        context_growth_tokens=None
        if previous is None
        else (current.prompt_tokens - previous.prompt_tokens),
        messages=len(state.messages),
        stop_reason=None if stop is None else stop.value,
    )
