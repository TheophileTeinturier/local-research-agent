"""Agent run state: `AgentState` is the ONLY mutable object of a run.

The loop (`agent/loop.py`) mutates it from the main thread only. The records it holds
(`LLMCallRecord`, `ToolCallRecord`) and the outputs (`AgentMetrics`, `AgentResult`) are frozen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ..llm import Message


class StopReason(StrEnum):
    """Why a run ended. Every run ends with exactly one of these."""

    FINAL_ANSWER = "final_answer"  # the model answered without requesting a tool
    MAX_ITERATIONS = "max_iterations"  # the model still wanted tools at the last iteration
    TIMEOUT = "timeout"  # total wall-clock budget exhausted
    REPETITION = "repetition"  # identical tool call repeated after a warning
    LLM_ERROR = "llm_error"  # the backend failed or an LLM call timed out
    TOO_MANY_TOOL_ERRORS = "too_many_tool_errors"  # consecutive failed tool calls


@dataclass(frozen=True)
class LLMCallRecord:
    """One successful LLM call (= one iteration)."""

    iteration: int
    tool_calls: list[str]  # names of the requested tools, empty for a final answer
    prompt_tokens: int  # 0 when the backend did not report it
    completion_tokens: int
    latency_ms: float  # client-side wall clock (HTTP + model load + generation)
    load_ms: float  # server-side model loading time (cold start)
    messages: int  # history length sent to the model


@dataclass(frozen=True)
class ToolCallRecord:
    """One tool call the loop handled: executed, or answered with a repetition warning."""

    iteration: int
    name: str
    arguments: str  # normalized (sorted keys, compact JSON): used to detect repetitions
    ok: bool
    content: str  # the observation sent to the model
    repeated: bool = False  # True: not executed, a repetition warning was sent instead
    error_type: str | None = None
    duration_ms: float = 0.0


@dataclass
class AgentState:
    """Everything that changes during a run."""

    messages: list[Message]
    started_at: float  # value of the run's clock (seconds) when the run started
    iteration: int = 0  # number of successful LLM calls so far
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_latency_ms: float = 0.0
    llm_load_ms: float = 0.0
    consecutive_tool_errors: int = 0
    final_answer: str | None = None
    error: str | None = None  # detail for LLM_ERROR / TIMEOUT
    stop_reason: StopReason | None = None


@dataclass(frozen=True)
class AgentMetrics:
    """Summary numbers of a finished run."""

    iterations: int
    tool_calls: int
    tool_errors: int
    prompt_tokens: int
    completion_tokens: int
    llm_latency_ms: float
    llm_load_ms: float
    wall_time_ms: float

    @classmethod
    def from_state(cls, state: AgentState, finished_at: float) -> AgentMetrics:
        """Build the summary from the final state and the clock value at the end."""
        return cls(
            iterations=state.iteration,
            tool_calls=len(state.tool_calls),
            tool_errors=sum(not c.ok and not c.repeated for c in state.tool_calls),
            prompt_tokens=state.prompt_tokens,
            completion_tokens=state.completion_tokens,
            llm_latency_ms=round(state.llm_latency_ms, 1),
            llm_load_ms=round(state.llm_load_ms, 1),
            wall_time_ms=round((finished_at - state.started_at) * 1000, 1),
        )


@dataclass(frozen=True)
class AgentResult:
    """What `run_agent` returns. `final_answer` is set only for `StopReason.FINAL_ANSWER`."""

    final_answer: str | None
    stop_reason: StopReason
    state: AgentState  # the full trace: messages, LLM calls, tool calls
    metrics: AgentMetrics
