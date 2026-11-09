"""Guardrails of the agent loop: limits plus small PURE functions.

No function here reads a clock, mutates anything or does I/O. The loop passes in the
values (iteration count, current time, past calls), so each guard is unit-tested directly.
A guard returns a `StopReason` when the run must stop, else None.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .state import StopReason, ToolCallRecord


@dataclass(frozen=True)
class AgentLimits:
    """Budgets of one run. All durations are in seconds."""

    max_iterations: int = 8  # max LLM calls
    # Per LLM call. Must cover a cold start (model loading takes several seconds).
    llm_timeout_s: float = 120.0
    tool_timeout_s: float = 10.0  # per tool call
    total_timeout_s: float = 300.0  # wall clock for the whole run
    # How many identical repeats of a successful call are answered with a warning
    # (not executed). The next identical repeat stops the run with REPETITION.
    max_repeated_calls: int = 1
    max_consecutive_tool_errors: int = 3

    def __post_init__(self) -> None:
        for name in ("max_iterations", "max_consecutive_tool_errors"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.max_repeated_calls < 0:
            raise ValueError("max_repeated_calls must be >= 0")
        for name in ("llm_timeout_s", "tool_timeout_s", "total_timeout_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")


class Repetition(Enum):
    """What to do with a tool call, given the calls already handled."""

    NEW = "new"  # execute it
    WARN = "warn"  # do not execute; tell the model it already has this result
    STOP = "stop"  # repeated again after the warning(s): stop the run


def check_iterations(iterations_done: int, limits: AgentLimits) -> StopReason | None:
    """MAX_ITERATIONS once `max_iterations` LLM calls have been made."""
    return StopReason.MAX_ITERATIONS if iterations_done >= limits.max_iterations else None


def check_deadline(started_at: float, now: float, limits: AgentLimits) -> StopReason | None:
    """TIMEOUT once `total_timeout_s` seconds have elapsed since `started_at`."""
    return StopReason.TIMEOUT if now - started_at >= limits.total_timeout_s else None


def check_error_budget(consecutive_errors: int, limits: AgentLimits) -> StopReason | None:
    """TOO_MANY_TOOL_ERRORS once `max_consecutive_tool_errors` tool calls failed in a row."""
    if consecutive_errors >= limits.max_consecutive_tool_errors:
        return StopReason.TOO_MANY_TOOL_ERRORS
    return None


def normalize_arguments(arguments: dict[str, Any] | str) -> str:
    """Canonical text of tool arguments: key order and whitespace do not matter.

    A string is decoded first when it is JSON, so '{"a": 1}' and {"a": 1} compare equal.
    Values are not rewritten: "17*23" and "17 * 23" stay different calls.
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments.strip()
    return json.dumps(
        arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )


def detect_repetition(
    previous: Sequence[ToolCallRecord], name: str, arguments: str, limits: AgentLimits
) -> Repetition:
    """Classify a call (name + NORMALIZED arguments) against the calls already handled.

    Only successful calls and earlier warnings count. Retrying a call that failed (an error, a
    timeout) is legitimate and is bounded by the error budget instead.
    With the default max_repeated_calls=1: 1st call NEW, 2nd WARN, 3rd STOP.
    """
    repeats = sum(
        1
        for record in previous
        if record.name == name and record.arguments == arguments and (record.ok or record.repeated)
    )
    if repeats == 0:
        return Repetition.NEW
    if repeats <= limits.max_repeated_calls:
        return Repetition.WARN
    return Repetition.STOP
