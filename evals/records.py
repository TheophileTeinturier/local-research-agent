"""Data types shared by the harness: what one run produced, and the verdict of each check.

Everything here is frozen and JSON-friendly, so a `RunRecord` can be stored in SQLite and
loaded back unchanged (`to_trace_json` / the store's loader). `passed` is derived from the
check results and the exclusion flags, never stored independently of them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCallTrace:
    """One tool call handled by the loop (a copy of `ToolCallRecord`, decoupled from it)."""

    iteration: int
    name: str
    arguments: str  # normalized JSON text, exactly as the loop recorded it
    ok: bool
    content: str  # the observation the model received
    repeated: bool = False  # True: not executed, a repetition warning was sent instead
    error_type: str | None = None

    def args(self) -> dict[str, Any]:
        """Decoded arguments; {} when they are not a JSON object (the call then failed)."""
        try:
            value = json.loads(self.arguments)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class RunTrace:
    """What the checks look at: the agent's observable behaviour in one run."""

    question: str
    final_answer: str | None
    stop_reason: str | None  # None only when the harness failed before the run ended
    iterations: int
    tool_calls: tuple[ToolCallTrace, ...]
    tool_errors: int  # failed executed calls (repetition warnings excluded), as AgentMetrics


@dataclass(frozen=True)
class CheckResult:
    """The verdict of one check on one run."""

    name: str  # readable label with params, e.g. "tool_called(tool=calculator, min=1)"
    passed: bool
    heuristic: bool
    detail: str


@dataclass(frozen=True)
class RunRecord:
    """Everything about one (case, repetition) run. Always produced, even on failure."""

    case_id: str
    category: str
    repetition: int
    model: str
    think: bool
    prompt_variant: str
    seed: int | None
    trace: RunTrace
    expected_stop_reason: str
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_latency_ms: float = 0.0
    llm_load_ms: float = 0.0
    wall_time_ms: float = 0.0
    agent_error: str | None = None  # AgentState.error (detail of LLM_ERROR / TIMEOUT)
    fixture_misses: tuple[str, ...] = ()  # arXiv requests no fixture covered
    harness_error: str | None = None  # an exception in the agent or the harness
    checks: tuple[CheckResult, ...] = field(default_factory=tuple)

    @property
    def excluded(self) -> bool:
        """True when the run says nothing about the agent: not counted in pass/fail."""
        return self.harness_error is not None or bool(self.fixture_misses)

    @property
    def passed(self) -> bool | None:
        """All deterministic checks passed. None when the run is excluded."""
        if self.excluded:
            return None
        return all(c.passed for c in self.checks if not c.heuristic)

    @property
    def passed_with_heuristics(self) -> bool | None:
        """All checks passed, heuristic ones included. None when the run is excluded."""
        if self.excluded:
            return None
        return all(c.passed for c in self.checks)

    def to_trace_json(self) -> str:
        """The parts of the record that have no column of their own in the store."""
        return json.dumps(
            {
                "question": self.trace.question,
                "expected_stop_reason": self.expected_stop_reason,
                "agent_error": self.agent_error,
                "fixture_misses": list(self.fixture_misses),
                "tool_calls": [asdict(call) for call in self.trace.tool_calls],
            },
            ensure_ascii=False,
        )
