"""The guards are pure functions: tested directly, without running the loop."""

import pytest

from local_research_agent.agent import AgentLimits, StopReason, ToolCallRecord
from local_research_agent.agent.guards import (
    Repetition,
    check_deadline,
    check_error_budget,
    check_iterations,
    detect_repetition,
    normalize_arguments,
)

LIMITS = AgentLimits(
    max_iterations=3, total_timeout_s=10, max_repeated_calls=1, max_consecutive_tool_errors=2
)


def record(
    *, ok: bool = True, repeated: bool = False, error_type: str | None = None
) -> ToolCallRecord:
    """A handled call to calc({"x": 1})."""
    return ToolCallRecord(
        iteration=1,
        name="calc",
        arguments='{"x":1}',
        ok=ok,
        content="",
        repeated=repeated,
        error_type=error_type,
    )


def test_defaults() -> None:
    limits = AgentLimits()
    assert limits.max_iterations == 8
    assert limits.max_consecutive_tool_errors == 3
    assert limits.max_repeated_calls == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iterations": 0},
        {"tool_timeout_s": 0},
        {"total_timeout_s": -1},
        {"max_repeated_calls": -1},
        {"max_consecutive_tool_errors": 0},
    ],
)
def test_invalid_limits_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        AgentLimits(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("done", "expected"), [(0, None), (2, None), (3, StopReason.MAX_ITERATIONS)]
)
def test_check_iterations(done: int, expected: StopReason | None) -> None:
    assert check_iterations(done, LIMITS) is expected


@pytest.mark.parametrize(
    ("now", "expected"), [(100.0, None), (109.9, None), (110.0, StopReason.TIMEOUT)]
)
def test_check_deadline(now: float, expected: StopReason | None) -> None:
    assert check_deadline(100.0, now, LIMITS) is expected


@pytest.mark.parametrize(
    ("errors", "expected"), [(0, None), (1, None), (2, StopReason.TOO_MANY_TOOL_ERRORS)]
)
def test_check_error_budget(errors: int, expected: StopReason | None) -> None:
    assert check_error_budget(errors, LIMITS) is expected


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"b": 2, "a": "x y"}, '{"a":"x y","b":2}'),
        ('{ "b": 2, "a": "x y" }', '{"a":"x y","b":2}'),  # JSON string: decoded first
        ("not json ", "not json"),
        ({}, "{}"),
    ],
)
def test_normalize_arguments(arguments: object, expected: str) -> None:
    assert normalize_arguments(arguments) == expected  # type: ignore[arg-type]


def test_detect_repetition_policy() -> None:
    first = record()
    assert detect_repetition([], "calc", '{"x":1}', LIMITS) is Repetition.NEW
    assert detect_repetition([first], "calc", '{"x":1}', LIMITS) is Repetition.WARN
    warned = record(ok=False, repeated=True)
    assert detect_repetition([first, warned], "calc", '{"x":1}', LIMITS) is Repetition.STOP


def test_detect_repetition_needs_same_name_and_args() -> None:
    previous = [record()]
    assert detect_repetition(previous, "calc", '{"x":2}', LIMITS) is Repetition.NEW
    assert detect_repetition(previous, "other", '{"x":1}', LIMITS) is Repetition.NEW


def test_failed_calls_do_not_count_as_repetitions() -> None:
    failed = record(ok=False, error_type="timeout")
    assert detect_repetition([failed, failed], "calc", '{"x":1}', LIMITS) is Repetition.NEW


def test_zero_tolerance_stops_on_first_repeat() -> None:
    strict = AgentLimits(max_repeated_calls=0)
    assert detect_repetition([record()], "calc", '{"x":1}', strict) is Repetition.STOP
